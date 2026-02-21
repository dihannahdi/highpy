"""
Deoptimizer
===========

Provides safe fallback from optimized code to interpreted execution
when speculative assumptions are violated at runtime.

This is critical for correctness: the optimizer may speculate that
a variable always holds an int, but if a float appears, we must
gracefully deoptimize rather than produce wrong results.

Design
------
Each optimized function carries a set of *guards*. A guard is a
predicate checked before (or during) execution. If any guard fails:
  1. The optimized variant is discarded
  2. Execution falls back to the original function
  3. A deoptimization event is recorded for profiling

Guard types:
  - TypeGuard: checks that argument types match expectations
  - ShapeGuard: checks object layout (e.g., __dict__ keys)
  - ValueGuard: checks that a value is within expected range
  - StabilityGuard: checks that a global/free variable hasn't changed
"""

import functools
import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type


class GuardKind(Enum):
    TYPE = auto()
    SHAPE = auto()
    VALUE = auto()
    STABILITY = auto()


@dataclass
class Guard:
    """A single deoptimization guard."""
    kind: GuardKind
    description: str
    check: Callable[..., bool]
    
    def test(self, *args, **kwargs) -> bool:
        try:
            return self.check(*args, **kwargs)
        except Exception:
            return False


@dataclass
class DeoptEvent:
    """Record of a deoptimization event."""
    func_name: str
    guard_kind: GuardKind
    guard_description: str
    args_repr: str
    
    def __str__(self):
        return (
            f"Deopt[{self.func_name}]: {self.guard_kind.name} guard failed "
            f"- {self.guard_description} (args: {self.args_repr})"
        )


class Deoptimizer:
    """
    Manages deoptimization for speculatively optimized functions.
    
    Usage:
        >>> deopt = Deoptimizer()
        >>> 
        >>> # Create a guarded optimized function
        >>> def fast_add(x, y):
        ...     return x + y  # Optimized for int+int
        ...
        >>> def slow_add(x, y):
        ...     return x + y  # General fallback
        ...
        >>> guarded = deopt.guard(
        ...     optimized=fast_add,
        ...     fallback=slow_add,
        ...     guards=[
        ...         deopt.type_guard(0, int),
        ...         deopt.type_guard(1, int),
        ...     ]
        ... )
        >>> guarded(1, 2)      # Uses fast_add
        3
        >>> guarded(1.0, 2.0)  # Deoptimizes to slow_add
        3.0
    """
    
    MAX_DEOPT_EVENTS = 1000
    
    def __init__(self):
        self.events: List[DeoptEvent] = []
        self.deopt_counts: Dict[str, int] = {}
    
    def guard(
        self,
        optimized: Callable,
        fallback: Callable,
        guards: List[Guard],
        max_deopts: int = 3,
    ) -> Callable:
        """
        Create a guarded function that deoptimizes on guard failure.
        
        After max_deopts failures, permanently falls back to the
        unoptimized version.
        """
        func_name = getattr(optimized, '__qualname__', str(optimized))
        deopt_count = [0]
        permanently_deopted = [False]
        
        @functools.wraps(fallback)
        def wrapper(*args, **kwargs):
            if permanently_deopted[0]:
                return fallback(*args, **kwargs)
            
            # Check all guards
            for g in guards:
                if not g.test(*args, **kwargs):
                    # Guard failed — deoptimize
                    deopt_count[0] += 1
                    self.deopt_counts[func_name] = deopt_count[0]
                    
                    event = DeoptEvent(
                        func_name=func_name,
                        guard_kind=g.kind,
                        guard_description=g.description,
                        args_repr=repr(args[:3]),
                    )
                    self.events.append(event)
                    if len(self.events) > self.MAX_DEOPT_EVENTS:
                        self.events = self.events[-500:]
                    
                    if deopt_count[0] >= max_deopts:
                        permanently_deopted[0] = True
                    
                    return fallback(*args, **kwargs)
            
            # All guards passed — use optimized version
            try:
                return optimized(*args, **kwargs)
            except Exception:
                # Runtime failure in optimized code
                deopt_count[0] += 1
                self.deopt_counts[func_name] = deopt_count[0]
                if deopt_count[0] >= max_deopts:
                    permanently_deopted[0] = True
                return fallback(*args, **kwargs)
        
        wrapper.__highpy_guarded__ = True
        wrapper.__highpy_guards__ = guards
        return wrapper
    
    # ------------------------------------------------------------------
    # Guard constructors
    # ------------------------------------------------------------------
    
    def type_guard(self, arg_index: int, expected_type: type) -> Guard:
        """Guard that checks the type of a positional argument."""
        def check(*args, **kwargs):
            if arg_index >= len(args):
                return False
            return type(args[arg_index]) is expected_type
        
        return Guard(
            kind=GuardKind.TYPE,
            description=f"arg[{arg_index}] is {expected_type.__name__}",
            check=check,
        )
    
    def shape_guard(self, arg_index: int, expected_attrs: Set[str]) -> Guard:
        """Guard that checks an object has specific attributes."""
        def check(*args, **kwargs):
            if arg_index >= len(args):
                return False
            obj = args[arg_index]
            obj_dict = getattr(obj, '__dict__', {})
            return expected_attrs.issubset(obj_dict.keys())
        
        return Guard(
            kind=GuardKind.SHAPE,
            description=f"arg[{arg_index}] has attrs {expected_attrs}",
            check=check,
        )
    
    def value_guard(
        self,
        arg_index: int,
        min_val: Any = None,
        max_val: Any = None,
    ) -> Guard:
        """Guard that checks a value is within range."""
        def check(*args, **kwargs):
            if arg_index >= len(args):
                return False
            val = args[arg_index]
            if min_val is not None and val < min_val:
                return False
            if max_val is not None and val > max_val:
                return False
            return True
        
        return Guard(
            kind=GuardKind.VALUE,
            description=f"arg[{arg_index}] in [{min_val}, {max_val}]",
            check=check,
        )
    
    def stability_guard(self, module: Any, name: str) -> Guard:
        """Guard that checks a module-level variable hasn't changed."""
        original = getattr(module, name)
        original_id = id(original)
        
        def check(*args, **kwargs):
            current = getattr(module, name, None)
            return id(current) == original_id
        
        return Guard(
            kind=GuardKind.STABILITY,
            description=f"{getattr(module, '__name__', '?')}.{name} unchanged",
            check=check,
        )
    
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deoptimization statistics."""
        return {
            'total_deopts': sum(self.deopt_counts.values()),
            'per_function': dict(self.deopt_counts),
            'recent_events': [str(e) for e in self.events[-10:]],
        }
    
    def get_hot_deopts(self, threshold: int = 2) -> List[str]:
        """Get functions that deoptimize frequently."""
        return [
            name for name, count in self.deopt_counts.items()
            if count >= threshold
        ]
