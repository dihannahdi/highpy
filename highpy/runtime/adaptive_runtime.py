"""
Adaptive Runtime Engine
=======================

A multi-tier adaptive execution engine that dynamically selects
the best execution strategy for each function based on runtime
profiling data.

Architecture
------------
The engine operates in tiers modeled after production JIT compilers:

  Tier 0: Interpreted (vanilla Python) — cold code
  Tier 1: Bytecode-optimized — warm code (>= warmup calls)
  Tier 2: Type-specialized — hot code with stable types
  Tier 3: Native-compiled — very hot numerical loops

Tier promotion is driven by:
  - Invocation count
  - Type stability (monomorphism across invocations)
  - Computational intensity (loop depth, arithmetic density)

Novel contribution: Automatic tier selection with speculative
optimization and deoptimization support, operating entirely within
standard CPython without interpreter modifications.
"""

import ast
import dis
import functools
import inspect
import textwrap
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from ..analysis.type_profiler import TypeProfiler, LatticeType, TypeTag
from ..compiler.ast_optimizer import ASTOptimizer
from ..compiler.bytecode_rewriter import BytecodeRewriter
from ..optimization.type_specializer import TypeSpecializer
from ..compiler.native_codegen import NativeCompiler

T = TypeVar('T')
logger = logging.getLogger(__name__)


class Tier(IntEnum):
    INTERPRETED = 0
    BYTECODE_OPTIMIZED = 1
    TYPE_SPECIALIZED = 2
    NATIVE_COMPILED = 3


@dataclass
class RuntimeProfile:
    """Runtime profiling data for a single function."""
    call_count: int = 0
    total_time_ns: int = 0
    avg_time_ns: float = 0.0
    current_tier: Tier = Tier.INTERPRETED
    type_stable: bool = False
    type_history: List[Tuple[tuple, type]] = field(default_factory=list)
    last_arg_types: Optional[tuple] = None
    polymorphism_degree: int = 0
    has_loops: bool = False
    loop_depth: int = 0
    deopt_count: int = 0
    optimized_variant: Optional[Callable] = None
    native_variant: Optional[Callable] = None
    bytecode_variant: Optional[Callable] = None


class AdaptiveRuntime:
    """
    Multi-tier adaptive execution engine.
    
    Monitors function execution and automatically promotes hot functions
    to higher optimization tiers when profiling data suggests benefit.
    
    Usage:
        >>> runtime = AdaptiveRuntime()
        >>> @runtime.optimize
        ... def compute(x, y):
        ...     total = 0
        ...     for i in range(x):
        ...         total += i * y
        ...     return total
        >>> compute(1000000, 42)  # First calls: interpreted
        >>> compute(1000000, 42)  # After warmup: auto-promoted
    """
    
    # Tier promotion thresholds
    TIER1_THRESHOLD = 5       # Calls before bytecode optimization
    TIER2_THRESHOLD = 20      # Calls before type specialization
    TIER3_THRESHOLD = 50      # Calls before native compilation
    TYPE_STABILITY_WINDOW = 10  # Consecutive calls with same types
    MAX_DEOPT = 3             # Maximum deoptimizations before giving up
    
    def __init__(
        self,
        tier1_threshold: int = 5,
        tier2_threshold: int = 20,
        tier3_threshold: int = 50,
        enable_native: bool = True,
        enable_logging: bool = False,
    ):
        self.TIER1_THRESHOLD = tier1_threshold
        self.TIER2_THRESHOLD = tier2_threshold
        self.TIER3_THRESHOLD = tier3_threshold
        self.enable_native = enable_native
        self.profiles: Dict[str, RuntimeProfile] = {}
        
        # Sub-engines
        self._type_profiler = TypeProfiler()
        self._ast_optimizer = ASTOptimizer()
        self._bytecode_rewriter = BytecodeRewriter()
        self._type_specializer = TypeSpecializer()
        self._native_compiler = NativeCompiler()
        
        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)
    
    def optimize(self, func: Callable = None, **kwargs) -> Callable:
        """
        Decorator that wraps a function with adaptive optimization.
        
        Usage:
            @runtime.optimize
            def f(x): ...
            
            @runtime.optimize(tier3_threshold=100)
            def f(x): ...
        """
        if func is None:
            return lambda f: self.optimize(f, **kwargs)
        
        func_id = f"{func.__module__}.{func.__qualname__}"
        self.profiles[func_id] = RuntimeProfile()
        
        # Pre-analyze function
        self._pre_analyze(func, func_id)
        
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return self._dispatch(func, func_id, args, kw)
        
        wrapper.__highpy_optimized__ = True
        wrapper.__highpy_func_id__ = func_id
        wrapper.__highpy_runtime__ = self
        wrapper.__wrapped__ = func
        return wrapper
    
    def get_profile(self, func_or_id) -> RuntimeProfile:
        """Get the runtime profile for a function."""
        if isinstance(func_or_id, str):
            return self.profiles.get(func_or_id)
        func_id = getattr(func_or_id, '__highpy_func_id__', None)
        if func_id:
            return self.profiles.get(func_id)
        return None
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tracked functions."""
        stats = {}
        for func_id, profile in self.profiles.items():
            stats[func_id] = {
                'call_count': profile.call_count,
                'tier': profile.current_tier.name,
                'avg_time_ns': profile.avg_time_ns,
                'type_stable': profile.type_stable,
                'deopt_count': profile.deopt_count,
            }
        return stats
    
    def _pre_analyze(self, func: Callable, func_id: str):
        """Pre-analyze function for optimization potential."""
        profile = self.profiles[func_id]
        
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    profile.has_loops = True
                    # Simple loop depth detection
                    depth = 0
                    parent = node
                    for child in ast.walk(node):
                        if isinstance(child, (ast.For, ast.While)) and child is not node:
                            depth += 1
                    profile.loop_depth = max(profile.loop_depth, depth + 1)
        except (TypeError, OSError):
            pass
    
    def _dispatch(self, func: Callable, func_id: str, args: tuple, kwargs: dict) -> Any:
        """
        Central dispatch: profile, potentially promote tier, execute.
        """
        profile = self.profiles[func_id]
        profile.call_count += 1
        
        # Track argument types
        arg_types = tuple(type(a).__name__ for a in args)
        
        # Check type stability
        if profile.last_arg_types is not None:
            if arg_types == profile.last_arg_types:
                stable_count = 0
                for entry in reversed(profile.type_history[-self.TYPE_STABILITY_WINDOW:]):
                    if entry[0] == arg_types:
                        stable_count += 1
                    else:
                        break
                profile.type_stable = stable_count >= self.TYPE_STABILITY_WINDOW - 1
            else:
                profile.type_stable = False
                profile.polymorphism_degree = len(
                    set(t for t, _ in profile.type_history[-20:])
                )
        
        profile.last_arg_types = arg_types
        profile.type_history.append((arg_types, None))
        if len(profile.type_history) > 100:
            profile.type_history = profile.type_history[-50:]
        
        # Tier promotion logic
        self._maybe_promote(func, func_id, profile)
        
        # Execute at current tier
        start = time.perf_counter_ns()
        try:
            result = self._execute_at_tier(func, func_id, profile, args, kwargs)
        except Exception as e:
            # Deoptimize on failure
            if profile.current_tier > Tier.INTERPRETED:
                profile.deopt_count += 1
                profile.current_tier = Tier.INTERPRETED
                logger.debug(f"Deoptimized {func_id}: {e}")
                result = func(*args, **kwargs)
            else:
                raise
        
        elapsed = time.perf_counter_ns() - start
        profile.total_time_ns += elapsed
        profile.avg_time_ns = profile.total_time_ns / profile.call_count
        
        return result
    
    def _maybe_promote(self, func: Callable, func_id: str, profile: RuntimeProfile):
        """Decide whether to promote to a higher tier."""
        cc = profile.call_count
        
        if profile.deopt_count >= self.MAX_DEOPT:
            return
        
        if profile.current_tier == Tier.INTERPRETED and cc >= self.TIER1_THRESHOLD:
            self._promote_to_tier1(func, func_id, profile)
        
        elif profile.current_tier == Tier.BYTECODE_OPTIMIZED and cc >= self.TIER2_THRESHOLD:
            if profile.type_stable:
                self._promote_to_tier2(func, func_id, profile)
        
        elif (
            profile.current_tier == Tier.TYPE_SPECIALIZED
            and cc >= self.TIER3_THRESHOLD
            and self.enable_native
            and profile.has_loops
            and profile.type_stable
        ):
            self._promote_to_tier3(func, func_id, profile)
    
    def _promote_to_tier1(self, func: Callable, func_id: str, profile: RuntimeProfile):
        """Promote to Tier 1: bytecode-optimized."""
        try:
            optimized = self._ast_optimizer.optimize(func)
            profile.bytecode_variant = optimized
            profile.current_tier = Tier.BYTECODE_OPTIMIZED
            logger.debug(f"Promoted {func_id} to Tier 1 (bytecode-optimized)")
        except Exception as e:
            logger.debug(f"Tier 1 promotion failed for {func_id}: {e}")
    
    def _promote_to_tier2(self, func: Callable, func_id: str, profile: RuntimeProfile):
        """Promote to Tier 2: type-specialized."""
        try:
            base = profile.bytecode_variant or func
            specialized = self._type_specializer.auto_specialize(base)
            profile.optimized_variant = specialized
            profile.current_tier = Tier.TYPE_SPECIALIZED
            logger.debug(f"Promoted {func_id} to Tier 2 (type-specialized)")
        except Exception as e:
            logger.debug(f"Tier 2 promotion failed for {func_id}: {e}")
    
    def _promote_to_tier3(self, func: Callable, func_id: str, profile: RuntimeProfile):
        """Promote to Tier 3: native-compiled."""
        try:
            native = self._native_compiler.compile(func)
            if native is not None:
                profile.native_variant = native
                profile.current_tier = Tier.NATIVE_COMPILED
                logger.debug(f"Promoted {func_id} to Tier 3 (native-compiled)")
        except Exception as e:
            logger.debug(f"Tier 3 promotion failed for {func_id}: {e}")
    
    def _execute_at_tier(
        self,
        func: Callable,
        func_id: str,
        profile: RuntimeProfile,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Execute function at its current optimization tier."""
        tier = profile.current_tier
        
        if tier == Tier.NATIVE_COMPILED and profile.native_variant:
            try:
                return profile.native_variant(*args, **kwargs)
            except (TypeError, ValueError):
                # Type guard failure — fall through
                pass
        
        if tier >= Tier.TYPE_SPECIALIZED and profile.optimized_variant:
            try:
                return profile.optimized_variant(*args, **kwargs)
            except (TypeError, ValueError):
                pass
        
        if tier >= Tier.BYTECODE_OPTIMIZED and profile.bytecode_variant:
            return profile.bytecode_variant(*args, **kwargs)
        
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_runtime = AdaptiveRuntime()


def optimize(func: Callable = None, **kwargs) -> Callable:
    """
    Module-level decorator for adaptive optimization.
    
    Usage:
        from highpy import optimize
        
        @optimize
        def compute(x, y):
            total = 0
            for i in range(x):
                total += i * y
            return total
    """
    if func is None:
        return lambda f: optimize(f, **kwargs)
    return _default_runtime.optimize(func)


def jit(func: Callable = None, **kwargs) -> Callable:
    """
    JIT-compile a function with aggressive optimization.
    
    Equivalent to optimize() with lower promotion thresholds
    and native compilation enabled.
    
    Usage:
        from highpy import jit
        
        @jit
        def numerical_kernel(n):
            total = 0.0
            for i in range(n):
                total += i * 0.5
            return total
    """
    aggressive_runtime = AdaptiveRuntime(
        tier1_threshold=1,
        tier2_threshold=2,
        tier3_threshold=5,
        enable_native=True,
    )
    if func is None:
        return lambda f: aggressive_runtime.optimize(f, **kwargs)
    return aggressive_runtime.optimize(func)
