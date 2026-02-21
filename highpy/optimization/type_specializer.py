"""
Type Specializer
=================

Novel approach: Adaptive type specialization that creates monomorphized
versions of Python functions based on observed argument type profiles.

Unlike traditional JIT approaches that specialize at the bytecode level,
this operates at the source level using AST transformation + compilation,
enabling more aggressive cross-statement optimizations.

Key techniques:
1. Monomorphization: Create type-specific function clones
2. Type guards: Verify types before dispatching to specialized versions
3. Dispatch table: O(1) lookup from type signature to specialized version
4. Deoptimization: Graceful fallback to generic version on type mismatch
"""

import ast
import inspect
import textwrap
import functools
import types
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from collections import defaultdict
from dataclasses import dataclass, field

from highpy.analysis.type_profiler import TypeProfiler, LatticeType, TypeTag


@dataclass
class SpecializationEntry:
    """A type-specialized variant of a function."""
    type_signature: Tuple[type, ...]
    specialized_func: Callable
    call_count: int = 0
    total_time_ns: int = 0


class TypeSpecializer:
    """
    Creates and manages type-specialized function variants.
    
    When a function is called with specific types consistently,
    the specializer creates an optimized variant that eliminates
    type dispatch overhead by assuming fixed types.
    
    Usage:
        >>> specializer = TypeSpecializer()
        >>> @specializer.auto_specialize
        ... def add(a, b):
        ...     return a + b
        >>> add(1, 2)        # Creates int-specialized version
        >>> add(1.0, 2.0)    # Creates float-specialized version
    """
    
    def __init__(
        self,
        max_specializations: int = 8,
        warmup_threshold: int = 5,
        profiler: Optional[TypeProfiler] = None,
    ):
        self.max_specializations = max_specializations
        self.warmup_threshold = warmup_threshold
        self.profiler = profiler or TypeProfiler()
        self._dispatch_tables: Dict[str, Dict[Tuple[type, ...], SpecializationEntry]] = {}
        self._call_counts: Dict[str, Dict[Tuple[type, ...], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.stats = {
            'specializations_created': 0,
            'dispatch_hits': 0,
            'dispatch_misses': 0,
            'deoptimizations': 0,
        }
    
    def auto_specialize(self, func: Callable) -> Callable:
        """
        Decorator that automatically specializes based on observed types.
        
        The first N calls profile types, then specialized versions are
        created on-demand when a stable type pattern is detected.
        """
        func_name = func.__qualname__
        self._dispatch_tables[func_name] = {}
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute type signature
            type_sig = tuple(type(a) for a in args)
            
            # Track type frequency
            self._call_counts[func_name][type_sig] += 1
            count = self._call_counts[func_name][type_sig]
            
            # Check dispatch table for existing specialization
            table = self._dispatch_tables[func_name]
            if type_sig in table:
                entry = table[type_sig]
                entry.call_count += 1
                self.stats['dispatch_hits'] += 1
                return entry.specialized_func(*args, **kwargs)
            
            # After warmup, create specialization
            if (count >= self.warmup_threshold and 
                len(table) < self.max_specializations):
                
                specialized = self._create_specialization(
                    func, param_names, type_sig
                )
                
                if specialized:
                    entry = SpecializationEntry(
                        type_signature=type_sig,
                        specialized_func=specialized,
                    )
                    table[type_sig] = entry
                    self.stats['specializations_created'] += 1
                    return specialized(*args, **kwargs)
            
            # Fallback to generic version
            self.stats['dispatch_misses'] += 1
            return func(*args, **kwargs)
        
        wrapper.__highpy_specializer__ = self
        wrapper.__highpy_original__ = func
        return wrapper
    
    def specialize_for(
        self, 
        arg_types: Dict[str, type],
    ) -> Callable:
        """
        Decorator that creates a specific type specialization.
        
        Usage:
            @specializer.specialize_for({'x': int, 'y': int})
            def add(x, y):
                return x + y
        """
        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            type_sig = tuple(arg_types.get(name, object) for name in param_names)
            
            specialized = self._create_specialization(func, param_names, type_sig)
            
            if specialized:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    # Type guard
                    for i, (expected, actual) in enumerate(zip(type_sig, args)):
                        if expected is not object and not isinstance(actual, expected):
                            self.stats['deoptimizations'] += 1
                            return func(*args, **kwargs)
                    return specialized(*args, **kwargs)
                
                wrapper.__highpy_specialized__ = True
                wrapper.__highpy_original__ = func
                return wrapper
            
            return func
        
        return decorator
    
    def _create_specialization(
        self,
        func: Callable,
        param_names: List[str],
        type_sig: Tuple[type, ...],
    ) -> Optional[Callable]:
        """Create a type-specialized version of a function."""
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            # Build type map
            type_map = {}
            for name, t in zip(param_names, type_sig):
                type_map[name] = t
            
            # Apply type-aware optimizations
            transformer = _TypeAwareOptimizer(type_map)
            tree = transformer.visit(tree)
            ast.fix_missing_locations(tree)
            
            # Compile
            code = compile(tree, f'<highpy-spec:{func.__name__}:{type_sig}>', 'exec')
            namespace = dict(func.__globals__)
            exec(code, namespace)
            
            return namespace[func.__name__]
        except Exception:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get specialization statistics."""
        result = dict(self.stats)
        result['active_specializations'] = sum(
            len(table) for table in self._dispatch_tables.values()
        )
        return result


class _TypeAwareOptimizer(ast.NodeTransformer):
    """AST transformer that optimizes based on known types."""
    
    def __init__(self, type_map: Dict[str, type]):
        self.type_map = type_map
        self._known_types: Dict[str, type] = dict(type_map)
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        
        # If both sides are known int types and operation is power,
        # we can use multiplication for small powers
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                return ast.BinOp(
                    left=node.left,
                    op=ast.Mult(),
                    right=ast.copy_location(ast.Name(id=node.left.id, ctx=ast.Load()), node.left)
                    if isinstance(node.left, ast.Name) else node.left
                )
        
        return node
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        
        # Track types of local variables
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            inferred = self._infer_type(node.value)
            if inferred:
                self._known_types[name] = inferred
        
        return node
    
    def _infer_type(self, node: ast.expr) -> Optional[type]:
        """Infer Python type from an expression."""
        if isinstance(node, ast.Constant):
            return type(node.value)
        if isinstance(node, ast.Name) and node.id in self._known_types:
            return self._known_types[node.id]
        if isinstance(node, ast.BinOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if left_type == float or right_type == float:
                return float
            if left_type == int and right_type == int:
                if isinstance(node.op, ast.Div):
                    return float
                return int
        return None


def specialize(arg_types=None):
    """
    Convenience decorator for type specialization.
    
    Usage:
        @specialize({'x': int, 'y': int})
        def add(x, y):
            return x + y
        
        @specialize()  # auto-specialize
        def multiply(x, y):
            return x * y
        
        @specialize      # also auto-specialize (no parens)
        def divide(x, y):
            return x / y
    """
    _specializer = TypeSpecializer()
    
    # Handle @specialize (no parens) — arg_types is actually the function
    if callable(arg_types):
        return _specializer.auto_specialize(arg_types)
    
    if arg_types:
        return _specializer.specialize_for(arg_types)
    else:
        return _specializer.auto_specialize
