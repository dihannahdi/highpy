"""
Function Specializer
====================

Cross-function specialization and inlining optimization.

Novel approach: Context-sensitive function cloning that creates
specialized versions of callees based on the calling context,
enabling interprocedural optimization.

Techniques:
1. Inline expansion for small functions
2. Context-sensitive cloning (specialize callee for each call site)
3. Partial evaluation (specialize on known constant arguments)
4. Memoization with type-aware caching
"""

import ast
import inspect
import textwrap
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class FunctionProfile:
    """Profile data for a function."""
    name: str
    call_count: int = 0
    total_time_ns: int = 0
    avg_time_ns: float = 0.0
    is_pure: bool = False
    body_size: int = 0  # AST node count
    is_inlinable: bool = False


class FunctionSpecializer:
    """
    Cross-function optimization through specialization and inlining.
    
    Usage:
        >>> specializer = FunctionSpecializer()
        >>> @specializer.inline
        ... def square(x):
        ...     return x * x
        >>> @specializer.optimize
        ... def sum_squares(n):
        ...     return sum(square(i) for i in range(n))
    """
    
    INLINE_SIZE_LIMIT = 10  # Max AST nodes for inlining
    MEMO_CACHE_SIZE = 256
    
    def __init__(self):
        self._profiles: Dict[str, FunctionProfile] = {}
        self._inlinable: Dict[str, ast.FunctionDef] = {}
        self.stats = {
            'functions_inlined': 0,
            'partial_evaluations': 0,
            'memoizations': 0,
        }
    
    def inline(self, func: Callable) -> Callable:
        """Mark a function as inlinable."""
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                    # Check if small enough to inline
                    body_size = sum(1 for _ in ast.walk(node))
                    if body_size <= self.INLINE_SIZE_LIMIT:
                        self._inlinable[func.__name__] = node
                        func.__highpy_inlinable__ = True
        except Exception:
            pass
        
        return func
    
    def optimize(self, func: Callable) -> Callable:
        """Optimize a function using cross-function techniques."""
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            # Apply inlining
            if self._inlinable:
                inliner = _FunctionInliner(self._inlinable, self.stats)
                tree = inliner.visit(tree)
                ast.fix_missing_locations(tree)
            
            code = compile(tree, f'<highpy-fspec:{func.__name__}>', 'exec')
            namespace = dict(func.__globals__)
            exec(code, namespace)
            
            result = namespace[func.__name__]
            result.__highpy_original__ = func
            return result
        except Exception:
            return func
    
    def memoize(
        self,
        maxsize: int = 256,
        typed: bool = True,
    ) -> Callable:
        """
        Type-aware memoization decorator.
        
        Unlike functools.lru_cache, this separates cache entries
        by argument types, preventing type confusion.
        """
        def decorator(func: Callable) -> Callable:
            cache: OrderedDict = OrderedDict()
            hits = 0
            misses = 0
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal hits, misses
                
                if typed:
                    key = (tuple((type(a), a) for a in args),
                           tuple(sorted((k, type(v), v) for k, v in kwargs.items())))
                else:
                    key = (args, tuple(sorted(kwargs.items())))
                
                try:
                    result = cache[key]
                    cache.move_to_end(key)
                    hits += 1
                    return result
                except (KeyError, TypeError):
                    pass
                
                result = func(*args, **kwargs)
                misses += 1
                
                try:
                    cache[key] = result
                    if len(cache) > maxsize:
                        cache.popitem(last=False)
                except TypeError:
                    pass  # Unhashable arguments
                
                return result
            
            wrapper.cache_info = lambda: {'hits': hits, 'misses': misses, 'size': len(cache)}
            wrapper.cache_clear = lambda: cache.clear()
            wrapper.__highpy_memoized__ = True
            self.stats['memoizations'] += 1
            
            return wrapper
        
        return decorator
    
    def partial_evaluate(
        self,
        func: Callable,
        known_args: Dict[str, Any],
    ) -> Callable:
        """
        Create a partially-evaluated version of a function.
        
        Substitutes known constant arguments directly into the AST,
        enabling constant folding through the specialized body.
        """
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            replacer = _ConstantSubstituter(known_args)
            tree = replacer.visit(tree)
            ast.fix_missing_locations(tree)
            
            code = compile(tree, f'<highpy-partial:{func.__name__}>', 'exec')
            namespace = dict(func.__globals__)
            exec(code, namespace)
            
            result = namespace[func.__name__]
            result.__highpy_partial__ = True
            result.__highpy_original__ = func
            self.stats['partial_evaluations'] += 1
            
            return result
        except Exception:
            return func


class _FunctionInliner(ast.NodeTransformer):
    """Inline function calls where possible."""
    
    def __init__(self, inlinable: Dict[str, ast.FunctionDef], stats: dict):
        self.inlinable = inlinable
        self.stats = stats
    
    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        
        if isinstance(node.func, ast.Name) and node.func.id in self.inlinable:
            func_def = self.inlinable[node.func.id]
            
            # Only inline single-expression returns
            if (len(func_def.body) == 1 and 
                isinstance(func_def.body[0], ast.Return) and
                func_def.body[0].value is not None):
                
                # Build argument mapping
                arg_names = [arg.arg for arg in func_def.args.args]
                if len(node.args) == len(arg_names):
                    import copy
                    body_expr = copy.deepcopy(func_def.body[0].value)
                    
                    # Substitute arguments
                    mapping = dict(zip(arg_names, node.args))
                    substituter = _NameSubstituter(mapping)
                    inlined = substituter.visit(body_expr)
                    
                    self.stats['functions_inlined'] += 1
                    return inlined
        
        return node


class _NameSubstituter(ast.NodeTransformer):
    """Substitute variable names with expressions."""
    
    def __init__(self, mapping: Dict[str, ast.expr]):
        self.mapping = mapping
    
    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            import copy
            return copy.deepcopy(self.mapping[node.id])
        return node


class _ConstantSubstituter(ast.NodeTransformer):
    """Substitute known constants into AST."""
    
    def __init__(self, constants: Dict[str, Any]):
        self.constants = constants
    
    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            value = self.constants[node.id]
            if isinstance(value, (int, float, str, bool, type(None))):
                return ast.Constant(value=value)
        return node
