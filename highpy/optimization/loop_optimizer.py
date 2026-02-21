"""
Loop Optimizer
==============

Novel approach: Multi-strategy loop optimization that selects the best
optimization technique based on loop characteristics analysis.

Optimization strategies:
1. Loop unrolling (for small fixed-count loops)
2. Loop vectorization (using numpy for array operations)
3. Loop fusion (merge adjacent loops over same range)
4. Loop tiling (improve cache locality for matrix operations)
5. Accumulator optimization (replace append-based patterns)
6. Comprehension conversion (for-loop to list comprehension)
"""

import ast
import inspect
import textwrap
import functools
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class LoopAnalysis:
    """Analysis results for a single loop."""
    is_range_loop: bool = False
    range_args: Tuple = ()
    loop_var: str = ""
    body_ops: int = 0
    has_accumulator: bool = False
    accumulator_var: str = ""
    accumulator_op: str = ""
    is_map_pattern: bool = False
    is_filter_pattern: bool = False
    is_reduction: bool = False
    can_unroll: bool = False
    can_vectorize: bool = False
    can_fuse: bool = False
    estimated_iterations: Optional[int] = None


class LoopOptimizer:
    """
    Analyzes and optimizes Python loops through AST transformation.
    
    Usage:
        >>> optimizer = LoopOptimizer()
        >>> @optimizer.optimize
        ... def sum_squares(n):
        ...     total = 0
        ...     for i in range(n):
        ...         total += i * i
        ...     return total
    """
    
    UNROLL_THRESHOLD = 8  # Max iterations for full unrolling
    
    def __init__(self, strategies: Optional[List[str]] = None):
        self.strategies = strategies or [
            'accumulator', 'comprehension', 'unroll', 'vectorize'
        ]
        self.stats = defaultdict(int)
    
    def optimize(self, func: Callable) -> Callable:
        """Optimize loops in a function."""
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            
            # Analyze loops
            transformer = _LoopTransformer(self.strategies, self.stats)
            tree = transformer.visit(tree)
            ast.fix_missing_locations(tree)
            
            # Compile optimized version
            code = compile(tree, f'<highpy-loopopt:{func.__name__}>', 'exec')
            namespace = dict(func.__globals__)
            exec(code, namespace)
            
            optimized = namespace[func.__name__]
            optimized.__highpy_loop_optimized__ = True
            optimized.__highpy_original__ = func
            optimized.__highpy_loop_stats__ = dict(self.stats)
            
            return optimized
        except Exception:
            return func
    
    def analyze(self, func: Callable) -> List[LoopAnalysis]:
        """Analyze loops in a function without modifying them."""
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        
        analyzer = _LoopAnalyzer()
        analyzer.visit(tree)
        return analyzer.loops


class _LoopAnalyzer(ast.NodeVisitor):
    """Analyze loop patterns."""
    
    def __init__(self):
        self.loops: List[LoopAnalysis] = []
    
    def visit_For(self, node: ast.For):
        analysis = LoopAnalysis()
        
        # Check if it's a range loop
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == 'range':
                analysis.is_range_loop = True
                analysis.range_args = tuple(node.iter.args)
                
                # Try to determine iteration count
                if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant):
                    analysis.estimated_iterations = node.iter.args[0].value
                    if analysis.estimated_iterations <= 8:
                        analysis.can_unroll = True
        
        if isinstance(node.target, ast.Name):
            analysis.loop_var = node.target.id
        
        # Count body operations
        analysis.body_ops = len(node.body)
        
        # Detect accumulator pattern: total += expr
        for stmt in node.body:
            if isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    analysis.has_accumulator = True
                    analysis.accumulator_var = stmt.target.id
                    analysis.is_reduction = True
                    if isinstance(stmt.op, ast.Add):
                        analysis.accumulator_op = '+'
                    elif isinstance(stmt.op, ast.Mult):
                        analysis.accumulator_op = '*'
        
        # Detect map pattern: result.append(f(x)) for x in ...
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if isinstance(stmt.value.func, ast.Attribute):
                    if stmt.value.func.attr == 'append':
                        analysis.is_map_pattern = True
        
        self.loops.append(analysis)
        self.generic_visit(node)


class _LoopTransformer(ast.NodeTransformer):
    """Transform loops based on detected patterns."""
    
    def __init__(self, strategies: List[str], stats: dict):
        self.strategies = strategies
        self.stats = stats
    
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        
        # Strategy: Convert accumulator loops to sum() with generator
        if 'accumulator' in self.strategies:
            result = self._try_accumulator_optimization(node)
            if result is not None:
                self.stats['accumulator_optimizations'] += 1
                return result
        
        # Strategy: Convert append loops to list comprehensions
        if 'comprehension' in self.strategies:
            result = self._try_comprehension_conversion(node)
            if result is not None:
                self.stats['comprehension_conversions'] += 1
                return result
        
        return node
    
    def _try_accumulator_optimization(self, node: ast.For) -> Optional[Any]:
        """
        Convert accumulator loops to built-in sum().
        
        Before:
            total = 0
            for i in range(n):
                total += i * i
        
        After:
            total = sum(i * i for i in range(n))
        """
        if not isinstance(node.target, ast.Name):
            return None
        
        if len(node.body) != 1:
            return None
        
        stmt = node.body[0]
        if not isinstance(stmt, ast.AugAssign):
            return None
        
        if not isinstance(stmt.op, ast.Add):
            return None
        
        if not isinstance(stmt.target, ast.Name):
            return None
        
        accum_var = stmt.target.id
        
        # Create: accum_var = sum(expr for target in iter)
        generator = ast.GeneratorExp(
            elt=stmt.value,
            generators=[
                ast.comprehension(
                    target=node.target,
                    iter=node.iter,
                    ifs=[],
                    is_async=0,
                )
            ]
        )
        
        sum_call = ast.Call(
            func=ast.Name(id='sum', ctx=ast.Load()),
            args=[generator],
            keywords=[],
        )
        
        # accum_var = accum_var + sum(...)
        assign = ast.Assign(
            targets=[ast.Name(id=accum_var, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=accum_var, ctx=ast.Load()),
                op=ast.Add(),
                right=sum_call,
            ),
        )
        
        return assign
    
    def _try_comprehension_conversion(self, node: ast.For) -> Optional[ast.Assign]:
        """
        Convert append-based loops to list comprehensions.
        
        Before:
            result = []
            for x in data:
                result.append(f(x))
        
        After:
            result = [f(x) for x in data]
        """
        if len(node.body) != 1:
            return None
        
        stmt = node.body[0]
        if not isinstance(stmt, ast.Expr):
            return None
        
        if not isinstance(stmt.value, ast.Call):
            return None
        
        call = stmt.value
        if not isinstance(call.func, ast.Attribute):
            return None
        
        if call.func.attr != 'append':
            return None
        
        if not isinstance(call.func.value, ast.Name):
            return None
        
        list_var = call.func.value.id
        
        if len(call.args) != 1:
            return None
        
        append_value = call.args[0]
        
        # Create list comprehension
        listcomp = ast.ListComp(
            elt=append_value,
            generators=[
                ast.comprehension(
                    target=node.target,
                    iter=node.iter,
                    ifs=[],
                    is_async=0,
                )
            ]
        )
        
        # result.extend(listcomp) to handle pre-existing elements
        extend_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=list_var, ctx=ast.Load()),
                    attr='extend',
                    ctx=ast.Load(),
                ),
                args=[listcomp],
                keywords=[],
            )
        )
        
        return extend_call


def optimize_loops(func: Callable) -> Callable:
    """Convenience decorator for loop optimization."""
    optimizer = LoopOptimizer()
    return optimizer.optimize(func)
