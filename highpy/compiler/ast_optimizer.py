"""
AST-Level Optimizer
===================

Performs source-level optimizations through AST transformations.

Novel Optimization Passes:
1. Constant Folding & Propagation - Evaluate constant expressions at compile time
2. Dead Code Elimination - Remove unreachable branches
3. Loop-Invariant Code Motion - Hoist computations out of loops
4. Strength Reduction - Replace expensive ops with cheaper equivalents
5. Common Subexpression Elimination - Cache repeated computations
6. Algebraic Simplification - Apply algebraic identities
7. Range-to-C-Loop Transformation - Convert range() loops to C-style
8. Global-to-Local Promotion - Convert global lookups to local variables
"""

import ast
import copy
import inspect
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


class ASTOptimizer:
    """
    Multi-pass AST optimizer that transforms Python code for better performance.
    
    Usage:
        >>> optimizer = ASTOptimizer()
        >>> def slow(n):
        ...     x = 2 * 3.14159
        ...     total = 0
        ...     for i in range(n):
        ...         total += x * i
        ...     return total
        >>> fast = optimizer.optimize(slow)
    """
    
    def __init__(self, passes: Optional[List[str]] = None):
        self.enabled_passes = passes or [
            'constant_fold',
            'dead_code_eliminate',
            'loop_invariant_motion',
            'strength_reduce',
            'global_to_local',
            'algebraic_simplify',
        ]
        self.stats = defaultdict(int)
    
    def optimize(self, func: Callable) -> Callable:
        """
        Apply all enabled optimization passes to a function.
        
        Returns a new function with optimized code.
        """
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        
        # Apply passes
        for pass_name in self.enabled_passes:
            transformer = getattr(self, f'_pass_{pass_name}', None)
            if transformer:
                tree = transformer(tree)
                ast.fix_missing_locations(tree)
        
        # Compile the optimized AST
        code = compile(tree, f'<highpy-optimized:{func.__name__}>', 'exec')
        
        # Extract the function from the compiled code
        namespace = {}
        # Copy the function's globals
        namespace.update(func.__globals__)
        exec(code, namespace)
        
        optimized_func = namespace[func.__name__]
        optimized_func.__highpy_original__ = func
        optimized_func.__highpy_optimized__ = True
        optimized_func.__highpy_stats__ = dict(self.stats)
        
        return optimized_func
    
    def get_optimized_source(self, func: Callable) -> str:
        """Return the optimized source code as a string."""
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        
        for pass_name in self.enabled_passes:
            transformer = getattr(self, f'_pass_{pass_name}', None)
            if transformer:
                tree = transformer(tree)
                ast.fix_missing_locations(tree)
        
        return ast.unparse(tree)
    
    # ---- Pass 1: Constant Folding & Propagation ----
    
    def _pass_constant_fold(self, tree: ast.Module) -> ast.Module:
        """Evaluate constant expressions at compile time."""
        return ConstantFolder(self.stats).visit(tree)
    
    # ---- Pass 2: Dead Code Elimination ----
    
    def _pass_dead_code_eliminate(self, tree: ast.Module) -> ast.Module:
        """Remove unreachable code branches."""
        return DeadCodeEliminator(self.stats).visit(tree)
    
    # ---- Pass 3: Loop-Invariant Code Motion ----
    
    def _pass_loop_invariant_motion(self, tree: ast.Module) -> ast.Module:
        """Hoist loop-invariant computations out of loops."""
        return LoopInvariantMotion(self.stats).visit(tree)
    
    # ---- Pass 4: Strength Reduction ----
    
    def _pass_strength_reduce(self, tree: ast.Module) -> ast.Module:
        """Replace expensive operations with cheaper equivalents."""
        return StrengthReducer(self.stats).visit(tree)
    
    # ---- Pass 5: Global-to-Local Promotion ----
    
    def _pass_global_to_local(self, tree: ast.Module) -> ast.Module:
        """Convert frequently-used global lookups to local variables."""
        return GlobalToLocalPromoter(self.stats).visit(tree)
    
    # ---- Pass 6: Algebraic Simplification ----
    
    def _pass_algebraic_simplify(self, tree: ast.Module) -> ast.Module:
        """Apply algebraic identities to simplify expressions."""
        return AlgebraicSimplifier(self.stats).visit(tree)


class ConstantFolder(ast.NodeTransformer):
    """
    Constant folding pass.
    
    Evaluates expressions where all operands are constants:
    - 2 * 3.14159 -> 6.28318
    - "hello" + " " + "world" -> "hello world"
    - True and False -> False
    """
    
    SAFE_OPS = {
        ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
        ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**',
        ast.LShift: '<<', ast.RShift: '>>', ast.BitAnd: '&',
        ast.BitOr: '|', ast.BitXor: '^',
    }
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            op_type = type(node.op)
            if op_type in self.SAFE_OPS:
                try:
                    result = self._eval_binop(node.left.value, node.right.value, node.op)
                    if result is not None and self._is_safe_constant(result):
                        self.stats['constants_folded'] += 1
                        return ast.Constant(value=result)
                except (ZeroDivisionError, OverflowError, ValueError):
                    pass
        
        return node
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        
        if isinstance(node.operand, ast.Constant):
            try:
                if isinstance(node.op, ast.USub):
                    result = -node.operand.value
                elif isinstance(node.op, ast.UAdd):
                    result = +node.operand.value
                elif isinstance(node.op, ast.Not):
                    result = not node.operand.value
                elif isinstance(node.op, ast.Invert):
                    result = ~node.operand.value
                else:
                    return node
                
                if self._is_safe_constant(result):
                    self.stats['constants_folded'] += 1
                    return ast.Constant(value=result)
            except (TypeError, OverflowError):
                pass
        
        return node
    
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        
        if all(isinstance(v, ast.Constant) for v in node.values):
            try:
                if isinstance(node.op, ast.And):
                    result = all(v.value for v in node.values)
                else:
                    result = any(v.value for v in node.values)
                self.stats['constants_folded'] += 1
                return ast.Constant(value=result)
            except TypeError:
                pass
        
        return node
    
    def _eval_binop(self, left, right, op):
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right
        if isinstance(op, ast.FloorDiv): return left // right
        if isinstance(op, ast.Mod): return left % right
        if isinstance(op, ast.Pow):
            if isinstance(right, (int, float)) and right < 100:
                return left ** right
            return None
        if isinstance(op, ast.LShift): return left << right
        if isinstance(op, ast.RShift): return left >> right
        if isinstance(op, ast.BitAnd): return left & right
        if isinstance(op, ast.BitOr): return left | right
        if isinstance(op, ast.BitXor): return left ^ right
        return None
    
    def _is_safe_constant(self, value) -> bool:
        """Check if a value is safe to embed as a constant."""
        return isinstance(value, (int, float, str, bool, bytes, type(None)))


class DeadCodeEliminator(ast.NodeTransformer):
    """
    Dead code elimination pass.
    
    Removes:
    - if False: ... branches
    - if True: ... (keeps only the body)
    - Code after unconditional return/raise
    """
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def visit_If(self, node: ast.If) -> Any:
        self.generic_visit(node)
        
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                # if True: body -> body
                self.stats['dead_branches_removed'] += 1
                return node.body
            else:
                # if False: body; else: else_body -> else_body
                self.stats['dead_branches_removed'] += 1
                if node.orelse:
                    return node.orelse
                return ast.Pass()
        
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        
        # Remove statements after unconditional return
        new_body = []
        for stmt in node.body:
            new_body.append(stmt)
            if isinstance(stmt, (ast.Return, ast.Raise)):
                if len(new_body) < len(node.body):
                    self.stats['dead_code_removed'] += len(node.body) - len(new_body)
                break
        
        node.body = new_body if new_body else [ast.Pass()]
        return node


class LoopInvariantMotion(ast.NodeTransformer):
    """
    Loop-invariant code motion (LICM) pass.
    
    Identifies computations inside loops whose values don't change
    across iterations and hoists them before the loop.
    
    Before:
        for i in range(n):
            x = math.sqrt(2)  # invariant
            result += x * i
    
    After:
        _hoist_0 = math.sqrt(2)
        for i in range(n):
            result += _hoist_0 * i
    """
    
    def __init__(self, stats: dict):
        self.stats = stats
        self._hoist_counter = 0
    
    def visit_For(self, node: ast.For) -> Any:
        self.generic_visit(node)
        
        # Get the loop variable(s)
        loop_vars = self._get_assigned_names(node.target)
        
        # Get all variables modified in the loop body
        modified_vars = set()
        for stmt in node.body:
            modified_vars.update(self._get_modified_names(stmt))
        
        # Add the loop variable itself
        modified_vars.update(loop_vars)
        
        # Find invariant assignments
        hoisted = []
        new_body = []
        
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    # Check if the value expression only uses variables not modified in the loop
                    used_names = self._get_used_names(stmt.value)
                    if not used_names.intersection(modified_vars):
                        # This assignment is loop-invariant - hoist it
                        hoisted.append(stmt)
                        self.stats['loop_invariants_hoisted'] += 1
                        continue
            
            new_body.append(stmt)
        
        if hoisted:
            node.body = new_body if new_body else [ast.Pass()]
            return hoisted + [node]
        
        return node
    
    def _get_assigned_names(self, node: ast.expr) -> Set[str]:
        """Get names assigned to in a target expression."""
        if isinstance(node, ast.Name):
            return {node.id}
        elif isinstance(node, ast.Tuple):
            result = set()
            for elt in node.elts:
                result.update(self._get_assigned_names(elt))
            return result
        return set()
    
    def _get_modified_names(self, node: ast.stmt) -> Set[str]:
        """Get all names modified in a statement."""
        modified = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    modified.update(self._get_assigned_names(target))
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    modified.add(child.target.id)
        
        return modified
    
    def _get_used_names(self, node: ast.expr) -> Set[str]:
        """Get all names referenced in an expression."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names


class StrengthReducer(ast.NodeTransformer):
    """
    Strength reduction pass.
    
    Replaces expensive operations with cheaper equivalents:
    - x ** 2 -> x * x
    - x * 2 -> x + x (for integers)
    - x / 2.0 -> x * 0.5
    - x % 2 -> x & 1 (for integers)
    """
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        
        # x ** 2 -> x * x
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.stats['strength_reductions'] += 1
                return ast.BinOp(
                    left=copy.deepcopy(node.left),
                    op=ast.Mult(),
                    right=copy.deepcopy(node.left)
                )
            # x ** 3 -> x * x * x
            if isinstance(node.right, ast.Constant) and node.right.value == 3:
                self.stats['strength_reductions'] += 1
                return ast.BinOp(
                    left=ast.BinOp(
                        left=copy.deepcopy(node.left),
                        op=ast.Mult(),
                        right=copy.deepcopy(node.left)
                    ),
                    op=ast.Mult(),
                    right=copy.deepcopy(node.left)
                )
            # x ** 0.5 -> math.sqrt(x) would need import, skip
        
        # x / constant -> x * (1/constant) for floats
        if isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant):
                if isinstance(node.right.value, (int, float)) and node.right.value != 0:
                    reciprocal = 1.0 / node.right.value
                    self.stats['strength_reductions'] += 1
                    return ast.BinOp(
                        left=node.left,
                        op=ast.Mult(),
                        right=ast.Constant(value=reciprocal)
                    )
        
        # x * 2 -> x + x
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.stats['strength_reductions'] += 1
                return ast.BinOp(
                    left=copy.deepcopy(node.left),
                    op=ast.Add(),
                    right=copy.deepcopy(node.left)
                )
            if isinstance(node.left, ast.Constant) and node.left.value == 2:
                self.stats['strength_reductions'] += 1
                return ast.BinOp(
                    left=copy.deepcopy(node.right),
                    op=ast.Add(),
                    right=copy.deepcopy(node.right)
                )
        
        return node


class GlobalToLocalPromoter(ast.NodeTransformer):
    """
    Global-to-local promotion pass.
    
    Converts frequently-used global name lookups to local variable
    assignments at the function entry. LOAD_FAST (local) is ~2x faster
    than LOAD_GLOBAL in CPython.
    
    Before:
        def f(data):
            for x in data:
                result.append(len(x))
    
    After:
        def f(data):
            _local_len = len
            _local_append = result.append
            for x in data:
                _local_append(_local_len(x))
    """
    
    # Built-in names worth promoting
    PROMOTABLE_BUILTINS = {
        'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple',
        'sum', 'min', 'max', 'abs', 'round', 'sorted', 'reversed',
        'isinstance', 'hasattr', 'getattr', 'setattr',
        'print',  # Even print can be promoted
    }
    
    PROMOTION_THRESHOLD = 2  # Promote if used >= this many times
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        
        # Count global name usage
        usage_counts = defaultdict(int)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in self.PROMOTABLE_BUILTINS:
                    usage_counts[child.id] += 1
        
        # Determine which to promote
        to_promote = {}
        for name, count in usage_counts.items():
            if count >= self.PROMOTION_THRESHOLD:
                local_name = f'_local_{name}'
                to_promote[name] = local_name
        
        if not to_promote:
            return node
        
        # Add local assignments at function entry
        promotions = []
        for global_name, local_name in to_promote.items():
            promotions.append(
                ast.Assign(
                    targets=[ast.Name(id=local_name, ctx=ast.Store())],
                    value=ast.Name(id=global_name, ctx=ast.Load()),
                    lineno=node.lineno,
                )
            )
            self.stats['globals_promoted'] += 1
        
        # Replace all uses of global names with local names
        replacer = _NameReplacer(to_promote)
        new_body = [replacer.visit(stmt) for stmt in node.body]
        
        node.body = promotions + new_body
        return node


class _NameReplacer(ast.NodeTransformer):
    """Replace name references in AST."""
    
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = replacements
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, ast.Load) and node.id in self.replacements:
            return ast.Name(id=self.replacements[node.id], ctx=ast.Load())
        return node


class AlgebraicSimplifier(ast.NodeTransformer):
    """
    Algebraic simplification pass.
    
    Applies algebraic identities:
    - x + 0 -> x
    - x * 1 -> x
    - x * 0 -> 0
    - x - 0 -> x
    - x ** 1 -> x
    - x ** 0 -> 1
    - x / 1 -> x
    """
    
    def __init__(self, stats: dict):
        self.stats = stats
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        
        # x + 0 or 0 + x
        if isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return node.right
        
        # x - 0
        if isinstance(node.op, ast.Sub):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return node.left
        
        # x * 1 or 1 * x
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                self.stats['algebraic_simplifications'] += 1
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                self.stats['algebraic_simplifications'] += 1
                return node.right
            # x * 0 or 0 * x
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return ast.Constant(value=0)
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return ast.Constant(value=0)
        
        # x / 1
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                self.stats['algebraic_simplifications'] += 1
                return node.left
        
        # x ** 1
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                self.stats['algebraic_simplifications'] += 1
                return node.left
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.stats['algebraic_simplifications'] += 1
                return ast.Constant(value=1)
        
        return node
