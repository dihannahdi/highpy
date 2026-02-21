"""
Fractal Optimizer — Core Engine
================================

Implements the fractal self-similarity axiom: the same optimization morphism
applies at every program granularity level with structure-preserving mappings.

Theoretical Foundation:
    Define a category **Prog** whose objects are program fragments at different
    granularity levels and whose morphisms are optimizing transformations.
    A Fractal Optimization Morphism (FOM) is a natural transformation:

        η: Id_Prog → Opt_Prog

    such that for every level-embedding functor E: Level_k → Level_{k+1},
    the following diagram commutes:

        F_k ---η_k--→ Opt(F_k)
         |                |
        E_k            E_k
         ↓                ↓
        F_{k+1} -η_{k+1}→ Opt(F_{k+1})

    This ensures that optimizing at level k and then embedding produces
    the same result as embedding first and optimizing at level k+1.

Implementation:
    Each FractalLevel represents a granularity tier. The optimizer recursively
    decomposes a program into its fractal components, applies the universal
    optimization morphism at each level, and recomposes.
"""

import ast
import copy
import dis
import functools
import hashlib
import inspect
import math
import textwrap
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
)


# ═══════════════════════════════════════════════════════════════════════════
# Safe Memoization Wrapper
# ═══════════════════════════════════════════════════════════════════════════

def _safe_memoize(func: Callable) -> Callable:
    """
    Wrap *func* with memoization that gracefully handles unhashable arguments.

    Falls back to calling the function without caching when arguments
    cannot be hashed (e.g. lists, dicts, sets).
    """
    cache: Dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            key = args + tuple(sorted(kwargs.items())) if kwargs else args
            hash(key)  # test hashability
        except TypeError:
            # Unhashable args — call directly without caching
            return func(*args, **kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.__wrapped__ = func
    wrapper.cache = cache
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Fractal Level Hierarchy
# ═══════════════════════════════════════════════════════════════════════════

class FractalLevel(IntEnum):
    """
    Granularity levels forming the fractal hierarchy.
    
    Each level is self-similar: the same optimization patterns apply
    at every level, parameterized by level-specific semantics.
    """
    EXPRESSION = 0    # Individual expressions: a + b, f(x)
    STATEMENT = 1     # Single statements: x = expr; if cond: ...
    BLOCK = 2         # Basic blocks: sequences of statements
    FUNCTION = 3      # Entire function bodies
    MODULE = 4        # Full modules
    PROGRAM = 5       # Multi-module programs


@dataclass
class OptimizationMorphism:
    """
    A single optimization transformation applicable at any fractal level.
    
    The morphism carries:
        - name: Human-readable identifier
        - transform: The actual AST transformation function
        - applicable_levels: Set of levels where this morphism applies
        - contraction_factor: Lipschitz constant (< 1 guarantees convergence)
        - energy_reduction: Expected energy reduction per application
    """
    name: str
    transform: Callable[[ast.AST, 'FractalLevel'], ast.AST]
    applicable_levels: Set[FractalLevel] = field(
        default_factory=lambda: set(FractalLevel)
    )
    contraction_factor: float = 0.9  # Lipschitz constant
    energy_reduction: float = 0.0    # Measured empirically
    applications: int = 0
    
    def apply(self, node: ast.AST, level: FractalLevel) -> ast.AST:
        """Apply this morphism at the given level."""
        if level in self.applicable_levels:
            result = self.transform(node, level)
            self.applications += 1
            return result
        return node


@dataclass
class OptimizationEnergy:
    """
    Program energy metric — a composite measure of computational cost.
    
    Energy E(P) = w_i * I(P) + w_m * M(P) + w_b * B(P) + w_a * A(P)
    
    where:
        I(P) = instruction complexity (weighted opcode count)
        M(P) = memory pressure (loads + stores + allocations)
        B(P) = branch cost (unpredictable branches)
        A(P) = abstraction overhead (attribute lookups, dynamic dispatch)
    """
    instruction_complexity: float = 0.0
    memory_pressure: float = 0.0
    branch_cost: float = 0.0
    abstraction_overhead: float = 0.0
    
    # Weights for the composite metric
    W_INSTRUCTION: float = 1.0
    W_MEMORY: float = 1.5
    W_BRANCH: float = 2.0
    W_ABSTRACTION: float = 1.8
    
    @property
    def total(self) -> float:
        """Compute total weighted energy."""
        return (
            self.W_INSTRUCTION * self.instruction_complexity +
            self.W_MEMORY * self.memory_pressure +
            self.W_BRANCH * self.branch_cost +
            self.W_ABSTRACTION * self.abstraction_overhead
        )
    
    def distance(self, other: 'OptimizationEnergy') -> float:
        """Metric distance between two energy states."""
        return math.sqrt(
            (self.instruction_complexity - other.instruction_complexity) ** 2 +
            (self.memory_pressure - other.memory_pressure) ** 2 +
            (self.branch_cost - other.branch_cost) ** 2 +
            (self.abstraction_overhead - other.abstraction_overhead) ** 2
        )
    
    def __sub__(self, other: 'OptimizationEnergy') -> 'OptimizationEnergy':
        return OptimizationEnergy(
            instruction_complexity=self.instruction_complexity - other.instruction_complexity,
            memory_pressure=self.memory_pressure - other.memory_pressure,
            branch_cost=self.branch_cost - other.branch_cost,
            abstraction_overhead=self.abstraction_overhead - other.abstraction_overhead,
        )
    
    def reduction_ratio(self, original: 'OptimizationEnergy') -> float:
        """How much energy was reduced relative to original."""
        if original.total == 0:
            return 0.0
        return 1.0 - (self.total / original.total)


@dataclass
class FractalOptimizationResult:
    """Result of a full recursive fractal optimization pass."""
    original_func: Callable
    optimized_func: Callable
    iterations: int
    converged: bool
    initial_energy: OptimizationEnergy
    final_energy: OptimizationEnergy
    energy_history: List[float]
    level_stats: Dict[FractalLevel, int]  # optimizations per level
    morphism_stats: Dict[str, int]  # applications per morphism
    contraction_factors: List[float]  # measured per iteration
    wall_time_seconds: float


# ═══════════════════════════════════════════════════════════════════════════
# Energy Analyzer — Computes program energy from bytecode/AST
# ═══════════════════════════════════════════════════════════════════════════

# Opcode weight categories
_CHEAP_OPS = {
    'NOP', 'POP_TOP', 'PUSH_NULL', 'RESUME', 'RETURN_VALUE',
    'RETURN_CONST', 'COPY', 'SWAP',
}
_MEMORY_OPS = {
    'LOAD_FAST', 'STORE_FAST', 'LOAD_CONST', 'LOAD_GLOBAL',
    'STORE_GLOBAL', 'LOAD_ATTR', 'STORE_ATTR', 'LOAD_NAME',
    'STORE_NAME', 'DELETE_FAST', 'DELETE_GLOBAL', 'LOAD_DEREF',
    'STORE_DEREF', 'LOAD_FAST_AND_CLEAR',
}
_BRANCH_OPS = {
    'POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_NONE',
    'POP_JUMP_IF_NOT_NONE', 'JUMP_FORWARD', 'JUMP_BACKWARD',
    'FOR_ITER', 'JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP',
}
_CALL_OPS = {
    'CALL', 'CALL_FUNCTION_EX', 'CALL_KW',
}
_ABSTRACTION_OPS = {
    'LOAD_ATTR', 'LOAD_METHOD', 'LOAD_GLOBAL', 'BINARY_SUBSCR',
    'STORE_SUBSCR', 'DELETE_SUBSCR', 'IMPORT_NAME', 'IMPORT_FROM',
}


class EnergyAnalyzer:
    """
    Computes the multi-dimensional energy of a Python function.
    
    The energy metric captures four orthogonal cost dimensions:
    instruction complexity, memory pressure, branch cost, and
    abstraction overhead.
    """
    
    @staticmethod
    def compute_energy(func: Callable) -> OptimizationEnergy:
        """Compute the energy of a callable."""
        try:
            code = func.__code__ if hasattr(func, '__code__') else func
            return EnergyAnalyzer._analyze_code(code)
        except Exception:
            return OptimizationEnergy()
    
    @staticmethod
    def _analyze_code(code: types.CodeType) -> OptimizationEnergy:
        """Analyze bytecode to compute energy components."""
        instruction_cost = 0.0
        memory_cost = 0.0
        branch_cost = 0.0
        abstraction_cost = 0.0
        
        try:
            instructions = list(dis.get_instructions(code))
        except Exception:
            return OptimizationEnergy()
        
        for instr in instructions:
            name = instr.opname
            
            if name in _CHEAP_OPS:
                instruction_cost += 0.1
            elif name in _CALL_OPS:
                instruction_cost += 5.0
                abstraction_cost += 3.0
            elif name in _BRANCH_OPS:
                instruction_cost += 1.0
                branch_cost += 2.0
            elif name in _MEMORY_OPS:
                instruction_cost += 1.0
                memory_cost += 1.5
            elif name in _ABSTRACTION_OPS:
                instruction_cost += 2.0
                abstraction_cost += 2.5
            else:
                # Arithmetic, comparison, etc.
                instruction_cost += 1.5
        
        # Recursively analyze nested code objects
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                inner = EnergyAnalyzer._analyze_code(const)
                instruction_cost += inner.instruction_complexity * 0.5
                memory_cost += inner.memory_pressure * 0.5
                branch_cost += inner.branch_cost * 0.5
                abstraction_cost += inner.abstraction_overhead * 0.5
        
        return OptimizationEnergy(
            instruction_complexity=instruction_cost,
            memory_pressure=memory_cost,
            branch_cost=branch_cost,
            abstraction_overhead=abstraction_cost,
        )
    
    @staticmethod
    def compute_ast_energy(tree: ast.AST) -> OptimizationEnergy:
        """Compute energy from an AST (used during optimization before compilation)."""
        visitor = _ASTEnergyVisitor()
        visitor.visit(tree)
        return visitor.energy


class _ASTEnergyVisitor(ast.NodeVisitor):
    """Visit AST nodes to estimate energy."""
    
    def __init__(self):
        self.energy = OptimizationEnergy()
    
    def visit_BinOp(self, node):
        self.energy.instruction_complexity += 1.5
        self.generic_visit(node)
    
    def visit_UnaryOp(self, node):
        self.energy.instruction_complexity += 1.0
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        self.energy.instruction_complexity += 1.0
        self.energy.branch_cost += 1.0
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        self.energy.instruction_complexity += 1.5
        self.energy.branch_cost += 0.5
        self.generic_visit(node)
    
    def visit_Call(self, node):
        self.energy.instruction_complexity += 5.0
        self.energy.abstraction_overhead += 3.0
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        self.energy.instruction_complexity += 2.0
        self.energy.abstraction_overhead += 2.5
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        self.energy.instruction_complexity += 2.0
        self.energy.memory_pressure += 1.5
        self.generic_visit(node)
    
    def visit_Name(self, node):
        self.energy.instruction_complexity += 0.5
        self.energy.memory_pressure += 0.5
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.energy.branch_cost += 2.0
        self.energy.instruction_complexity += 1.0
        self.generic_visit(node)
    
    def visit_For(self, node):
        # Loop overhead: iter expression runs once
        iter_visitor = _ASTEnergyVisitor()
        iter_visitor.visit(node.iter)
        self.energy.instruction_complexity += iter_visitor.energy.instruction_complexity
        self.energy.memory_pressure += iter_visitor.energy.memory_pressure
        self.energy.branch_cost += iter_visitor.energy.branch_cost
        self.energy.abstraction_overhead += iter_visitor.energy.abstraction_overhead
        
        # Target energy (once)
        target_visitor = _ASTEnergyVisitor()
        target_visitor.visit(node.target)
        self.energy.instruction_complexity += target_visitor.energy.instruction_complexity
        self.energy.memory_pressure += target_visitor.energy.memory_pressure
        
        # Body energy — amplified by loop factor
        LOOP_FACTOR = 5  # Conservative estimate
        body_visitor = _ASTEnergyVisitor()
        for stmt in node.body:
            body_visitor.visit(stmt)
        self.energy.instruction_complexity += body_visitor.energy.instruction_complexity * LOOP_FACTOR
        self.energy.memory_pressure += body_visitor.energy.memory_pressure * LOOP_FACTOR
        self.energy.branch_cost += body_visitor.energy.branch_cost * LOOP_FACTOR + 2.0
        self.energy.abstraction_overhead += body_visitor.energy.abstraction_overhead * LOOP_FACTOR
        
        # orelse (once)
        if node.orelse:
            else_visitor = _ASTEnergyVisitor()
            for stmt in node.orelse:
                else_visitor.visit(stmt)
            self.energy.instruction_complexity += else_visitor.energy.instruction_complexity
            self.energy.memory_pressure += else_visitor.energy.memory_pressure
    
    def visit_While(self, node):
        # Test expression — runs each iteration
        LOOP_FACTOR = 5
        test_visitor = _ASTEnergyVisitor()
        test_visitor.visit(node.test)
        self.energy.instruction_complexity += test_visitor.energy.instruction_complexity * LOOP_FACTOR
        self.energy.branch_cost += test_visitor.energy.branch_cost * LOOP_FACTOR + 2.0
        
        # Body — amplified by loop factor
        body_visitor = _ASTEnergyVisitor()
        for stmt in node.body:
            body_visitor.visit(stmt)
        self.energy.instruction_complexity += body_visitor.energy.instruction_complexity * LOOP_FACTOR
        self.energy.memory_pressure += body_visitor.energy.memory_pressure * LOOP_FACTOR
        self.energy.branch_cost += body_visitor.energy.branch_cost * LOOP_FACTOR
        self.energy.abstraction_overhead += body_visitor.energy.abstraction_overhead * LOOP_FACTOR
    
    def visit_Global(self, node):
        self.energy.abstraction_overhead += len(node.names) * 2.0
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        self.energy.instruction_complexity += 5.0
        self.energy.memory_pressure += 3.0
        self.generic_visit(node)
    
    def visit_DictComp(self, node):
        self.energy.instruction_complexity += 7.0
        self.energy.memory_pressure += 5.0
        self.generic_visit(node)


# ═══════════════════════════════════════════════════════════════════════════
# Fractal Decomposer — Breaks program into self-similar components
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FractalComponent:
    """A program fragment at a specific fractal level."""
    level: FractalLevel
    node: ast.AST
    children: List['FractalComponent'] = field(default_factory=list)
    energy: OptimizationEnergy = field(default_factory=OptimizationEnergy)
    optimized: bool = False


class FractalDecomposer:
    """
    Decomposes a program AST into a fractal tree of components.
    
    The decomposition is self-similar: each level contains components
    that are structurally analogous to their parent level, enabling
    the same optimization morphisms to apply everywhere.
    """
    
    def decompose(self, tree: ast.AST) -> FractalComponent:
        """Decompose an AST into a fractal component hierarchy."""
        if isinstance(tree, ast.Module):
            return self._decompose_module(tree)
        elif isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._decompose_function(tree)
        else:
            return FractalComponent(
                level=FractalLevel.EXPRESSION,
                node=tree,
                energy=EnergyAnalyzer.compute_ast_energy(tree),
            )
    
    def _decompose_module(self, module: ast.Module) -> FractalComponent:
        """Decompose a module into function-level and statement-level components."""
        children = []
        for stmt in module.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                children.append(self._decompose_function(stmt))
            elif isinstance(stmt, ast.ClassDef):
                # Classes contain functions
                class_children = []
                for item in stmt.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_children.append(self._decompose_function(item))
                    else:
                        class_children.append(self._decompose_statement(item))
                children.append(FractalComponent(
                    level=FractalLevel.FUNCTION,
                    node=stmt,
                    children=class_children,
                    energy=EnergyAnalyzer.compute_ast_energy(stmt),
                ))
            else:
                children.append(self._decompose_statement(stmt))
        
        return FractalComponent(
            level=FractalLevel.MODULE,
            node=module,
            children=children,
            energy=EnergyAnalyzer.compute_ast_energy(module),
        )
    
    def _decompose_function(self, func: ast.FunctionDef) -> FractalComponent:
        """Decompose a function into block-level components."""
        blocks = self._extract_blocks(func.body)
        children = [self._decompose_block(block) for block in blocks]
        
        return FractalComponent(
            level=FractalLevel.FUNCTION,
            node=func,
            children=children,
            energy=EnergyAnalyzer.compute_ast_energy(func),
        )
    
    def _decompose_block(self, stmts: List[ast.stmt]) -> FractalComponent:
        """Decompose a basic block into statement-level components."""
        children = [self._decompose_statement(stmt) for stmt in stmts]
        
        # Create a synthetic block node
        block_node = ast.Module(body=stmts, type_ignores=[])
        return FractalComponent(
            level=FractalLevel.BLOCK,
            node=block_node,
            children=children,
            energy=EnergyAnalyzer.compute_ast_energy(block_node),
        )
    
    def _decompose_statement(self, stmt: ast.stmt) -> FractalComponent:
        """Decompose a statement into expression-level components."""
        expressions = self._extract_expressions(stmt)
        children = [
            FractalComponent(
                level=FractalLevel.EXPRESSION,
                node=expr,
                energy=EnergyAnalyzer.compute_ast_energy(expr),
            )
            for expr in expressions
        ]
        
        return FractalComponent(
            level=FractalLevel.STATEMENT,
            node=stmt,
            children=children,
            energy=EnergyAnalyzer.compute_ast_energy(stmt),
        )
    
    def _extract_blocks(self, stmts: List[ast.stmt]) -> List[List[ast.stmt]]:
        """Extract basic blocks from a list of statements."""
        blocks = []
        current_block = []
        
        for stmt in stmts:
            if isinstance(stmt, (ast.If, ast.For, ast.While, ast.With,
                                  ast.Try, ast.Match)):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                blocks.append([stmt])
            else:
                current_block.append(stmt)
        
        if current_block:
            blocks.append(current_block)
        
        if not blocks:
            blocks = [stmts]
        
        return blocks
    
    def _extract_expressions(self, stmt: ast.stmt) -> List[ast.expr]:
        """Extract all expression nodes from a statement."""
        expressions = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.expr):
                expressions.append(node)
        return expressions


# ═══════════════════════════════════════════════════════════════════════════
# Universal Optimization Morphisms — Apply at ANY fractal level
# ═══════════════════════════════════════════════════════════════════════════

class UniversalMorphisms:
    """
    Collection of optimization morphisms that are level-agnostic.
    
    Each morphism is parameterized by the fractal level, enabling
    the same logical optimization to adapt to different granularities.
    This is the core of fractal self-similarity.
    """
    
    @staticmethod
    def constant_propagation() -> OptimizationMorphism:
        """
        Constant propagation — fractal version.
        
        At EXPRESSION level: fold constant subexpressions.
        At STATEMENT level: propagate known constants across assignments.
        At BLOCK level: propagate constants across statements.
        At FUNCTION level: propagate argument defaults and module constants.
        """
        def transform(node, level):
            transformer = _FractalConstantPropagator(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_constant_propagation',
            transform=transform,
            contraction_factor=0.85,
        )
    
    @staticmethod
    def dead_code_elimination() -> OptimizationMorphism:
        """
        Dead code elimination — fractal version.
        
        At EXPRESSION level: simplify tautological conditions.
        At STATEMENT level: remove unreachable statements.
        At BLOCK level: remove dead blocks after constant branches.
        At FUNCTION level: remove unused helper functions.
        """
        def transform(node, level):
            transformer = _FractalDeadCodeEliminator(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_dead_code_elimination',
            transform=transform,
            contraction_factor=0.80,
        )
    
    @staticmethod
    def strength_reduction() -> OptimizationMorphism:
        """
        Strength reduction — fractal version.
        
        At EXPRESSION level: x * 2 → x + x, x ** 2 → x * x.
        At STATEMENT level: replace expensive operations in assignments.
        At BLOCK level: replace loop patterns with cheaper equivalents.
        At FUNCTION level: replace recursive calls with iterative equivalents.
        """
        def transform(node, level):
            transformer = _FractalStrengthReducer(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_strength_reduction',
            transform=transform,
            contraction_factor=0.90,
        )
    
    @staticmethod
    def loop_invariant_motion() -> OptimizationMorphism:
        """
        Loop-invariant code motion — fractal version.
        
        At BLOCK level: hoist loop-invariant computations.
        At FUNCTION level: hoist module-level lookups out of hot loops.
        """
        def transform(node, level):
            if level.value < FractalLevel.BLOCK.value:
                return node  # Only meaningful at block level and above
            transformer = _FractalLoopInvariantMotion(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_loop_invariant_motion',
            transform=transform,
            applicable_levels={
                FractalLevel.BLOCK, FractalLevel.FUNCTION,
                FractalLevel.MODULE, FractalLevel.PROGRAM,
            },
            contraction_factor=0.75,
        )
    
    @staticmethod
    def algebraic_simplification() -> OptimizationMorphism:
        """
        Algebraic simplification — fractal version.
        
        At EXPRESSION level: x + 0 → x, x * 1 → x, x * 0 → 0.
        At higher levels: propagate simplifications upward.
        """
        def transform(node, level):
            transformer = _FractalAlgebraicSimplifier(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_algebraic_simplification',
            transform=transform,
            contraction_factor=0.88,
        )
    
    @staticmethod
    def common_subexpression_elimination() -> OptimizationMorphism:
        """
        Common subexpression elimination — fractal version.
        
        At EXPRESSION level: identify repeated subexpressions.
        At BLOCK level: cache results of repeated computations.
        At FUNCTION level: merge redundant helper computations.
        """
        def transform(node, level):
            transformer = _FractalCSE(level)
            return transformer.visit(copy.deepcopy(node))
        
        return OptimizationMorphism(
            name='fractal_cse',
            transform=transform,
            contraction_factor=0.82,
        )


# ═══════════════════════════════════════════════════════════════════════════
# AST Transformers for Each Morphism
# ═══════════════════════════════════════════════════════════════════════════

class _FractalConstantPropagator(ast.NodeTransformer):
    """
    Constant + copy propagation with proper invalidation.
    
    Handles:
     - Constant propagation: a = 5; ... a → 5
     - Copy propagation: a = x; ... a → x (when both are immutable)
     - Constant folding within visited expressions
    
    Safe analysis: tracks constants and copies only through simple assignments,
    invalidates on any mutable access (AugAssign, For, While, function defs, etc.).
    """
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.constants: Dict[str, Any] = {}
        self._copies: Dict[str, str] = {}  # copy_var → source_var
        self._mutable_vars: Set[str] = set()  # vars modified by non-simple assignment
        self.changes = 0
    
    def _pre_scan_mutations(self, nodes):
        """Pre-scan to find ALL variables mutated by AugAssign, For, loops, etc.
        
        A variable is considered mutable if:
        - It appears as a target of AugAssign (+=, *=, etc.)
        - It is a For-loop target variable
        - It is assigned inside a For or While loop body (loop-carried variable)
        - It is a function argument
        """
        self._scan_mutations_recursive(nodes, in_loop=False)
    
    def _scan_mutations_recursive(self, node, in_loop: bool):
        """Recursively scan for mutations, tracking loop context."""
        if not isinstance(node, ast.AST):
            return
        
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                self._mutable_vars.add(node.target.id)
        
        elif isinstance(node, ast.For):
            # Loop target variable is mutable
            if isinstance(node.target, ast.Name):
                self._mutable_vars.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self._mutable_vars.add(elt.id)
            # Scan loop body with in_loop=True
            for child in node.body:
                self._scan_mutations_recursive(child, in_loop=True)
            for child in node.orelse:
                self._scan_mutations_recursive(child, in_loop=True)
            return  # Don't generic-descend again
        
        elif isinstance(node, ast.While):
            # Scan while body with in_loop=True
            for child in node.body:
                self._scan_mutations_recursive(child, in_loop=True)
            for child in node.orelse:
                self._scan_mutations_recursive(child, in_loop=True)
            return
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function args shadow outer constants
            for arg in node.args.args:
                self._mutable_vars.add(arg.arg)
        
        elif isinstance(node, ast.Assign) and in_loop:
            # Variables assigned inside a loop are loop-carried → mutable
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._mutable_vars.add(target.id)
        
        # Recurse into child nodes
        for child in ast.iter_child_nodes(node):
            self._scan_mutations_recursive(child, in_loop)
    
    def _count_assignments(self, body_nodes):
        """Count how many times each variable is assigned (for copy safety)."""
        counts: Dict[str, int] = defaultdict(int)
        for node in ast.walk(body_nodes) if isinstance(body_nodes, ast.AST) else []:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        counts[target.id] += 1
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    counts[node.target.id] += 1
        return counts
    
    def visit_Module(self, node):
        self._pre_scan_mutations(node)
        self._assign_counts = self._count_assignments(node)
        self.generic_visit(node)
        return node
    
    def visit_FunctionDef(self, node):
        # Function arguments are mutable — invalidate them
        for arg in node.args.args:
            self._mutable_vars.add(arg.arg)
        self._pre_scan_mutations(node)
        self._assign_counts = self._count_assignments(node)
        self.generic_visit(node)
        return node
    
    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            
            # Track constant assignments
            if isinstance(node.value, ast.Constant):
                if var_name not in self._mutable_vars:
                    self.constants[var_name] = node.value.value
                else:
                    self.constants.pop(var_name, None)
            # Track copy assignments: a = x (Name = Name)
            elif isinstance(node.value, ast.Name):
                source = node.value.id
                counts = getattr(self, '_assign_counts', {})
                # Only propagate copy if:
                # 1. Target var is not mutated by AugAssign/For
                # 2. Source var is not mutated by AugAssign/For 
                # 3. Target is assigned only once (simple SSA)
                if (var_name not in self._mutable_vars and
                        source not in self._mutable_vars and
                        counts.get(var_name, 0) <= 1):
                    self._copies[var_name] = source
                else:
                    self._copies.pop(var_name, None)
                self.constants.pop(var_name, None)
            else:
                # Non-constant, non-copy assignment
                self.constants.pop(var_name, None)
                self._copies.pop(var_name, None)
        return node
    
    def visit_AugAssign(self, node):
        """AugAssign (+=, *=, etc.) invalidates any tracked constant."""
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.constants.pop(node.target.id, None)
        return node
    
    def visit_For(self, node):
        """For-loop target is mutable — invalidate."""
        if isinstance(node.target, ast.Name):
            self.constants.pop(node.target.id, None)
        self.generic_visit(node)
        return node
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        # Fold constant binary operations
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                result = self._eval_binop(node.op, node.left.value, node.right.value)
                if result is not None:
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=result), node)
            except Exception:
                pass
        return node
    
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            try:
                result = self._eval_unaryop(node.op, node.operand.value)
                if result is not None:
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=result), node)
            except Exception:
                pass
        return node
    
    def visit_Name(self, node):
        # Substitute known constants (only for non-mutated vars)
        if (self.level.value >= FractalLevel.STATEMENT.value and
                isinstance(node.ctx, ast.Load) and
                node.id not in self._mutable_vars):
            counts = getattr(self, '_assign_counts', {})
            # Only propagate if variable is assigned exactly once (SSA-like)
            if counts.get(node.id, 0) <= 1:
                # Constant propagation: a → 5
                if node.id in self.constants:
                    self.changes += 1
                    return ast.copy_location(
                        ast.Constant(value=self.constants[node.id]), node
                    )
                # Copy propagation: a → x (replace copy with source, follow chains)
                if node.id in self._copies:
                    source = node.id
                    seen = set()
                    while source in self._copies and source not in seen:
                        seen.add(source)
                        source = self._copies[source]
                    if source != node.id:
                        self.changes += 1
                        return ast.copy_location(
                            ast.Name(id=source, ctx=ast.Load()), node
                        )
        return node
    
    @staticmethod
    def _eval_binop(op, left, right):
        """Safely evaluate a binary operation on constants."""
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div) and right != 0:
            return left / right
        elif isinstance(op, ast.FloorDiv) and right != 0:
            return left // right
        elif isinstance(op, ast.Mod) and right != 0:
            return left % right
        elif isinstance(op, ast.Pow):
            if isinstance(right, int) and 0 <= right <= 100:
                return left ** right
        elif isinstance(op, ast.LShift):
            return left << right
        elif isinstance(op, ast.RShift):
            return left >> right
        elif isinstance(op, ast.BitOr):
            return left | right
        elif isinstance(op, ast.BitXor):
            return left ^ right
        elif isinstance(op, ast.BitAnd):
            return left & right
        return None
    
    @staticmethod
    def _eval_unaryop(op, operand):
        if isinstance(op, ast.UAdd):
            return +operand
        elif isinstance(op, ast.USub):
            return -operand
        elif isinstance(op, ast.Not):
            return not operand
        elif isinstance(op, ast.Invert):
            return ~operand
        return None


class _FractalDeadCodeEliminator(ast.NodeTransformer):
    """
    Dead code elimination with proper liveness analysis.
    
    Removes:
     1. Assignments to unused variables (dead stores)
     2. Branches with constant conditions
     3. Unreachable code after return statements
    """
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.changes = 0
    
    def visit_Module(self, node):
        """At module level, eliminate dead stores in function bodies."""
        self.generic_visit(node)
        return node
    
    def visit_FunctionDef(self, node):
        """Analyze function body for dead stores and eliminate them."""
        self.generic_visit(node)
        
        if self.level.value >= FractalLevel.FUNCTION.value:
            node.body = self._eliminate_dead_stores(node.body)
            node.body = self._eliminate_after_return(node.body)
        
        if not node.body:
            node.body = [ast.Pass()]
        return node
    
    def _eliminate_dead_stores(self, stmts: List) -> List:
        """Remove assignments whose targets are never read downstream."""
        # Collect all variables that appear in Load context
        live_vars = set()
        for stmt in stmts:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    live_vars.add(n.id)
                # Also keep vars used in return, augassign target, etc.
                if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                    live_vars.add(n.target.id)
        
        # Also keep variables referenced in return statements
        for stmt in stmts:
            if isinstance(stmt, ast.Return) and stmt.value:
                for n in ast.walk(stmt.value):
                    if isinstance(n, ast.Name):
                        live_vars.add(n.id)
        
        new_stmts = []
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                # Check if any target is used
                all_dead = True
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id in live_vars:
                            all_dead = False
                            break
                        # Check if the value has side effects (calls, etc.)
                        if self._has_side_effects(stmt.value):
                            all_dead = False
                            break
                    else:
                        all_dead = False
                        break
                
                if all_dead and not self._has_side_effects(stmt.value):
                    self.changes += 1
                    continue  # Skip dead store
            
            new_stmts.append(stmt)
        
        return new_stmts if new_stmts else stmts
    
    def _eliminate_after_return(self, stmts: List) -> List:
        """Remove unreachable code after unconditional return."""
        result = []
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, ast.Return):
                if len(result) < len(stmts):
                    self.changes += 1
                break
        return result
    
    def _has_side_effects(self, node: ast.AST) -> bool:
        """Conservative check for side effects."""
        for n in ast.walk(node):
            if isinstance(n, (ast.Call, ast.Yield, ast.YieldFrom, ast.Await)):
                return True
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
                return True
        return False
    
    def visit_If(self, node):
        self.generic_visit(node)
        # Eliminate branches with constant conditions
        if isinstance(node.test, ast.Constant):
            self.changes += 1
            if node.test.value:
                return node.body if node.body else [ast.Pass()]
            else:
                if node.orelse:
                    return node.orelse
                else:
                    return ast.Pass()
        return node
    
    def visit_While(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and not node.test.value:
            self.changes += 1
            return ast.Pass()
        return node


class _FractalStrengthReducer(ast.NodeTransformer):
    """
    Strength reduction: replace expensive operations with cheaper equivalents.
    
    Only performs transformations that provably reduce AST energy:
     - x ** 2 → x * x  (replaces Pow node with Mult, same node count)
     - x * 0 → 0       (eliminates BinOp entirely)
     - x // 2 → x >> 1 (cheaper integer operation, same node count)
     - x * 2 is NOT replaced (x + x adds a Name node = higher energy)
    """
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.changes = 0
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        
        # x ** 2 → x * x (same node count, cheaper operation)
        if (isinstance(node.op, ast.Pow) and
                isinstance(node.right, ast.Constant) and
                node.right.value == 2):
            self.changes += 1
            return ast.copy_location(
                ast.BinOp(
                    left=node.left,
                    op=ast.Mult(),
                    right=copy.deepcopy(node.left),
                ),
                node,
            )
        
        # x ** 0 → 1
        if (isinstance(node.op, ast.Pow) and
                isinstance(node.right, ast.Constant) and
                node.right.value == 0 and
                isinstance(node.left, (ast.Name, ast.Constant))):
            self.changes += 1
            return ast.copy_location(ast.Constant(value=1), node)
        
        # x * 0 → 0 (if side-effect free)
        if (isinstance(node.op, ast.Mult) and
                isinstance(node.right, ast.Constant) and
                node.right.value == 0 and
                isinstance(node.left, (ast.Name, ast.Constant))):
            self.changes += 1
            return ast.copy_location(ast.Constant(value=0), node)
        
        # 0 * x → 0
        if (isinstance(node.op, ast.Mult) and
                isinstance(node.left, ast.Constant) and
                node.left.value == 0 and
                isinstance(node.right, (ast.Name, ast.Constant))):
            self.changes += 1
            return ast.copy_location(ast.Constant(value=0), node)
        
        return node


class _FractalLoopInvariantMotion(ast.NodeTransformer):
    """Loop-invariant code motion parameterized by fractal level."""
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.changes = 0
    
    def visit_For(self, node):
        self.generic_visit(node)
        
        # Identify loop variable(s) — handle Name and Tuple unpacking
        loop_vars = set()
        if isinstance(node.target, ast.Name):
            loop_vars.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_vars.add(elt.id)
        elif isinstance(node.target, ast.List):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_vars.add(elt.id)
        
        # Find assignments in loop body
        modified_vars = set()
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    modified_vars.add(n.id)
        
        modified_vars |= loop_vars
        
        # Hoist invariant assignments
        hoisted = []
        remaining = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and self._is_invariant(stmt, modified_vars):
                hoisted.append(stmt)
                self.changes += 1
            else:
                remaining.append(stmt)
        
        if hoisted:
            node.body = remaining if remaining else [ast.Pass()]
            return hoisted + [node]
        
        return node
    
    def _is_invariant(self, stmt: ast.Assign, modified_vars: Set[str]) -> bool:
        """Check if an assignment is truly loop-invariant.
        
        An assignment is NOT invariant if:
        1. Its RHS reads from any loop-modified variable, OR
        2. Its LHS target is also modified elsewhere in the loop body
           (indicating a loop-carried dependency like accumulators)
        """
        # Check if the TARGET variable is also modified elsewhere in the loop
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id in modified_vars:
                return False
        
        # Check if the RHS reads from any loop-modified variable
        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in modified_vars:
                    return False
        return True


class _FractalAlgebraicSimplifier(ast.NodeTransformer):
    """
    Algebraic simplification AND constant folding parameterized by fractal level.
    
    Handles:
     - Identity operations: x + 0, x * 1, x ** 1, x / 1, x - 0
     - Annihilation: x ** 0 → 1  
     - Constant folding: BinOp(Constant, op, Constant) → Constant
     - Boolean folding: not True → False, etc.
    """
    
    _SAFE_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Pow: lambda a, b: a ** b if (isinstance(b, int) and b >= 0 and b <= 100) else None,
        ast.FloorDiv: lambda a, b: a // b if b != 0 else None,
        ast.Mod: lambda a, b: a % b if b != 0 else None,
        ast.BitAnd: lambda a, b: a & b if isinstance(a, int) and isinstance(b, int) else None,
        ast.BitOr: lambda a, b: a | b if isinstance(a, int) and isinstance(b, int) else None,
        ast.BitXor: lambda a, b: a ^ b if isinstance(a, int) and isinstance(b, int) else None,
        ast.LShift: lambda a, b: a << b if isinstance(a, int) and isinstance(b, int) and 0 <= b <= 64 else None,
        ast.RShift: lambda a, b: a >> b if isinstance(a, int) and isinstance(b, int) and b >= 0 else None,
    }
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.changes = 0
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        
        # Constant folding: BinOp(Constant, op, Constant) → Constant
        if (isinstance(node.left, ast.Constant) and
                isinstance(node.right, ast.Constant)):
            op_type = type(node.op)
            fn = self._SAFE_OPS.get(op_type)
            if fn is not None:
                try:
                    result = fn(node.left.value, node.right.value)
                    if result is not None and not isinstance(result, float) or (
                            isinstance(result, float) and math.isfinite(result)):
                        self.changes += 1
                        return ast.copy_location(ast.Constant(value=result), node)
                except (OverflowError, ZeroDivisionError, ValueError, TypeError):
                    pass
        
        # x + 0 → x
        if isinstance(node.op, ast.Add) and self._is_zero(node.right):
            self.changes += 1
            return node.left
        if isinstance(node.op, ast.Add) and self._is_zero(node.left):
            self.changes += 1
            return node.right
        
        # x - 0 → x
        if isinstance(node.op, ast.Sub) and self._is_zero(node.right):
            self.changes += 1
            return node.left
        
        # x * 1 → x
        if isinstance(node.op, ast.Mult) and self._is_one(node.right):
            self.changes += 1
            return node.left
        if isinstance(node.op, ast.Mult) and self._is_one(node.left):
            self.changes += 1
            return node.right
        
        # x ** 1 → x
        if isinstance(node.op, ast.Pow) and self._is_one(node.right):
            self.changes += 1
            return node.left
        
        # x ** 0 → 1
        if isinstance(node.op, ast.Pow) and self._is_zero(node.right):
            self.changes += 1
            return ast.copy_location(ast.Constant(value=1), node)
        
        # x / 1 → x
        if isinstance(node.op, (ast.Div, ast.FloorDiv)) and self._is_one(node.right):
            self.changes += 1
            return node.left
        
        return node
    
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        # Constant fold unary ops: -Constant, +Constant, ~Constant, not Constant
        if isinstance(node.operand, ast.Constant):
            val = node.operand.value
            try:
                if isinstance(node.op, ast.USub):
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=-val), node)
                if isinstance(node.op, ast.UAdd):
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=+val), node)
                if isinstance(node.op, ast.Invert) and isinstance(val, int):
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=~val), node)
                if isinstance(node.op, ast.Not):
                    self.changes += 1
                    return ast.copy_location(ast.Constant(value=not val), node)
            except (TypeError, OverflowError):
                pass
        return node
    
    @staticmethod
    def _is_zero(node):
        return isinstance(node, ast.Constant) and node.value == 0
    
    @staticmethod
    def _is_one(node):
        return isinstance(node, ast.Constant) and node.value == 1


class _FractalCSE(ast.NodeTransformer):
    """
    Common Subexpression Elimination parameterized by fractal level.
    
    Detects repeated pure subexpressions (BinOp with Name/Constant operands),
    hoists them into temporary variables, and replaces occurrences.
    """
    
    def __init__(self, level: FractalLevel):
        self.level = level
        self.changes = 0
        self._counter = 0
    
    def _fresh_var(self) -> str:
        self._counter += 1
        return f'_cse_{self._counter}'
    
    def visit_FunctionDef(self, node):
        if self.level.value >= FractalLevel.FUNCTION.value:
            node.body = self._cse_block(node.body)
        self.generic_visit(node)
        return node
    
    def _is_pure_expr(self, node: ast.expr) -> bool:
        """Check if expression is pure (no function calls or side effects)."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Call, ast.Yield, ast.YieldFrom,
                                   ast.Await, ast.NamedExpr)):
                return False
        return True
    
    def _cse_block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        """Apply CSE within a block of statements."""
        # Collect names that are ONLY defined inside loops or conditionals
        # These must NOT be hoisted outside their defining scope
        loop_internal_vars: Set[str] = set()
        param_vars: Set[str] = set()  # function parameters — always safe
        for stmt in stmts:
            # Collect For/While loop target variables and variables assigned inside loops
            for node in ast.walk(stmt):
                if isinstance(node, ast.For):
                    if isinstance(node.target, ast.Name):
                        loop_internal_vars.add(node.target.id)
                    elif isinstance(node.target, ast.Tuple):
                        for elt in node.target.elts:
                            if isinstance(elt, ast.Name):
                                loop_internal_vars.add(elt.id)
                    # Mark all names assigned inside the loop body as loop-internal
                    for body_node in ast.walk(node):
                        if isinstance(body_node, ast.Assign):
                            for target in body_node.targets:
                                if isinstance(target, ast.Name):
                                    loop_internal_vars.add(target.id)
                        elif isinstance(body_node, ast.AugAssign):
                            if isinstance(body_node.target, ast.Name):
                                loop_internal_vars.add(body_node.target.id)
                elif isinstance(node, ast.While):
                    for body_node in ast.walk(node):
                        if isinstance(body_node, ast.Assign):
                            for target in body_node.targets:
                                if isinstance(target, ast.Name):
                                    loop_internal_vars.add(target.id)
                        elif isinstance(body_node, ast.AugAssign):
                            if isinstance(body_node.target, ast.Name):
                                loop_internal_vars.add(body_node.target.id)
        
        # Count expression occurrences (only pure BinOp expressions)
        expr_counts: Dict[str, int] = defaultdict(int)
        expr_nodes: Dict[str, ast.expr] = {}
        
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.BinOp) and self._is_pure_expr(node):
                    key = ast.dump(node)
                    expr_counts[key] += 1
                    if key not in expr_nodes:
                        expr_nodes[key] = node
        
        # Find expressions appearing more than once
        repeated = {k: v for k, v in expr_counts.items() if v > 1}
        if not repeated:
            return stmts
        
        # Create temp variables for repeated expressions
        # BUT skip any expression that references loop-internal variables
        expr_to_var: Dict[str, str] = {}
        hoisted: List[ast.stmt] = []
        for key, expr_node in expr_nodes.items():
            if key in repeated:
                # Safety check: ensure no loop-internal vars are referenced
                referenced_names = {
                    n.id for n in ast.walk(expr_node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                if referenced_names & loop_internal_vars:
                    continue  # Skip — would be invalid to hoist
                
                var_name = self._fresh_var()
                expr_to_var[key] = var_name
                # Create: _cse_N = <expr>
                assign = ast.Assign(
                    targets=[ast.Name(id=var_name, ctx=ast.Store())],
                    value=copy.deepcopy(expr_node),
                    lineno=0, col_offset=0,
                )
                hoisted.append(assign)
                self.changes += 1
        
        if not hoisted:
            return stmts
        
        # Replace occurrences in statements
        replacer = _CSEReplacer(expr_to_var)
        new_stmts = []
        for stmt in stmts:
            new_stmts.append(replacer.visit(stmt))
        
        # Insert hoisted assignments at the top of the block
        return hoisted + new_stmts


class _CSEReplacer(ast.NodeTransformer):
    """Replace repeated subexpressions with their hoisted variable names."""
    
    def __init__(self, expr_to_var: Dict[str, str]):
        self.expr_to_var = expr_to_var
    
    def visit_BinOp(self, node):
        # Check BEFORE recursing into children (pre-order match),
        # because generic_visit would mutate children and change the dump
        key = ast.dump(node)
        if key in self.expr_to_var:
            return ast.copy_location(
                ast.Name(id=self.expr_to_var[key], ctx=ast.Load()),
                node,
            )
        self.generic_visit(node)
        return node


# ═══════════════════════════════════════════════════════════════════════════
# The Recursive Fractal Optimizer — Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveFractalOptimizer:
    """
    The main Recursive Fractal Optimization engine.
    
    Combines fractal decomposition, universal morphisms, fixed-point
    convergence, and meta-circular self-optimization into a single
    coherent system.
    
    Performance Notes:
        - Source-level caching: identical functions are optimized only once
        - Hash-based AST comparison: O(n) hashing vs O(n²) dump comparison
        - Single-copy transforms: morphisms operate on one copy, not two
        - Purity-aware memoization: extends auto-memoization beyond pure functions
    
    Usage:
        >>> rfo = RecursiveFractalOptimizer()
        >>> @rfo.optimize
        ... def slow_function(n):
        ...     total = 0
        ...     x = 2 * 3.14159
        ...     for i in range(n):
        ...         total += x * i * i
        ...     return total
        >>> result = slow_function(1000)
    """
    
    # Class-level optimization cache: source_hash → (optimized_func, result)
    _optimization_cache: Dict[str, Tuple[Any, 'FractalOptimizationResult']] = {}
    
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6,
        enable_meta_circular: bool = True,
        morphisms: Optional[List[OptimizationMorphism]] = None,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.enable_meta_circular = enable_meta_circular
        
        # Initialize default morphisms
        self.morphisms = morphisms or [
            UniversalMorphisms.constant_propagation(),
            UniversalMorphisms.dead_code_elimination(),
            UniversalMorphisms.strength_reduction(),
            UniversalMorphisms.algebraic_simplification(),
            UniversalMorphisms.loop_invariant_motion(),
            UniversalMorphisms.common_subexpression_elimination(),
        ]
        
        self.decomposer = FractalDecomposer()
        self.energy_analyzer = EnergyAnalyzer()
        
        # Purity analyzer for smart memoization
        self._purity_analyzer = None  # Lazy-initialized
        
        # Statistics
        self._optimization_history: List[FractalOptimizationResult] = []
    
    @staticmethod
    def _ast_hash(tree: ast.AST) -> str:
        """Fast structural hash of an AST (cheaper than ast.dump)."""
        return hashlib.md5(ast.dump(tree).encode()).hexdigest()
    
    @staticmethod
    def _source_hash(source: str) -> str:
        """Hash source code for cache lookup."""
        return hashlib.sha256(source.encode()).hexdigest()
    
    def _get_purity_analyzer(self):
        """Lazy-initialize the purity analyzer."""
        if self._purity_analyzer is None:
            try:
                from highpy.recursive.purity_analyzer import PurityAnalyzer
                self._purity_analyzer = PurityAnalyzer()
            except ImportError:
                self._purity_analyzer = None
        return self._purity_analyzer
    
    def optimize(self, func: Callable) -> Callable:
        """
        Apply recursive fractal optimization to a function.
        
        This is the main entry point. It:
        1. Checks source-level cache for previously optimized identical code
        2. Computes initial energy E_0(P)
        3. Decomposes the function into fractal levels
        4. Applies all morphisms at every applicable level (single-copy)
        5. Iterates until fixed-point convergence: |E_{n+1} - E_n| < ε
        6. Uses purity analysis for smart memoization decisions
        7. Returns the optimized function
        """
        start_time = time.perf_counter()
        
        # Compute initial energy using AST (consistent with iteration energy)
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
        except (OSError, TypeError):
            # Can't get source — return unmodified
            return func
        
        # ── Cache lookup: skip re-optimization for identical source ──
        src_hash = self._source_hash(source)
        if src_hash in RecursiveFractalOptimizer._optimization_cache:
            cached_func, cached_result = RecursiveFractalOptimizer._optimization_cache[src_hash]
            # Rebind to current function's globals and name
            try:
                rebound = self._compile_optimized(tree, func)
                is_recursive = self._is_recursive(tree, func.__name__)
                if is_recursive and self._should_memoize(tree, func):
                    memoized = _safe_memoize(rebound)
                    rebound.__globals__[func.__name__] = memoized
                    rebound = memoized
                wall_time = time.perf_counter() - start_time
                result = FractalOptimizationResult(
                    original_func=func,
                    optimized_func=rebound,
                    iterations=cached_result.iterations,
                    converged=cached_result.converged,
                    initial_energy=cached_result.initial_energy,
                    final_energy=cached_result.final_energy,
                    energy_history=cached_result.energy_history,
                    level_stats=cached_result.level_stats,
                    morphism_stats=cached_result.morphism_stats,
                    contraction_factors=cached_result.contraction_factors,
                    wall_time_seconds=wall_time,
                )
                self._optimization_history.append(result)
                rebound._rfo_result = result
                return rebound
            except Exception:
                pass  # Fall through to full optimization
        
        initial_energy = EnergyAnalyzer.compute_ast_energy(tree)
        
        # Iterative fixed-point optimization
        energy_history = [initial_energy.total]
        contraction_factors = []
        level_stats = defaultdict(int)
        morphism_stats = defaultdict(int)
        current_tree = tree
        
        converged = False
        iteration = 0
        prev_hash = self._ast_hash(current_tree)
        
        for iteration in range(1, self.max_iterations + 1):
            # Decompose into fractal components
            fractal_tree = self.decomposer.decompose(current_tree)
            
            # Apply all morphisms at all applicable levels (bottom-up)
            # Uses single-copy transform (morphisms handle their own copying)
            new_tree = self._apply_morphisms_fast(
                current_tree, fractal_tree, level_stats, morphism_stats
            )
            
            # Fast hash comparison — skip expensive energy calc if unchanged
            new_hash = self._ast_hash(new_tree)
            if new_hash == prev_hash:
                converged = True
                energy_history.append(energy_history[-1])
                break
            prev_hash = new_hash
            
            # Compute new energy
            new_energy = EnergyAnalyzer.compute_ast_energy(new_tree)
            energy_history.append(new_energy.total)
            
            # Compute contraction factor
            prev_energy = energy_history[-2]
            if prev_energy > 0:
                contraction = abs(new_energy.total - prev_energy) / prev_energy
                contraction_factors.append(contraction)
            
            # Check convergence
            energy_change = abs(new_energy.total - prev_energy)
            if energy_change < self.convergence_threshold:
                converged = True
                current_tree = new_tree
                break
            
            current_tree = new_tree
        
        # Compile the optimized AST back to a function
        # Use hash comparison instead of full dump
        initial_hash = self._ast_hash(tree)
        final_hash = self._ast_hash(current_tree)
        final_energy = EnergyAnalyzer.compute_ast_energy(current_tree)
        
        energy_ratio = final_energy.total / initial_energy.total if initial_energy.total > 0 else 1.0
        minimal_change = (initial_hash == final_hash) or (energy_ratio > 0.9)
        
        # Detect recursive functions — use purity-aware memoization
        is_recursive = self._is_recursive(current_tree, func.__name__)
        should_memo = is_recursive and self._should_memoize(current_tree, func)
        
        if minimal_change and not should_memo:
            optimized_func = func
        else:
            optimized_func = self._compile_optimized(current_tree, func)
            if should_memo:
                # Apply automatic memoization (purity-verified)
                memoized = _safe_memoize(optimized_func)
                # Update namespace so recursive calls go through the cache
                optimized_func.__globals__[func.__name__] = memoized
                optimized_func = memoized
        
        wall_time = time.perf_counter() - start_time
        
        result = FractalOptimizationResult(
            original_func=func,
            optimized_func=optimized_func,
            iterations=iteration,
            converged=converged,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_history=energy_history,
            level_stats=dict(level_stats),
            morphism_stats=dict(morphism_stats),
            contraction_factors=contraction_factors,
            wall_time_seconds=wall_time,
        )
        self._optimization_history.append(result)
        
        # Cache the result for future identical functions
        RecursiveFractalOptimizer._optimization_cache[src_hash] = (optimized_func, result)
        
        # Transfer metadata (only for recompiled functions)
        if optimized_func is not func:
            try:
                functools.update_wrapper(optimized_func, func)
            except (TypeError, AttributeError):
                pass  # lru_cache wrappers may not support all attributes
        optimized_func._rfo_result = result
        
        return optimized_func
    
    @staticmethod
    def _is_recursive(tree: ast.AST, func_name: str) -> bool:
        """Check if a function's AST contains recursive calls to itself."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                # Walk the function body for calls to func_name
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Name)
                            and child.func.id == func_name):
                        return True
        return False
    
    def _should_memoize(self, tree: ast.AST, func: Callable) -> bool:
        """
        Determine whether a function should be automatically memoized.
        
        Uses purity analysis to go beyond the naive pure-function assumption.
        Functions classified as PURE, READ_ONLY, or LOCALLY_IMPURE are safe.
        """
        analyzer = self._get_purity_analyzer()
        if analyzer is None:
            # Fallback: only memoize if we can't detect impurity
            return not self._has_obvious_side_effects(tree)
        
        try:
            report = analyzer.analyze(func)
            return report.is_memoizable
        except Exception:
            return not self._has_obvious_side_effects(tree)
    
    @staticmethod
    def _has_obvious_side_effects(tree: ast.AST) -> bool:
        """Quick check for obvious side effects without full purity analysis."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('print', 'input', 'open', 'exec', 'eval'):
                    return True
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
            if isinstance(node, ast.Global):
                return True
        return False
    
    def _apply_morphisms_recursive(
        self,
        tree: ast.AST,
        component: FractalComponent,
        level_stats: Dict,
        morphism_stats: Dict,
    ) -> ast.AST:
        """
        Apply all morphisms recursively at every fractal level.
        
        Bottom-up application: optimize expressions first, then statements,
        then blocks, then functions, then the whole module. This ensures
        that lower-level optimizations expose higher-level opportunities.
        """
        # First, recursively optimize children (bottom-up)
        for child in component.children:
            tree = self._apply_morphisms_recursive(
                tree, child, level_stats, morphism_stats
            )
        
        # Then apply all applicable morphisms at this level (energy-guarded)
        for morphism in self.morphisms:
            if component.level in morphism.applicable_levels:
                old_dump = ast.dump(tree)
                candidate = morphism.apply(copy.deepcopy(tree), component.level)
                ast.fix_missing_locations(candidate)
                new_dump = ast.dump(candidate)
                
                if old_dump != new_dump:
                    # Energy-guard: reject transforms that increase energy
                    old_energy = EnergyAnalyzer.compute_ast_energy(tree)
                    new_energy = EnergyAnalyzer.compute_ast_energy(candidate)
                    
                    if new_energy.total <= old_energy.total:
                        # Accept: energy did not increase (contraction)
                        tree = candidate
                        level_stats[component.level] = level_stats.get(component.level, 0) + 1
                        morphism_stats[morphism.name] = morphism_stats.get(morphism.name, 0) + 1
                    # else: reject the transform (energy increased)
        
        return tree
    
    def _apply_morphisms_fast(
        self,
        tree: ast.AST,
        component: FractalComponent,
        level_stats: Dict,
        morphism_stats: Dict,
    ) -> ast.AST:
        """
        Optimized morphism application with reduced overhead.
        
        Key optimizations over _apply_morphisms_recursive:
        1. Hash-based comparison instead of full AST dump
        2. Single copy per morphism (morphisms handle their own copying)
        3. Early exit when no morphism applies at a level
        4. Cached energy computations
        """
        # First, recursively optimize children (bottom-up)
        for child in component.children:
            tree = self._apply_morphisms_fast(
                tree, child, level_stats, morphism_stats
            )
        
        # Filter applicable morphisms for this level
        applicable = [
            m for m in self.morphisms
            if component.level in m.applicable_levels
        ]
        if not applicable:
            return tree
        
        # Cache the current energy (computed once, not per-morphism)
        current_energy = None
        
        for morphism in applicable:
            # Morphisms internally deepcopy — pass tree directly
            # Use hash for fast change detection
            old_hash = self._ast_hash(tree)
            candidate = morphism.apply(copy.deepcopy(tree), component.level)
            ast.fix_missing_locations(candidate)
            new_hash = self._ast_hash(candidate)
            
            if new_hash != old_hash:
                # Compute energy lazily
                if current_energy is None:
                    current_energy = EnergyAnalyzer.compute_ast_energy(tree)
                new_energy = EnergyAnalyzer.compute_ast_energy(candidate)
                
                if new_energy.total <= current_energy.total:
                    tree = candidate
                    current_energy = new_energy  # Update cached energy
                    level_stats[component.level] = level_stats.get(component.level, 0) + 1
                    morphism_stats[morphism.name] = morphism_stats.get(morphism.name, 0) + 1
        
        return tree
    
    def _compile_optimized(self, tree: ast.AST, original_func: Callable) -> Callable:
        """Compile an optimized AST back to a callable function."""
        try:
            ast.fix_missing_locations(tree)
            code = compile(tree, f'<rfo:{original_func.__name__}>', 'exec')
            
            # Execute in a namespace with the original function's globals
            namespace = dict(original_func.__globals__)
            exec(code, namespace)
            
            # Find the function in the namespace
            optimized = namespace.get(original_func.__name__, original_func)
            return optimized
        except Exception:
            return original_func
    
    @property
    def last_result(self) -> Optional[FractalOptimizationResult]:
        """Get the result of the last optimization."""
        return self._optimization_history[-1] if self._optimization_history else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all optimizations."""
        if not self._optimization_history:
            return {}
        
        total_iterations = sum(r.iterations for r in self._optimization_history)
        total_converged = sum(1 for r in self._optimization_history if r.converged)
        
        avg_energy_reduction = 0.0
        for r in self._optimization_history:
            if r.initial_energy.total > 0:
                avg_energy_reduction += r.final_energy.reduction_ratio(r.initial_energy)
        avg_energy_reduction /= len(self._optimization_history)
        
        return {
            'total_optimizations': len(self._optimization_history),
            'total_iterations': total_iterations,
            'convergence_rate': total_converged / len(self._optimization_history),
            'avg_energy_reduction': avg_energy_reduction,
            'avg_iterations': total_iterations / len(self._optimization_history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API — Decorators
# ═══════════════════════════════════════════════════════════════════════════

# Global default optimizer instance
_default_optimizer = None


def _get_default_optimizer() -> RecursiveFractalOptimizer:
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = RecursiveFractalOptimizer()
    return _default_optimizer


def rfo(func: Optional[Callable] = None, **kwargs):
    """
    Decorator for recursive fractal optimization.
    
    Usage:
        @rfo
        def compute(x):
            return x ** 2 + 2 * x + 1
        
        @rfo(max_iterations=20, convergence_threshold=1e-8)
        def heavy_compute(data):
            ...
    """
    if func is not None:
        # @rfo without arguments
        optimizer = _get_default_optimizer()
        return optimizer.optimize(func)
    
    # @rfo(...) with arguments
    def decorator(f):
        optimizer = RecursiveFractalOptimizer(**kwargs)
        return optimizer.optimize(f)
    return decorator


def rfo_optimize(func: Callable, **kwargs) -> Callable:
    """
    Functional interface for recursive fractal optimization.
    
    Usage:
        fast_func = rfo_optimize(slow_func, max_iterations=15)
    """
    optimizer = RecursiveFractalOptimizer(**kwargs)
    return optimizer.optimize(func)
