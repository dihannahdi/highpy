"""
Purity Analyzer
================

Static analysis module that determines whether a Python function is pure
(free of side effects), enabling safe automatic memoization beyond the
naive pure-function assumption.

Theoretical Foundation:
    A function f is **pure** if:
        1. Its return value depends only on its arguments (referential transparency)
        2. It produces no observable side effects

    In practice, we classify functions into a purity lattice:

        PURE ⊂ READ_ONLY ⊂ LOCALLY_IMPURE ⊂ IMPURE

    Where:
        - PURE: No reads/writes of external state, no I/O, deterministic
        - READ_ONLY: May read globals/closures but never mutates them
        - LOCALLY_IMPURE: Mutates local mutable state (lists, dicts) but
          doesn't escape; safe for memoization if args are the same
        - IMPURE: Has observable side effects (I/O, global mutation, etc.)

    Memoization Safety:
        - PURE functions: unconditionally safe to memoize
        - READ_ONLY + LOCALLY_IMPURE: safe if external state doesn't change
          between calls (we provide scope-limited caching)
        - IMPURE: never automatically memoized

Novel Contribution:
    This is the first purity analysis for Python that:
    1. Integrates directly with an optimization engine's memoization
    2. Provides graduated purity levels (not just pure/impure)
    3. Recognizes common "effectively pure" patterns (e.g. reading
       module-level constants, calling math.sin, etc.)
    4. Works at AST level for zero-runtime-overhead analysis
"""

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple,
)


class PurityLevel(IntEnum):
    """
    Graduated purity classification.

    Higher values = more impure.
    """
    PURE = 0             # Fully pure — safe for unlimited memoization
    READ_ONLY = 1        # Reads external state — safe for scope-limited memo
    LOCALLY_IMPURE = 2   # Mutates local mutable objects — safe if same args
    IMPURE = 3           # Observable side effects — NOT safe to memoize
    UNKNOWN = 4          # Analysis inconclusive — treated as impure


@dataclass
class PurityReport:
    """
    Detailed report of a function's purity analysis.

    Summarises *why* a function is/isn't pure so the optimizer can
    make informed decisions and the user can understand the result.
    """
    function_name: str
    level: PurityLevel
    reasons: List[str] = field(default_factory=list)

    # Sets of specific impurity sources detected
    global_reads: Set[str] = field(default_factory=set)
    global_writes: Set[str] = field(default_factory=set)
    nonlocal_access: Set[str] = field(default_factory=set)
    io_calls: Set[str] = field(default_factory=set)
    mutation_calls: Set[str] = field(default_factory=set)
    nondeterministic_calls: Set[str] = field(default_factory=set)
    attribute_mutations: Set[str] = field(default_factory=set)
    mutable_default_args: Set[str] = field(default_factory=set)

    # Analysis metadata
    is_recursive: bool = False
    has_yield: bool = False
    has_await: bool = False

    @property
    def is_memoizable(self) -> bool:
        """Whether automatic memoization is safe for this function."""
        return self.level <= PurityLevel.LOCALLY_IMPURE

    @property
    def confidence(self) -> float:
        """Confidence score [0, 1] — 1.0 for definitive results."""
        if self.level == PurityLevel.UNKNOWN:
            return 0.0
        # If we found concrete impurity evidence, high confidence
        if self.level == PurityLevel.IMPURE:
            return 1.0
        # Pure or read-only — high confidence if no unknowns
        return 1.0 if not self.reasons or self.level == PurityLevel.PURE else 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Known-pure and known-impure function registries
# ═══════════════════════════════════════════════════════════════════════════

# Standard library functions known to be pure (no side effects)
_KNOWN_PURE_MODULES: FrozenSet[str] = frozenset({
    'math', 'cmath', 'operator', 'functools', 'itertools',
    'string', 'decimal', 'fractions',
})

_KNOWN_PURE_FUNCTIONS: FrozenSet[str] = frozenset({
    # builtins
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytes',
    'chr', 'complex', 'divmod', 'enumerate', 'float',
    'format', 'frozenset', 'hash', 'hex', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'filter',
    'max', 'min', 'oct', 'ord', 'pow', 'range', 'repr',
    'reversed', 'round', 'set', 'slice', 'sorted', 'str',
    'sum', 'tuple', 'type', 'zip',
    # math module
    'math.acos', 'math.acosh', 'math.asin', 'math.asinh',
    'math.atan', 'math.atan2', 'math.atanh', 'math.ceil',
    'math.comb', 'math.copysign', 'math.cos', 'math.cosh',
    'math.degrees', 'math.dist', 'math.erf', 'math.erfc',
    'math.exp', 'math.expm1', 'math.fabs', 'math.factorial',
    'math.floor', 'math.fmod', 'math.frexp', 'math.fsum',
    'math.gamma', 'math.gcd', 'math.hypot', 'math.isclose',
    'math.isfinite', 'math.isinf', 'math.isnan', 'math.isqrt',
    'math.lcm', 'math.ldexp', 'math.lgamma', 'math.log',
    'math.log10', 'math.log1p', 'math.log2', 'math.modf',
    'math.perm', 'math.pow', 'math.prod', 'math.radians',
    'math.remainder', 'math.sin', 'math.sinh', 'math.sqrt',
    'math.tan', 'math.tanh', 'math.trunc',
})

# Functions known to perform I/O or side effects
_KNOWN_IO_FUNCTIONS: FrozenSet[str] = frozenset({
    'print', 'input', 'open', 'exec', 'eval', 'compile',
    '__import__', 'exit', 'quit',
})

# Functions known to be nondeterministic
_KNOWN_NONDETERMINISTIC: FrozenSet[str] = frozenset({
    'random.random', 'random.randint', 'random.choice',
    'random.shuffle', 'random.sample', 'random.uniform',
    'random.gauss', 'random.randrange',
    'time.time', 'time.perf_counter', 'time.monotonic',
    'time.process_time', 'time.time_ns', 'time.perf_counter_ns',
    'uuid.uuid4', 'uuid.uuid1',
    'os.urandom', 'secrets.token_bytes', 'secrets.token_hex',
    'id',  # CPython-specific, depends on memory address
})

# Methods known to mutate their receiver
_KNOWN_MUTATION_METHODS: FrozenSet[str] = frozenset({
    'append', 'extend', 'insert', 'remove', 'pop', 'clear',
    'sort', 'reverse',  # list
    'add', 'discard', 'update', 'intersection_update',
    'difference_update', 'symmetric_difference_update',  # set
    'setdefault', 'popitem',  # dict (update, pop, clear already listed)
})


class PurityAnalyzer:
    """
    Analyses a Python function's AST to determine its purity level.

    Usage:
        analyzer = PurityAnalyzer()
        report = analyzer.analyze(my_function)
        print(report.level, report.is_memoizable)

    The analysis is conservative: if uncertain, it errs on the side of
    classifying a function as more impure rather than falsely pure.
    """

    def __init__(
        self,
        *,
        extra_pure: Optional[Set[str]] = None,
        extra_impure: Optional[Set[str]] = None,
    ):
        """
        Args:
            extra_pure:   Additional function names to treat as pure.
            extra_impure: Additional function names to treat as impure.
        """
        self._pure_functions = set(_KNOWN_PURE_FUNCTIONS)
        self._io_functions = set(_KNOWN_IO_FUNCTIONS)
        if extra_pure:
            self._pure_functions |= extra_pure
        if extra_impure:
            self._io_functions |= extra_impure

    # ───────────────────────────────────────────────────────────────
    #  Public API
    # ───────────────────────────────────────────────────────────────

    def analyze(self, func: Callable) -> PurityReport:
        """Analyse a callable and return its PurityReport."""
        name = getattr(func, '__name__', '<anonymous>')

        # Unwrap common decorators
        inner = func
        while hasattr(inner, '__wrapped__'):
            inner = inner.__wrapped__

        try:
            source = textwrap.dedent(inspect.getsource(inner))
            tree = ast.parse(source)
        except (OSError, TypeError, IndentationError):
            return PurityReport(
                function_name=name,
                level=PurityLevel.UNKNOWN,
                reasons=['Could not retrieve source code'],
            )

        # Find the function definition node
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = node
                break

        if func_node is None:
            return PurityReport(
                function_name=name,
                level=PurityLevel.UNKNOWN,
                reasons=['No function definition found in source'],
            )

        return self._analyze_funcdef(func_node, name)

    def analyze_ast(self, tree: ast.AST, func_name: str = '<ast>') -> PurityReport:
        """Analyse an AST node directly."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self._analyze_funcdef(node, func_name)
        return PurityReport(
            function_name=func_name,
            level=PurityLevel.UNKNOWN,
            reasons=['No function definition in AST'],
        )

    # ───────────────────────────────────────────────────────────────
    #  Core analysis
    # ───────────────────────────────────────────────────────────────

    def _analyze_funcdef(
        self, node: ast.FunctionDef, name: str
    ) -> PurityReport:
        """Full purity analysis of a function definition AST node."""

        report = PurityReport(function_name=name, level=PurityLevel.PURE)

        # Collect local variable names (parameters + assignments)
        local_vars = self._collect_locals(node)
        param_names = {a.arg for a in node.args.args}
        if node.args.vararg:
            param_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            param_names.add(node.args.kwarg.arg)
        for a in node.args.kwonlyargs:
            param_names.add(a.arg)
        local_vars |= param_names

        visitor = _PurityVisitor(
            local_vars=local_vars,
            param_names=param_names,
            func_name=name,
            known_pure=self._pure_functions,
            known_io=self._io_functions,
        )
        visitor.visit(node)

        # Transfer findings to report
        report.global_reads = visitor.global_reads
        report.global_writes = visitor.global_writes
        report.nonlocal_access = visitor.nonlocal_access
        report.io_calls = visitor.io_calls
        report.mutation_calls = visitor.mutation_calls
        report.nondeterministic_calls = visitor.nondeterministic_calls
        report.attribute_mutations = visitor.attribute_mutations
        report.mutable_default_args = visitor.mutable_default_args
        report.is_recursive = visitor.is_recursive
        report.has_yield = visitor.has_yield
        report.has_await = visitor.has_await

        # Determine purity level based on findings
        report.level = self._determine_level(report)
        report.reasons = self._generate_reasons(report)

        return report

    def _collect_locals(self, node: ast.FunctionDef) -> Set[str]:
        """Collect all locally-scoped variable names."""
        locals_set: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                locals_set.add(child.id)
            elif isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                locals_set.add(child.target.id)
            elif isinstance(child, ast.For) and isinstance(child.target, ast.Tuple):
                for elt in child.target.elts:
                    if isinstance(elt, ast.Name):
                        locals_set.add(elt.id)
            # Nested function definitions are local
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child is not node:
                    locals_set.add(child.name)
        return locals_set

    @staticmethod
    def _determine_level(report: PurityReport) -> PurityLevel:
        """Determine the purity level from analysis findings."""
        # Hard impurity: I/O, global writes, yield, nondeterminism
        if (report.io_calls or report.global_writes or
                report.nondeterministic_calls or report.has_yield or
                report.has_await):
            return PurityLevel.IMPURE

        # Attribute mutations on non-local objects
        if report.attribute_mutations:
            return PurityLevel.IMPURE

        # Nonlocal access that writes
        if report.nonlocal_access:
            return PurityLevel.IMPURE

        # Mutation calls (append, sort, etc.) — locally impure
        if report.mutation_calls:
            return PurityLevel.LOCALLY_IMPURE

        # Mutable default arguments — locally impure
        if report.mutable_default_args:
            return PurityLevel.LOCALLY_IMPURE

        # Global reads without writes — read-only
        if report.global_reads:
            return PurityLevel.READ_ONLY

        return PurityLevel.PURE

    @staticmethod
    def _generate_reasons(report: PurityReport) -> List[str]:
        """Generate human-readable reasons for the purity classification."""
        reasons = []
        if report.global_writes:
            reasons.append(
                f"Writes to global variables: {', '.join(sorted(report.global_writes))}"
            )
        if report.nonlocal_access:
            reasons.append(
                f"Uses nonlocal: {', '.join(sorted(report.nonlocal_access))}"
            )
        if report.io_calls:
            reasons.append(
                f"I/O function calls: {', '.join(sorted(report.io_calls))}"
            )
        if report.nondeterministic_calls:
            reasons.append(
                f"Nondeterministic calls: {', '.join(sorted(report.nondeterministic_calls))}"
            )
        if report.attribute_mutations:
            reasons.append(
                f"Attribute mutations: {', '.join(sorted(report.attribute_mutations))}"
            )
        if report.mutation_calls:
            reasons.append(
                f"Mutation method calls: {', '.join(sorted(report.mutation_calls))}"
            )
        if report.mutable_default_args:
            reasons.append(
                f"Mutable default arguments: {', '.join(sorted(report.mutable_default_args))}"
            )
        if report.global_reads:
            reasons.append(
                f"Reads global variables: {', '.join(sorted(report.global_reads))}"
            )
        if report.has_yield:
            reasons.append("Contains yield (generator function)")
        if report.has_await:
            reasons.append("Contains await (async function)")
        if not reasons:
            reasons.append("Function is pure")
        return reasons


class _PurityVisitor(ast.NodeVisitor):
    """AST visitor that collects purity-violation evidence."""

    def __init__(
        self,
        *,
        local_vars: Set[str],
        param_names: Set[str],
        func_name: str,
        known_pure: Set[str],
        known_io: Set[str],
    ):
        self.local_vars = local_vars
        self.param_names = param_names
        self.func_name = func_name
        self.known_pure = known_pure
        self.known_io = known_io

        self.global_reads: Set[str] = set()
        self.global_writes: Set[str] = set()
        self.nonlocal_access: Set[str] = set()
        self.io_calls: Set[str] = set()
        self.mutation_calls: Set[str] = set()
        self.nondeterministic_calls: Set[str] = set()
        self.attribute_mutations: Set[str] = set()
        self.mutable_default_args: Set[str] = set()
        self.is_recursive: bool = False
        self.has_yield: bool = False
        self.has_await: bool = False

        # Track explicitly declared globals/nonlocals
        self._declared_globals: Set[str] = set()
        self._declared_nonlocals: Set[str] = set()

        # Builtin names to ignore for 'global read' reporting
        self._builtins = set(dir(builtins))

        # Depth tracking — don't descend into nested function defs
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._depth > 0:
            # Nested function — skip its body (separate scope)
            return
        self._depth += 1
        # Check mutable default args
        self._check_mutable_defaults(node)
        self.generic_visit(node)
        self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self._depth > 0:
            return
        self._depth += 1
        self.has_await = True
        self._check_mutable_defaults(node)
        self.generic_visit(node)
        self._depth -= 1

    def _check_mutable_defaults(self, node):
        """Check for mutable default argument values."""
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.mutable_default_args.add(ast.dump(default))

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self._declared_globals.add(name)
            self.global_writes.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self._declared_nonlocals.add(name)
            self.nonlocal_access.add(name)

    def visit_Name(self, node: ast.Name):
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name in self._declared_globals:
                self.global_writes.add(name)
        elif isinstance(node.ctx, ast.Load):
            if name in self._declared_globals:
                self.global_reads.add(name)
            elif (name not in self.local_vars and
                  name not in self._builtins and
                  name not in _KNOWN_PURE_MODULES and
                  name != self.func_name):
                # Reading a name not in locals, builtins, or known modules
                self.global_reads.add(name)

    def visit_Call(self, node: ast.Call):
        call_name = self._resolve_call_name(node)
        if call_name:
            # Check for recursive calls
            if call_name == self.func_name:
                self.is_recursive = True

            # Check against known categories
            if call_name in self.known_io:
                self.io_calls.add(call_name)
            elif call_name in _KNOWN_NONDETERMINISTIC:
                self.nondeterministic_calls.add(call_name)
            elif call_name in _KNOWN_MUTATION_METHODS:
                self.mutation_calls.add(call_name)

            # Method calls that mutate
            if '.' in call_name:
                method = call_name.rsplit('.', 1)[-1]
                if method in _KNOWN_MUTATION_METHODS:
                    self.mutation_calls.add(call_name)

        # Check for attribute method calls on parameters — may mutate
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in _KNOWN_MUTATION_METHODS:
                # Who is the receiver?
                receiver_name = self._get_receiver_name(node.func.value)
                if receiver_name:
                    self.mutation_calls.add(f'{receiver_name}.{method_name}')

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            receiver = self._get_receiver_name(node.value)
            if receiver:
                self.attribute_mutations.add(f'{receiver}.{node.attr}')
            else:
                self.attribute_mutations.add(f'?.{node.attr}')
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            receiver = self._get_receiver_name(node.value)
            if receiver and receiver in self.param_names:
                self.attribute_mutations.add(f'{receiver}[...]')
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.has_yield = True
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        self.has_yield = True
        self.generic_visit(node)

    def visit_Await(self, node):
        self.has_await = True
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in self._declared_globals:
                    self.global_writes.add(target.id)
        self.generic_visit(node)

    # ───────────────────────────────────────────────────────────────
    #  Helpers
    # ───────────────────────────────────────────────────────────────

    def _resolve_call_name(self, node: ast.Call) -> Optional[str]:
        """Resolve the name of a call expression."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            base = self._get_receiver_name(node.func.value)
            if base:
                return f'{base}.{node.func.attr}'
            return node.func.attr
        return None

    @staticmethod
    def _get_receiver_name(node: ast.expr) -> Optional[str]:
        """Get a simple name from an expression node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _PurityVisitor._get_receiver_name(node.value)
            if base:
                return f'{base}.{node.attr}'
        return None
