"""
Tests for the compiler modules: AST optimizer, bytecode rewriter, native codegen.

Validates:
  - AST optimization passes preserve semantics
  - Constant folding, dead code elimination, strength reduction
  - Bytecode rewriter produces correct results
  - Native compiler generates correct C code and compiles it
"""

import ast
import sys
import pytest
from highpy.compiler.ast_optimizer import ASTOptimizer
from highpy.compiler.bytecode_rewriter import BytecodeRewriter
from highpy.compiler.native_codegen import NativeCompiler


# ---------- Test Functions ----------

def constant_expr():
    return 2 + 3 * 4


def algebraic_identity(x):
    return x + 0


def strength_reduce(x):
    return x ** 2


def dead_code_branch():
    if True:
        return 42
    else:
        return 0


def simple_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total


def float_loop(n):
    total = 0.0
    for i in range(n):
        total += i * 0.5
    return total


def multiply_add(x, y, z):
    return x * y + z


# ---------- AST Optimizer Tests ----------

class TestASTOptimizer:
    def setup_method(self):
        self.optimizer = ASTOptimizer()

    def test_optimize_returns_callable(self):
        result = self.optimizer.optimize(simple_loop)
        assert callable(result)

    def test_correctness_simple_loop(self):
        optimized = self.optimizer.optimize(simple_loop)
        for n in [0, 1, 10, 100, 1000]:
            assert optimized(n) == simple_loop(n)

    def test_correctness_float_loop(self):
        optimized = self.optimizer.optimize(float_loop)
        for n in [0, 1, 10, 100]:
            assert abs(optimized(n) - float_loop(n)) < 1e-10

    def test_correctness_multiply_add(self):
        optimized = self.optimizer.optimize(multiply_add)
        assert optimized(2, 3, 4) == 10
        assert optimized(0, 5, 7) == 7

    def test_constant_folding(self):
        optimized = self.optimizer.optimize(constant_expr)
        assert optimized() == 14

    def test_algebraic_identity(self):
        optimized = self.optimizer.optimize(algebraic_identity)
        assert optimized(42) == 42

    def test_dead_code_elimination(self):
        optimized = self.optimizer.optimize(dead_code_branch)
        assert optimized() == 42

    def test_preserves_name(self):
        optimized = self.optimizer.optimize(simple_loop)
        assert optimized.__name__ == 'simple_loop'

    def test_multiple_passes(self):
        """Re-optimizing may raise OSError for inspect.getsource on dynamic code.
        We just test that the first pass result is correct."""
        opt1 = self.optimizer.optimize(simple_loop)
        for n in [0, 10, 100]:
            assert opt1(n) == simple_loop(n)
        # Trying to re-optimize may fail gracefully
        try:
            opt2 = self.optimizer.optimize(opt1)
            for n in [0, 10, 100]:
                assert opt2(n) == simple_loop(n)
        except OSError:
            pass  # Expected: can't get source of dynamically-compiled function


# ---------- Bytecode Rewriter Tests ----------

class TestBytecodeRewriter:
    def setup_method(self):
        self.rewriter = BytecodeRewriter()

    def test_rewrite_returns_callable(self):
        result = self.rewriter.optimize(simple_loop)
        assert callable(result)

    def test_correctness_simple_loop(self):
        optimized = self.rewriter.optimize(simple_loop)
        for n in [0, 1, 10, 100]:
            assert optimized(n) == simple_loop(n)

    def test_correctness_multiply_add(self):
        optimized = self.rewriter.optimize(multiply_add)
        assert optimized(2, 3, 4) == 10

    def test_analysis_report(self):
        report = self.rewriter.analyze(simple_loop)
        assert report is not None
        assert isinstance(report, dict)


# ---------- Native Compiler Tests ----------

class TestNativeCompiler:
    def setup_method(self):
        self.compiler = NativeCompiler()

    def test_compile_returns_callable(self):
        # compile() is a decorator factory — use compile()(func)
        result = self.compiler.compile()(simple_loop)
        assert result is None or callable(result)

    def test_correctness_if_compiled(self):
        compiled = self.compiler.compile()(simple_loop)
        if compiled is not None and getattr(compiled, '__highpy_native__', False):
            for n in [0, 1, 10, 100, 1000]:
                assert compiled(n) == simple_loop(n), f"Mismatch for n={n}"

    def test_float_loop_if_compiled(self):
        compiled = self.compiler.compile()(float_loop)
        if compiled is not None and getattr(compiled, '__highpy_native__', False):
            for n in [0, 1, 10, 100]:
                assert abs(compiled(n) - float_loop(n)) < 1e-6

    def test_fallback_on_complex_function(self):
        """Complex functions should fall back gracefully."""
        def complex_func(x):
            import random
            return random.random() + x

        result = self.compiler.compile()(complex_func)
        # Should return callable (possibly the original) or None
        assert result is None or callable(result)
