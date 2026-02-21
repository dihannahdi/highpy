"""
Tests for the function specializer.

Validates:
  - Function inlining
  - Memoization with type-aware caching
  - Partial evaluation
  - Cross-function optimization
"""

import pytest
from highpy.optimization.function_specializer import FunctionSpecializer


def helper(x):
    return x * 2


def caller(y):
    return helper(y) + 1


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def polynomial(x, a=1, b=0, c=0):
    return a * x * x + b * x + c


class TestFunctionSpecializer:
    def setup_method(self):
        self.spec = FunctionSpecializer()

    def test_memoize_basic(self):
        call_count = [0]
        def expensive(x):
            call_count[0] += 1
            return x * x
        
        # memoize() returns a decorator factory, so we need ()
        memoized = self.spec.memoize()(expensive)
        assert memoized(5) == 25
        assert memoized(5) == 25  # Should be cached
        assert memoized(3) == 9

    def test_memoize_fibonacci(self):
        """Memoized fibonacci should be fast."""
        memoized_fib = self.spec.memoize()(fib)
        assert memoized_fib(10) == 55
        assert memoized_fib(20) == 6765

    def test_partial_evaluate(self):
        """Partial evaluation with known constants."""
        partial = self.spec.partial_evaluate(polynomial, {'a': 2, 'b': 3, 'c': 1})
        assert callable(partial)
        # f(x) = 2x^2 + 3x + 1
        assert partial(0) == 1
        assert partial(1) == 6
        assert partial(2) == 15

    def test_inline_small_function(self):
        """Should be able to inline small helper functions."""
        inlined = self.spec.inline(helper)
        assert callable(inlined)
        assert inlined(5) == 10

    def test_optimize_preserves_semantics(self):
        """Optimization should not change results."""
        optimized = self.spec.optimize(caller)
        assert callable(optimized)
        for v in range(-10, 10):
            assert optimized(v) == caller(v)
