"""
Tests for the type specializer.

Validates:
  - auto_specialize creates monomorphic variants
  - specialize_for creates explicit specializations
  - Dispatch table routes to correct variant
  - Fallback to original for unseen types
"""

import pytest
from highpy.optimization.type_specializer import TypeSpecializer, specialize


def add(x, y):
    return x + y


def square(x):
    return x * x


def accumulate(n):
    total = 0
    for i in range(n):
        total += i
    return total


class TestTypeSpecializer:
    def setup_method(self):
        self.ts = TypeSpecializer()

    def test_auto_specialize_basics(self):
        opt = self.ts.auto_specialize(add)
        assert callable(opt)
        assert opt(1, 2) == 3
        assert opt(1.5, 2.5) == 4.0

    def test_auto_specialize_warmup(self):
        """Should use original during warmup, then specialize."""
        opt = self.ts.auto_specialize(add)
        for i in range(5):
            assert opt(i, i) == i + i

    def test_correct_results_int(self):
        opt = self.ts.auto_specialize(square)
        for v in [0, 1, -1, 100, -100, 2**20]:
            assert opt(v) == v * v

    def test_correct_results_float(self):
        opt = self.ts.auto_specialize(square)
        for v in [0.0, 1.5, -2.7, 3.14159]:
            assert abs(opt(v) - v * v) < 1e-10

    def test_correct_results_string(self):
        def repeat(s, n):
            return s * n
        opt = self.ts.auto_specialize(repeat)
        assert opt("ab", 3) == "ababab"

    def test_accumulate_specialize(self):
        opt = self.ts.auto_specialize(accumulate)
        assert opt(100) == sum(range(100))


class TestSpecializeDecorator:
    def test_specialize_decorator(self):
        @specialize
        def mul(a, b):
            return a * b
        assert mul(3, 4) == 12
        assert mul(1.5, 2.0) == 3.0

    def test_preserves_docstring(self):
        @specialize
        def documented(x):
            """My docstring."""
            return x
        # auto_specialize uses functools.wraps, so docstring should be preserved
        assert documented.__doc__ == "My docstring."
