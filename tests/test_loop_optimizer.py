"""
Tests for the loop optimizer.

Validates:
  - Accumulator detection and optimization
  - List comprehension conversion
  - Loop analysis (range detection, reduction patterns)
  - Correctness after optimization
"""

import pytest
from highpy.optimization.loop_optimizer import LoopOptimizer, optimize_loops


def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total


def build_list(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result


def dot_product(xs, ys):
    total = 0
    for i in range(len(xs)):
        total += xs[i] * ys[i]
    return total


def nested_sum(matrix):
    total = 0
    for row in matrix:
        for val in row:
            total += val
    return total


class TestLoopOptimizer:
    def setup_method(self):
        self.optimizer = LoopOptimizer()

    def test_optimize_sum_range(self):
        optimized = self.optimizer.optimize(sum_range)
        assert callable(optimized)
        assert optimized(100) == sum_range(100)
        assert optimized(0) == sum_range(0)
        assert optimized(1) == sum_range(1)

    def test_optimize_build_list(self):
        optimized = self.optimizer.optimize(build_list)
        assert callable(optimized)
        result = optimized(50)
        expected = build_list(50)
        assert result == expected

    def test_optimize_dot_product(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        optimized = self.optimizer.optimize(dot_product)
        assert callable(optimized)
        assert optimized(xs, ys) == dot_product(xs, ys)

    def test_correctness_large(self):
        optimized = self.optimizer.optimize(sum_range)
        assert optimized(10000) == sum_range(10000)

    def test_empty_input(self):
        optimized = self.optimizer.optimize(sum_range)
        assert optimized(0) == 0


class TestOptimizeLoopsDecorator:
    def test_decorator(self):
        @optimize_loops
        def count(n):
            total = 0
            for i in range(n):
                total += i
            return total

        assert count(100) == sum(range(100))
