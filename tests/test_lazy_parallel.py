"""
Tests for the lazy evaluator and parallel executor.
"""

import pytest
from highpy.optimization.lazy_evaluator import LazyChain, lazy
from highpy.optimization.parallel_executor import ParallelExecutor, auto_parallel


class TestLazyChain:
    def test_collect(self):
        assert lazy(range(5)).collect() == [0, 1, 2, 3, 4]

    def test_map(self):
        result = lazy(range(5)).map(lambda x: x * 2).collect()
        assert result == [0, 2, 4, 6, 8]

    def test_filter(self):
        result = lazy(range(10)).filter(lambda x: x % 2 == 0).collect()
        assert result == [0, 2, 4, 6, 8]

    def test_take(self):
        result = lazy(range(100)).take(3).collect()
        assert result == [0, 1, 2]

    def test_skip(self):
        result = lazy(range(10)).skip(7).collect()
        assert result == [7, 8, 9]

    def test_sum(self):
        assert lazy(range(10)).sum() == 45

    def test_count(self):
        assert lazy(range(10)).count() == 10

    def test_first(self):
        assert lazy(range(5)).first() == 0
        assert lazy(range(10)).filter(lambda x: x > 5).first() == 6

    def test_chain_operations(self):
        result = (
            lazy(range(20))
            .filter(lambda x: x % 2 == 0)
            .map(lambda x: x * x)
            .take(5)
            .collect()
        )
        assert result == [0, 4, 16, 36, 64]

    def test_reduce(self):
        result = lazy(range(1, 6)).reduce(lambda a, b: a * b, 1)
        assert result == 120  # 5!

    def test_empty(self):
        assert lazy([]).collect() == []
        assert lazy([]).sum() == 0
        assert lazy([]).count() == 0

    def test_enumerate(self):
        result = lazy(['a', 'b', 'c']).enumerate().collect()
        assert result == [(0, 'a'), (1, 'b'), (2, 'c')]

    def test_flat_map(self):
        result = lazy([[1, 2], [3, 4], [5]]).flat_map(lambda x: x).collect()
        assert result == [1, 2, 3, 4, 5]


class TestParallelExecutor:
    def setup_method(self):
        self.executor = ParallelExecutor(workers=2, min_parallel_size=100)

    def test_map_small(self):
        """Small data should not trigger parallel execution."""
        result = self.executor.map(lambda x: x * 2, range(10))
        assert result == [x * 2 for x in range(10)]

    def test_map_correctness(self):
        """Results should be correct regardless of parallel/sequential."""
        data = list(range(50))
        result = self.executor.map(lambda x: x * x, data)
        expected = [x * x for x in data]
        assert result == expected

    def test_purity_check_pure(self):
        def pure_func(x):
            return x * 2 + 1
        assert self.executor._check_purity(pure_func) is True

    def test_purity_check_impure_global(self):
        def impure_func(x):
            global some_var
            return x
        assert self.executor._check_purity(impure_func) is False

    def test_purity_check_impure_print(self):
        def impure_func(x):
            print(x)
            return x
        assert self.executor._check_purity(impure_func) is False

    def test_parallel_map_decorator(self):
        @self.executor.parallel_map
        def double(x):
            return x * 2
        result = double(range(50))
        assert result == [x * 2 for x in range(50)]

    def test_reduce_small(self):
        result = self.executor.reduce(lambda a, b: a + b, range(10), 0)
        assert result == 45
