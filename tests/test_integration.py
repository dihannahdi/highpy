"""
Integration tests for the HighPy framework.

End-to-end tests that validate the full optimization pipeline:
  Analysis -> Type Profiling -> AST Optimization -> Specialization -> Execution
"""

import math
import pytest
from highpy.runtime.adaptive_runtime import AdaptiveRuntime, optimize, jit
from highpy.optimization.lazy_evaluator import lazy, LazyChain
from highpy.optimization.memory_pool import CompactArray, ArenaAllocator
from highpy.analysis.cpython_bottlenecks import CPythonAnalyzer, BottleneckReport
from highpy.analysis.type_profiler import TypeProfiler, LatticeType, TypeTag


# ---------- Realistic Workloads ----------

def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def matrix_multiply_naive(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def numerical_integration(f, a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def prime_sieve(limit):
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(limit + 1) if is_prime[i]]


def monte_carlo_pi(n):
    import random
    rng = random.Random(42)
    inside = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


# ---------- Integration Tests ----------

class TestEndToEndOptimize:
    def test_fibonacci_optimized(self):
        optimized = optimize(fibonacci_iterative)
        for n in [0, 1, 2, 5, 10, 20, 30]:
            assert optimized(n) == fibonacci_iterative(n)

    def test_fibonacci_jit(self):
        jitted = jit(fibonacci_iterative)
        for n in [0, 1, 2, 5, 10, 20]:
            assert jitted(n) == fibonacci_iterative(n)

    def test_prime_sieve_optimized(self):
        optimized = optimize(prime_sieve)
        expected = prime_sieve(100)
        assert optimized(100) == expected

    def test_monte_carlo_deterministic(self):
        optimized = optimize(monte_carlo_pi)
        result1 = monte_carlo_pi(1000)
        result2 = optimized(1000)
        assert abs(result1 - result2) < 1e-10

    def test_numerical_integration(self):
        optimized = optimize(numerical_integration)
        result = optimized(lambda x: x * x, 0.0, 1.0, 10000)
        assert abs(result - 1.0 / 3.0) < 1e-4

    def test_matrix_multiply(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        expected = matrix_multiply_naive(A, B)
        optimized = optimize(matrix_multiply_naive)
        result = optimized(A, B)
        assert result == expected


class TestLazyEvaluation:
    def test_lazy_chain(self):
        result = (
            lazy(range(100))
            .map(lambda x: x * 2)
            .filter(lambda x: x % 3 == 0)
            .take(10)
            .collect()
        )
        expected = [x * 2 for x in range(100) if (x * 2) % 3 == 0][:10]
        assert result == expected

    def test_lazy_sum(self):
        result = lazy(range(1000)).map(lambda x: x * x).sum()
        expected = sum(x * x for x in range(1000))
        assert result == expected

    def test_lazy_count(self):
        result = lazy(range(100)).filter(lambda x: x % 7 == 0).count()
        expected = len([x for x in range(100) if x % 7 == 0])
        assert result == expected

    def test_lazy_first(self):
        result = lazy(range(1, 100)).filter(lambda x: x > 50).first()
        assert result == 51

    def test_lazy_foreach(self):
        collected = []
        lazy(range(5)).foreach(collected.append)
        assert collected == [0, 1, 2, 3, 4]


class TestCompactArrayIntegration:
    def test_dot_product_workload(self):
        n = 1000
        a = CompactArray.from_list([float(i) for i in range(n)])
        b = CompactArray.from_list([float(n - i) for i in range(n)])
        result = a.dot(b)
        expected = sum(float(i) * float(n - i) for i in range(n))
        assert abs(result - expected) < 1e-6

    def test_scale_and_sum(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = arr.scale(10.0)
        assert abs(scaled.sum() - 150.0) < 1e-10


class TestTypeProfilerIntegration:
    def test_profile_fibonacci(self):
        profiler = TypeProfiler()
        profiled = profiler.profile(fibonacci_iterative)
        for n in range(10):
            profiled(n)
        report = profiler.get_type_info(fibonacci_iterative)
        assert report is not None

    def test_type_lattice(self):
        int_type = LatticeType.from_python_type(int)
        float_type = LatticeType.from_python_type(float)
        joined = int_type.join(float_type)
        assert joined is not None


class TestCPythonAnalyzerIntegration:
    def test_full_analysis_pipeline(self):
        analyzer = CPythonAnalyzer()
        report = analyzer.analyze(fibonacci_iterative)
        assert report is not None
        assert isinstance(report, BottleneckReport)
        assert report.function_name == 'fibonacci_iterative'

    def test_analysis_with_loops(self):
        analyzer = CPythonAnalyzer()
        report = analyzer.analyze(prime_sieve)
        assert report is not None
        assert isinstance(report, BottleneckReport)


class TestArenaIntegration:
    def test_arena_with_computation(self):
        arena = ArenaAllocator()
        with arena.scope() as a:
            buf = a.alloc_doubles(1000)
            assert len(buf) == 1000
            # Write and read back
            for i in range(100):
                buf[i] = float(i)
            for i in range(100):
                assert abs(buf[i] - float(i)) < 1e-10


class TestMultipleOptimizations:
    def test_optimize_then_jit(self):
        @optimize
        def f(x):
            return x * x + x

        for i in range(100):
            assert f(i) == i * i + i

    def test_consistent_results(self):
        def workload(n):
            total = 0
            for i in range(n):
                total += i * (i + 1)
            return total

        original_results = [workload(n) for n in range(50)]
        
        optimized = optimize(workload)
        for run in range(3):
            for n in range(50):
                assert optimized(n) == original_results[n], \
                    f"Mismatch at n={n}, run={run}"
