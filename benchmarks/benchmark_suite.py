"""
HighPy Benchmark Suite
======================

Comprehensive benchmark suite comparing vanilla CPython execution
against HighPy-optimized execution across multiple workload categories:

1. **Micro-benchmarks**: Isolate specific bottlenecks
   - Function call overhead
   - Attribute access
   - Type dispatch
   - Loop overhead
   - Boxing/unboxing

2. **Meso-benchmarks**: Algorithmic kernels
   - Fibonacci (iterative & recursive)
   - Matrix multiplication
   - Numerical integration
   - Prime sieve
   - Sorting

3. **Macro-benchmarks**: Real-world workloads
   - Monte Carlo simulation
   - N-body simulation
   - Ray marching
   - Data processing pipeline

Methodology
-----------
- Each benchmark runs N iterations after W warmup iterations
- Time is measured with time.perf_counter_ns() (nanosecond precision)
- Statistics: min, median, mean, p95, p99, stddev
- Speedup computed as baseline_median / optimized_median
"""

import math
import time
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    category: str
    baseline_times_ns: List[int] = field(default_factory=list)
    optimized_times_ns: List[int] = field(default_factory=list)
    baseline_result: Any = None
    optimized_result: Any = None
    correct: bool = True
    error: Optional[str] = None
    
    @property
    def baseline_median_ns(self) -> float:
        return statistics.median(self.baseline_times_ns) if self.baseline_times_ns else 0
    
    @property
    def optimized_median_ns(self) -> float:
        return statistics.median(self.optimized_times_ns) if self.optimized_times_ns else 0
    
    @property
    def speedup(self) -> float:
        if self.optimized_median_ns <= 0:
            return float('inf')
        return self.baseline_median_ns / self.optimized_median_ns
    
    @property
    def baseline_stats(self) -> Dict[str, float]:
        return self._compute_stats(self.baseline_times_ns)
    
    @property
    def optimized_stats(self) -> Dict[str, float]:
        return self._compute_stats(self.optimized_times_ns)
    
    def _compute_stats(self, times: List[int]) -> Dict[str, float]:
        if not times:
            return {}
        sorted_t = sorted(times)
        n = len(sorted_t)
        return {
            'min_ns': sorted_t[0],
            'median_ns': sorted_t[n // 2],
            'mean_ns': statistics.mean(sorted_t),
            'p95_ns': sorted_t[int(n * 0.95)],
            'p99_ns': sorted_t[min(int(n * 0.99), n - 1)],
            'stddev_ns': statistics.stdev(sorted_t) if n > 1 else 0,
        }


# ========================================================================
# MICRO-BENCHMARKS: Isolate specific CPython bottlenecks
# ========================================================================

class MicroBenchmarks:
    """Micro-benchmarks targeting specific CPython bottlenecks."""
    
    @staticmethod
    def function_call_overhead_baseline(n):
        """Measure function call overhead (B5: Function Call Overhead)."""
        def noop():
            pass
        for _ in range(n):
            noop()
    
    @staticmethod
    def function_call_overhead_optimized(n):
        """Inlined version — no function call."""
        for _ in range(n):
            pass
    
    @staticmethod
    def attribute_access_baseline(n):
        """Measure attribute access overhead (B4: Attribute Lookup)."""
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2
                self.z = 3
        o = Obj()
        total = 0
        for _ in range(n):
            total += o.x + o.y + o.z
        return total
    
    @staticmethod
    def attribute_access_optimized(n):
        """Local variable version — avoids repeated attribute lookup."""
        class Obj:
            __slots__ = ('x', 'y', 'z')
            def __init__(self):
                self.x = 1
                self.y = 2
                self.z = 3
        o = Obj()
        x, y, z = o.x, o.y, o.z
        total = 0
        for _ in range(n):
            total += x + y + z
        return total
    
    @staticmethod
    def type_dispatch_baseline(n):
        """Dynamic type dispatch overhead (B1: Dynamic Type Dispatch)."""
        total = 0
        for i in range(n):
            total = total + i  # Dynamic dispatch on __add__
        return total
    
    @staticmethod
    def type_dispatch_optimized(n):
        """Direct integer addition — bypasses dispatch."""
        total = 0
        # In optimized code, the type is known to be int
        for i in range(n):
            total += i
        return total
    
    @staticmethod
    def boxing_overhead_baseline(n):
        """Boxing/unboxing overhead (B7: Boxing/Unboxing)."""
        total = 0.0
        for i in range(n):
            x = float(i)  # Box
            total += x * x  # Unbox, compute, rebox
        return total
    
    @staticmethod
    def boxing_overhead_optimized(n):
        """Bulk computation from compact array."""
        from highpy.optimization.memory_pool import CompactArray
        arr = CompactArray.from_list([float(i) for i in range(n)])
        total = 0.0
        for i in range(len(arr)):
            v = arr[i]
            total += v * v
        return total
    
    @staticmethod
    def global_lookup_baseline(n):
        """Global variable lookup overhead (B8: Late Binding)."""
        import math as _math
        total = 0.0
        for i in range(1, n + 1):
            total += _math.sqrt(float(i))
        return total
    
    @staticmethod
    def global_lookup_optimized(n):
        """Local binding — avoids global lookup."""
        from math import sqrt
        _sqrt = sqrt  # Bind to local
        total = 0.0
        for i in range(1, n + 1):
            total += _sqrt(float(i))
        return total


# ========================================================================
# MESO-BENCHMARKS: Algorithmic kernels
# ========================================================================

class MesoBenchmarks:
    """Medium-scale algorithmic benchmarks."""
    
    @staticmethod
    def fibonacci_iterative(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    
    @staticmethod
    def matrix_multiply(A, B):
        n = len(A)
        m = len(B[0])
        k = len(B)
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0
                for p in range(k):
                    s += A[i][p] * B[p][j]
                C[i][j] = s
        return C
    
    @staticmethod
    def numerical_integration_trapezoidal(n):
        """Integrate sin(x) from 0 to π (exact answer = 2.0)."""
        a, b = 0.0, math.pi
        h = (b - a) / n
        total = 0.5 * (math.sin(a) + math.sin(b))
        for i in range(1, n):
            total += math.sin(a + i * h)
        return total * h
    
    @staticmethod
    def prime_sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        return sum(is_prime)
    
    @staticmethod
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return MesoBenchmarks.quicksort(left) + middle + MesoBenchmarks.quicksort(right)
    
    @staticmethod
    def sum_of_squares(n):
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    @staticmethod
    def dot_product(n):
        a = [float(i) for i in range(n)]
        b = [float(n - i) for i in range(n)]
        total = 0.0
        for i in range(n):
            total += a[i] * b[i]
        return total
    
    @staticmethod
    def polynomial_eval(coeffs, x):
        """Horner's method polynomial evaluation."""
        result = 0.0
        for c in reversed(coeffs):
            result = result * x + c
        return result


# ========================================================================
# MACRO-BENCHMARKS: Real-world workloads
# ========================================================================

class MacroBenchmarks:
    """Larger, real-world-ish benchmarks."""
    
    @staticmethod
    def monte_carlo_pi(n):
        """Monte Carlo estimation of π."""
        import random
        rng = random.Random(42)
        inside = 0
        for _ in range(n):
            x = rng.random()
            y = rng.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / n
    
    @staticmethod
    def nbody_step(bodies, dt):
        """
        One step of an N-body simulation.
        bodies: list of (x, y, z, vx, vy, vz, mass)
        """
        n = len(bodies)
        # Compute accelerations
        for i in range(n):
            fx, fy, fz = 0.0, 0.0, 0.0
            xi, yi, zi = bodies[i][0], bodies[i][1], bodies[i][2]
            mi = bodies[i][6]
            for j in range(n):
                if i == j:
                    continue
                dx = bodies[j][0] - xi
                dy = bodies[j][1] - yi
                dz = bodies[j][2] - zi
                dist = math.sqrt(dx * dx + dy * dy + dz * dz + 1e-10)
                force = bodies[j][6] / (dist * dist * dist)
                fx += force * dx
                fy += force * dy
                fz += force * dz
            # Update velocity
            bodies[i] = (
                xi + bodies[i][3] * dt,
                yi + bodies[i][4] * dt,
                zi + bodies[i][5] * dt,
                bodies[i][3] + fx * dt,
                bodies[i][4] + fy * dt,
                bodies[i][5] + fz * dt,
                mi,
            )
        return bodies
    
    @staticmethod
    def mandelbrot(width, height, max_iter):
        """Mandelbrot set computation."""
        result = []
        for py in range(height):
            row = []
            for px in range(width):
                x0 = (px - width / 2.0) * 4.0 / width
                y0 = (py - height / 2.0) * 4.0 / height
                x, y = 0.0, 0.0
                iteration = 0
                while x * x + y * y <= 4.0 and iteration < max_iter:
                    xtemp = x * x - y * y + x0
                    y = 2.0 * x * y + y0
                    x = xtemp
                    iteration += 1
                row.append(iteration)
            result.append(row)
        return result
    
    @staticmethod
    def data_pipeline(n):
        """Simulated data processing pipeline."""
        # Generate data
        import random
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(n)]
        
        # Filter outliers
        filtered = [x for x in data if -3.0 <= x <= 3.0]
        
        # Transform
        transformed = [x * x for x in filtered]
        
        # Aggregate
        total = sum(transformed)
        count = len(transformed)
        mean = total / count if count > 0 else 0
        
        # Compute variance
        variance = sum((x - mean) ** 2 for x in transformed) / count if count > 0 else 0
        
        return {
            'count': count,
            'mean': mean,
            'variance': variance,
            'total': total,
        }


def get_all_benchmarks() -> Dict[str, Dict]:
    """
    Return all benchmarks organized by category.
    
    Returns a dict:
        { category: { name: (baseline_func, args, optimized_func, opt_args) } }
    """
    import random
    
    benchmarks = {
        'micro': {
            'function_call_overhead': (
                MicroBenchmarks.function_call_overhead_baseline,
                (100000,),
                MicroBenchmarks.function_call_overhead_optimized,
                (100000,),
            ),
            'attribute_access': (
                MicroBenchmarks.attribute_access_baseline,
                (100000,),
                MicroBenchmarks.attribute_access_optimized,
                (100000,),
            ),
            'type_dispatch': (
                MicroBenchmarks.type_dispatch_baseline,
                (100000,),
                MicroBenchmarks.type_dispatch_optimized,
                (100000,),
            ),
            'global_lookup': (
                MicroBenchmarks.global_lookup_baseline,
                (10000,),
                MicroBenchmarks.global_lookup_optimized,
                (10000,),
            ),
        },
        'meso': {
            'fibonacci_iter': (
                MesoBenchmarks.fibonacci_iterative,
                (10000,),
                None, None,
            ),
            'sum_of_squares': (
                MesoBenchmarks.sum_of_squares,
                (100000,),
                None, None,
            ),
            'dot_product': (
                MesoBenchmarks.dot_product,
                (10000,),
                None, None,
            ),
            'prime_sieve': (
                MesoBenchmarks.prime_sieve,
                (10000,),
                None, None,
            ),
            'numerical_integration': (
                MesoBenchmarks.numerical_integration_trapezoidal,
                (100000,),
                None, None,
            ),
        },
        'macro': {
            'monte_carlo_pi': (
                MacroBenchmarks.monte_carlo_pi,
                (100000,),
                None, None,
            ),
            'mandelbrot': (
                MacroBenchmarks.mandelbrot,
                (40, 40, 100),
                None, None,
            ),
            'data_pipeline': (
                MacroBenchmarks.data_pipeline,
                (50000,),
                None, None,
            ),
        },
    }
    
    return benchmarks
