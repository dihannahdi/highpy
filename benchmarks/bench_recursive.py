"""
╔════════════════════════════════════════════════════════════════════════════╗
║  RFOE Benchmark Suite                                                      ║
║  Recursive Fractal Optimization Engine — Performance Evaluation            ║
║                                                                            ║
║  Benchmarks:                                                               ║
║   1. Optimization overhead (analysis + compile time)                       ║
║   2. Runtime speedup on optimized functions                                ║
║   3. Fixed-point convergence rate on standard contractions                  ║
║   4. Meta-circular self-optimization efficiency                            ║
║   5. Energy reduction across function families                             ║
║   6. Convergence proof generation time                                     ║
║   7. Fractal analysis overhead                                             ║
║   8. Comparison: RFOE vs HighPy v1 vs CPython baseline                     ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import ast
import math
import time
import textwrap
import statistics
import sys
import os

# Ensure highpy is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from highpy.recursive.fractal_optimizer import (
    RecursiveFractalOptimizer, EnergyAnalyzer, OptimizationEnergy,
    FractalDecomposer, UniversalMorphisms, FractalLevel, rfo_optimize,
)
from highpy.recursive.fixed_point_engine import (
    FixedPointEngine, AdaptiveFixedPointEngine, ConvergenceStatus,
)
from highpy.recursive.meta_circular import (
    MetaCircularOptimizer, RecursiveMetaOptimizer,
)
from highpy.recursive.fractal_analyzer import FractalAnalyzer
from highpy.recursive.convergence_prover import ConvergenceProver


# ═══════════════════════════════════════════════════════════════════
#  Benchmark Targets — Diverse Function Families
# ═══════════════════════════════════════════════════════════════════

def bench_arithmetic(x, y):
    """Heavy arithmetic with algebraic identities + constant folding."""
    a = x + 0
    b = y * 1
    c = a - 0
    d = b * 1
    e = c + d
    f = e ** 1
    g = f + 0
    h = g * 1
    k = 2 * 3 + 4 * 5
    m = 10 + 20 + 30
    return h + k + m

def bench_dead_code(x):
    """Function with extensive dead code and dead stores."""
    a = x * 2
    b = x + 1       # unused
    c = x * 3       # unused
    d = x + 10      # unused
    e = x ** 2      # unused
    f = x - 7       # unused
    g = x + 100     # unused
    h = x * x       # unused
    k = x + x + x   # unused
    m = x - x       # unused
    return a

def bench_cse(x):
    """Common subexpressions — 5 copies of same computation."""
    a = x * x + 1
    b = x * x + 1
    c = x * x + 1
    d = x * x + 1
    e = x * x + 1
    return a + b + c + d + e

def bench_loop_compute(n):
    """Loop with optimizable interior expressions."""
    total = 0
    for i in range(n):
        total += i * 1 + 0 + i * 0
    return total

def bench_nested_branches(x):
    """Nested control flow with simplifiable expressions."""
    if x > 100:
        if x > 200:
            return x * 1 + 0
        else:
            return x * 1 + 0
    elif x > 50:
        return x - 0 + 0 * x
    else:
        return 0 + 0

def bench_matrix_like(n):
    """Matrix-like nested computation with heavy redundancy."""
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j * 1 + 0 + j * 0
    return result

def bench_fibonacci_iterative(n):
    """Iterative Fibonacci with redundant operations."""
    if n <= 1:
        return n * 1 + 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b + 0

def bench_polynomial(x):
    """Polynomial with heavy redundancy and constant folding."""
    a = x * 1 + 0
    b = a * 1 + 0
    c = b ** 2
    d = b * 3
    e = 5 * 1 + 2 * 3 + 4 * 2
    return c + d + e

def bench_constant_heavy(x):
    """Function dominated by constant computation."""  
    a = 2 * 3 + 4 * 5 + 6 * 7
    b = 10 + 20 + 30 + 40 + 50
    c = 100 * 2 + 50 * 3
    d = a + b + c
    e = d + 0
    f = e * 1
    return x + f

def bench_identity_chain(x):
    """Long chain of identity operations."""
    a = x + 0
    b = a * 1
    c = b - 0
    d = c ** 1
    e = d + 0
    f = e * 1
    g = f - 0
    h = g ** 1
    k = h + 0
    m = k * 1
    n = m - 0
    p = n ** 1
    return p

def bench_dead_heavy(x):
    """90% dead computation."""
    r = x * 2
    d1 = x * 3 + 4
    d2 = x ** 2 + x + 1
    d3 = x * 5 - 2
    d4 = x + x + x
    d5 = x * x * x
    d6 = x - 1 + 2
    d7 = x * 7 + 3
    d8 = x // 1 + x
    d9 = x * 9 - x
    d10 = x + 10
    d11 = x * 11
    d12 = x + 12 * x
    d13 = x ** 2
    d14 = x * 14 + 1
    d15 = x + 15
    d16 = x * 16
    d17 = x - 17
    d18 = x + 18
    d19 = x * 19
    d20 = x + 20
    return r

def bench_mixed_heavy(x, y):
    """Heavy mix of all optimization types."""
    # Constants to fold
    c1 = 2 * 3
    c2 = 4 + 5
    c3 = 10 * 10
    # Identities to simplify
    a = x + 0
    b = y * 1
    c = a - 0
    d = b ** 1
    # Dead code
    dead1 = x * 7
    dead2 = y + 3
    dead3 = x * y
    # Actual computation
    result = c + d + c1 + c2 + c3
    return result


# ═══════════════════════════════════════════════════════════════════
#  Recursive Benchmark Targets — Memoization Candidates
# ═══════════════════════════════════════════════════════════════════

def bench_fibonacci_recursive(n):
    """Classic recursive Fibonacci — exponential without memoization."""
    if n <= 1:
        return n
    return bench_fibonacci_recursive(n - 1) + bench_fibonacci_recursive(n - 2)

def bench_tribonacci(n):
    """Tribonacci sequence — 3-way exponential recursion."""
    if n <= 0:
        return 0
    if n == 1 or n == 2:
        return 1
    return bench_tribonacci(n - 1) + bench_tribonacci(n - 2) + bench_tribonacci(n - 3)

def bench_grid_paths(m, n):
    """Count unique paths in m×n grid (only right/down moves)."""
    if m == 0 or n == 0:
        return 1
    return bench_grid_paths(m - 1, n) + bench_grid_paths(m, n - 1)

def bench_binomial(n, k):
    """Recursive binomial coefficient C(n, k) — Pascal's triangle."""
    if k == 0 or k == n:
        return 1
    if k < 0 or k > n:
        return 0
    return bench_binomial(n - 1, k - 1) + bench_binomial(n - 1, k)

def bench_subset_sum(n, target):
    """Count subsets of {1,...,n} summing to target — exponential recursive."""
    if target == 0:
        return 1
    if n == 0 or target < 0:
        return 0
    return bench_subset_sum(n - 1, target - n) + bench_subset_sum(n - 1, target)


# ═══════════════════════════════════════════════════════════════════
#  Timing Utilities
# ═══════════════════════════════════════════════════════════════════

def time_function(func, args, iterations=10000):
    """Time a function call over many iterations and return median time in µs."""
    times = []
    for _ in range(5):  # 5 rounds
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / iterations / 1000)  # ns → µs
    return statistics.median(times)


def run_benchmarks():
    """Run the complete RFOE benchmark suite."""
    
    print("=" * 80)
    print("  RECURSIVE FRACTAL OPTIMIZATION ENGINE (RFOE) — BENCHMARK SUITE")
    print("=" * 80)
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 1: Optimization Compile Time
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 1: Optimization Compile Time                     │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    targets = [
        ("arithmetic", bench_arithmetic),
        ("dead_code", bench_dead_code),
        ("cse", bench_cse),
        ("loop_compute", bench_loop_compute),
        ("nested_branches", bench_nested_branches),
        ("matrix_like", bench_matrix_like),
        ("fibonacci", bench_fibonacci_iterative),
        ("polynomial", bench_polynomial),
        ("constant_heavy", bench_constant_heavy),
        ("identity_chain", bench_identity_chain),
        ("dead_heavy", bench_dead_heavy),
        ("mixed_heavy", bench_mixed_heavy),
        ("fib_recursive", bench_fibonacci_recursive),
        ("tribonacci", bench_tribonacci),
        ("grid_paths", bench_grid_paths),
        ("binomial", bench_binomial),
        ("subset_sum", bench_subset_sum),
    ]
    
    optimizer = RecursiveFractalOptimizer(max_iterations=10)
    compile_times = {}
    optimized_funcs = {}
    
    print(f"  {'Function':<20} {'Compile Time (ms)':>18} {'Iterations':>12}")
    print(f"  {'─' * 20} {'─' * 18} {'─' * 12}")
    
    for name, func in targets:
        start = time.perf_counter()
        optimized = optimizer.optimize(func)
        elapsed = (time.perf_counter() - start) * 1000
        result = optimized._rfo_result
        compile_times[name] = elapsed
        optimized_funcs[name] = (func, optimized)
        print(f"  {name:<20} {elapsed:>18.2f} {result.iterations:>12}")
    
    avg_compile = statistics.mean(compile_times.values())
    print(f"\n  Average compile time: {avg_compile:.2f} ms")
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 2: Runtime Speedup (RFOE vs Baseline)
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 2: Runtime Speedup (RFOE vs CPython Baseline)    │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    test_args = {
        "arithmetic": (42, 17),
        "dead_code": (100,),
        "cse": (7,),
        "loop_compute": (100,),
        "nested_branches": (150,),
        "matrix_like": (10,),
        "fibonacci": (20,),
        "polynomial": (5.0,),
        "constant_heavy": (10,),
        "identity_chain": (42,),
        "dead_heavy": (100,),
        "mixed_heavy": (42, 17),
        "fib_recursive": (10,),
        "tribonacci": (10,),
        "grid_paths": (5, 5),
        "binomial": (10, 5),
        "subset_sum": (10, 10),
    }
    
    speedups = []
    print(f"  {'Function':<20} {'Baseline (µs)':>14} {'RFOE (µs)':>12} {'Speedup':>10} {'Correct':>8}")
    print(f"  {'─' * 20} {'─' * 14} {'─' * 12} {'─' * 10} {'─' * 8}")
    
    for name, (orig, optim) in optimized_funcs.items():
        args = test_args[name]
        t_base = time_function(orig, args)
        t_rfoe = time_function(optim, args)
        speedup = t_base / t_rfoe if t_rfoe > 0 else float('inf')
        speedups.append(speedup)
        
        # Correctness check
        try:
            correct = orig(*args) == optim(*args)
        except Exception:
            correct = False
        
        print(f"  {name:<20} {t_base:>14.3f} {t_rfoe:>12.3f} {speedup:>10.2f}x {'✓' if correct else '✗':>8}")
    
    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f"\n  Geometric mean speedup: {geo_mean:.3f}x")
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 3: Fixed-Point Convergence Rate
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 3: Fixed-Point Convergence Rate                  │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    contractions = [
        ("f(x)=x/2+1 (fp=2)",     lambda x: x/2 + 1,     10.0, 2.0),
        ("f(x)=cos(x) (fp≈0.739)", math.cos,              0.0,  0.7390851),
        ("f(x)=x/3+2 (fp=3)",     lambda x: x/3 + 2,     10.0, 3.0),
        ("f(x)=√(x+1) (fp≈φ)",    lambda x: math.sqrt(x+1), 1.0, (1+math.sqrt(5))/2),
        ("f(x)=1/(1+x) (fp≈0.618)", lambda x: 1/(1+x),   0.5, (math.sqrt(5)-1)/2),
    ]
    
    basic_engine = FixedPointEngine(threshold=1e-10, max_iterations=500)
    accel_engine = AdaptiveFixedPointEngine(threshold=1e-10, max_iterations=500)
    
    print(f"  {'Contraction':<28} {'Basic iters':>12} {'Accel iters':>12} {'Speedup':>10} {'Error':>12}")
    print(f"  {'─' * 28} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 12}")
    
    for label, f, x0, fp in contractions:
        r1 = basic_engine.iterate(x0, f)
        r2 = accel_engine.accelerated_iterate(x0, f)
        
        sp = r1.iterations / r2.iterations if r2.iterations > 0 else float('inf')
        err = abs(r2.estimated_fixed_point - fp)
        
        print(f"  {label:<28} {r1.iterations:>12} {r2.iterations:>12} {sp:>10.2f}x {err:>12.2e}")
    
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 4: Energy Reduction Analysis
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 4: Energy Reduction per Function                 │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    analyzer = FractalAnalyzer()
    
    print(f"  {'Function':<20} {'Initial Energy':>15} {'Final Energy':>14} {'Reduction %':>12} {'Converged':>10}")
    print(f"  {'─' * 20} {'─' * 15} {'─' * 14} {'─' * 12} {'─' * 10}")
    
    reductions = []
    for name, (orig, optim) in optimized_funcs.items():
        result = optim._rfo_result
        if result.energy_history and len(result.energy_history) >= 2:
            e0 = result.energy_history[0]
            ef = result.energy_history[-1]
            reduction = (1 - ef / e0) * 100 if e0 > 0 else 0
            reductions.append(max(reduction, 0))
        else:
            e0 = ef = 0
            reduction = 0
        converged = "Yes" if result.converged else "No"
        print(f"  {name:<20} {e0:>15.2f} {ef:>14.2f} {reduction:>11.1f}% {converged:>10}")
    
    if reductions:
        avg_red = statistics.mean(reductions)
        print(f"\n  Average energy reduction: {avg_red:.1f}%")
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 5: Meta-Circular Self-Optimization
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 5: Meta-Circular Self-Optimization               │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    mco = MetaCircularOptimizer()
    
    start = time.perf_counter()
    results = mco.self_optimize(generations=5)
    meta_time = (time.perf_counter() - start) * 1000
    
    print(f"  Self-optimization time: {meta_time:.2f} ms")
    print(f"  Generations completed:  {len(results)}")
    
    if results:
        print(f"\n  {'Generation':>12} {'Original E':>12} {'Optimized E':>12} {'Speedup':>10}")
        print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10}")
        for r in results:
            print(f"  {r.generation:>12} {r.original_energy:>12.2f} {r.optimized_energy:>12.2f} {r.speedup:>10.3f}x")
    
    # Recursive meta-optimizer convergence
    rmo = RecursiveMetaOptimizer(max_meta_iterations=5)
    start = time.perf_counter()
    convergence = rmo.converge()
    rmo_time = (time.perf_counter() - start) * 1000
    
    print(f"\n  Recursive meta-convergence:")
    print(f"    Time:       {rmo_time:.2f} ms")
    print(f"    Converged:  {convergence['converged']}")
    print(f"    Generations: {convergence['generations']}")
    print(f"    Final energy: {convergence['final_energy']:.2f}")
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 6: Convergence Proof Generation
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 6: Convergence Proof Generation                  │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    prover = ConvergenceProver()
    samples = [bench_arithmetic, bench_dead_code, bench_cse, bench_loop_compute]
    
    # Individual morphism verification
    morphisms = [
        UniversalMorphisms.constant_propagation(),
        UniversalMorphisms.dead_code_elimination(),
        UniversalMorphisms.strength_reduction(),
        UniversalMorphisms.algebraic_simplification(),
    ]
    
    print(f"  {'Morphism':<30} {'Contraction k':>14} {'Is Contraction':>15} {'Time (ms)':>12}")
    print(f"  {'─' * 30} {'─' * 14} {'─' * 15} {'─' * 12}")
    
    for m in morphisms:
        start = time.perf_counter()
        cert = prover.verify_morphism(m, samples)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {m.name:<30} {cert.mean_factor:>14.4f} {'Yes' if cert.is_contraction else 'No':>15} {elapsed:>12.2f}")
    
    # Full pipeline proof
    print()
    opt_for_proof = RecursiveFractalOptimizer(max_iterations=5)
    start = time.perf_counter()
    proof = prover.prove_convergence(opt_for_proof, samples, iterations=5)
    proof_time = (time.perf_counter() - start) * 1000
    
    print(f"  Full pipeline convergence proof:")
    print(f"    Status:              {proof.status.name}")
    print(f"    Contraction factor:  {proof.contraction_factor:.4f}")
    print(f"    Confidence:          {proof.confidence:.1%}")
    print(f"    Est. iterations:     {proof.estimated_iterations_to_convergence}")
    print(f"    Proof time:          {proof_time:.2f} ms")
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 7: Fractal Analysis
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK 7: Fractal Analysis                              │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    print(f"  {'Function':<20} {'Fractal Dim':>12} {'Is Fractal':>11} {'Self-Sim':>10} {'Total E':>10} {'Time (ms)':>12}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 11} {'─' * 10} {'─' * 10} {'─' * 12}")
    
    for name, func in targets:
        start = time.perf_counter()
        field = analyzer.analyze_function(func)
        elapsed = (time.perf_counter() - start) * 1000
        fd = field.fractal_dimension
        print(f"  {name:<20} {fd.dimension:>12.4f} {'Yes' if fd.is_fractal else 'No':>11} {field.self_similarity_index:>10.4f} {field.total_energy:>10.1f} {elapsed:>12.2f}")
    
    print()
    
    # ─────────────────────────────────────────────────────────
    #  Benchmark 8: Overall Summary
    # ─────────────────────────────────────────────────────────
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│  SUMMARY                                                    │")
    print("└──────────────────────────────────────────────────────────────┘")
    
    print(f"  Total functions benchmarked:   {len(targets)}")
    print(f"  Geometric mean speedup:        {geo_mean:.3f}x")
    print(f"  Avg compile time:              {avg_compile:.2f} ms")
    print(f"  Avg energy reduction:          {statistics.mean(reductions) if reductions else 0:.1f}%")
    print(f"  Convergence proof status:      {proof.status.name}")
    print(f"  Meta self-optimization gens:   {len(results)}")
    print(f"  Fixed-point acceleration:      Aitken Δ² enabled")
    print(f"  Mathematical foundation:       Banach Contraction Mapping Theorem")
    print()
    print("=" * 80)
    print("  RFOE BENCHMARK SUITE COMPLETE")
    print("=" * 80)
    
    return {
        "geo_mean_speedup": geo_mean,
        "avg_compile_ms": avg_compile,
        "avg_energy_reduction_pct": statistics.mean(reductions) if reductions else 0,
        "proof_status": proof.status.name,
        "convergence_factor": proof.contraction_factor,
    }


if __name__ == "__main__":
    run_benchmarks()
