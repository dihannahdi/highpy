"""
HighPy Benchmark Runner
=======================

Runs all benchmarks and collects results for report generation.

Usage:
    python -m benchmarks.benchmark_runner
"""

import gc
import os
import sys
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_suite import (
    BenchmarkResult,
    MicroBenchmarks,
    MesoBenchmarks,
    MacroBenchmarks,
    get_all_benchmarks,
)


ITERATIONS = 50      # Benchmark iterations
WARMUP = 10          # Warmup iterations


def time_function(func: Callable, args: tuple, iterations: int, warmup: int) -> List[int]:
    """Time a function call over multiple iterations, returning list of ns times."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    times = []
    for _ in range(iterations):
        gc.disable()
        start = time.perf_counter_ns()
        func(*args)
        end = time.perf_counter_ns()
        gc.enable()
        times.append(end - start)
    
    return times


def check_correctness(result1: Any, result2: Any) -> bool:
    """Check if two results are equivalent."""
    if result1 is None and result2 is None:
        return True
    if isinstance(result1, float) and isinstance(result2, float):
        if abs(result1) < 1e-10 and abs(result2) < 1e-10:
            return True
        return abs(result1 - result2) / max(abs(result1), abs(result2), 1e-10) < 0.01
    if isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
        if len(result1) != len(result2):
            return False
        for a, b in zip(result1, result2):
            if not check_correctness(a, b):
                return False
        return True
    if isinstance(result1, dict) and isinstance(result2, dict):
        if set(result1.keys()) != set(result2.keys()):
            return False
        for key in result1:
            if not check_correctness(result1[key], result2[key]):
                return False
        return True
    return result1 == result2


def run_benchmark_pair(
    name: str,
    category: str,
    baseline_func: Callable,
    baseline_args: tuple,
    optimized_func: Optional[Callable],
    optimized_args: Optional[tuple],
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> BenchmarkResult:
    """Run a single benchmark (baseline vs optimized)."""
    result = BenchmarkResult(name=name, category=category)
    
    # Run baseline
    try:
        baseline_result = baseline_func(*baseline_args)
        result.baseline_result = baseline_result
        result.baseline_times_ns = time_function(
            baseline_func, baseline_args, iterations, warmup
        )
    except Exception as e:
        result.error = f"Baseline error: {e}"
        result.correct = False
        return result
    
    # If there's an explicit optimized version, use it
    if optimized_func is not None:
        try:
            opt_result = optimized_func(*optimized_args)
            result.optimized_result = opt_result
            result.optimized_times_ns = time_function(
                optimized_func, optimized_args, iterations, warmup
            )
            # Check correctness (if applicable)
            if baseline_result is not None and opt_result is not None:
                result.correct = check_correctness(baseline_result, opt_result)
        except Exception as e:
            result.error = f"Optimized error: {e}"
            result.correct = False
        return result
    
    # Otherwise, apply HighPy optimization automatically
    try:
        from highpy.runtime.adaptive_runtime import optimize
        
        opt_func = optimize(baseline_func)
        # Warm up the adaptive runtime
        for _ in range(warmup + 30):
            opt_func(*baseline_args)
        
        opt_result = opt_func(*baseline_args)
        result.optimized_result = opt_result
        result.optimized_times_ns = time_function(
            opt_func, baseline_args, iterations, warmup
        )
        if baseline_result is not None and opt_result is not None:
            result.correct = check_correctness(baseline_result, opt_result)
    except Exception as e:
        result.error = f"HighPy optimization error: {e}"
        # Still have baseline times, just no optimized
    
    return result


def run_all_benchmarks(
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    all_benchmarks = get_all_benchmarks()
    results = []
    
    total = sum(len(v) for v in all_benchmarks.values())
    completed = 0
    
    for category, benchmarks in all_benchmarks.items():
        print(f"\n{'='*60}")
        print(f"  Category: {category.upper()}")
        print(f"{'='*60}")
        
        for name, (baseline_func, baseline_args, opt_func, opt_args) in benchmarks.items():
            completed += 1
            print(f"  [{completed}/{total}] Running {name}...", end=" ", flush=True)
            
            result = run_benchmark_pair(
                name=name,
                category=category,
                baseline_func=baseline_func,
                baseline_args=baseline_args,
                optimized_func=opt_func,
                optimized_args=opt_args,
                iterations=iterations,
                warmup=warmup,
            )
            results.append(result)
            
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                speedup = result.speedup
                bl = result.baseline_median_ns
                op = result.optimized_median_ns
                correct = "OK" if result.correct else "MISMATCH"
                print(
                    f"baseline={_fmt(bl)}, optimized={_fmt(op)}, "
                    f"speedup={speedup:.2f}x [{correct}]"
                )
    
    return results


def _fmt(ns: float) -> str:
    """Format nanoseconds."""
    if ns < 1_000:
        return f"{ns:.0f}ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.1f}µs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f}ms"
    else:
        return f"{ns/1_000_000_000:.3f}s"


def save_results(results: List[BenchmarkResult], output_dir: str = "reports"):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    data = {
        'timestamp': timestamp,
        'python_version': sys.version,
        'platform': sys.platform,
        'results': [],
    }
    
    for r in results:
        entry = {
            'name': r.name,
            'category': r.category,
            'correct': r.correct,
            'error': r.error,
            'baseline_stats': r.baseline_stats,
            'optimized_stats': r.optimized_stats,
            'speedup': r.speedup if r.optimized_times_ns else None,
        }
        data['results'].append(entry)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\nResults saved to {filename}")
    return filename


def print_summary(results: List[BenchmarkResult]):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"  HIGHPY BENCHMARK SUMMARY")
    print(f"  Python {sys.version.split()[0]} | {sys.platform}")
    print(f"{'='*80}")
    
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    
    for cat, cat_results in categories.items():
        print(f"\n  --- {cat.upper()} ---")
        print(f"  {'Benchmark':<30} {'Baseline':>12} {'Optimized':>12} {'Speedup':>10} {'Status':>8}")
        print(f"  {'-'*72}")
        
        for r in cat_results:
            bl = _fmt(r.baseline_median_ns) if r.baseline_times_ns else "N/A"
            op = _fmt(r.optimized_median_ns) if r.optimized_times_ns else "N/A"
            sp = f"{r.speedup:.2f}x" if r.optimized_times_ns else "N/A"
            status = "OK" if r.correct and not r.error else "FAIL"
            if r.error:
                status = "ERR"
            print(f"  {r.name:<30} {bl:>12} {op:>12} {sp:>10} {status:>8}")
    
    # Overall statistics
    valid = [r for r in results if r.optimized_times_ns and r.correct]
    if valid:
        speedups = [r.speedup for r in valid]
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 1.0
        print(f"\n  {'='*72}")
        print(f"  Geometric mean speedup: {geomean:.2f}x")
        print(f"  Max speedup: {max(speedups):.2f}x ({max(valid, key=lambda r: r.speedup).name})")
        print(f"  Min speedup: {min(speedups):.2f}x ({min(valid, key=lambda r: r.speedup).name})")
        print(f"  Benchmarks passed: {len(valid)}/{len(results)}")


import math


def main():
    """Main entry point."""
    print("HighPy Performance Benchmark Suite")
    print(f"Python {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    results = run_all_benchmarks()
    print_summary(results)
    filepath = save_results(results)
    
    return results


if __name__ == '__main__':
    main()
