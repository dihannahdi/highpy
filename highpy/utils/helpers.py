"""Utility helpers for HighPy."""

import time
import functools
from contextlib import contextmanager
from typing import Any, Callable


class Timer:
    """High-resolution timer for benchmarking."""
    
    def __init__(self):
        self.start_ns = 0
        self.end_ns = 0
    
    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self
    
    def __exit__(self, *exc):
        self.end_ns = time.perf_counter_ns()
    
    @property
    def elapsed_ns(self) -> int:
        return self.end_ns - self.start_ns
    
    @property
    def elapsed_us(self) -> float:
        return self.elapsed_ns / 1000.0
    
    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000.0
    
    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ns / 1_000_000_000.0


def benchmark(func: Callable = None, *, iterations: int = 100, warmup: int = 10):
    """
    Decorator that benchmarks a function.
    
    Usage:
        @benchmark(iterations=1000)
        def my_func():
            ...
    """
    if func is None:
        return lambda f: benchmark(f, iterations=iterations, warmup=warmup)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            with Timer() as t:
                result = func(*args, **kwargs)
            times.append(t.elapsed_ns)
        
        times.sort()
        median = times[len(times) // 2]
        mean = sum(times) / len(times)
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        minimum = times[0]
        
        wrapper.__benchmark_results__ = {
            'median_ns': median,
            'mean_ns': mean,
            'p95_ns': p95,
            'p99_ns': p99,
            'min_ns': minimum,
            'iterations': iterations,
        }
        
        return result
    
    return wrapper


def format_ns(ns: float) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns < 1_000:
        return f"{ns:.0f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.3f} s"


def format_speedup(baseline_ns: float, optimized_ns: float) -> str:
    """Format a speedup ratio."""
    if optimized_ns <= 0:
        return "∞x"
    ratio = baseline_ns / optimized_ns
    if ratio >= 1:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"
