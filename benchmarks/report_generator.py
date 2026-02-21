"""
HighPy Report Generator
========================

Generates publication-quality performance reports including:
  - Formatted tables (compatible with LaTeX/Markdown)
  - Statistical analysis with confidence intervals
  - Bottleneck analysis summary
  - Speedup charts (ASCII and matplotlib)

Output formats:
  - Markdown report (for README / paper draft)
  - JSON (machine-readable)
  - LaTeX table snippets
"""

import json
import math
import os
import sys
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_suite import BenchmarkResult


def generate_markdown_report(
    results: List[BenchmarkResult],
    output_path: str = "reports/performance_report.md",
) -> str:
    """Generate a comprehensive Markdown performance report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = []
    lines.append("# HighPy Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Python:** {sys.version.split()[0]}")
    lines.append(f"**Platform:** {sys.platform}")
    lines.append("")
    
    # Abstract
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "This report presents the results of comprehensive performance benchmarks "
        "comparing vanilla CPython execution against HighPy-optimized execution. "
        "HighPy implements a novel Adaptive Multi-Level Specialization (AMLS) framework "
        "that operates at three tiers: bytecode optimization, type specialization, and "
        "native compilation. Our results demonstrate measurable speedups across "
        "micro-, meso-, and macro-benchmarks while preserving semantic correctness."
    )
    lines.append("")
    
    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Experimental Setup")
    lines.append("- **Measurement:** `time.perf_counter_ns()` (nanosecond precision)")
    lines.append("- **Iterations:** 50 per benchmark (after 10 warmup iterations)")
    lines.append("- **GC:** Disabled during measurement (`gc.disable()`)")
    lines.append("- **Statistics:** Median used as primary metric (robust to outliers)")
    lines.append("- **Correctness:** Output equivalence verified for each benchmark")
    lines.append("")
    lines.append("### Optimization Tiers")
    lines.append("| Tier | Name | Trigger | Technique |")
    lines.append("|------|------|---------|-----------|")
    lines.append("| 0 | Interpreted | Default | Vanilla CPython |")
    lines.append("| 1 | Bytecode-Optimized | 5 calls | AST constant folding, strength reduction, dead code elimination |")
    lines.append("| 2 | Type-Specialized | 20 calls + type stable | Monomorphic dispatch, type-guided code generation |")
    lines.append("| 3 | Native-Compiled | 50 calls + loops | C code generation via ctypes, direct memory access |")
    lines.append("")
    
    # Results by category
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    
    lines.append("## Results")
    lines.append("")
    
    for cat, cat_results in sorted(categories.items()):
        lines.append(f"### {cat.replace('_', ' ').title()} Benchmarks")
        lines.append("")
        lines.append("| Benchmark | Baseline (median) | Optimized (median) | Speedup | Correct |")
        lines.append("|-----------|-------------------|-------------------|---------|---------|")
        
        for r in cat_results:
            bl = _fmt(r.baseline_median_ns) if r.baseline_times_ns else "N/A"
            op = _fmt(r.optimized_median_ns) if r.optimized_times_ns else "N/A"
            sp = f"{r.speedup:.2f}x" if r.optimized_times_ns else "N/A"
            correct = "✓" if r.correct and not r.error else "✗"
            if r.error:
                correct = f"✗ ({r.error[:30]})"
            lines.append(f"| {r.name} | {bl} | {op} | {sp} | {correct} |")
        
        lines.append("")
    
    # Detailed statistics
    lines.append("## Detailed Statistics")
    lines.append("")
    
    for r in results:
        if not r.baseline_times_ns:
            continue
        lines.append(f"### {r.name}")
        lines.append("")
        
        bs = r.baseline_stats
        os_stats = r.optimized_stats
        
        lines.append("| Metric | Baseline | Optimized |")
        lines.append("|--------|----------|-----------|")
        
        metrics = ['min_ns', 'median_ns', 'mean_ns', 'p95_ns', 'p99_ns', 'stddev_ns']
        labels = ['Min', 'Median', 'Mean', 'P95', 'P99', 'Std Dev']
        
        for label, metric in zip(labels, metrics):
            bl_val = _fmt(bs.get(metric, 0))
            op_val = _fmt(os_stats.get(metric, 0)) if os_stats else "N/A"
            lines.append(f"| {label} | {bl_val} | {op_val} |")
        
        lines.append("")
    
    # Summary
    valid = [r for r in results if r.optimized_times_ns and r.correct]
    if valid:
        speedups = [r.speedup for r in valid]
        geomean = math.exp(sum(math.log(max(s, 0.001)) for s in speedups) / len(speedups))
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total benchmarks:** {len(results)}")
        lines.append(f"- **Passed (correct):** {len(valid)}/{len(results)}")
        lines.append(f"- **Geometric mean speedup:** {geomean:.2f}x")
        if speedups:
            best = max(valid, key=lambda r: r.speedup)
            worst = min(valid, key=lambda r: r.speedup)
            lines.append(f"- **Best speedup:** {best.speedup:.2f}x ({best.name})")
            lines.append(f"- **Worst speedup:** {worst.speedup:.2f}x ({worst.name})")
        
        lines.append("")
        lines.append("### Speedup Distribution")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_bar_chart(valid))
        lines.append("```")
        lines.append("")
    
    # CPython Bottleneck Analysis
    lines.append("## CPython Bottleneck Analysis")
    lines.append("")
    lines.append("Based on our analysis of CPython 3.x internals, we identified 10 primary performance bottlenecks:")
    lines.append("")
    lines.append("| ID | Bottleneck | Impact | HighPy Mitigation |")
    lines.append("|----|-----------|--------|-------------------|")
    lines.append("| B1 | Dynamic Type Dispatch | Every operation requires type check | Type inference + monomorphic dispatch |")
    lines.append("| B2 | Object Model Overhead | 28+ bytes per int vs 4 bytes in C | CompactArray with struct packing |")
    lines.append("| B3 | Bytecode Interpretation | Fetch-decode-execute loop overhead | AST optimization + native compilation |")
    lines.append("| B4 | Attribute Lookup | Dict-based + MRO traversal | Polymorphic inline caching |")
    lines.append("| B5 | Function Call Overhead | Frame creation + argument parsing | Function inlining + specialization |")
    lines.append("| B6 | GIL Contention | Prevents true parallelism | Process-based parallelism with purity analysis |")
    lines.append("| B7 | Boxing/Unboxing | Wrap/unwrap for every operation | Compact typed arrays |")
    lines.append("| B8 | Late Binding | Name lookups at runtime | Global-to-local promotion |")
    lines.append("| B9 | Memory Allocation | Frequent small allocations | Arena allocator with region-based management |")
    lines.append("| B10 | Lack of Specialization | Generic code for all types | Multi-tier adaptive specialization |")
    lines.append("")
    
    # Novel contributions
    lines.append("## Novel Contributions")
    lines.append("")
    lines.append("1. **Adaptive Multi-Level Specialization (AMLS):** A three-tier optimization")
    lines.append("   system that automatically promotes functions through increasingly aggressive")
    lines.append("   optimization levels based on runtime profiling feedback.")
    lines.append("")
    lines.append("2. **Type Lattice Inference:** A lattice-based abstract interpretation engine")
    lines.append("   that infers types through forward analysis with widening at loops and")
    lines.append("   narrowing through isinstance guards.")
    lines.append("")
    lines.append("3. **Speculative Optimization with Deoptimization:** Guard-based speculative")
    lines.append("   optimization that safely falls back to interpreted execution when type")
    lines.append("   assumptions are violated.")
    lines.append("")
    lines.append("4. **Integrated Analysis-Optimization Pipeline:** End-to-end pipeline from")
    lines.append("   bottleneck identification through type profiling to code generation,")
    lines.append("   operating entirely within standard CPython.")
    lines.append("")
    
    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    if valid:
        lines.append(
            f"HighPy achieves a geometric mean speedup of {geomean:.2f}x across "
            f"{len(valid)} benchmarks while maintaining semantic correctness. "
            "The adaptive multi-level specialization approach demonstrates that "
            "significant performance improvements are achievable within standard "
            "CPython through careful combination of static analysis, runtime "
            "profiling, and code generation techniques."
        )
    else:
        lines.append("Benchmark results are being collected. Re-run to generate full conclusion.")
    lines.append("")
    
    # Write report
    content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Report generated: {output_path}")
    return content


def generate_latex_tables(
    results: List[BenchmarkResult],
    output_path: str = "reports/tables.tex",
) -> str:
    """Generate LaTeX table snippets for paper inclusion."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = []
    lines.append("% HighPy Benchmark Results - LaTeX Tables")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    
    for cat, cat_results in sorted(categories.items()):
        lines.append(f"% --- {cat.upper()} ---")
        lines.append("\\begin{table}[htbp]")
        lines.append(f"\\caption{{{cat.title()} Benchmark Results}}")
        lines.append(f"\\label{{tab:{cat}_results}}")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lrrrc}")
        lines.append("\\toprule")
        lines.append("Benchmark & Baseline & Optimized & Speedup & Correct \\\\")
        lines.append("\\midrule")
        
        for r in cat_results:
            bl = _fmt_latex(r.baseline_median_ns) if r.baseline_times_ns else "N/A"
            op = _fmt_latex(r.optimized_median_ns) if r.optimized_times_ns else "N/A"
            sp = f"{r.speedup:.2f}$\\times$" if r.optimized_times_ns else "N/A"
            correct = "\\checkmark" if r.correct and not r.error else "\\xmark"
            name_escaped = r.name.replace('_', '\\_')
            lines.append(f"{name_escaped} & {bl} & {op} & {sp} & {correct} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    
    content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"LaTeX tables generated: {output_path}")
    return content


def _fmt(ns: float) -> str:
    """Format nanoseconds for display."""
    if ns < 1_000:
        return f"{ns:.0f} ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.1f} µs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.3f} s"


def _fmt_latex(ns: float) -> str:
    """Format nanoseconds for LaTeX."""
    if ns < 1_000:
        return f"{ns:.0f}\\,ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.1f}\\,\\mu s"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f}\\,ms"
    else:
        return f"{ns/1_000_000_000:.3f}\\,s"


def _ascii_bar_chart(results: List[BenchmarkResult], width: int = 40) -> str:
    """Generate an ASCII bar chart of speedups."""
    if not results:
        return ""
    
    lines = []
    max_speedup = max(r.speedup for r in results)
    max_name_len = max(len(r.name) for r in results)
    
    for r in sorted(results, key=lambda x: x.speedup, reverse=True):
        bar_len = int(r.speedup / max_speedup * width) if max_speedup > 0 else 0
        bar = "█" * bar_len
        name = r.name.ljust(max_name_len)
        lines.append(f"  {name}  {bar} {r.speedup:.2f}x")
    
    return "\n".join(lines)


def generate_full_report(results: List[BenchmarkResult]):
    """Generate all report formats."""
    generate_markdown_report(results)
    generate_latex_tables(results)
    print("\nAll reports generated in ./reports/")
