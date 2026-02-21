# HighPy: Recursive Fractal Optimization Engine for Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-green" alt="Version 1.0.0"/>
  <img src="https://img.shields.io/badge/Tests-266%20unit%20tests-brightgreen" alt="266 unit tests"/>
  <img src="https://img.shields.io/badge/Benchmarks-58%20functions-orange" alt="58 benchmark functions"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Journal-JSS%20(Elsevier)-red" alt="Submitted to JSS"/>
</p>

> **Submitted to:** *Journal of Systems and Software* (Elsevier)
>
> **Paper:** *Recursive Fractal Optimization Engine: Banach Contraction Convergence Guarantees and Automatic Memoization for Python Program Optimization*
>
> **Author:** Farid Dihan Nahdi — Universitas Gadjah Mada, Yogyakarta, Indonesia (`fariddihannahdi@mail.ugm.ac.id`)

---

## Abstract

Python's interpreted nature incurs significant performance penalties compared to compiled languages, yet existing optimization approaches — JIT compilers and single-pass AST rewriters — lack formal convergence guarantees. We present the **Recursive Fractal Optimization Engine (RFOE)**, a novel framework that unifies three mathematically grounded pillars:

1. **Fractal Self-Similar Decomposition** — programs are hierarchically decomposed across six granularity levels (*expression → statement → block → function → module → program*) with identical optimization morphisms applied at every level.
2. **Fixed-point convergence via Banach's Contraction Mapping Theorem** — each optimization pass is modeled as a contraction operator in the complete metric space of program energy vectors, providing existence, uniqueness, and geometric convergence-rate guarantees.
3. **Meta-circular self-optimization** — the optimizer applies its own passes to its own source code, converging to a Futamura-projection-inspired fixed point.

RFOE additionally incorporates **purity-aware automatic memoization**: a static purity analyzer classifies functions into a four-level lattice (`PURE`, `READ_ONLY`, `LOCALLY_IMPURE`, `IMPURE`), enabling safe memoization decisions without runtime overhead. SHA-256 source-level caching eliminates recompilation overhead for previously optimized functions.

**Key Results:**

| Metric | Value |
|---|---|
| Geometric mean speedup (core suite) | **6.755×** |
| Speedup across 41 large-scale functions | **3.402×** |
| Peak speedup (dynamic programming via memoization) | **39,072×** |
| Average energy reduction | **44.4%** |
| Aitken Δ² convergence acceleration | up to **12.3×** |
| Proven pipeline contraction factor | **k = 0.7989 < 1** (100% confidence) |
| Cache hit speedup | **>130×** |

---

## Table of Contents

- [Theoretical Foundations](#theoretical-foundations)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Test Suite](#test-suite)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Theoretical Foundations

### 1. Fractal Self-Similar Decomposition

The RFOE decomposes a program $P$ into a hierarchy of optimization levels $\mathcal{L} = \{L_0, L_1, \ldots, L_5\}$, where $L_0$ = expression, $L_5$ = program. An **optimization morphism** $\phi: \mathcal{L}_i \to \mathcal{L}_{i+1}$ preserves semantic equivalence while reducing the program energy metric:

$$E(P) = \alpha \cdot C_{\text{instr}}(P) + \beta \cdot M_{\text{pressure}}(P) + \gamma \cdot B_{\text{cost}}(P)$$

where $C_{\text{instr}}$, $M_{\text{pressure}}$, and $B_{\text{cost}}$ denote instruction complexity, memory pressure, and branch prediction cost respectively.

### 2. Fixed-Point Convergence (Banach Contraction Theorem)

Let $(X, d)$ be the complete metric space of program energy vectors. An optimization pass $O: X \to X$ is a *contraction mapping* if:

$$\exists\, k \in [0, 1): \quad d(O(P_1), O(P_2)) \leq k \cdot d(P_1, P_2) \quad \forall P_1, P_2 \in X$$

By Banach's theorem, repeated application converges to a **unique fixed point** $P^* = O(P^*)$:

$$P^* = \lim_{n \to \infty} O^n(P_0), \quad d(O^n(P_0), P^*) \leq \frac{k^n}{1-k} d(O(P_0), P_0)$$

The full optimization pipeline achieves a proven contraction factor of $k = 0.7989 < 1$.

### 3. Meta-Circular Self-Optimization

Inspired by Futamura projections, the optimizer $O$ is itself subject to optimization:

$$O' = O(O) \implies O'' = O(O') \implies \cdots \implies O^* = \lim_{n \to \infty} O^n(O_0)$$

This yields progressively more efficient optimizer instances converging to a self-consistent fixed point.

### 4. Purity Lattice for Safe Memoization

Functions are classified into a four-level lattice for memoization eligibility:

```
PURE  ⊑  READ_ONLY  ⊑  LOCALLY_IMPURE  ⊑  IMPURE
 ↑                                              ↑
safe to memoize                        unsafe to memoize
```

---

## Architecture

```
highpy/
├── analysis/              # CPython bottleneck identification & type profiling
│   ├── cpython_bottlenecks.py     # CPythonAnalyzer: identifies performance hotspots
│   └── type_profiler.py           # TypeProfiler: runtime/static type inference
│
├── compiler/              # AST/bytecode optimization & native code generation
│   ├── ast_optimizer.py           # Constant folding, dead code elimination, CSE
│   ├── bytecode_rewriter.py       # Peephole optimization on CPython bytecode
│   └── native_codegen.py          # C code generation via ctypes (Tier 3)
│
├── optimization/          # Core optimization passes
│   ├── type_specializer.py        # Type lattice inference & monomorphic dispatch
│   ├── loop_optimizer.py          # Loop unrolling, vectorization hints
│   ├── function_specializer.py    # Cross-function interprocedural specialization
│   ├── memory_pool.py             # Region-based arena memory management
│   ├── lazy_evaluator.py          # Lazy chain evaluation / thunk memoization
│   └── parallel_executor.py       # Automatic parallelization of pure functions
│
├── runtime/               # Adaptive execution engine
│   ├── adaptive_runtime.py        # 3-tier JIT: bytecode → type-specialized → native
│   ├── inline_cache.py            # Polymorphic inline caching (PIC) for attr access
│   └── deoptimizer.py             # Guard-based deoptimization on type violation
│
└── recursive/             # Recursive Fractal Optimization Engine (RFOE)
    ├── fractal_optimizer.py       # RecursiveFractalOptimizer, OptimizationMorphism
    ├── fixed_point_engine.py      # FixedPointEngine, Banach convergence, Aitken Δ²
    ├── meta_circular.py           # MetaCircularOptimizer, SelfOptimizingPass
    ├── fractal_analyzer.py        # FractalDimension, OptimizationEnergyField
    ├── convergence_prover.py      # ConvergenceProver, ContractionCertificate
    └── purity_analyzer.py         # PurityAnalyzer, PurityLevel lattice
```

### Optimization Tiers

| Tier | Name | Trigger | Technique |
|---|---|---|---|
| 0 | Interpreted | Default | Vanilla CPython |
| 1 | Bytecode-Optimized | 5 calls | AST constant folding, strength reduction, dead-code elimination |
| 2 | Type-Specialized | 20 calls + type stable | Monomorphic dispatch, type-guided code generation |
| 3 | Native-Compiled | 50 calls + loops | C code generation via ctypes, direct memory access |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A C compiler (for Tier 3 native compilation, optional)

### From source

```bash
git clone https://github.com/dihannahdi/highpy.git
cd highpy
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### Dependencies

```
numpy>=1.24.0
cffi>=1.15.0
```

Dev dependencies: `pytest`, `pytest-benchmark`, `matplotlib`, `tabulate`, `psutil`

---

## Quick Start

### Basic optimization with `@optimize` decorator

```python
import highpy

@highpy.optimize
def compute(x, y):
    return x * x + y * y

result = compute(3.14, 2.71)
```

### JIT compilation with type specialization

```python
@highpy.jit(specialize=True, native=True)
def matrix_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

### Recursive Fractal Optimization Engine (RFOE)

```python
from highpy.recursive import rfo, rfo_optimize, RecursiveFractalOptimizer

# Decorator usage — automatic fractal optimization + memoization
@rfo(levels=6, memoize=True)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Returns instantly after the first call (SHA-256 cache hit)
result = fibonacci(35)

# Programmatic usage
optimizer = RecursiveFractalOptimizer(max_levels=6)
optimized_fn = rfo_optimize(fibonacci, optimizer)
```

### Purity analysis

```python
from highpy.recursive import PurityAnalyzer, PurityLevel

analyzer = PurityAnalyzer()
report = analyzer.analyze(my_function)
print(report.level)          # PurityLevel.PURE | READ_ONLY | LOCALLY_IMPURE | IMPURE
print(report.safe_to_memoize)  # True / False
```

### Convergence certificate

```python
from highpy.recursive import ConvergenceProver

prover = ConvergenceProver()
cert = prover.prove(optimizer_pipeline)
print(cert.contraction_factor)  # e.g. 0.7989
print(cert.is_proven)           # True
```

---

## API Reference

### Core Decorators

| Symbol | Module | Description |
|---|---|---|
| `@optimize` | `highpy.runtime` | Adaptive 3-tier JIT optimizer |
| `@jit(specialize, native)` | `highpy.runtime` | JIT with explicit tier control |
| `@rfo(levels, memoize)` | `highpy.recursive` | Recursive fractal optimizer decorator |
| `@specialize` | `highpy.optimization` | Type-lattice specialization |
| `@lazy` | `highpy.optimization` | Lazy chain evaluation |
| `@auto_parallel` | `highpy.optimization` | Automatic parallelization |

### Key Classes

| Class | Module | Description |
|---|---|---|
| `RecursiveFractalOptimizer` | `highpy.recursive` | Main RFOE orchestrator |
| `FixedPointEngine` | `highpy.recursive` | Banach contraction iteration + Aitken Δ² |
| `ConvergenceProver` | `highpy.recursive` | Formal contraction certificate issuer |
| `PurityAnalyzer` | `highpy.recursive` | Static purity lattice classification |
| `MetaCircularOptimizer` | `highpy.recursive` | Self-optimizing meta-circular pass |
| `FractalAnalyzer` | `highpy.recursive` | Fractal dimension & energy field analysis |
| `AdaptiveRuntime` | `highpy.runtime` | Tiered JIT runtime engine |
| `PolymorphicInlineCache` | `highpy.runtime` | PIC for attribute/method dispatch |
| `TypeSpecializer` | `highpy.optimization` | Type lattice inference engine |
| `ASTOptimizer` | `highpy.compiler` | AST-level constant folding & DCE |
| `NativeCompiler` | `highpy.compiler` | C code generation backend |
| `CPythonAnalyzer` | `highpy.analysis` | CPython bottleneck profiler |

---

## Benchmarks

Benchmarks are located in `benchmarks/` and cover nine real-world categories:

1. **Numerical computation** — matrix operations, numerical integration
2. **Dynamic programming** — memoized recursion, knapsack, LCS
3. **String processing** — pattern matching, parsing
4. **Data pipeline** — transformation chains, sorting
5. **Graph algorithms** — BFS/DFS, shortest paths
6. **Scientific computing** — Monte Carlo, Mandelbrot
7. **Function call overhead** — dispatch microbenchmarks
8. **Memory allocation** — arena vs. heap allocation patterns
9. **Large-scale** — 41 complex real-world functions

### Running benchmarks

```bash
# Full benchmark suite
python benchmarks/benchmark_runner.py

# Large-scale suite
python benchmarks/bench_large_scale.py

# Recursive / RFOE suite
python benchmarks/bench_recursive.py

# Generate report
python benchmarks/report_generator.py
```

### Selected Results

| Category | Baseline | Optimized | Speedup |
|---|---|---|---|
| Dynamic programming (DP) | baseline | optimized | **39,072×** (memoization) |
| Mandelbrot | 4.60 ms | 483 µs | **9.52×** |
| Function call overhead | 5.35 ms | 2.04 ms | **2.62×** |
| Data pipeline | 46.93 ms | 34.55 ms | **1.36×** |
| **Geometric mean (core)** | — | — | **6.755×** |
| **Geometric mean (large-scale)** | — | — | **3.402×** |

---

## Test Suite

The test suite comprises **266 unit tests** across all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=highpy --cov-report=html

# Run specific module tests
python -m pytest tests/test_recursive.py -v
python -m pytest tests/test_purity_largescale.py -v
```

| Test file | Description |
|---|---|
| `test_recursive.py` | RFOE core: fractal optimizer, fixed-point, meta-circular |
| `test_purity_largescale.py` | Large-scale purity analysis (58-function benchmark) |
| `test_analysis.py` | CPython bottleneck analysis & type profiling |
| `test_compiler.py` | AST optimizer, bytecode rewriter, native codegen |
| `test_type_specializer.py` | Type lattice inference |
| `test_loop_optimizer.py` | Loop unrolling and vectorization |
| `test_function_specializer.py` | Interprocedural specialization |
| `test_memory_pool.py` | Arena allocator benchmarks |
| `test_lazy_parallel.py` | Lazy evaluator and parallel executor |
| `test_runtime.py` | Adaptive runtime and inline cache |
| `test_integration.py` | End-to-end integration tests |

---

## Project Structure

```
highpy/
├── highpy/              # Main library (4,300+ lines)
│   ├── analysis/
│   ├── compiler/
│   ├── optimization/
│   ├── runtime/
│   └── recursive/       # RFOE (primary contribution)
├── benchmarks/          # Benchmark suite (58 functions, 9 categories)
├── tests/               # 266 unit tests
├── reports/             # Benchmark output and performance reports
├── manuscript/          # JSS paper (LaTeX source + bibliography)
│   ├── manuscript_jss.tex
│   └── references.bib
├── requirements.txt
└── setup.py
```

---

## Contributing

This repository accompanies a research paper submission. Contributions that improve reproducibility, add benchmarks, or extend the theoretical framework are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-contribution`
3. Run the tests: `python -m pytest tests/ -q`
4. Submit a pull request

---

## Citation

If you use HighPy or RFOE in your research, please cite:

```bibtex
@article{nahdi2026rfoe,
  title   = {Recursive Fractal Optimization Engine: Banach Contraction
             Convergence Guarantees and Automatic Memoization for
             Python Program Optimization},
  author  = {Nahdi, Farid Dihan},
  journal = {Journal of Systems and Software},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/dihannahdi/highpy}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Universitas Gadjah Mada · Yogyakarta, Indonesia · 2026</sub>
</p>
