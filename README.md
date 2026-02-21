# HighPy — Recursive Fractal Optimization Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-green" alt="Version 1.0.0"/>
  <img src="https://img.shields.io/badge/Tests-266%20passing-brightgreen" alt="266 tests passing"/>
  <img src="https://img.shields.io/badge/Benchmarks-58%20functions%20%C2%B7%209%20categories-orange" alt="58 benchmark functions"/>
  <img src="https://img.shields.io/badge/Coverage-RFOE%204%2C558%20lines-blueviolet" alt="4558 lines"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Journal-JSS%20(Elsevier)-red" alt="Submitted to JSS"/>
</p>

> **Submitted to:** *Journal of Systems and Software* (Elsevier)
>
> **Paper:** *Recursive Fractal Optimization Engine: Banach Contraction Convergence Guarantees and Automatic Memoization for Python Program Optimization*
>
> **Author:** Farid Dihan Nahdi · Universitas Gadjah Mada, Yogyakarta, Indonesia · `fariddihannahdi@mail.ugm.ac.id`

---

## Overview

**HighPy** is a multi-level Python optimization framework. Its primary research contribution is the **Recursive Fractal Optimization Engine (RFOE)** — a novel system that combines fractal self-similar program decomposition, Banach contraction mapping convergence theory, and meta-circular self-optimization to deliver both formally guaranteed and empirically validated performance improvements.

RFOE differs from JIT compilers (PyPy, Numba) and single-pass AST rewriters (Nuitka) in three key respects:

| Property | JIT (PyPy/Numba) | Single-pass AST | **RFOE** |
|---|:---:|:---:|:---:|
| Formal convergence proof | ✗ | ✗ | **✓** |
| Self-similar multi-level passes | ✗ | ✗ | **✓** |
| Meta-circular self-optimization | ✗ | ✗ | **✓** |
| Static purity-aware memoization | ✗ | partial | **✓** |
| Quantitative convergence rate | ✗ | ✗ | **✓** |

---

## Key Results

| Metric | Result |
|---|---|
| Geometric mean speedup — core 17-function suite | **6.755×** |
| Geometric mean speedup — 41 large-scale functions | **3.402×** |
| Peak speedup (DP via automatic memoization) | **39,072×** |
| Average energy reduction (AST-optimized functions) | **44.4%** (peak 95.4%) |
| Aitken Δ² convergence acceleration | up to **12.3×** fewer iterations |
| Proven pipeline contraction factor | **k = 0.7989 < 1** at 100% confidence |
| Cache-hit recompilation speedup | **> 130×** (SHA-256 source hashing) |
| Test suite | **266 / 266 passing** |
| Correctness | **58 / 58 benchmark functions verified** |

---

## Table of Contents

- [Theoretical Foundations](#theoretical-foundations)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Benchmark Results](#benchmark-results)
- [Test Suite](#test-suite)
- [Project Structure](#project-structure)
- [Changelog](#changelog)
- [Citation](#citation)
- [License](#license)

---

## Theoretical Foundations

### 1. Program Energy Metric Space

For a program `P` represented as an AST, the *optimization energy* is the four-dimensional vector:

$$E(P) = (e_{\text{instr}},\; e_{\text{mem}},\; e_{\text{branch}},\; e_{\text{abstract}}) \in \mathbb{R}^4_+$$

with weights **w** = (1.0, 1.5, 2.0, 1.8) giving total scalar energy:

$$E_{\text{total}}(P) = \mathbf{w} \cdot E(P)$$

The space $(M, d)$ with $d(E_1, E_2) = \sqrt{\sum_i w_i (E_{1,i} - E_{2,i})^2}$ (weighted Euclidean) is a complete metric space — the precondition for Banach's theorem.

### 2. Fractal Self-Similar Decomposition

A program is recursively decomposed across six granularity levels:

```
Level 0  EXPRESSION   —  individual expressions  (x+1, f(x))
Level 1  STATEMENT    —  single statements        (assignments, returns)
Level 2  BLOCK        —  basic blocks             (sequences of statements)
Level 3  FUNCTION     —  function definitions
Level 4  MODULE       —  module-level code
Level 5  PROGRAM      —  entire program
```

The *same* six morphisms are applied identically at **every** level. This self-similar structure is the defining fractal property: constant propagation at the expression level folds `1+2 → 3`; at the module level it propagates global constants.

### 3. Banach Contraction Convergence (Theorem 1)

An optimization morphism $T^*: M \to M$ is a *contraction mapping* if:

$$d(T^*(E(P_1)),\, T^*(E(P_2))) \leq k \cdot d(E(P_1),\, E(P_2)) \quad \forall\, P_1, P_2 \in M,\;\; k \in [0,1)$$

By **Banach's Fixed-Point Theorem**, this implies:

- A **unique fixed point** $E^* \in M$ exists with $T^*(E^*) = E^*$.
- Convergence is **geometric**: $d(E_n, E^*) \leq \dfrac{k^n}{1-k} \cdot d(E_0, E_1)$.
- Iteration count to $\varepsilon$-accuracy is $O\!\left(\dfrac{\log(1/\varepsilon)}{\log(1/k)}\right)$.

RFOE's *energy-guarded* morphism application (deep-copy → transform → compare energy → accept only if non-increasing) **structurally enforces** the contraction property. The empirically proven pipeline contraction factor is **k = 0.7989** (pairwise Lipschitz factors, 100% confidence).

### 4. Aitken Δ² Acceleration

For a linearly convergent sequence $x_n \to x^*$:

$$\tilde{x}_n = x_n - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}$$

`FixedPointEngine` adaptively switches between basic Banach iteration and Aitken acceleration, achieving up to **12.3×** fewer iterations on strongly linear contractions.

### 5. Meta-Circular Self-Optimization

Let $O$ be the optimizer with source code $S_O$. Define $\Phi(O) = O(S_O)$. RFOE computes $\Phi, \Phi^2, \ldots$ and reaches the meta-circular fixed point $O^*$ in **two generations** (final optimizer energy 306.75, time 6.11 ms).

### 6. Purity Lattice for Safe Automatic Memoization

Functions are classified by the static `PurityAnalyzer` into a four-level lattice:

```
PURE  ⊑  READ_ONLY  ⊑  LOCALLY_IMPURE  ⊑  IMPURE
 ↑              ↑
safe           safe        unsafe — aliasing risk    unsafe
```

Only `PURE` and `READ_ONLY` functions are memoized. `LOCALLY_IMPURE` functions (those performing local mutations, e.g., list appends, that may return the mutated object) are **excluded** because caching their return value could expose the internal mutable state across calls.

---

## Architecture

```
highpy/                          #  ~11,100 lines total
│
├── recursive/                   #  RFOE core — 4,558 lines (6 modules)
│   ├── fractal_optimizer.py     #  1,934 lines — RecursiveFractalOptimizer, UniversalMorphisms,
│   │                            #                 FractalDecomposer, OptimizationEnergy
│   ├── convergence_prover.py    #    680 lines — ConvergenceProver, BanachProof,
│   │                            #                 ContractionCertificate (pairwise Lipschitz)
│   ├── fixed_point_engine.py    #    463 lines — FixedPointEngine, Banach + Aitken Δ²
│   ├── meta_circular.py         #    394 lines — MetaCircularOptimizer, SelfOptimizationResult
│   ├── fractal_analyzer.py      #    501 lines — FractalAnalyzer, OptimizationEnergyField
│   └── purity_analyzer.py       #    586 lines — PurityAnalyzer, PurityReport, PurityLevel
│
├── compiler/                    #  AST / bytecode / native code generation
│   ├── ast_optimizer.py         #  Constant folding, DCE, CSE
│   ├── bytecode_rewriter.py     #  CPython bytecode peephole passes
│   └── native_codegen.py        #  C code generation via ctypes (Tier 3)
│
├── optimization/                #  Interprocedural & runtime passes
│   ├── type_specializer.py      #  Type lattice inference & monomorphic dispatch
│   ├── loop_optimizer.py        #  Loop unrolling and vectorization hints
│   ├── function_specializer.py  #  Cross-function specialization
│   ├── memory_pool.py           #  Region-based arena allocator
│   ├── lazy_evaluator.py        #  Lazy chain / thunk memoization
│   └── parallel_executor.py     #  Automatic parallelization of pure functions
│
├── runtime/                     #  Adaptive 3-tier execution engine
│   ├── adaptive_runtime.py      #  Bytecode → type-specialized → native JIT
│   ├── inline_cache.py          #  Polymorphic inline caching for attribute dispatch
│   └── deoptimizer.py           #  Guard-based deoptimization on type violation
│
└── analysis/                    #  Profiling & analysis
    ├── cpython_bottlenecks.py   #  CPython performance hotspot identification
    └── type_profiler.py         #  Static & runtime type inference
```

### Optimization tiers

| Tier | Name | Trigger | Technique |
|:---:|---|---|---|
| 0 | Interpreted | Default | Vanilla CPython |
| 1 | Bytecode-Optimized | First call | AST constant folding, strength reduction, DCE |
| 2 | Type-Specialized | Type-stable code | Monomorphic dispatch, type-guided codegen |
| 3 | Native-Compiled | Hot loops | C code generation via ctypes |
| — | **RFOE** | `@rfo` decorator | Fractal decomposition + Banach convergence + memoization |

---

## Installation

### Prerequisites

- Python 3.10 – 3.14
- A C compiler (optional — Tier 3 native compilation only)

### From source

```bash
git clone https://github.com/dihannahdi/highpy.git
cd highpy

python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### Runtime dependencies

```
numpy>=1.24.0
cffi>=1.15.0
```

Dev extras: `pytest`, `pytest-benchmark`, `matplotlib`, `tabulate`, `psutil`

---

## Quick Start

### 1. Basic optimization — `@rfo` decorator

```python
from highpy.recursive import rfo

@rfo
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# First call: RFOE detects recursion, applies memoization → O(n) instead of O(2^n)
print(fibonacci(40))   # returns instantly

# Subsequent processes with the same source: SHA-256 cache hit (>130× faster compilation)
```

### 2. Programmatic optimizer

```python
from highpy.recursive import RecursiveFractalOptimizer

optimizer = RecursiveFractalOptimizer(max_iterations=10)
optimized_fn = optimizer.optimize(fibonacci)   # returns callable

print(optimized_fn(35))   # verified correct output
```

### 3. Purity analysis

```python
from highpy.recursive import PurityAnalyzer, PurityLevel

analyzer = PurityAnalyzer()
report = analyzer.analyze(fibonacci)

print(report.level)           # PurityLevel.PURE
print(report.is_memoizable)   # True
print(report.reasons)         # [] — no impurity sources found
print(report.confidence)      # 0.95
```

### 4. Convergence certificate

```python
from highpy.recursive import ConvergenceProver

prover = ConvergenceProver()
proof = prover.prove_convergence(optimizer, sample_functions=[fibonacci])

print(proof.status)              # 'PROVEN'
print(proof.contraction_factor)  # e.g. 0.7989
print(proof.confidence)          # 1.0
print(proof.to_certificate())    # human-readable ASCII certificate
```

### 5. Fixed-point iteration with Aitken Δ² acceleration

```python
from highpy.recursive import FixedPointEngine
import math

engine = FixedPointEngine()
result = engine.iterate(0.5, math.cos)

print(result.estimated_fixed_point)   # ≈ 0.7390851332151607
print(result.iterations)              # typically 3–27 (adaptive)
print(result.status)                  # 'converged'
```

### 6. Meta-circular self-optimization

```python
from highpy.recursive import MetaCircularOptimizer

mco = MetaCircularOptimizer()
generations = mco.self_optimize(generations=3)

for gen in generations:
    print(f"Gen {gen.generation}:  "
          f"energy {gen.original_energy:.1f} → {gen.optimized_energy:.1f}  "
          f"({gen.energy_reduction:.1%} reduction)")
```

### 7. Fractal energy field analysis

```python
from highpy.recursive import FractalAnalyzer

analyzer = FractalAnalyzer()
field = analyzer.analyze_function(fibonacci)
print(analyzer.generate_report(field))
```

### 8. HighPy v1 API (unchanged)

```python
import highpy

@highpy.optimize
def compute(x, y):
    return x * x + y * y

@highpy.jit(specialize=True, native=True)
def matrix_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

---

## API Reference

### Top-level decorators (`import highpy`)

| Symbol | Description |
|---|---|
| `@highpy.optimize` | Adaptive 3-tier JIT optimizer |
| `@highpy.jit(specialize, native)` | JIT with explicit tier control |
| `@highpy.rfo` | Recursive fractal optimizer (RFOE) |
| `@highpy.specialize` | Type-lattice specialization |
| `@highpy.lazy` | Lazy chain evaluation |
| `@highpy.auto_parallel` | Automatic parallelization |

### RFOE — `highpy.recursive`

| Class / Function | Signature | Description |
|---|---|---|
| `rfo` | `@rfo` or `@rfo(...)` | Decorator — full RFOE pipeline |
| `rfo_optimize` | `(func, optimizer)` | Functional form of `@rfo` |
| `RecursiveFractalOptimizer` | `.optimize(func) → Callable` | Main RFOE orchestrator |
| `FixedPointEngine` | `.iterate(x0, f) → ConvergenceResult` | Banach + Aitken iteration |
| `ConvergenceProver` | `.prove_convergence(opt, funcs) → BanachProof` | Formal certificate issuer |
| `PurityAnalyzer` | `.analyze(func) → PurityReport` | Static purity classifier |
| `MetaCircularOptimizer` | `.self_optimize(generations) → List[SelfOptimizationResult]` | Self-optimization |
| `FractalAnalyzer` | `.analyze_function(func) → OptimizationEnergyField` | Energy field analysis |

### RFOE data types

| Type | Key attributes |
|---|---|
| `BanachProof` | `status`, `contraction_factor`, `confidence`, `estimated_iterations_to_convergence`, `to_certificate()` |
| `ContractionCertificate` | Per-morphism contraction factor and confidence |
| `ConvergenceResult` | `estimated_fixed_point`, `iterations`, `status`, `error_bound`, `contraction_factors` |
| `PurityReport` | `level`, `is_memoizable`, `confidence`, `reasons`, `global_reads`, `global_writes`, `io_calls` |
| `PurityLevel` | `PURE=0`, `READ_ONLY=1`, `LOCALLY_IMPURE=2`, `IMPURE=3` |
| `SelfOptimizationResult` | `generation`, `original_energy`, `optimized_energy`, `energy_reduction`, `speedup` |
| `OptimizationEnergyField` | Per-level energy vectors and fractal dimension estimate |

### Other subpackages

| Class | Module | Description |
|---|---|---|
| `ASTOptimizer` | `highpy.compiler` | AST constant folding, DCE, CSE |
| `BytecodeRewriter` | `highpy.compiler` | Bytecode peephole passes |
| `NativeCompiler` | `highpy.compiler` | C code generation backend |
| `AdaptiveRuntime` | `highpy.runtime` | Tiered JIT execution engine |
| `PolymorphicInlineCache` | `highpy.runtime` | PIC for attribute / method dispatch |
| `TypeSpecializer` | `highpy.optimization` | Type lattice inference |
| `LoopOptimizer` | `highpy.optimization` | Loop unrolling and vectorization |
| `MemoryPool` | `highpy.optimization` | Region-based arena allocator |
| `CPythonAnalyzer` | `highpy.analysis` | Performance hotspot identifier |
| `TypeProfiler` | `highpy.analysis` | Static and runtime type inference |

---

## Benchmark Results

All experiments: Windows, Python 3.14.2. Each function executed 1,000 times. Geometric means reported.

### Core suite — 17 functions

| Function | Baseline (µs) | RFOE (µs) | Speedup | Correct |
|---|---:|---:|---:|:---:|
| *AST-optimized functions* | | | | |
| arithmetic | 0.445 | 0.189 | 2.35× | ✓ |
| dead_code | 0.486 | 0.152 | 3.21× | ✓ |
| cse | 0.436 | 0.226 | 1.92× | ✓ |
| loop_compute | 13.103 | 5.745 | 2.28× | ✓ |
| nested_branches | 0.221 | 0.184 | 1.20× | ✓ |
| matrix_like | 19.076 | 10.034 | 1.90× | ✓ |
| fibonacci_iter | 1.584 | 1.607 | 0.99× | ✓ |
| polynomial | 0.679 | 0.433 | 1.57× | ✓ |
| constant_heavy | 0.355 | 0.173 | 2.05× | ✓ |
| identity_chain | 0.558 | 0.134 | 4.16× | ✓ |
| dead_heavy | 1.381 | 0.152 | **9.08×** | ✓ |
| mixed_heavy | 0.516 | 0.233 | 2.21× | ✓ |
| *Automatic memoization* | | | | |
| fib_recursive | 20.943 | 0.413 | 50.70× | ✓ |
| tribonacci | 48.207 | 0.514 | 93.82× | ✓ |
| grid_paths | 79.117 | 0.529 | 149.54× | ✓ |
| binomial | 100.661 | 0.491 | **204.82×** | ✓ |
| subset_sum | 23.203 | 0.536 | 43.31× | ✓ |
| **Geometric mean** | | | **6.755×** | **17/17** |

### Large-scale suite — 41 functions across 9 categories

| Category | Representative functions | Geo. Mean | Peak |
|---|---|---:|---:|
| A. Sorting | quicksort, mergesort, insertion, heapsort | 0.68× | 1.14× |
| B. Graph Algorithms | DFS, shortest path, components, topo-sort | 0.76× | 1.03× |
| C. Dynamic Programming | LCS, edit dist., coin change, matrix chain | **557.3×** | **39,072×** |
| D. String Processing | palindrome, vowels, RLE, word freq | 1.18× | 2.37× |
| E. Numerical | matrix mult, Newton sqrt, trapezoidal | 1.09× | 2.93× |
| F. Data Processing | moving avg, normalize, group-by, flatten | 0.88× | 1.55× |
| G. Tree Operations | depth, flatten, count, search | 0.37× | 0.49× |
| H. Combinatorial | Stirling, Bell, derangements, partitions | **135.3×** | **7,868×** |
| I. Real-World | CSV parse, email valid., Levenshtein | 1.02× | 1.09× |
| **Overall geometric mean** | 41 functions | **3.402×** | **39,072×** |

> Categories A, B, F, and G contain impure functions, data-structure-heavy code, and tree
> traversals where memoization is inapplicable and AST overhead slightly exceeds gains.
> The 3.402× overall mean is computed across *all* 41 functions including these slowdowns.

### Energy reduction — AST-optimized functions

| Function | Initial E | Final E | Reduction |
|---|---:|---:|---:|
| dead_heavy | 113.25 | 5.25 | **95.4%** |
| constant_heavy | 41.75 | 2.75 | 93.4% |
| identity_chain | 49.25 | 3.75 | 92.4% |
| dead_code | 47.75 | 5.25 | 89.0% |
| arithmetic | 50.00 | 14.50 | 71.0% |
| mixed_heavy | 52.25 | 16.00 | 69.4% |
| cse | 46.00 | 19.00 | 58.7% |
| polynomial | 34.50 | 16.00 | 53.6% |
| loop_compute | 69.40 | 33.15 | 52.2% |
| matrix_like | 423.90 | 242.65 | 42.8% |
| nested_branches | 43.25 | 30.00 | 30.6% |
| fibonacci_iter | 76.65 | 72.15 | 5.9% |
| **Average** | | | **62.9%** |

### Aitken Δ² acceleration

| Contraction f(x) | Basic iters | Aitken iters | Speedup |
|---|---:|---:|---:|
| x/2 + 1  (fp = 2.0) | 37 | 3 | **12.3×** |
| x/3 + 2  (fp = 3.0) | 24 | 3 | **8.0×** |
| √(x+1)  (fp ≈ φ) | 20 | 11 | 1.8× |
| cos(x)  (fp ≈ 0.739) | 3 | 27 | 0.1× |
| 1/(1+x)  (fp ≈ 0.618) | 5 | 12 | 0.4× |

Aitken acceleration excels for strongly linear contractions and may be counterproductive for near-quadratic maps. The `FixedPointEngine` selects the better strategy automatically.

### Running benchmarks

```bash
python benchmarks/benchmark_runner.py     # full suite
python benchmarks/bench_recursive.py      # RFOE + memoization (17 functions)
python benchmarks/bench_large_scale.py    # large-scale 41-function suite
python benchmarks/report_generator.py     # formatted performance report
```

---

## Test Suite

**266 / 266 passing** — Python 3.14.2, pytest 9.0.2.

```bash
python -m pytest tests/ -q               # fast run
python -m pytest tests/ -v               # verbose
python -m pytest tests/test_recursive.py -v       # RFOE only
python -m pytest tests/test_purity_largescale.py  # purity analysis
python -m pytest tests/ --cov=highpy --cov-report=html
```

| Test file | Description |
|---|---|
| `test_recursive.py` | RFOE core: fractal optimizer, convergence prover, Aitken, meta-circular |
| `test_purity_largescale.py` | Purity analyzer against all 58 large-scale benchmark functions |
| `test_analysis.py` | CPython bottleneck analysis and type profiling |
| `test_compiler.py` | AST optimizer, bytecode rewriter, native code generation |
| `test_integration.py` | End-to-end integration across all optimization tiers |
| `test_runtime.py` | Adaptive runtime and polymorphic inline cache |
| `test_memory_pool.py` | Arena allocator correctness and benchmarks |
| `test_type_specializer.py` | Type lattice inference engine |
| `test_loop_optimizer.py` | Loop unrolling and vectorization hints |
| `test_function_specializer.py` | Interprocedural function specialization |
| `test_lazy_parallel.py` | Lazy evaluator and parallel executor |
| `test_type_profiler.py` | Static and runtime type profiler |

---

## Project Structure

```
highpy/
├── highpy/                        # Main package (~11,100 lines)
│   ├── __init__.py                # Top-level API — exposes all 40+ public symbols
│   ├── recursive/                 # RFOE — 4,558 lines (primary research contribution)
│   │   ├── fractal_optimizer.py   # 1,934 lines
│   │   ├── convergence_prover.py  #   680 lines
│   │   ├── fixed_point_engine.py  #   463 lines
│   │   ├── meta_circular.py       #   394 lines
│   │   ├── fractal_analyzer.py    #   501 lines
│   │   └── purity_analyzer.py     #   586 lines
│   ├── compiler/
│   ├── optimization/
│   ├── runtime/
│   └── analysis/
│
├── benchmarks/                    # Benchmark suite (~2,438 lines)
│   ├── bench_recursive.py         # RFOE + memoization benchmarks
│   ├── bench_large_scale.py       # 41-function diverse benchmark
│   ├── benchmark_suite.py         # Core benchmark runner
│   ├── benchmark_runner.py        # Orchestrator + timing harness
│   └── report_generator.py        # Formatted output generator
│
├── tests/                         # 266 unit tests (~2,879 lines)
├── reports/                       # Benchmark outputs, JSON results, performance report
│
├── manuscript/                    # JSS submission package
│   ├── manuscript_jss.tex         # LaTeX source (elsarticle, single-column 12pt)
│   ├── manuscript_jss.docx        # DOCX version
│   ├── references.bib             # BibTeX (13 references)
│   ├── compile.bat                # pdflatex → bibtex → pdflatex×2
│   ├── generate_docx.py           # DOCX generator (python-docx, ~1,200 lines)
│   ├── cover_letter.txt
│   ├── title_page.txt
│   ├── highlights.txt
│   ├── biography.txt
│   ├── credit_author_statement.txt
│   ├── declaration_ai_use.txt
│   ├── declaration_competing_interests.txt
│   └── SUBMISSION_CHECKLIST.md
│
├── pyproject.toml                 # Build system (PEP 517/518, replaces setup.py)
├── setup.py                       # Legacy build (kept for compatibility)
├── requirements.txt               # Pinned runtime dependencies
├── LICENSE                        # MIT
└── README.md
```

---

## Changelog

### v1.0.0 — February 2026 (initial release)

**RFOE (6 modules, 4,558 lines):**
- Fractal decomposition across 6 levels with 6 universal optimization morphisms
- Energy-guarded morphism application structurally enforces the contraction property
- Purity-aware automatic memoization (`PURE` + `READ_ONLY` only — `LOCALLY_IMPURE` excluded for aliasing safety)
- SHA-256 source-level compilation cache eliminates recompilation overhead
- Aitken Δ² adaptive fixed-point acceleration

**Correctness fixes (post-audit v1.0.0):**
- `convergence_prover.py` — `verify_morphism()` and `verify_pipeline()` now use **pairwise Lipschitz factors** (correct Banach definition; previous implementation used single-point energy ratios which is not the Lipschitz constant)
- `fractal_optimizer.py` — `LOAD_ATTR` removed from `_ABSTRACTION_OPS` (was double-counted with `_MEMORY_OPS`); `OptimizationEnergy.distance()` uses weighted Euclidean metric consistent with `total`
- `purity_analyzer.py` — `is_memoizable` returns `False` for `LOCALLY_IMPURE` (previously returned `True`, causing aliasing safety bug)
- Tests updated to match corrected behavior: 266 / 266 passing

**Package:**
- Added `pyproject.toml` (PEP 517/518 build system; Python 3.10–3.14 classifiers; correct author metadata)
- DOCX manuscript generator (`manuscript/generate_docx.py`)

---

## Contributing

This repository accompanies a research paper under review at the Journal of Systems and Software. Contributions that improve reproducibility, add benchmarks, fix bugs, or extend the framework are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-contribution`
3. Ensure all tests pass: `python -m pytest tests/ -q`
4. Submit a pull request with a clear description

---

## Citation

If you use HighPy / RFOE in your research, please cite:

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

Released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Universitas Gadjah Mada &nbsp;·&nbsp; Yogyakarta, Indonesia &nbsp;·&nbsp; 2026</sub>
</p>
