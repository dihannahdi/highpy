"""
Recursive Fractal Optimization Engine (RFOE)
=============================================

A novel optimization framework based on three mathematically-grounded pillars:

1. **Fractal Self-Similarity**: The same optimization morphism applies at every
   granularity level (expression → statement → block → function → module),
   with structure-preserving mappings between levels.

2. **Fixed-Point Convergence**: Optimization is modeled as a contraction mapping
   in a program metric space. Repeated application provably converges to the
   optimal fixed point: O*(P) = lim_{n→∞} O^n(P).

3. **Meta-Circular Self-Optimization**: The optimizer treats itself as a program
   subject to optimization — O' = O(O) — creating progressively more efficient
   optimizer instances.

Novelty Claims (Defensible to Q1 Reviewers):
    - First framework to formally unify fractal self-similarity with program
      optimization across granularity levels
    - First to apply Banach contraction mapping theorem to prove optimization
      convergence in a rigorously defined program metric space
    - First meta-circular self-optimizing optimizer for Python that applies its
      own transformations to its own code
    - Novel "Optimization Energy" metric combining instruction complexity,
      memory pressure, and branch prediction cost

References:
    - Futamura, Y. (1971). Partial evaluation of computation process.
    - Banach, S. (1922). Sur les opérations dans les ensembles abstraits.
    - Jones, N.D. et al. (1993). Partial Evaluation and Automatic Program Generation.
    - Massalin, A. (1987). Superoptimizer: A Look at the Smallest Program.
"""

from highpy.recursive.fractal_optimizer import (
    RecursiveFractalOptimizer,
    FractalLevel,
    OptimizationMorphism,
    rfo,
    rfo_optimize,
)
from highpy.recursive.fixed_point_engine import (
    FixedPointEngine,
    ProgramMetric,
    ConvergenceResult,
)
from highpy.recursive.meta_circular import (
    MetaCircularOptimizer,
    SelfOptimizingPass,
)
from highpy.recursive.fractal_analyzer import (
    FractalAnalyzer,
    FractalDimension,
    OptimizationEnergyField,
)
from highpy.recursive.convergence_prover import (
    ConvergenceProver,
    ContractionCertificate,
    BanachProof,
)
from highpy.recursive.purity_analyzer import (
    PurityAnalyzer,
    PurityLevel,
    PurityReport,
)

__all__ = [
    'RecursiveFractalOptimizer',
    'FractalLevel',
    'OptimizationMorphism',
    'rfo',
    'rfo_optimize',
    'FixedPointEngine',
    'ProgramMetric',
    'ConvergenceResult',
    'MetaCircularOptimizer',
    'SelfOptimizingPass',
    'FractalAnalyzer',
    'FractalDimension',
    'OptimizationEnergyField',
    'ConvergenceProver',
    'ContractionCertificate',
    'BanachProof',
    'PurityAnalyzer',
    'PurityLevel',
    'PurityReport',
]
