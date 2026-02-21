"""
HighPy: Adaptive Multi-Level Specialization Framework for Python Performance Optimization
========================================================================================

HighPy addresses Python's fundamental performance bottlenecks through a novel
multi-level adaptive specialization approach that combines static analysis,
runtime profiling, and native code generation.

Core Components:
    - analysis: CPython bottleneck identification and profiling
    - compiler: AST/bytecode optimization and native code generation
    - optimization: Type specialization, loop optimization, memory management
    - runtime: Adaptive execution engine with inline caching

Usage:
    >>> import highpy
    >>> @highpy.optimize
    ... def compute(x, y):
    ...     return x * x + y * y
    >>> result = compute(3.14, 2.71)

    >>> @highpy.jit(specialize=True, native=True)
    ... def matrix_multiply(a, b):
    ...     ...
"""

__version__ = "1.0.0"
__author__ = "HighPy Research Team"

from highpy.optimization.type_specializer import TypeSpecializer, specialize
from highpy.optimization.loop_optimizer import LoopOptimizer, optimize_loops
from highpy.optimization.function_specializer import FunctionSpecializer
from highpy.optimization.memory_pool import ArenaAllocator, MemoryPool
from highpy.optimization.lazy_evaluator import LazyChain, lazy
from highpy.optimization.parallel_executor import ParallelExecutor, auto_parallel
from highpy.compiler.ast_optimizer import ASTOptimizer
from highpy.compiler.native_codegen import NativeCompiler
from highpy.compiler.bytecode_rewriter import BytecodeRewriter
from highpy.runtime.adaptive_runtime import AdaptiveRuntime, optimize, jit
from highpy.runtime.inline_cache import PolymorphicInlineCache
from highpy.analysis.cpython_bottlenecks import CPythonAnalyzer
from highpy.analysis.type_profiler import TypeProfiler

# Recursive Fractal Optimization Engine (RFOE)
from highpy.recursive import (
    RecursiveFractalOptimizer,
    FractalLevel,
    OptimizationMorphism,
    rfo,
    rfo_optimize,
    FixedPointEngine,
    ProgramMetric,
    ConvergenceResult,
    MetaCircularOptimizer,
    SelfOptimizingPass,
    FractalAnalyzer,
    FractalDimension,
    OptimizationEnergyField,
    ConvergenceProver,
    ContractionCertificate,
    BanachProof,
    PurityAnalyzer,
    PurityLevel,
    PurityReport,
)
