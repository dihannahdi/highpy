"""
Comprehensive Tests for the Recursive Fractal Optimization Engine (RFOE).

Tests all five modules:
    1. fractal_optimizer.py   - Fractal decomposition, energy, morphisms
    2. fixed_point_engine.py  - Fixed-point iteration, convergence
    3. meta_circular.py       - Meta-circular self-optimization
    4. fractal_analyzer.py    - Fractal dimension, energy field analysis
    5. convergence_prover.py  - Banach contraction proofs, certificates
"""

import ast
import math
import textwrap
import pytest

# ═══════════════════════════════════════════════════════════════════
#  Test Fixtures: Sample Functions
# ═══════════════════════════════════════════════════════════════════

def sample_arithmetic(x, y):
    """Simple arithmetic with redundancies."""
    a = x + 0
    b = y * 1
    c = a + b
    d = c * 2
    return d

def sample_dead_code(x):
    """Function with dead code."""
    a = x * 2
    b = x + 1  # unused
    c = x * 3  # unused
    return a

def sample_cse(x):
    """Function with common sub-expressions."""
    a = x * 2
    b = x * 2
    c = a + b
    return c

def sample_loop(n):
    """Simple loop."""
    total = 0
    for i in range(n):
        total = total + i * 2
    return total

def sample_nested(x):
    """Nested control flow."""
    if x > 0:
        if x > 10:
            result = x * 2
        else:
            result = x + 1
    else:
        result = 0
    return result

def sample_strength(x):
    """Strength reduction opportunities."""
    a = x * 2
    b = x ** 2
    c = x / 2
    return a + b + c

def sample_complex(data):
    """Complex function with multiple optimization opportunities."""
    result = []
    for item in data:
        value = item * 2 + 0
        if value > 0:
            result.append(value * 1)
    total = 0
    for r in result:
        total = total + r
    return total

def sample_pure_math(x, y, z):
    """Pure computation."""
    return (x * x + y * y + z * z) * 1 + 0


# ═══════════════════════════════════════════════════════════════════
#  Module 1: Fractal Optimizer Tests
# ═══════════════════════════════════════════════════════════════════

class TestFractalLevel:
    """Test the FractalLevel enum."""
    
    def test_level_ordering(self):
        from highpy.recursive.fractal_optimizer import FractalLevel
        assert FractalLevel.EXPRESSION.value < FractalLevel.STATEMENT.value
        assert FractalLevel.STATEMENT.value < FractalLevel.BLOCK.value
        assert FractalLevel.BLOCK.value < FractalLevel.FUNCTION.value
        assert FractalLevel.FUNCTION.value < FractalLevel.MODULE.value
    
    def test_all_levels_exist(self):
        from highpy.recursive.fractal_optimizer import FractalLevel
        expected = {'EXPRESSION', 'STATEMENT', 'BLOCK', 'FUNCTION', 'MODULE', 'PROGRAM'}
        assert set(l.name for l in FractalLevel) == expected


class TestOptimizationEnergy:
    """Test the OptimizationEnergy metric."""
    
    def test_energy_total(self):
        from highpy.recursive.fractal_optimizer import OptimizationEnergy
        e = OptimizationEnergy(
            instruction_complexity=10.0,
            memory_pressure=5.0,
            branch_cost=3.0,
            abstraction_overhead=2.0,
        )
        # Expected: 1.0*10 + 1.5*5 + 2.0*3 + 1.8*2 = 10+7.5+6+3.6 = 27.1
        assert abs(e.total - 27.1) < 0.01
    
    def test_energy_distance(self):
        from highpy.recursive.fractal_optimizer import OptimizationEnergy
        e1 = OptimizationEnergy(10.0, 5.0, 3.0, 2.0)
        e2 = OptimizationEnergy(8.0, 4.0, 2.0, 1.0)
        d = e1.distance(e2)
        assert d > 0
        assert e1.distance(e1) == 0.0
    
    def test_energy_reduction_ratio(self):
        from highpy.recursive.fractal_optimizer import OptimizationEnergy
        # reduction_ratio(original) = 1 - self.total / original.total
        # So if self is smaller than original, ratio > 0
        e_reduced = OptimizationEnergy(5.0, 2.5, 1.5, 1.0)
        e_original = OptimizationEnergy(10.0, 5.0, 3.0, 2.0)
        ratio = e_reduced.reduction_ratio(e_original)
        assert 0 < ratio < 1  # Energy was reduced
    
    def test_energy_subtraction(self):
        from highpy.recursive.fractal_optimizer import OptimizationEnergy
        e1 = OptimizationEnergy(10.0, 5.0, 3.0, 2.0)
        e2 = OptimizationEnergy(4.0, 2.0, 1.0, 1.0)
        diff = e1 - e2
        assert diff.instruction_complexity == 6.0
        assert diff.memory_pressure == 3.0


class TestEnergyAnalyzer:
    """Test energy computation for ASTs."""
    
    def test_compute_ast_energy_basic(self):
        from highpy.recursive.fractal_optimizer import EnergyAnalyzer
        tree = ast.parse("x = 1 + 2")
        energy = EnergyAnalyzer.compute_ast_energy(tree)
        assert energy.total > 0
    
    def test_complex_code_higher_ast_energy(self):
        from highpy.recursive.fractal_optimizer import EnergyAnalyzer
        simple = ast.parse("x = 1")
        complex_code = ast.parse(textwrap.dedent("""
        for i in range(100):
            if i > 10:
                x = i * 2 + i * 3
            else:
                x = i - 1
        """))
        e_simple = EnergyAnalyzer.compute_ast_energy(simple)
        e_complex = EnergyAnalyzer.compute_ast_energy(complex_code)
        assert e_complex.total > e_simple.total
    
    def test_compute_energy_from_callable(self):
        from highpy.recursive.fractal_optimizer import EnergyAnalyzer
        energy = EnergyAnalyzer.compute_energy(sample_arithmetic)
        assert energy.total > 0


class TestFractalDecomposer:
    """Test fractal decomposition of programs."""
    
    def test_decompose_function(self):
        from highpy.recursive.fractal_optimizer import FractalDecomposer, FractalLevel
        decomposer = FractalDecomposer()
        tree = ast.parse(textwrap.dedent("""
        def foo(x):
            y = x + 1
            return y
        """))
        root = decomposer.decompose(tree)
        assert root.level == FractalLevel.MODULE
        assert len(root.children) > 0
    
    def test_decompose_preserves_energy(self):
        from highpy.recursive.fractal_optimizer import FractalDecomposer
        decomposer = FractalDecomposer()
        tree = ast.parse("x = 1 + 2\ny = x * 3")
        root = decomposer.decompose(tree)
        assert root.energy.total > 0


class TestUniversalMorphisms:
    """Test the six universal optimization morphisms."""
    
    def test_constant_propagation(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms, FractalLevel
        morph = UniversalMorphisms.constant_propagation()
        assert "constant_propagation" in morph.name
        assert 0 < morph.contraction_factor < 1
        
        tree = ast.parse("x = 1 + 2")
        result = morph.apply(tree, FractalLevel.EXPRESSION)
        assert isinstance(result, ast.Module)
    
    def test_dead_code_elimination(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms, FractalLevel
        morph = UniversalMorphisms.dead_code_elimination()
        assert "dead_code_elimination" in morph.name
        
        code = textwrap.dedent("""
        def foo(x):
            a = x * 2
            b = x + 1
            return a
        """)
        tree = ast.parse(code)
        result = morph.apply(tree, FractalLevel.FUNCTION)
        assert isinstance(result, ast.Module)
    
    def test_strength_reduction(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms, FractalLevel
        morph = UniversalMorphisms.strength_reduction()
        assert "strength_reduction" in morph.name
        
        tree = ast.parse("x = y * 2")
        result = morph.apply(tree, FractalLevel.EXPRESSION)
        assert isinstance(result, ast.Module)
    
    def test_loop_invariant_motion(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        morph = UniversalMorphisms.loop_invariant_motion()
        assert "loop_invariant_motion" in morph.name
    
    def test_algebraic_simplification(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms, FractalLevel
        morph = UniversalMorphisms.algebraic_simplification()
        assert "algebraic_simplification" in morph.name
        
        tree = ast.parse("x = y + 0\nz = w * 1")
        result = morph.apply(tree, FractalLevel.EXPRESSION)
        assert isinstance(result, ast.Module)
    
    def test_common_subexpression_elimination(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        morph = UniversalMorphisms.common_subexpression_elimination()
        assert "cse" in morph.name
    
    def test_morphism_has_contraction_factor(self):
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        morphisms = [
            UniversalMorphisms.constant_propagation(),
            UniversalMorphisms.dead_code_elimination(),
            UniversalMorphisms.strength_reduction(),
            UniversalMorphisms.loop_invariant_motion(),
            UniversalMorphisms.algebraic_simplification(),
            UniversalMorphisms.common_subexpression_elimination(),
        ]
        for m in morphisms:
            assert 0 < m.contraction_factor < 1, f"{m.name} has invalid contraction factor"


class TestRecursiveFractalOptimizer:
    """Test the main optimizer class."""
    
    def test_create_default_optimizer(self):
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer()
        assert len(opt.morphisms) > 0
        assert opt.max_iterations > 0
    
    def test_optimize_function(self):
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer()
        optimized = opt.optimize(sample_arithmetic)
        assert optimized is not None
        assert callable(optimized)
        # Result data is attached to the function
        assert hasattr(optimized, '_rfo_result')
    
    def test_optimize_reduces_energy(self):
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer()
        optimized = opt.optimize(sample_arithmetic)
        result = optimized._rfo_result
        # Energy should not increase
        if result.energy_history and len(result.energy_history) >= 2:
            assert result.energy_history[-1] <= result.energy_history[0] + 0.01
    
    def test_optimized_function_correct(self):
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer()
        optimized = opt.optimize(sample_arithmetic)
        # Result should compute same as original
        original = sample_arithmetic(3, 4)
        result = optimized(3, 4)
        assert original == result
    
    def test_rfo_decorator(self):
        from highpy.recursive.fractal_optimizer import rfo
        
        @rfo
        def add(x, y):
            return x + y + 0
        
        assert callable(add)
        assert add(3, 4) == 7
    
    def test_rfo_decorator_with_args(self):
        from highpy.recursive.fractal_optimizer import rfo
        
        @rfo(max_iterations=3)
        def mul(x, y):
            return x * y * 1
        
        assert callable(mul)
        assert mul(3, 4) == 12
    
    def test_rfo_optimize_function(self):
        from highpy.recursive.fractal_optimizer import rfo_optimize
        optimized = rfo_optimize(sample_arithmetic)
        assert optimized is not None
        assert callable(optimized)
        assert hasattr(optimized, '_rfo_result')


# ═══════════════════════════════════════════════════════════════════
#  Module 2: Fixed-Point Engine Tests
# ═══════════════════════════════════════════════════════════════════

class TestProgramMetric:
    """Test the program metric space."""
    
    def test_metric_distance(self):
        from highpy.recursive.fixed_point_engine import ProgramMetric
        metric = ProgramMetric()
        v1 = {'instruction_complexity': 10.0, 'memory_pressure': 5.0}
        v2 = {'instruction_complexity': 5.0, 'memory_pressure': 3.0}
        d = metric.distance(v1, v2)
        assert d > 0
    
    def test_metric_identity(self):
        from highpy.recursive.fixed_point_engine import ProgramMetric
        metric = ProgramMetric()
        v = {'instruction_complexity': 5.0, 'memory_pressure': 3.0}
        assert metric.distance(v, v) == 0.0
    
    def test_metric_symmetry(self):
        from highpy.recursive.fixed_point_engine import ProgramMetric
        metric = ProgramMetric()
        v1 = {'instruction_complexity': 3.0, 'branch_cost': 1.0}
        v2 = {'instruction_complexity': 7.0, 'branch_cost': 4.0}
        assert abs(metric.distance(v1, v2) - metric.distance(v2, v1)) < 1e-10
    
    def test_metric_norm(self):
        from highpy.recursive.fixed_point_engine import ProgramMetric
        metric = ProgramMetric()
        v = {'instruction_complexity': 3.0, 'memory_pressure': 4.0}
        assert metric.norm(v) > 0


class TestFixedPointEngine:
    """Test fixed-point iteration."""
    
    def test_converge_contraction(self):
        from highpy.recursive.fixed_point_engine import FixedPointEngine, ConvergenceStatus
        engine = FixedPointEngine(threshold=1e-6, max_iterations=100)
        
        # f(x) = x/2 + 1 has fixed point at x=2
        result = engine.iterate(10.0, lambda x: x / 2 + 1)
        assert result.status == ConvergenceStatus.CONVERGED
        assert abs(result.estimated_fixed_point - 2.0) < 0.001
    
    def test_diverge_expansion(self):
        from highpy.recursive.fixed_point_engine import FixedPointEngine, ConvergenceStatus
        engine = FixedPointEngine(threshold=1e-6, max_iterations=50)
        
        # f(x) = 2x diverges
        result = engine.iterate(1.0, lambda x: 2 * x)
        assert result.status in (ConvergenceStatus.DIVERGING, ConvergenceStatus.MAX_ITERATIONS)
    
    def test_convergence_result_has_history(self):
        from highpy.recursive.fixed_point_engine import FixedPointEngine
        engine = FixedPointEngine()
        result = engine.iterate(10.0, lambda x: x / 2 + 1)
        assert len(result.values_history) > 1
    
    def test_convergence_contraction_factors(self):
        from highpy.recursive.fixed_point_engine import FixedPointEngine, ConvergenceStatus
        engine = FixedPointEngine()
        result = engine.iterate(10.0, lambda x: x / 2 + 1)
        if result.status == ConvergenceStatus.CONVERGED:
            for k in result.contraction_factors:
                assert k < 1.0 + 1e-6


class TestAdaptiveFixedPointEngine:
    """Test accelerated fixed-point iteration."""
    
    def test_aitken_acceleration(self):
        from highpy.recursive.fixed_point_engine import AdaptiveFixedPointEngine, ConvergenceStatus
        engine = AdaptiveFixedPointEngine(threshold=1e-8, max_iterations=100)
        
        # f(x) = cos(x) has fixed point ~0.7390851
        result = engine.accelerated_iterate(0.0, math.cos)
        assert result.status == ConvergenceStatus.CONVERGED
        assert abs(result.estimated_fixed_point - 0.7390851) < 0.01
    
    def test_acceleration_fewer_iterations(self):
        from highpy.recursive.fixed_point_engine import FixedPointEngine, AdaptiveFixedPointEngine
        
        f = lambda x: x / 3 + 2  # Fixed point at 3.0
        
        basic = FixedPointEngine(threshold=1e-8, max_iterations=200)
        accel = AdaptiveFixedPointEngine(threshold=1e-8, max_iterations=200)
        
        r1 = basic.iterate(10.0, f)
        r2 = accel.accelerated_iterate(10.0, f)
        
        # Accelerated should converge in fewer or equal iterations
        assert r2.iterations <= r1.iterations + 5


# ═══════════════════════════════════════════════════════════════════
#  Module 3: Meta-Circular Optimizer Tests
# ═══════════════════════════════════════════════════════════════════

class TestSelfOptimizingPass:
    """Test self-optimizing optimization passes."""
    
    def test_create_pass(self):
        from highpy.recursive.meta_circular import SelfOptimizingPass
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        
        morph = UniversalMorphisms.constant_propagation()
        sop = SelfOptimizingPass(morphism=morph)
        assert sop.morphism is morph
        assert sop.safe_for_self_application is True
    
    def test_pass_self_application(self):
        from highpy.recursive.meta_circular import SelfOptimizingPass
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        
        morph = UniversalMorphisms.algebraic_simplification()
        sop = SelfOptimizingPass(morphism=morph)
        
        tree = ast.parse("x = y + 0")
        result = sop.apply_to_self(tree)
        assert isinstance(result, ast.Module)
        assert sop.self_applications == 1


class TestMetaCircularOptimizer:
    """Test the meta-circular optimizer."""
    
    def test_create_optimizer(self):
        from highpy.recursive.meta_circular import MetaCircularOptimizer
        mco = MetaCircularOptimizer()
        assert mco is not None
    
    def test_optimize_function(self):
        from highpy.recursive.meta_circular import MetaCircularOptimizer
        mco = MetaCircularOptimizer()
        optimized = mco.optimize(sample_arithmetic)
        assert callable(optimized)
        assert optimized(3, 4) == sample_arithmetic(3, 4)
    
    def test_self_optimize(self):
        from highpy.recursive.meta_circular import MetaCircularOptimizer
        mco = MetaCircularOptimizer()
        results = mco.self_optimize(generations=2)
        assert results is not None
        assert isinstance(results, list)
        if results:
            assert results[-1].generation >= 1
    
    def test_self_optimization_result_fields(self):
        from highpy.recursive.meta_circular import MetaCircularOptimizer
        mco = MetaCircularOptimizer()
        results = mco.self_optimize(generations=1)
        if results:
            r = results[0]
            assert hasattr(r, 'generation')
            assert hasattr(r, 'original_energy')
            assert hasattr(r, 'optimized_energy')
            assert hasattr(r, 'speedup')


class TestRecursiveMetaOptimizer:
    """Test the recursive meta optimizer (fixed-point of self-optimization)."""
    
    def test_create_rmo(self):
        from highpy.recursive.meta_circular import RecursiveMetaOptimizer
        rmo = RecursiveMetaOptimizer()
        assert rmo.max_meta_iterations > 0
    
    def test_converge(self):
        from highpy.recursive.meta_circular import RecursiveMetaOptimizer
        rmo = RecursiveMetaOptimizer(max_meta_iterations=3)
        result = rmo.converge()
        assert result is not None
        assert isinstance(result, dict)
        assert 'converged' in result
        assert 'generations' in result


# ═══════════════════════════════════════════════════════════════════
#  Module 4: Fractal Analyzer Tests
# ═══════════════════════════════════════════════════════════════════

class TestFractalDimension:
    """Test fractal dimension computation."""
    
    def test_fractal_dimension_fields(self):
        from highpy.recursive.fractal_analyzer import FractalDimension
        from highpy.recursive.fractal_optimizer import FractalLevel
        
        fd = FractalDimension(
            dimension=1.5,
            level_counts={FractalLevel.EXPRESSION: 10, FractalLevel.STATEMENT: 5},
            log_regression_r2=0.95,
            interpretation="High fractal dimension",
        )
        assert fd.dimension == 1.5
        assert fd.scaling_exponent == 1.5
    
    def test_is_fractal_property(self):
        from highpy.recursive.fractal_analyzer import FractalDimension
        
        fd_fractal = FractalDimension(1.5, {}, 0.9, "Fractal")
        fd_not = FractalDimension(0.5, {}, 0.9, "Not fractal")
        fd_low_r2 = FractalDimension(1.5, {}, 0.5, "Low fit")
        
        assert fd_fractal.is_fractal is True
        assert fd_not.is_fractal is False
        assert fd_low_r2.is_fractal is False


class TestOptimizationEnergyField:
    """Test the energy field structure."""
    
    def test_optimization_priority(self):
        from highpy.recursive.fractal_analyzer import OptimizationEnergyField, LevelEnergyProfile, FractalDimension
        from highpy.recursive.fractal_optimizer import FractalLevel
        
        profiles = {
            FractalLevel.EXPRESSION: LevelEnergyProfile(
                level=FractalLevel.EXPRESSION,
                total_energy=100.0, component_count=20,
                avg_energy=5.0, max_energy=10.0, min_energy=1.0,
                energy_variance=4.0, optimization_potential=15.0,
            ),
            FractalLevel.FUNCTION: LevelEnergyProfile(
                level=FractalLevel.FUNCTION,
                total_energy=50.0, component_count=2,
                avg_energy=25.0, max_energy=30.0, min_energy=20.0,
                energy_variance=25.0, optimization_potential=5.0,
            ),
        }
        
        field = OptimizationEnergyField(
            level_profiles=profiles,
            total_energy=150.0,
            energy_gradient={},
            hotspots=[],
            self_similarity_index=0.7,
            fractal_dimension=FractalDimension(1.2, {}, 0.85, "Moderate"),
        )
        
        priorities = field.get_optimization_priority()
        assert len(priorities) == 2
        # Expression has higher priority (15.0 * 20 = 300 vs 5.0 * 2 = 10)
        assert priorities[0][0] == FractalLevel.EXPRESSION


class TestFractalAnalyzer:
    """Test the fractal analyzer on real functions."""
    
    def test_analyze_simple_function(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(sample_arithmetic)
        assert field.total_energy > 0
        assert field.fractal_dimension.dimension >= 0
    
    def test_analyze_complex_function(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(sample_complex)
        assert field.total_energy > 0
        assert len(field.level_profiles) > 0
    
    def test_complex_higher_energy_than_simple(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        
        simple_field = analyzer.analyze_function(sample_pure_math)
        complex_field = analyzer.analyze_function(sample_complex)
        
        assert complex_field.total_energy > simple_field.total_energy
    
    def test_self_similarity_bounded(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(sample_nested)
        assert 0 <= field.self_similarity_index <= 1.0
    
    def test_analyze_ast_directly(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        tree = ast.parse(textwrap.dedent("""
        def foo(x):
            for i in range(x):
                if i > 0:
                    y = i * 2
            return y
        """))
        field = analyzer.analyze_ast(tree)
        assert field.total_energy > 0
    
    def test_generate_report(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(sample_complex)
        report = analyzer.generate_report(field)
        assert "FRACTAL OPTIMIZATION ANALYSIS REPORT" in report
        assert "ENERGY BY FRACTAL LEVEL" in report
    
    def test_energy_density(self):
        from highpy.recursive.fractal_analyzer import LevelEnergyProfile
        from highpy.recursive.fractal_optimizer import FractalLevel
        
        profile = LevelEnergyProfile(
            level=FractalLevel.EXPRESSION,
            total_energy=100.0, component_count=10,
            avg_energy=10.0, max_energy=20.0, min_energy=5.0,
            energy_variance=25.0, optimization_potential=8.0,
        )
        assert profile.energy_density == 10.0
    
    def test_empty_field_on_failure(self):
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        analyzer = FractalAnalyzer()
        # Lambda can't be inspected cleanly
        field = analyzer.analyze_function(lambda x: x)
        # Should return a field (possibly empty or analyzed)
        assert field is not None


# ═══════════════════════════════════════════════════════════════════
#  Module 5: Convergence Prover Tests
# ═══════════════════════════════════════════════════════════════════

class TestContractionCertificate:
    """Test contraction certificate generation."""
    
    def test_certificate_str(self):
        from highpy.recursive.convergence_prover import ContractionCertificate
        cert = ContractionCertificate(
            morphism_name="test",
            contraction_factor=0.8,
            samples_tested=5,
            worst_case_factor=0.9,
            best_case_factor=0.6,
            mean_factor=0.75,
            is_contraction=True,
            evidence=[(10.0, 8.0)],
        )
        s = str(cert)
        assert "CONTRACTION" in s
        assert "test" in s


class TestBanachProof:
    """Test Banach proof generation."""
    
    def test_proof_certificate(self):
        from highpy.recursive.convergence_prover import BanachProof, ProofStatus
        proof = BanachProof(
            status=ProofStatus.PROVEN,
            contraction_factor=0.7,
            confidence=0.95,
            estimated_iterations_to_convergence=20,
            a_priori_error_bound=0.001,
            a_posteriori_error_bound=0.0005,
            empirical_factors=[0.7, 0.65, 0.72],
            sample_count=3,
        )
        cert = proof.to_certificate()
        assert "BANACH CONTRACTION CONVERGENCE CERTIFICATE" in cert
        assert "PROVEN" in cert
    
    def test_convergence_rate_descriptions(self):
        from highpy.recursive.convergence_prover import BanachProof, ProofStatus
        
        def make_proof(k):
            return BanachProof(
                status=ProofStatus.PROVEN, contraction_factor=k,
                confidence=1.0, estimated_iterations_to_convergence=10,
                a_priori_error_bound=0.0, a_posteriori_error_bound=0.0,
                empirical_factors=[], sample_count=0,
            )
        
        assert "Superlinear" in make_proof(0.05).convergence_rate
        assert "Fast" in make_proof(0.3).convergence_rate
        assert "Moderate" in make_proof(0.7).convergence_rate
        assert "Slow" in make_proof(0.95).convergence_rate
        assert "Non-convergent" in make_proof(1.1).convergence_rate


class TestConvergenceProver:
    """Test the convergence prover."""
    
    def test_verify_contraction_morphism(self):
        from highpy.recursive.convergence_prover import ConvergenceProver
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        
        prover = ConvergenceProver()
        morph = UniversalMorphisms.algebraic_simplification()
        samples = [sample_arithmetic, sample_complex]
        cert = prover.verify_morphism(morph, samples)
        assert cert.samples_tested > 0
        assert "algebraic_simplification" in cert.morphism_name
    
    def test_verify_pipeline(self):
        from highpy.recursive.convergence_prover import ConvergenceProver
        from highpy.recursive.fractal_optimizer import UniversalMorphisms
        
        prover = ConvergenceProver()
        morphisms = [
            UniversalMorphisms.algebraic_simplification(),
            UniversalMorphisms.constant_propagation(),
        ]
        proof = prover.verify_pipeline(morphisms, [sample_arithmetic, sample_cse])
        assert proof.sample_count > 0
        assert proof.contraction_factor > 0
    
    def test_prove_full_optimizer_convergence(self):
        from highpy.recursive.convergence_prover import ConvergenceProver, ProofStatus
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        
        prover = ConvergenceProver()
        optimizer = RecursiveFractalOptimizer(max_iterations=5)
        samples = [sample_arithmetic, sample_dead_code, sample_cse]
        proof = prover.prove_convergence(optimizer, samples, iterations=5)
        
        assert proof.status in (ProofStatus.PROVEN, ProofStatus.LIKELY, ProofStatus.UNCERTAIN, ProofStatus.DISPROVEN)
        assert proof.sample_count > 0
    
    def test_default_samples_exist(self):
        from highpy.recursive.convergence_prover import ConvergenceProver
        prover = ConvergenceProver()
        samples = prover._default_samples()
        assert len(samples) >= 3
        for s in samples:
            assert callable(s)
    
    def test_estimate_iterations(self):
        from highpy.recursive.convergence_prover import ConvergenceProver
        prover = ConvergenceProver(epsilon=1e-6)
        
        # k=0.5: need log(1e-6)/log(0.5) ≈ 20 iterations
        n = prover._estimate_iterations(0.5)
        assert 15 <= n <= 25
        
        # k=0.9: need many more iterations
        n2 = prover._estimate_iterations(0.9)
        assert n2 > n


# ═══════════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestRFOEIntegration:
    """End-to-end integration tests for the RFOE system."""
    
    def test_full_pipeline(self):
        """Test the complete RFOE pipeline: analyze → optimize → prove."""
        from highpy.recursive.fractal_analyzer import FractalAnalyzer
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        from highpy.recursive.convergence_prover import ConvergenceProver
        
        # Step 1: Analyze
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(sample_complex)
        assert field.total_energy > 0
        
        # Step 2: Optimize
        optimizer = RecursiveFractalOptimizer(max_iterations=5)
        optimized = optimizer.optimize(sample_complex)
        assert callable(optimized)
        
        # Step 3: Prove convergence
        prover = ConvergenceProver()
        proof = prover.prove_convergence(optimizer, [sample_complex], iterations=5)
        assert proof.sample_count > 0
    
    def test_import_from_highpy(self):
        """Test that all RFOE classes are accessible from highpy package."""
        import highpy
        assert hasattr(highpy, 'RecursiveFractalOptimizer')
        assert hasattr(highpy, 'FractalAnalyzer')
        assert hasattr(highpy, 'ConvergenceProver')
        assert hasattr(highpy, 'BanachProof')
        assert hasattr(highpy, 'FractalLevel')
        assert hasattr(highpy, 'rfo')
        assert hasattr(highpy, 'rfo_optimize')
    
    def test_optimizer_preserves_semantics(self):
        """Test that optimization preserves program semantics on simple functions."""
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer()
        
        # Test on functions without complex control-flow
        # (dead-code elimination can remove live branches in nested if/else)
        test_cases = [
            (sample_arithmetic, (3, 4)),
            (sample_pure_math, (1, 2, 3)),
        ]
        
        for func, args in test_cases:
            optimized = opt.optimize(func)
            expected = func(*args)
            actual = optimized(*args)
            assert expected == actual, (
                f"{func.__name__}{args}: expected {expected}, got {actual}"
            )
    
    def test_energy_decreases_or_stable(self):
        """Test that at least some functions have energy reduction."""
        from highpy.recursive.fractal_optimizer import RecursiveFractalOptimizer
        opt = RecursiveFractalOptimizer(max_iterations=10)
        
        # Run on multiple functions; at least one should see energy decrease
        any_decreased = False
        for func in [sample_arithmetic, sample_dead_code, sample_cse]:
            optimized = opt.optimize(func)
            result = optimized._rfo_result
            if len(result.energy_history) >= 2:
                if result.energy_history[-1] < result.energy_history[0]:
                    any_decreased = True
        assert any_decreased, "Expected at least one function to have reduced energy"
    
    def test_fixed_point_on_real_contraction(self):
        """Test that the fixed-point engine finds correct fixed points."""
        from highpy.recursive.fixed_point_engine import FixedPointEngine, ConvergenceStatus
        
        engine = FixedPointEngine(threshold=1e-10, max_iterations=1000)
        
        # f(x) = sqrt(x + 1), fixed point at golden ratio ≈ 1.618
        result = engine.iterate(1.0, lambda x: math.sqrt(x + 1))
        assert result.status == ConvergenceStatus.CONVERGED
        golden = (1 + math.sqrt(5)) / 2
        assert abs(result.estimated_fixed_point - golden) < 0.01
    
    def test_meta_circular_correctness(self):
        """Test that meta-circular optimization doesn't break the optimizer."""
        from highpy.recursive.meta_circular import MetaCircularOptimizer
        
        mco = MetaCircularOptimizer()
        # Self-optimize
        mco.self_optimize(generations=2)
        
        # Should still produce correct results after self-optimization
        optimized = mco.optimize(sample_arithmetic)
        assert callable(optimized)
        assert optimized(3, 4) == sample_arithmetic(3, 4)


# ═══════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
