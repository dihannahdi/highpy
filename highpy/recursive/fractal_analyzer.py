"""
Fractal Analyzer
=================

Measures the fractal properties of optimization opportunities across
program granularity levels.

Theoretical Foundation:
    Define the "fractal dimension" of optimization opportunity as:

        D_opt = lim_{ε→0} log(N(ε)) / log(1/ε)

    where N(ε) is the number of optimization opportunities at
    granularity scale ε. This measures how optimization opportunities
    "scale" across levels of program decomposition.

    A fractal dimension D_opt > 1 indicates that optimization
    opportunities grow super-linearly as we examine finer granularity —
    meaning deeper fractal decomposition yields disproportionately
    more optimization opportunities.

Novel Concepts:
    1. **Optimization Energy Field**: A function E: Program × Level → ℝ
       that maps each (sub-program, level) pair to its energy. This
       creates a "field" over the program structure that we can analyze
       for gradients, critical points, and conservation laws.

    2. **Self-Similarity Index**: Measures how similar the optimization
       patterns are across different levels. High self-similarity means
       fractal optimization is especially effective.

    3. **Optimization Potential**: The maximum possible energy reduction
       at each level, analogous to potential energy in physics.
"""

import ast
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from highpy.recursive.fractal_optimizer import (
    FractalLevel,
    FractalComponent,
    FractalDecomposer,
    EnergyAnalyzer,
    OptimizationEnergy,
)


@dataclass
class FractalDimension:
    """
    The fractal dimension of optimization opportunities.
    
    Attributes:
        dimension: The computed fractal dimension D_opt
        level_counts: Number of optimization points at each level
        log_regression_r2: R² of the log-log regression (fit quality)
        interpretation: Human-readable interpretation
    """
    dimension: float
    level_counts: Dict[FractalLevel, int]
    log_regression_r2: float
    interpretation: str
    
    @property
    def is_fractal(self) -> bool:
        """True if optimization opportunities exhibit fractal scaling."""
        return self.dimension > 1.0 and self.log_regression_r2 > 0.8
    
    @property
    def scaling_exponent(self) -> float:
        """How quickly optimization opportunities grow with depth."""
        return self.dimension


@dataclass
class LevelEnergyProfile:
    """Energy profile for a single fractal level."""
    level: FractalLevel
    total_energy: float
    component_count: int
    avg_energy: float
    max_energy: float
    min_energy: float
    energy_variance: float
    optimization_potential: float  # Max possible reduction
    
    @property
    def energy_density(self) -> float:
        """Energy per component at this level."""
        if self.component_count == 0:
            return 0.0
        return self.total_energy / self.component_count


@dataclass
class OptimizationEnergyField:
    """
    The optimization energy field E: Program × Level → ℝ.
    
    This field captures the full energy landscape of a program across
    all fractal levels, enabling gradient-based optimization strategy.
    """
    level_profiles: Dict[FractalLevel, LevelEnergyProfile]
    total_energy: float
    energy_gradient: Dict[Tuple[FractalLevel, FractalLevel], float]
    hotspots: List[Tuple[FractalLevel, float]]  # Highest-energy points
    self_similarity_index: float
    fractal_dimension: FractalDimension
    
    @property
    def gradient_direction(self) -> FractalLevel:
        """
        The fractal level with the steepest energy gradient.
        
        This tells us where to focus optimization effort for
        maximum energy reduction per optimization applied.
        """
        if not self.level_profiles:
            return FractalLevel.FUNCTION
        return max(
            self.level_profiles.values(),
            key=lambda p: p.optimization_potential
        ).level
    
    def get_optimization_priority(self) -> List[Tuple[FractalLevel, float]]:
        """
        Rank fractal levels by optimization priority.
        
        Priority = potential_energy_reduction × component_count
        (maximize total impact).
        """
        priorities = []
        for level, profile in self.level_profiles.items():
            priority = profile.optimization_potential * profile.component_count
            priorities.append((level, priority))
        return sorted(priorities, key=lambda x: x[1], reverse=True)


@dataclass
class SelfSimilarityMetrics:
    """Metrics measuring self-similarity across fractal levels."""
    energy_pattern_similarity: float   # [0,1] how similar energy patterns are
    optimization_yield_similarity: float  # [0,1] how similar optimization yields are
    structural_similarity: float       # [0,1] how similar AST patterns are
    overall_index: float              # Weighted combination


class FractalAnalyzer:
    """
    Analyzes the fractal properties of optimization opportunities.
    
    This analyzer examines a program's AST at multiple granularity levels
    and computes:
    1. The fractal dimension of optimization opportunities
    2. The optimization energy field
    3. Self-similarity metrics across levels
    
    Usage:
        analyzer = FractalAnalyzer()
        field = analyzer.analyze_function(my_function)
        print(f"Fractal dimension: {field.fractal_dimension.dimension}")
        print(f"Self-similarity: {field.self_similarity_index}")
        print(f"Best optimization target: {field.gradient_direction}")
    """
    
    def __init__(self):
        self.decomposer = FractalDecomposer()
        self.energy_analyzer = EnergyAnalyzer()
    
    def analyze_function(self, func: Callable) -> OptimizationEnergyField:
        """Analyze the fractal optimization properties of a function."""
        import textwrap
        import inspect
        
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
        except (OSError, TypeError):
            return self._empty_field()
        
        return self.analyze_ast(tree)
    
    def analyze_ast(self, tree: ast.AST) -> OptimizationEnergyField:
        """Analyze the fractal optimization properties of an AST."""
        # Decompose into fractal components
        root = self.decomposer.decompose(tree)
        
        # Collect energy at each level
        level_energies = defaultdict(list)
        self._collect_level_energies(root, level_energies)
        
        # Compute level profiles
        level_profiles = {}
        for level, energies in level_energies.items():
            if energies:
                total = sum(energies)
                avg = total / len(energies)
                variance = sum((e - avg) ** 2 for e in energies) / max(len(energies), 1)
                
                # Estimate optimization potential (heuristic: 30% of variance)
                potential = math.sqrt(variance) * 0.3 + avg * 0.1
                
                level_profiles[level] = LevelEnergyProfile(
                    level=level,
                    total_energy=total,
                    component_count=len(energies),
                    avg_energy=avg,
                    max_energy=max(energies),
                    min_energy=min(energies),
                    energy_variance=variance,
                    optimization_potential=potential,
                )
        
        # Compute total energy
        total_energy = sum(p.total_energy for p in level_profiles.values())
        
        # Compute energy gradient between adjacent levels
        gradient = {}
        sorted_levels = sorted(level_profiles.keys(), key=lambda l: l.value)
        for i in range(len(sorted_levels) - 1):
            l1, l2 = sorted_levels[i], sorted_levels[i + 1]
            e1 = level_profiles[l1].avg_energy
            e2 = level_profiles[l2].avg_energy
            gradient[(l1, l2)] = e2 - e1
        
        # Find hotspots (highest energy components)
        hotspots = sorted(
            [(level, profile.max_energy) for level, profile in level_profiles.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        # Compute fractal dimension
        fractal_dim = self._compute_fractal_dimension(level_energies)
        
        # Compute self-similarity
        similarity = self._compute_self_similarity(level_energies, level_profiles)
        
        return OptimizationEnergyField(
            level_profiles=level_profiles,
            total_energy=total_energy,
            energy_gradient=gradient,
            hotspots=hotspots,
            self_similarity_index=similarity.overall_index,
            fractal_dimension=fractal_dim,
        )
    
    def compute_ast_energy(self, tree: ast.AST) -> OptimizationEnergy:
        """Convenience wrapper for EnergyAnalyzer.compute_ast_energy."""
        return EnergyAnalyzer.compute_ast_energy(tree)
    
    def _collect_level_energies(
        self,
        component: FractalComponent,
        level_energies: Dict[FractalLevel, List[float]],
    ):
        """Recursively collect energy values at each fractal level."""
        energy = component.energy.total
        level_energies[component.level].append(energy)
        
        for child in component.children:
            self._collect_level_energies(child, level_energies)
    
    def _compute_fractal_dimension(
        self,
        level_energies: Dict[FractalLevel, List[float]],
    ) -> FractalDimension:
        """
        Compute the fractal dimension of optimization opportunities.
        
        Uses log-log regression on the number of optimization points
        at each granularity scale.
        """
        level_counts = {}
        for level in sorted(level_energies.keys(), key=lambda l: l.value):
            level_counts[level] = len(level_energies[level])
        
        if len(level_counts) < 2:
            return FractalDimension(
                dimension=1.0,
                level_counts=level_counts,
                log_regression_r2=0.0,
                interpretation="Insufficient levels for fractal analysis",
            )
        
        # Log-log regression: log(count) vs log(level_value + 1)
        xs = []
        ys = []
        for level, count in sorted(level_counts.items(), key=lambda x: x[0].value):
            if count > 0:
                # Use inverse of level value as "scale" (finer = smaller number)
                scale = 1.0 / (level.value + 1)
                xs.append(math.log(1.0 / scale))
                ys.append(math.log(count))
        
        if len(xs) < 2:
            return FractalDimension(
                dimension=1.0,
                level_counts=level_counts,
                log_regression_r2=0.0,
                interpretation="Insufficient data points",
            )
        
        # Linear regression in log-log space
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x ** 2 for x in xs)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            slope = 0.0
            r2 = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # R² computation
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in ys)
            intercept = (sum_y - slope * sum_x) / n
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        dimension = max(slope, 0.0)
        
        # Interpretation
        if dimension > 2.0:
            interp = "Super-fractal: optimization opportunities grow explosively with depth"
        elif dimension > 1.5:
            interp = "High fractal dimension: deep decomposition reveals many opportunities"
        elif dimension > 1.0:
            interp = "Moderate fractal scaling: optimization benefits from multi-level approach"
        elif dimension > 0.5:
            interp = "Sub-fractal: optimization opportunities grow slowly with depth"
        else:
            interp = "Non-fractal: optimization opportunities don't scale with depth"
        
        return FractalDimension(
            dimension=dimension,
            level_counts=level_counts,
            log_regression_r2=max(r2, 0.0),
            interpretation=interp,
        )
    
    def _compute_self_similarity(
        self,
        level_energies: Dict[FractalLevel, List[float]],
        level_profiles: Dict[FractalLevel, LevelEnergyProfile],
    ) -> SelfSimilarityMetrics:
        """
        Compute self-similarity metrics across fractal levels.
        
        High self-similarity means the same optimization patterns
        appear at every level — the core property of fractals.
        """
        # Energy pattern similarity: how similar are the energy distributions?
        distributions = []
        for level in sorted(level_energies.keys(), key=lambda l: l.value):
            energies = level_energies[level]
            if energies:
                total = sum(energies)
                if total > 0:
                    normalized = sorted([e / total for e in energies], reverse=True)
                    distributions.append(normalized)
        
        energy_sim = self._distribution_similarity(distributions)
        
        # Optimization yield similarity
        yields = []
        for profile in level_profiles.values():
            if profile.total_energy > 0:
                yields.append(profile.optimization_potential / profile.total_energy)
        
        yield_sim = 1.0 - self._coefficient_of_variation(yields) if yields else 0.5
        yield_sim = max(0.0, min(1.0, yield_sim))
        
        # Structural similarity (based on component counts)
        counts = [len(level_energies[l]) for l in sorted(level_energies.keys(), key=lambda l: l.value)]
        struct_sim = 1.0 - self._coefficient_of_variation(counts) if len(counts) > 1 else 0.5
        struct_sim = max(0.0, min(1.0, struct_sim))
        
        overall = 0.4 * energy_sim + 0.35 * yield_sim + 0.25 * struct_sim
        
        return SelfSimilarityMetrics(
            energy_pattern_similarity=energy_sim,
            optimization_yield_similarity=yield_sim,
            structural_similarity=struct_sim,
            overall_index=overall,
        )
    
    @staticmethod
    def _distribution_similarity(distributions: List[List[float]]) -> float:
        """
        Compute pairwise similarity between distributions.
        
        Uses cosine similarity between normalized energy distributions.
        """
        if len(distributions) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                d1, d2 = distributions[i], distributions[j]
                # Pad to same length
                max_len = max(len(d1), len(d2))
                v1 = d1 + [0.0] * (max_len - len(d1))
                v2 = d2 + [0.0] * (max_len - len(d2))
                
                # Cosine similarity
                dot = sum(a * b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a ** 2 for a in v1))
                norm2 = math.sqrt(sum(b ** 2 for b in v2))
                
                if norm1 > 0 and norm2 > 0:
                    similarities.append(dot / (norm1 * norm2))
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    @staticmethod
    def _coefficient_of_variation(values: List[float]) -> float:
        """Compute coefficient of variation (std/mean)."""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / abs(mean)
    
    def _empty_field(self) -> OptimizationEnergyField:
        """Return an empty energy field."""
        return OptimizationEnergyField(
            level_profiles={},
            total_energy=0.0,
            energy_gradient={},
            hotspots=[],
            self_similarity_index=0.0,
            fractal_dimension=FractalDimension(
                dimension=0.0,
                level_counts={},
                log_regression_r2=0.0,
                interpretation="No data",
            ),
        )
    
    def generate_report(self, field: OptimizationEnergyField) -> str:
        """Generate a human-readable analysis report."""
        lines = [
            "═" * 60,
            "  FRACTAL OPTIMIZATION ANALYSIS REPORT",
            "═" * 60,
            "",
            f"  Total Program Energy: {field.total_energy:.2f}",
            f"  Fractal Dimension:    {field.fractal_dimension.dimension:.4f}",
            f"  Self-Similarity:      {field.self_similarity_index:.4f}",
            f"  Is Fractal:           {field.fractal_dimension.is_fractal}",
            "",
            f"  {field.fractal_dimension.interpretation}",
            "",
            "─" * 60,
            "  ENERGY BY FRACTAL LEVEL",
            "─" * 60,
        ]
        
        for level in sorted(field.level_profiles.keys(), key=lambda l: l.value):
            profile = field.level_profiles[level]
            lines.append(
                f"  {level.name:15s} │ E={profile.total_energy:8.2f} │ "
                f"N={profile.component_count:3d} │ "
                f"Potential={profile.optimization_potential:8.2f}"
            )
        
        lines.extend([
            "",
            "─" * 60,
            "  OPTIMIZATION PRIORITY (highest first)",
            "─" * 60,
        ])
        
        for level, priority in field.get_optimization_priority():
            lines.append(f"  {level.name:15s} │ Priority Score: {priority:.2f}")
        
        if field.hotspots:
            lines.extend([
                "",
                "─" * 60,
                "  ENERGY HOTSPOTS",
                "─" * 60,
            ])
            for level, energy in field.hotspots:
                lines.append(f"  {level.name:15s} │ Peak Energy: {energy:.2f}")
        
        lines.append("")
        lines.append("═" * 60)
        
        return "\n".join(lines)
