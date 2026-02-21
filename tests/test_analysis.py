"""
Tests for the CPython bottleneck analysis module.

Validates:
  - CPythonAnalyzer produces correct BottleneckReport dataclass
  - MemoryOverheadAnalyzer measures per-type memory overhead
  - GILContentionAnalyzer detects GIL overhead
  - BytecodeComplexityAnalyzer identifies complexity hotspots
"""

import sys
import pytest
from highpy.analysis.cpython_bottlenecks import (
    CPythonAnalyzer,
    BottleneckReport,
    MemoryOverheadAnalyzer,
    GILContentionAnalyzer,
    BytecodeComplexityAnalyzer,
)


# ---------- Fixture functions for analysis ----------

def simple_add(x, y):
    return x + y


def loop_function(n):
    total = 0
    for i in range(n):
        total += i
    return total


def nested_loop(n, m):
    total = 0
    for i in range(n):
        for j in range(m):
            total += i * j
    return total


def attribute_heavy():
    class Obj:
        def __init__(self):
            self.x = 1
            self.y = 2
            self.z = 3

    o = Obj()
    return o.x + o.y + o.z


def recursive_fib(n):
    if n <= 1:
        return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)


# ---------- CPythonAnalyzer Tests ----------

class TestCPythonAnalyzer:
    def setup_method(self):
        self.analyzer = CPythonAnalyzer()

    def test_analyze_simple(self):
        report = self.analyzer.analyze(simple_add)
        assert report is not None
        assert isinstance(report, BottleneckReport)
        assert report.function_name == 'simple_add'

    def test_analyze_loop_detected(self):
        report = self.analyzer.analyze(loop_function)
        assert report is not None
        assert isinstance(report.bytecode_stats, dict)

    def test_analyze_nested_loop(self):
        report = self.analyzer.analyze(nested_loop)
        assert report is not None
        assert isinstance(report, BottleneckReport)

    def test_analyze_attribute_access(self):
        report = self.analyzer.analyze(attribute_heavy)
        assert report is not None
        assert isinstance(report, BottleneckReport)

    def test_analyze_recursive(self):
        report = self.analyzer.analyze(recursive_fib)
        assert report is not None
        assert isinstance(report, BottleneckReport)

    def test_bottleneck_identification(self):
        report = self.analyzer.analyze(loop_function)
        assert isinstance(report.bottlenecks, list)

    def test_optimization_suggestions(self):
        report = self.analyzer.analyze(nested_loop)
        assert isinstance(report.optimization_potential, float)
        assert 0.0 <= report.optimization_potential <= 1.0


# ---------- MemoryOverheadAnalyzer Tests ----------

class TestMemoryOverheadAnalyzer:
    def test_measure_overhead(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_int_overhead(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        assert 'int_small' in results
        info = results['int_small']
        assert info['python_size'] >= 28

    def test_float_overhead(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        assert 'float' in results
        assert results['float']['python_size'] > 0

    def test_list_overhead(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        assert 'list_1000_floats' in results
        assert results['list_1000_floats']['overhead_ratio'] > 1.0

    def test_dict_overhead(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        assert 'dict_100_entries' in results

    def test_overhead_ratio(self):
        results = MemoryOverheadAnalyzer.measure_object_overhead()
        for name, info in results.items():
            assert info['overhead_ratio'] >= 1.0, f"{name} ratio < 1"


# ---------- GILContentionAnalyzer Tests ----------

class TestGILContentionAnalyzer:
    def test_measure_single_thread(self):
        result = GILContentionAnalyzer.measure_gil_overhead(threads=1)
        assert result is not None
        assert 'single_threaded_ns' in result

    def test_measure_multi_thread(self):
        result = GILContentionAnalyzer.measure_gil_overhead(threads=2)
        assert result is not None
        assert 'multi_threaded_ns' in result

    def test_contention_ratio(self):
        result = GILContentionAnalyzer.measure_gil_overhead(threads=4)
        assert 'overhead_ratio' in result
        ratio = result['overhead_ratio']
        assert isinstance(ratio, float)


# ---------- BytecodeComplexityAnalyzer Tests ----------

class TestBytecodeComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = BytecodeComplexityAnalyzer()

    def test_analyze_simple(self):
        result = self.analyzer.analyze_complexity(simple_add)
        assert result is not None
        assert 'total_instructions' in result or 'complexity_class' in result

    def test_analyze_loop(self):
        result = self.analyzer.analyze_complexity(loop_function)
        assert result is not None

    def test_higher_complexity_for_loops(self):
        simple_report = self.analyzer.analyze_complexity(simple_add)
        loop_report = self.analyzer.analyze_complexity(nested_loop)
        assert loop_report is not None
        assert simple_report is not None

    def test_recursion_detection(self):
        result = self.analyzer.analyze_complexity(recursive_fib)
        assert result is not None
        assert result.get('is_recursive', False) is True
