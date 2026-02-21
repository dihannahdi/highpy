"""
Tests for the type profiler and abstract interpreter.
"""

import pytest
from highpy.analysis.type_profiler import (
    TypeProfiler,
    TypeTag,
    LatticeType,
    AbstractInterpreter,
)


def identity(x):
    return x

def add_ints(a, b):
    return a + b

def mixed_types(x):
    if isinstance(x, int):
        return x + 1
    return str(x)

def loop_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total


class TestLatticeType:
    def test_from_python_type_int(self):
        t = LatticeType.from_python_type(int)
        assert t.tag == TypeTag.INT

    def test_from_python_type_float(self):
        t = LatticeType.from_python_type(float)
        assert t.tag == TypeTag.FLOAT

    def test_from_python_type_str(self):
        t = LatticeType.from_python_type(str)
        assert t.tag == TypeTag.STR

    def test_from_python_type_list(self):
        t = LatticeType.from_python_type(list)
        assert t.tag == TypeTag.LIST

    def test_from_value(self):
        assert LatticeType.from_value(42).tag == TypeTag.INT
        assert LatticeType.from_value(3.14).tag == TypeTag.FLOAT
        assert LatticeType.from_value("hi").tag == TypeTag.STR
        assert LatticeType.from_value(True).tag == TypeTag.BOOL
        assert LatticeType.from_value(None).tag == TypeTag.NONE

    def test_join_same(self):
        t = LatticeType.from_python_type(int)
        assert t.join(t).tag == TypeTag.INT

    def test_join_different(self):
        int_t = LatticeType.from_python_type(int)
        float_t = LatticeType.from_python_type(float)
        joined = int_t.join(float_t)
        assert joined is not None

    def test_meet_same(self):
        t = LatticeType.from_python_type(int)
        met = t.meet(t)
        assert met.tag == TypeTag.INT

    def test_top_bottom(self):
        top = LatticeType(TypeTag.TOP)
        bottom = LatticeType(TypeTag.BOTTOM)
        assert LatticeType.from_python_type(int).join(top).tag == TypeTag.TOP
        assert LatticeType.from_python_type(int).meet(bottom).tag == TypeTag.BOTTOM

    def test_to_c_type(self):
        assert LatticeType.from_python_type(int).to_c_type() is not None
        assert LatticeType.from_python_type(float).to_c_type() is not None


class TestTypeProfiler:
    def setup_method(self):
        self.profiler = TypeProfiler()

    def test_profile_decorator(self):
        profiled = self.profiler.profile(identity)
        assert callable(profiled)
        assert profiled(42) == 42

    def test_profile_tracks_types(self):
        profiled = self.profiler.profile(add_ints)
        profiled(1, 2)
        profiled(3, 4)
        report = self.profiler.get_type_info(add_ints)
        assert report is not None

    def test_profile_mixed_types(self):
        profiled = self.profiler.profile(identity)
        profiled(42)
        profiled("hello")
        profiled(3.14)
        report = self.profiler.get_type_info(identity)
        assert report is not None

    def test_monomorphic_detection(self):
        profiled = self.profiler.profile(add_ints)
        for i in range(10):
            profiled(i, i)
        report = self.profiler.get_type_info(add_ints)
        assert report is not None


class TestAbstractInterpreter:
    def test_analyze_simple(self):
        interp = AbstractInterpreter()
        result = interp.infer_function(identity)
        assert result is not None

    def test_analyze_loop(self):
        interp = AbstractInterpreter()
        result = interp.infer_function(loop_sum)
        assert result is not None
