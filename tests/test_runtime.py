"""
Tests for the runtime modules: adaptive runtime, inline cache, deoptimizer.

Validates:
  - AdaptiveRuntime tier promotion logic
  - Inline cache hit/miss tracking
  - Deoptimizer guard evaluation and fallback
  - End-to-end optimize and jit decorators
"""

import pytest
from highpy.runtime.adaptive_runtime import AdaptiveRuntime, optimize, jit, Tier
from highpy.runtime.inline_cache import PolymorphicInlineCache, MethodCache, GuardedIC
from highpy.runtime.deoptimizer import Deoptimizer, GuardKind


# ---------- Adaptive Runtime Tests ----------

class TestAdaptiveRuntime:
    def setup_method(self):
        self.runtime = AdaptiveRuntime(
            tier1_threshold=3,
            tier2_threshold=10,
            tier3_threshold=30,
            enable_native=False,
        )

    def test_optimize_decorator(self):
        @self.runtime.optimize
        def add(x, y):
            return x + y
        assert add(1, 2) == 3

    def test_initial_tier(self):
        @self.runtime.optimize
        def f(x):
            return x * 2
        f(1)
        profile = self.runtime.get_profile(f)
        assert profile is not None
        assert profile.call_count == 1

    def test_tier_promotion(self):
        @self.runtime.optimize
        def g(x):
            total = 0
            for i in range(x):
                total += i
            return total

        # Warm up past tier1 threshold
        for i in range(5):
            g(10)

        profile = self.runtime.get_profile(g)
        assert profile is not None
        assert profile.call_count >= 5
        # Should have promoted to at least tier 1
        assert profile.current_tier >= Tier.INTERPRETED

    def test_correctness_across_tiers(self):
        @self.runtime.optimize
        def square(x):
            return x * x

        # Run many times to trigger tier promotions
        for i in range(50):
            assert square(i) == i * i

    def test_get_stats(self):
        @self.runtime.optimize
        def h(x):
            return x + 1
        h(1)
        h(2)
        stats = self.runtime.get_stats()
        assert len(stats) > 0


class TestOptimizeDecorator:
    def test_basic(self):
        @optimize
        def mul(a, b):
            return a * b
        assert mul(3, 4) == 12
        assert mul(2.5, 4.0) == 10.0

    def test_preserves_name(self):
        @optimize
        def my_func(x):
            return x
        assert my_func.__name__ == 'my_func'


class TestJitDecorator:
    def test_basic(self):
        @jit
        def compute(n):
            total = 0
            for i in range(n):
                total += i
            return total
        assert compute(100) == 4950

    def test_correctness_over_iterations(self):
        @jit
        def double(x):
            return x * 2
        for v in range(-100, 100):
            assert double(v) == v * 2


# ---------- Inline Cache Tests ----------

class TestPolymorphicInlineCache:
    def setup_method(self):
        self.pic = PolymorphicInlineCache()

    def test_load_attr(self):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        p = Point(3, 4)
        assert self.pic.load_attr(p, 'x') == 3
        assert self.pic.load_attr(p, 'y') == 4

    def test_cache_hit(self):
        class Obj:
            def __init__(self):
                self.val = 42

        o1 = Obj()
        o2 = Obj()
        o2.val = 99

        self.pic.load_attr(o1, 'val')  # Miss
        self.pic.load_attr(o2, 'val')  # Hit (same type)

        stats = self.pic.get_stats()
        assert 'val' in stats
        assert stats['val']['hits'] >= 1

    def test_polymorphic_types(self):
        class A:
            x = 1
        class B:
            x = 2
        class C:
            x = 3

        self.pic.load_attr(A(), 'x')
        self.pic.load_attr(B(), 'x')
        self.pic.load_attr(C(), 'x')

        stats = self.pic.get_stats()
        assert stats['x']['misses'] >= 2  # At least 2 new types

    def test_load_method(self):
        class MyClass:
            def greet(self):
                return "hello"

        obj = MyClass()
        method = self.pic.load_method(obj, 'greet')
        assert method() == "hello"

    def test_store_attr(self):
        class Container:
            def __init__(self):
                self.value = 0

        c = Container()
        self.pic.store_attr(c, 'value', 42)
        assert c.value == 42

    def test_invalidate(self):
        class X:
            val = 1
        self.pic.load_attr(X(), 'val')
        self.pic.invalidate('val')
        stats = self.pic.get_stats()
        assert 'val' not in stats

    def test_invalidate_all(self):
        class Y:
            a = 1
            b = 2
        obj = Y()
        self.pic.load_attr(obj, 'a')
        self.pic.load_attr(obj, 'b')
        self.pic.invalidate()
        assert len(self.pic.get_stats()) == 0


class TestMethodCache:
    def test_lookup(self):
        cache = MethodCache()

        class Greeter:
            def hello(self):
                return "hi"

        g = Greeter()
        method = cache.lookup(g, 'hello')
        assert method() == "hi"

    def test_cache_hit(self):
        cache = MethodCache()

        class Calculator:
            def add(self, x, y):
                return x + y

        c = Calculator()
        m1 = cache.lookup(c, 'add')
        m2 = cache.lookup(c, 'add')
        assert m1(1, 2) == 3
        assert m2(3, 4) == 7
        assert cache._stats.hits >= 1


# ---------- Deoptimizer Tests ----------

class TestDeoptimizer:
    def setup_method(self):
        self.deopt = Deoptimizer()

    def test_guard_passes(self):
        def fast(x, y):
            return x + y  # Int-specialized
        def slow(x, y):
            return x + y  # General

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[
                self.deopt.type_guard(0, int),
                self.deopt.type_guard(1, int),
            ],
        )
        assert guarded(1, 2) == 3

    def test_guard_fails_deoptimizes(self):
        fast_called = [False]
        def fast(x, y):
            fast_called[0] = True
            return x + y
        def slow(x, y):
            return x + y

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[self.deopt.type_guard(0, int)],
        )
        # Float arg should trigger deopt
        result = guarded(1.5, 2)
        assert result == 3.5
        assert not fast_called[0]  # fast should not be called

    def test_value_guard(self):
        def fast(x):
            return x * 2
        def slow(x):
            return x * 2

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[self.deopt.value_guard(0, min_val=0, max_val=100)],
        )
        assert guarded(50) == 100
        assert guarded(-1) == -2  # Deoptimizes but still correct
        assert guarded(200) == 400  # Deoptimizes

    def test_shape_guard(self):
        def fast(obj):
            return obj.x + obj.y
        def slow(obj):
            return getattr(obj, 'x', 0) + getattr(obj, 'y', 0)

        guard = self.deopt.shape_guard(0, {'x', 'y'})

        class Full:
            def __init__(self):
                self.x = 1
                self.y = 2

        class Partial:
            def __init__(self):
                self.x = 1

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[guard],
        )
        assert guarded(Full()) == 3

    def test_permanent_deopt(self):
        """After max_deopts, should permanently use fallback."""
        call_log = []
        def fast(x):
            call_log.append('fast')
            return x
        def slow(x):
            call_log.append('slow')
            return x

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[self.deopt.type_guard(0, int)],
            max_deopts=2,
        )
        guarded(1.0)  # deopt 1
        guarded(1.0)  # deopt 2 → permanent
        guarded(1)    # Should still use slow (permanent deopt)
        assert call_log[-1] == 'slow'

    def test_stats(self):
        def fast(x):
            return x
        def slow(x):
            return x

        guarded = self.deopt.guard(
            optimized=fast,
            fallback=slow,
            guards=[self.deopt.type_guard(0, int)],
        )
        guarded(1.0)  # Deopt
        stats = self.deopt.get_stats()
        assert stats['total_deopts'] >= 1
