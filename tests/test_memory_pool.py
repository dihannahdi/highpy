"""
Tests for the memory pool and arena allocator.

Validates:
  - ArenaAllocator allocates and frees correctly
  - CompactArray provides correct operations  
  - MemoryPool recycles objects
  - StructOfArrays provides correct field access
"""

import pytest
from highpy.optimization.memory_pool import (
    ArenaAllocator,
    CompactArray,
    MemoryPool,
    StructOfArrays,
)


class TestArenaAllocator:
    def test_create(self):
        arena = ArenaAllocator()
        assert arena is not None

    def test_scope(self):
        arena = ArenaAllocator()
        with arena.scope() as a:
            assert a is not None

    def test_allocate_in_scope(self):
        arena = ArenaAllocator()
        with arena.scope() as a:
            buf = a.alloc_doubles(128)  # 128 doubles = 1024 bytes
            assert buf is not None
            assert len(buf) == 128

    def test_multiple_allocations(self):
        arena = ArenaAllocator()
        with arena.scope() as a:
            b1 = a.alloc_doubles(100)
            b2 = a.alloc_doubles(200)
            b3 = a.alloc_doubles(300)
            assert len(b1) == 100
            assert len(b2) == 200
            assert len(b3) == 300


class TestCompactArray:
    def test_create_int(self):
        arr = CompactArray.from_list([1, 2, 3, 4, 5], 'i')
        assert len(arr) == 5

    def test_create_float(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0])
        assert len(arr) == 3

    def test_getitem(self):
        arr = CompactArray.from_list([10.0, 20.0, 30.0])
        assert abs(arr[0] - 10.0) < 1e-10
        assert abs(arr[1] - 20.0) < 1e-10
        assert abs(arr[2] - 30.0) < 1e-10

    def test_setitem(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0])
        arr[1] = 99.0
        assert abs(arr[1] - 99.0) < 1e-10

    def test_sum(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0, 4.0])
        assert abs(arr.sum() - 10.0) < 1e-10

    def test_add(self):
        a = CompactArray.from_list([1.0, 2.0, 3.0])
        b = CompactArray.from_list([4.0, 5.0, 6.0])
        c = a.add(b)
        assert abs(c[0] - 5.0) < 1e-10
        assert abs(c[1] - 7.0) < 1e-10
        assert abs(c[2] - 9.0) < 1e-10

    def test_multiply(self):
        a = CompactArray.from_list([1.0, 2.0, 3.0])
        b = CompactArray.from_list([4.0, 5.0, 6.0])
        c = a.multiply(b)
        assert abs(c[0] - 4.0) < 1e-10
        assert abs(c[1] - 10.0) < 1e-10
        assert abs(c[2] - 18.0) < 1e-10

    def test_dot(self):
        a = CompactArray.from_list([1.0, 2.0, 3.0])
        b = CompactArray.from_list([4.0, 5.0, 6.0])
        assert abs(a.dot(b) - 32.0) < 1e-10

    def test_scale(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0])
        scaled = arr.scale(2.0)
        assert abs(scaled[0] - 2.0) < 1e-10
        assert abs(scaled[1] - 4.0) < 1e-10

    def test_map(self):
        arr = CompactArray.from_list([1.0, 4.0, 9.0])
        import math
        result = arr.map(math.sqrt)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[2] - 3.0) < 1e-10

    def test_reduce(self):
        arr = CompactArray.from_list([1.0, 2.0, 3.0, 4.0])
        total = arr.reduce(lambda a, b: a + b, 0.0)
        assert abs(total - 10.0) < 1e-10

    def test_iter(self):
        arr = CompactArray.from_list([10.0, 20.0, 30.0])
        vals = list(arr)
        assert all(abs(a - b) < 1e-10 for a, b in zip(vals, [10.0, 20.0, 30.0]))

    def test_memory_efficiency(self):
        """CompactArray should use less memory than a Python list."""
        import sys
        n = 1000
        py_list = list(range(n))
        compact = CompactArray.from_list([float(i) for i in range(n)])
        compact_size = compact.memory_usage()
        assert compact_size < sys.getsizeof(py_list)


class TestMemoryPool:
    def test_create(self):
        pool = MemoryPool(list)
        assert pool is not None

    def test_acquire_release(self):
        pool = MemoryPool(list)
        obj = pool.acquire()
        assert isinstance(obj, list)
        pool.release(obj)

    def test_reuse(self):
        pool = MemoryPool(list)
        obj1 = pool.acquire()
        pool.release(obj1)
        obj2 = pool.acquire()
        assert obj2 is obj1

    def test_scoped(self):
        pool = MemoryPool(dict)
        with pool.scoped() as obj:
            assert isinstance(obj, dict)
            obj['key'] = 'value'


class TestStructOfArrays:
    def test_create(self):
        soa = StructOfArrays({'x': 'd', 'y': 'd', 'z': 'd'}, count=0)
        assert soa is not None

    def test_set_get(self):
        soa = StructOfArrays({'x': 'd', 'y': 'd'}, count=2)
        soa.set(0, 'x', 1.0)
        soa.set(0, 'y', 2.0)
        soa.set(1, 'x', 3.0)
        soa.set(1, 'y', 4.0)
        assert abs(soa.get(0, 'x') - 1.0) < 1e-10
        assert abs(soa.get(1, 'x') - 3.0) < 1e-10
        assert abs(soa.get(0, 'y') - 2.0) < 1e-10
        assert abs(soa.get(1, 'y') - 4.0) < 1e-10

    def test_len(self):
        soa = StructOfArrays({'a': 'd', 'b': 'd'}, count=5)
        assert len(soa) == 5

    def test_get_array(self):
        soa = StructOfArrays({'x': 'd', 'y': 'd'}, count=3)
        soa.set(0, 'x', 10.0)
        soa.set(1, 'x', 20.0)
        soa.set(2, 'x', 30.0)
        x_arr = soa.get_array('x')
        assert abs(x_arr[0] - 10.0) < 1e-10
        assert abs(x_arr[1] - 20.0) < 1e-10
        assert abs(x_arr[2] - 30.0) < 1e-10
