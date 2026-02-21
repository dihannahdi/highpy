"""
Memory Pool & Arena Allocator
==============================

Novel contribution: Region-based arena memory management for Python
numerical workloads that eliminates per-object allocation/deallocation
overhead and reference counting costs.

CPython's memory overhead per object:
- PyObject_HEAD: 16 bytes (refcount + type pointer)
- Even small integers: 28 bytes
- float: 24 bytes for 8 bytes of data
- list: 56+ bytes overhead

Our arena allocator:
- Pre-allocates contiguous memory regions
- Amortizes allocation cost over many objects
- Enables bulk deallocation (free entire arena at once)
- Improves cache locality for numerical data

This module provides:
1. ArenaAllocator: Region-based allocator for numerical arrays
2. MemoryPool: Object pool for frequently-allocated types  
3. CompactArray: Cache-friendly array with minimal overhead
4. StructOfArrays: SoA layout for better vectorization
"""

import array
import ctypes
import struct
import math
import sys
import weakref
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, Type, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager

T = TypeVar('T')


@dataclass
class ArenaStats:
    """Statistics for arena allocation."""
    total_allocated_bytes: int = 0
    total_freed_bytes: int = 0
    current_usage_bytes: int = 0
    peak_usage_bytes: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    arena_count: int = 0


class ArenaAllocator:
    """
    Region-based arena memory allocator for numerical data.
    
    Allocates memory in large contiguous blocks (arenas) and
    sub-allocates from them with minimal overhead. Entire arenas
    are freed at once, eliminating individual deallocation cost.
    
    This is particularly effective for:
    - Temporary arrays in numerical computations
    - Matrix operations that create many intermediate arrays
    - Loop-intensive code with frequent small allocations
    
    Usage:
        >>> allocator = ArenaAllocator(arena_size=1024*1024)  # 1MB arenas
        >>> with allocator.scope() as arena:
        ...     arr1 = arena.alloc_doubles(1000)
        ...     arr2 = arena.alloc_doubles(1000)
        ...     # ... compute with arr1, arr2 ...
        ...     # Both freed automatically at scope exit
    """
    
    DEFAULT_ARENA_SIZE = 1024 * 1024  # 1 MB
    
    def __init__(self, arena_size: int = DEFAULT_ARENA_SIZE):
        self.arena_size = arena_size
        self._arenas: List['Arena'] = []
        self._current_arena: Optional['Arena'] = None
        self.stats = ArenaStats()
    
    @contextmanager
    def scope(self):
        """
        Create a scoped arena context.
        
        All allocations within the scope are freed when the
        context manager exits, regardless of exceptions.
        """
        arena = Arena(self.arena_size)
        self._arenas.append(arena)
        self._current_arena = arena
        self.stats.arena_count += 1
        
        try:
            yield arena
        finally:
            self._free_arena(arena)
    
    def alloc_doubles(self, count: int) -> 'CompactArray':
        """Allocate an array of doubles from the current arena."""
        if self._current_arena is None:
            self._current_arena = Arena(self.arena_size)
            self._arenas.append(self._current_arena)
            self.stats.arena_count += 1
        
        return self._current_arena.alloc_doubles(count)
    
    def alloc_longs(self, count: int) -> 'CompactArray':
        """Allocate an array of long integers from the current arena."""
        if self._current_arena is None:
            self._current_arena = Arena(self.arena_size)
            self._arenas.append(self._current_arena)
            self.stats.arena_count += 1
        
        return self._current_arena.alloc_longs(count)
    
    def _free_arena(self, arena: 'Arena'):
        """Free an entire arena and all its allocations."""
        freed = arena.used_bytes
        arena.reset()
        self.stats.total_freed_bytes += freed
        self.stats.deallocation_count += 1
        
        if arena in self._arenas:
            self._arenas.remove(arena)
        
        if self._current_arena is arena:
            self._current_arena = self._arenas[-1] if self._arenas else None
    
    def get_stats(self) -> ArenaStats:
        """Get allocation statistics."""
        self.stats.current_usage_bytes = sum(a.used_bytes for a in self._arenas)
        return self.stats


class Arena:
    """
    A contiguous memory region for sub-allocation.
    
    Memory layout:
    [header][alloc1][alloc2][...][free space...]
    
    Allocations within an arena are contiguous, improving
    cache locality for sequential access patterns.
    """
    
    def __init__(self, size: int):
        self.size = size
        self._buffer = bytearray(size)
        self._offset = 0
        self._allocations: List[Tuple[int, int]] = []  # (offset, size)
    
    @property
    def used_bytes(self) -> int:
        return self._offset
    
    @property  
    def free_bytes(self) -> int:
        return self.size - self._offset
    
    def alloc_doubles(self, count: int) -> 'CompactArray':
        """Allocate a contiguous array of doubles."""
        nbytes = count * 8  # sizeof(double)
        
        # Align to 8 bytes
        aligned_offset = (self._offset + 7) & ~7
        
        if aligned_offset + nbytes > self.size:
            raise MemoryError(
                f"Arena overflow: need {nbytes} bytes, "
                f"have {self.size - aligned_offset} free"
            )
        
        self._offset = aligned_offset + nbytes
        self._allocations.append((aligned_offset, nbytes))
        
        return CompactArray('d', count, self._buffer, aligned_offset)
    
    def alloc_longs(self, count: int) -> 'CompactArray':
        """Allocate a contiguous array of long integers."""
        nbytes = count * 8  # sizeof(long long)
        
        aligned_offset = (self._offset + 7) & ~7
        
        if aligned_offset + nbytes > self.size:
            raise MemoryError(
                f"Arena overflow: need {nbytes} bytes, "
                f"have {self.size - aligned_offset} free"
            )
        
        self._offset = aligned_offset + nbytes
        self._allocations.append((aligned_offset, nbytes))
        
        return CompactArray('q', count, self._buffer, aligned_offset)
    
    def reset(self):
        """Free all allocations in this arena."""
        self._offset = 0
        self._allocations.clear()


class CompactArray:
    """
    Cache-friendly compact array with minimal overhead.
    
    Unlike Python lists which store pointers to heap-allocated PyObjects,
    CompactArray stores raw values contiguously in memory, similar to
    C arrays or numpy ndarrays.
    
    Memory comparison for 1000 floats:
    - Python list: ~32,000 bytes (8 bytes ptr + 24 bytes per float object)
    - CompactArray: ~8,000 bytes (8 bytes per double, contiguous)
    - Overhead ratio: 4x less memory, much better cache behavior
    
    Usage:
        >>> arr = CompactArray.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> arr[2]
        3.0
        >>> arr.sum()
        15.0
    """
    
    TYPE_SIZES = {'d': 8, 'q': 8, 'l': 4, 'i': 4, 'f': 4}
    
    def __init__(
        self,
        typecode: str,
        count: int,
        buffer: Optional[bytearray] = None,
        offset: int = 0,
    ):
        self.typecode = typecode
        self.count = count
        self.item_size = self.TYPE_SIZES.get(typecode, 8)
        self._offset = offset
        
        if buffer is not None:
            self._buffer = buffer
            self._owned = False
        else:
            self._buffer = bytearray(count * self.item_size)
            self._offset = 0
            self._owned = True
    
    @classmethod
    def from_list(cls, data: list, typecode: str = 'd') -> 'CompactArray':
        """Create a CompactArray from a Python list."""
        arr = cls(typecode, len(data))
        for i, val in enumerate(data):
            arr[i] = val
        return arr
    
    @classmethod
    def zeros(cls, count: int, typecode: str = 'd') -> 'CompactArray':
        """Create a zero-initialized CompactArray."""
        return cls(typecode, count)
    
    @classmethod
    def range(cls, start: int, stop: int, step: int = 1) -> 'CompactArray':
        """Create a CompactArray with range values."""
        count = max(0, (stop - start + step - 1) // step)
        arr = cls('q', count)
        for i, val in enumerate(range(start, stop, step)):
            arr[i] = val
        return arr
    
    def __getitem__(self, index: int):
        if index < 0:
            index += self.count
        if not (0 <= index < self.count):
            raise IndexError(f"index {index} out of range [0, {self.count})")
        
        byte_offset = self._offset + index * self.item_size
        return struct.unpack_from(self.typecode, self._buffer, byte_offset)[0]
    
    def __setitem__(self, index: int, value):
        if index < 0:
            index += self.count
        if not (0 <= index < self.count):
            raise IndexError(f"index {index} out of range [0, {self.count})")
        
        byte_offset = self._offset + index * self.item_size
        struct.pack_into(self.typecode, self._buffer, byte_offset, value)
    
    def __len__(self) -> int:
        return self.count
    
    def __iter__(self) -> Iterator:
        for i in range(self.count):
            yield self[i]
    
    def sum(self) -> float:
        """Compute sum of all elements."""
        total = 0.0
        for i in range(self.count):
            byte_offset = self._offset + i * self.item_size
            total += struct.unpack_from(self.typecode, self._buffer, byte_offset)[0]
        return total
    
    def dot(self, other: 'CompactArray') -> float:
        """Compute dot product with another array."""
        if self.count != other.count:
            raise ValueError("Arrays must have same length")
        
        total = 0.0
        for i in range(self.count):
            a = struct.unpack_from(self.typecode, self._buffer, self._offset + i * self.item_size)[0]
            b = struct.unpack_from(other.typecode, other._buffer, other._offset + i * other.item_size)[0]
            total += a * b
        return total
    
    def add(self, other: 'CompactArray') -> 'CompactArray':
        """Element-wise addition."""
        if self.count != other.count:
            raise ValueError("Arrays must have same length")
        
        result = CompactArray(self.typecode, self.count)
        for i in range(self.count):
            a = self[i]
            b = other[i]
            result[i] = a + b
        return result
    
    def multiply(self, other: 'CompactArray') -> 'CompactArray':
        """Element-wise multiplication."""
        if self.count != other.count:
            raise ValueError("Arrays must have same length")
        
        result = CompactArray(self.typecode, self.count)
        for i in range(self.count):
            result[i] = self[i] * other[i]
        return result
    
    def scale(self, factor: float) -> 'CompactArray':
        """Scalar multiplication."""
        result = CompactArray(self.typecode, self.count)
        for i in range(self.count):
            result[i] = self[i] * factor
        return result
    
    def map(self, func: Callable) -> 'CompactArray':
        """Apply a function to every element."""
        result = CompactArray(self.typecode, self.count)
        for i in range(self.count):
            result[i] = func(self[i])
        return result
    
    def reduce(self, func: Callable, initial: float = 0.0) -> float:
        """Reduce the array with a binary function."""
        acc = initial
        for i in range(self.count):
            acc = func(acc, self[i])
        return acc
    
    def to_list(self) -> list:
        """Convert back to a Python list."""
        return [self[i] for i in range(self.count)]
    
    def memory_usage(self) -> int:
        """Return memory usage in bytes (just the data, no Python overhead)."""
        return self.count * self.item_size
    
    def __repr__(self):
        if self.count <= 10:
            vals = ', '.join(f'{self[i]:.4g}' for i in range(self.count))
        else:
            first = ', '.join(f'{self[i]:.4g}' for i in range(5))
            last = ', '.join(f'{self[i]:.4g}' for i in range(self.count-3, self.count))
            vals = f'{first}, ..., {last}'
        return f'CompactArray({self.typecode}, [{vals}])'


class MemoryPool:
    """
    Object pool for frequently-allocated Python objects.
    
    Reduces allocation overhead by recycling objects instead of
    creating new ones. Particularly effective for objects that are
    created and destroyed in tight loops.
    
    Usage:
        >>> pool = MemoryPool(list, initial_size=100)
        >>> obj = pool.acquire()  # Get from pool (fast)
        >>> obj.append(42)
        >>> pool.release(obj)     # Return to pool (fast)
    """
    
    def __init__(
        self,
        factory: Callable[[], Any],
        initial_size: int = 64,
        max_size: int = 1024,
        reset_func: Optional[Callable[[Any], None]] = None,
    ):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func or self._default_reset
        self._pool: List[Any] = []
        self._in_use: int = 0
        self.stats = {
            'acquires': 0,
            'releases': 0,
            'allocations': 0,
            'pool_hits': 0,
        }
        
        # Pre-populate pool
        for _ in range(initial_size):
            self._pool.append(factory())
            self.stats['allocations'] += 1
    
    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        self.stats['acquires'] += 1
        
        if self._pool:
            obj = self._pool.pop()
            self.stats['pool_hits'] += 1
        else:
            obj = self.factory()
            self.stats['allocations'] += 1
        
        self._in_use += 1
        return obj
    
    def release(self, obj: Any):
        """Release an object back to the pool."""
        self.stats['releases'] += 1
        self._in_use -= 1
        
        if len(self._pool) < self.max_size:
            self.reset_func(obj)
            self._pool.append(obj)
    
    @contextmanager
    def scoped(self):
        """Context manager for automatic release."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)
    
    def _default_reset(self, obj: Any):
        """Default reset: try to clear the object."""
        if hasattr(obj, 'clear'):
            obj.clear()
    
    @property
    def pool_size(self) -> int:
        return len(self._pool)
    
    @property
    def in_use(self) -> int:
        return self._in_use
    
    def hit_rate(self) -> float:
        """Pool hit rate (higher is better)."""
        if self.stats['acquires'] == 0:
            return 0.0
        return self.stats['pool_hits'] / self.stats['acquires']


class StructOfArrays:
    """
    Struct-of-Arrays (SoA) layout for better cache performance.
    
    Instead of storing N objects with M fields each (Array of Structs),
    stores M arrays of N elements each. This dramatically improves
    cache locality when iterating over a single field.
    
    AoS (Python default):
        [{x: 1, y: 2}, {x: 3, y: 4}, ...] -> poor locality for x-only access
    
    SoA (HighPy):
        {x: [1, 3, ...], y: [2, 4, ...]} -> excellent locality for x-only access
    
    Usage:
        >>> soa = StructOfArrays({'x': 'd', 'y': 'd', 'z': 'd'}, count=1000)
        >>> for i in range(1000):
        ...     soa.set(i, 'x', float(i))
        ...     soa.set(i, 'y', float(i * 2))
        >>> x_sum = soa.get_array('x').sum()
    """
    
    def __init__(self, fields: Dict[str, str], count: int):
        """
        Args:
            fields: Mapping of field name to typecode ('d' for double, 'q' for long)
            count: Number of elements
        """
        self.fields = fields
        self.count = count
        self._arrays: Dict[str, CompactArray] = {}
        
        for name, typecode in fields.items():
            self._arrays[name] = CompactArray(typecode, count)
    
    def get(self, index: int, field: str):
        """Get a field value for an element."""
        return self._arrays[field][index]
    
    def set(self, index: int, field: str, value):
        """Set a field value for an element."""
        self._arrays[field][index] = value
    
    def get_array(self, field: str) -> CompactArray:
        """Get the entire array for a field (zero-copy)."""
        return self._arrays[field]
    
    def memory_usage(self) -> int:
        """Total memory usage in bytes."""
        return sum(arr.memory_usage() for arr in self._arrays.values())
    
    def __len__(self) -> int:
        return self.count
