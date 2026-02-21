"""
CPython Performance Bottleneck Analyzer
========================================

Deep analysis of CPython's fundamental performance bottlenecks through
bytecode inspection, type system overhead measurement, memory allocation
profiling, and GIL contention analysis.

This module provides systematic identification and quantification of
the 10 primary reasons CPython underperforms compared to statically-typed,
compiled languages.

Bottleneck Taxonomy:
    B1 - Dynamic Type Dispatch:      Runtime type checking on every operation
    B2 - Object Model Overhead:      PyObject header + reference counting
    B3 - Bytecode Interpretation:    Eval loop dispatch overhead
    B4 - Attribute Lookup:           Dictionary-based __dict__ access
    B5 - Function Call Overhead:     Frame allocation + argument parsing
    B6 - Global Interpreter Lock:    Thread serialization
    B7 - Boxing/Unboxing:            Wrapping primitives as heap objects
    B8 - Late Binding:               Name resolution at runtime
    B9 - Memory Allocation Pattern:  Frequent small allocations
    B10 - Lack of Specialization:    Generic bytecode for all types
"""

import dis
import sys
import time
import types
import struct
import ctypes
import inspect
import opcode
import weakref
import gc
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class BottleneckReport:
    """Comprehensive report of identified performance bottlenecks."""
    
    function_name: str
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    bytecode_stats: Dict[str, int] = field(default_factory=dict)
    type_dispatch_count: int = 0
    attribute_lookup_count: int = 0
    function_call_count: int = 0
    boxing_operations: int = 0
    late_binding_count: int = 0
    loop_count: int = 0
    estimated_overhead_ns: float = 0.0
    optimization_potential: float = 0.0  # 0.0 to 1.0
    
    def summary(self) -> str:
        lines = [
            f"╔══════════════════════════════════════════════════════════╗",
            f"║  HighPy Bottleneck Analysis: {self.function_name:<28s}║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Type dispatches:      {self.type_dispatch_count:<8d}                       ║",
            f"║  Attribute lookups:    {self.attribute_lookup_count:<8d}                       ║",
            f"║  Function calls:       {self.function_call_count:<8d}                       ║",
            f"║  Boxing operations:    {self.boxing_operations:<8d}                       ║",
            f"║  Late bindings:        {self.late_binding_count:<8d}                       ║",
            f"║  Loops detected:       {self.loop_count:<8d}                       ║",
            f"║  Est. overhead:        {self.estimated_overhead_ns:>8.1f} ns/call               ║",
            f"║  Optimization pot.:    {self.optimization_potential*100:>6.1f}%                      ║",
            f"╠══════════════════════════════════════════════════════════╣",
        ]
        for b in self.bottlenecks[:5]:
            cat = b['category'][:20]
            sev = b['severity']
            lines.append(f"║  [{sev:>4s}] {cat:<46s}║")
        lines.append(f"╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ---- Overhead cost model (empirically calibrated for CPython 3.11+) ----
# Costs in nanoseconds per operation
COST_MODEL = {
    'type_check':        15.0,   # COMPARE_OP type dispatch
    'attribute_dict':    45.0,   # __dict__ lookup
    'attribute_slot':    12.0,   # __slots__ lookup
    'function_call':     80.0,   # Full Python call
    'c_function_call':   25.0,   # C-extension call
    'box_int':           35.0,   # int object creation
    'box_float':         40.0,   # float object creation
    'unbox':             10.0,   # PyLong_AsLong etc.
    'name_lookup':       30.0,   # LOAD_GLOBAL / LOAD_NAME
    'bytecode_dispatch': 8.0,    # Eval loop iteration
    'refcount_inc':      3.0,    # Py_INCREF
    'refcount_dec':      5.0,    # Py_DECREF (may trigger dealloc)
    'gc_overhead':       2.0,    # Amortized GC cost per object
    'loop_overhead':     12.0,   # FOR_ITER dispatch
    'binary_op':         25.0,   # BINARY_OP with type dispatch
    'compare_op':        20.0,   # COMPARE_OP
    'store_fast':        5.0,    # STORE_FAST (direct)
    'load_fast':         4.0,    # LOAD_FAST (direct)
}

# Bytecodes that involve dynamic type dispatch
TYPE_DISPATCH_OPS = {
    'BINARY_OP', 'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY',
    'BINARY_FLOOR_DIVIDE', 'BINARY_TRUE_DIVIDE', 'BINARY_MODULO',
    'BINARY_POWER', 'BINARY_LSHIFT', 'BINARY_RSHIFT', 'BINARY_AND',
    'BINARY_OR', 'BINARY_XOR', 'BINARY_SUBSCR',
    'COMPARE_OP', 'CONTAINS_OP', 'IS_OP',
    'UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT',
    'BINARY_OP',
}

# Bytecodes that involve attribute/name lookup
ATTR_LOOKUP_OPS = {
    'LOAD_ATTR', 'STORE_ATTR', 'DELETE_ATTR',
    'LOAD_METHOD',
}

# Bytecodes involving function calls
CALL_OPS = {
    'CALL', 'CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX',
    'CALL_METHOD',
}

# Bytecodes involving name resolution (late binding)
LATE_BINDING_OPS = {
    'LOAD_GLOBAL', 'LOAD_NAME', 'STORE_GLOBAL', 'STORE_NAME',
    'LOAD_DEREF', 'STORE_DEREF', 'LOAD_CLOSURE',
}

# Bytecodes involving boxing/object creation
BOXING_OPS = {
    'LOAD_CONST', 'BUILD_LIST', 'BUILD_TUPLE', 'BUILD_SET',
    'BUILD_MAP', 'BUILD_CONST_KEY_MAP', 'BUILD_STRING',
    'BUILD_SLICE', 'LIST_APPEND', 'SET_ADD', 'MAP_ADD',
}

# Loop-related bytecodes
LOOP_OPS = {
    'FOR_ITER', 'GET_ITER', 'SETUP_LOOP', 'JUMP_BACKWARD',
    'JUMP_BACKWARD_NO_INTERRUPT',
}


class CPythonAnalyzer:
    """
    Deep analyzer for CPython performance bottlenecks.
    
    Performs multi-level analysis:
    1. Bytecode-level: Instruction frequency and cost estimation
    2. Type-level: Dynamic dispatch overhead quantification
    3. Memory-level: Object allocation pattern analysis
    4. Call-level: Function call overhead profiling
    5. GIL-level: Thread contention measurement
    
    Example:
        >>> analyzer = CPythonAnalyzer()
        >>> def target(n):
        ...     total = 0
        ...     for i in range(n):
        ...         total += i * i
        ...     return total
        >>> report = analyzer.analyze(target)
        >>> print(report.summary())
    """
    
    def __init__(self, cost_model: Optional[Dict[str, float]] = None):
        self.cost_model = cost_model or COST_MODEL
        self._calibrated = False
        self._calibration_factor = 1.0
    
    def calibrate(self, iterations: int = 1000000) -> float:
        """
        Calibrate the cost model against the actual machine.
        
        Runs a calibration microbenchmark to determine the ratio between
        the theoretical cost model and actual execution time.
        
        Returns:
            Calibration factor (actual_time / model_time)
        """
        def _calibration_workload():
            x = 0
            for i in range(100):
                x = x + i
                x = x * 2
                x = x - i
            return x
        
        # Warm up
        for _ in range(1000):
            _calibration_workload()
        
        # Measure
        start = time.perf_counter_ns()
        for _ in range(iterations):
            _calibration_workload()
        elapsed_ns = time.perf_counter_ns() - start
        actual_per_call = elapsed_ns / iterations
        
        # Estimate model cost
        model_cost = self._estimate_bytecode_cost(_calibration_workload)
        
        if model_cost > 0:
            self._calibration_factor = actual_per_call / model_cost
        else:
            self._calibration_factor = 1.0
        
        self._calibrated = True
        return self._calibration_factor
    
    def analyze(self, func: Callable) -> BottleneckReport:
        """
        Perform comprehensive bottleneck analysis on a function.
        
        Args:
            func: The Python function to analyze
            
        Returns:
            BottleneckReport with detailed bottleneck identification
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func).__name__}")
        
        # Get the underlying code object
        code = self._get_code_object(func)
        if code is None:
            raise ValueError(f"Cannot extract code object from {func}")
        
        report = BottleneckReport(function_name=getattr(func, '__name__', str(func)))
        
        # Phase 1: Bytecode analysis
        self._analyze_bytecode(code, report)
        
        # Phase 2: Type dispatch analysis
        self._analyze_type_dispatch(code, report)
        
        # Phase 3: Attribute access analysis
        self._analyze_attribute_access(code, report)
        
        # Phase 4: Function call analysis
        self._analyze_function_calls(code, report)
        
        # Phase 5: Boxing/unboxing analysis
        self._analyze_boxing(code, report)
        
        # Phase 6: Late binding analysis
        self._analyze_late_binding(code, report)
        
        # Phase 7: Loop analysis
        self._analyze_loops(code, report)
        
        # Phase 8: Memory allocation pattern
        self._analyze_memory_pattern(code, report)
        
        # Phase 9: Cost estimation
        self._estimate_cost(code, report)
        
        # Phase 10: Optimization potential
        self._compute_optimization_potential(report)
        
        return report
    
    def analyze_module(self, module) -> List[BottleneckReport]:
        """Analyze all functions in a module."""
        reports = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            try:
                report = self.analyze(obj)
                reports.append(report)
            except (ValueError, TypeError):
                continue
        return reports
    
    def compare_implementations(
        self,
        funcs: List[Callable],
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        iterations: int = 10000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple implementations of the same functionality.
        
        Returns timing, bottleneck analysis, and speedup ratios.
        """
        kwargs = kwargs or {}
        results = {}
        
        for func in funcs:
            name = getattr(func, '__name__', str(func))
            
            # Warm up
            for _ in range(min(100, iterations)):
                func(*args, **kwargs)
            
            # Time
            start = time.perf_counter_ns()
            for _ in range(iterations):
                func(*args, **kwargs)
            elapsed_ns = time.perf_counter_ns() - start
            
            # Analyze
            try:
                report = self.analyze(func)
            except Exception:
                report = None
            
            results[name] = {
                'total_ns': elapsed_ns,
                'per_call_ns': elapsed_ns / iterations,
                'report': report,
            }
        
        # Compute speedup ratios
        if results:
            baseline_time = list(results.values())[0]['per_call_ns']
            for name, data in results.items():
                data['speedup'] = baseline_time / max(data['per_call_ns'], 1)
        
        return results
    
    def _get_code_object(self, func: Callable) -> Optional[types.CodeType]:
        """Extract code object from various callable types."""
        if hasattr(func, '__code__'):
            return func.__code__
        if hasattr(func, '__func__'):
            return func.__func__.__code__
        if hasattr(func, '__wrapped__'):
            return self._get_code_object(func.__wrapped__)
        return None
    
    def _get_instructions(self, code: types.CodeType) -> List:
        """Get bytecode instructions for a code object."""
        return list(dis.get_instructions(code))
    
    def _analyze_bytecode(self, code: types.CodeType, report: BottleneckReport):
        """Analyze bytecode instruction distribution."""
        instructions = self._get_instructions(code)
        stats = defaultdict(int)
        
        for instr in instructions:
            stats[instr.opname] += 1
        
        report.bytecode_stats = dict(stats)
        
        total_instructions = len(instructions)
        if total_instructions > 0:
            # Compute instruction mix metrics
            compute_ops = sum(stats.get(op, 0) for op in TYPE_DISPATCH_OPS)
            overhead_ops = sum(stats.get(op, 0) for op in 
                            ATTR_LOOKUP_OPS | CALL_OPS | LATE_BINDING_OPS)
            
            overhead_ratio = overhead_ops / total_instructions if total_instructions > 0 else 0
            
            report.bottlenecks.append({
                'category': 'B3-Bytecode Interpretation',
                'severity': 'HIGH' if total_instructions > 50 else 'MED',
                'detail': f'{total_instructions} instructions, {overhead_ratio:.1%} overhead',
                'instruction_count': total_instructions,
                'overhead_ratio': overhead_ratio,
            })
    
    def _analyze_type_dispatch(self, code: types.CodeType, report: BottleneckReport):
        """Quantify dynamic type dispatch overhead."""
        instructions = self._get_instructions(code)
        dispatch_count = 0
        
        for instr in instructions:
            if instr.opname in TYPE_DISPATCH_OPS:
                dispatch_count += 1
        
        report.type_dispatch_count = dispatch_count
        
        if dispatch_count > 0:
            cost = dispatch_count * self.cost_model['type_check']
            severity = 'CRIT' if dispatch_count > 20 else ('HIGH' if dispatch_count > 10 else 'MED')
            report.bottlenecks.append({
                'category': 'B1-Dynamic Type Dispatch',
                'severity': severity,
                'detail': f'{dispatch_count} dispatch sites, ~{cost:.0f}ns overhead',
                'count': dispatch_count,
                'estimated_cost_ns': cost,
            })
    
    def _analyze_attribute_access(self, code: types.CodeType, report: BottleneckReport):
        """Analyze attribute access patterns."""
        instructions = self._get_instructions(code)
        attr_count = 0
        attr_names = []
        
        for instr in instructions:
            if instr.opname in ATTR_LOOKUP_OPS:
                attr_count += 1
                if instr.argval:
                    attr_names.append(str(instr.argval))
        
        report.attribute_lookup_count = attr_count
        
        if attr_count > 0:
            cost = attr_count * self.cost_model['attribute_dict']
            severity = 'CRIT' if attr_count > 15 else ('HIGH' if attr_count > 5 else 'MED')
            report.bottlenecks.append({
                'category': 'B4-Attribute Dict Lookup',
                'severity': severity,
                'detail': f'{attr_count} lookups ({", ".join(attr_names[:5])}), ~{cost:.0f}ns',
                'count': attr_count,
                'attributes': attr_names,
                'estimated_cost_ns': cost,
            })
    
    def _analyze_function_calls(self, code: types.CodeType, report: BottleneckReport):
        """Analyze function call overhead."""
        instructions = self._get_instructions(code)
        call_count = 0
        
        for instr in instructions:
            if instr.opname in CALL_OPS:
                call_count += 1
        
        report.function_call_count = call_count
        
        if call_count > 0:
            cost = call_count * self.cost_model['function_call']
            severity = 'HIGH' if call_count > 5 else 'MED'
            report.bottlenecks.append({
                'category': 'B5-Function Call Overhead',
                'severity': severity,
                'detail': f'{call_count} calls, ~{cost:.0f}ns frame+arg overhead',
                'count': call_count,
                'estimated_cost_ns': cost,
            })
    
    def _analyze_boxing(self, code: types.CodeType, report: BottleneckReport):
        """Analyze boxing/unboxing operations."""
        instructions = self._get_instructions(code)
        boxing_count = 0
        
        for instr in instructions:
            if instr.opname in BOXING_OPS:
                boxing_count += 1
        
        report.boxing_operations = boxing_count
        
        if boxing_count > 0:
            cost = boxing_count * self.cost_model['box_int']
            severity = 'HIGH' if boxing_count > 10 else 'MED'
            report.bottlenecks.append({
                'category': 'B7-Boxing/Object Creation',
                'severity': severity,
                'detail': f'{boxing_count} boxing ops, ~{cost:.0f}ns alloc overhead',
                'count': boxing_count,
                'estimated_cost_ns': cost,
            })
    
    def _analyze_late_binding(self, code: types.CodeType, report: BottleneckReport):
        """Analyze late binding / name resolution overhead."""
        instructions = self._get_instructions(code)
        late_count = 0
        global_names = []
        
        for instr in instructions:
            if instr.opname in LATE_BINDING_OPS:
                late_count += 1
                if instr.argval and instr.opname in ('LOAD_GLOBAL', 'STORE_GLOBAL'):
                    global_names.append(str(instr.argval))
        
        report.late_binding_count = late_count
        
        if late_count > 0:
            cost = late_count * self.cost_model['name_lookup']
            severity = 'HIGH' if late_count > 10 else 'MED'
            report.bottlenecks.append({
                'category': 'B8-Late Binding/Name Lookup',
                'severity': severity,
                'detail': f'{late_count} lookups (globals: {", ".join(global_names[:5])})',
                'count': late_count,
                'global_names': global_names,
                'estimated_cost_ns': cost,
            })
    
    def _analyze_loops(self, code: types.CodeType, report: BottleneckReport):
        """Detect and analyze loop structures."""
        instructions = self._get_instructions(code)
        loop_count = 0
        
        for instr in instructions:
            if instr.opname in LOOP_OPS:
                loop_count += 1
        
        # Also detect backward jumps as loop indicators
        offsets = {instr.offset: instr for instr in instructions}
        for instr in instructions:
            if instr.opname.startswith('JUMP') and instr.argval is not None:
                try:
                    target = int(instr.argval) if not isinstance(instr.argval, int) else instr.argval
                    if target < instr.offset:
                        loop_count += 1
                except (ValueError, TypeError):
                    pass
        
        report.loop_count = max(loop_count, 0)
        
        if loop_count > 0:
            report.bottlenecks.append({
                'category': 'B3-Loop Interpretation Overhead',
                'severity': 'CRIT',
                'detail': f'{loop_count} loop structures detected - prime optimization target',
                'count': loop_count,
            })
    
    def _analyze_memory_pattern(self, code: types.CodeType, report: BottleneckReport):
        """Analyze memory allocation patterns from bytecode."""
        instructions = self._get_instructions(code)
        alloc_ops = 0
        
        alloc_opcodes = {
            'BUILD_LIST', 'BUILD_TUPLE', 'BUILD_SET', 'BUILD_MAP',
            'BUILD_CONST_KEY_MAP', 'BUILD_STRING', 'BUILD_SLICE',
            'LIST_APPEND', 'SET_ADD', 'MAP_ADD', 'LIST_EXTEND',
        }
        
        for instr in instructions:
            if instr.opname in alloc_opcodes:
                alloc_ops += 1
        
        if alloc_ops > 0:
            report.bottlenecks.append({
                'category': 'B9-Memory Allocation Pattern',
                'severity': 'HIGH' if alloc_ops > 5 else 'LOW',
                'detail': f'{alloc_ops} allocation ops - candidates for arena allocation',
                'count': alloc_ops,
            })
    
    def _estimate_bytecode_cost(self, func: Callable) -> float:
        """Estimate total bytecode execution cost in nanoseconds."""
        code = self._get_code_object(func)
        if code is None:
            return 0.0
        
        instructions = self._get_instructions(code)
        total_cost = 0.0
        
        for instr in instructions:
            total_cost += self.cost_model['bytecode_dispatch']
            
            if instr.opname in TYPE_DISPATCH_OPS:
                total_cost += self.cost_model['binary_op']
            elif instr.opname in ATTR_LOOKUP_OPS:
                total_cost += self.cost_model['attribute_dict']
            elif instr.opname in CALL_OPS:
                total_cost += self.cost_model['function_call']
            elif instr.opname in LATE_BINDING_OPS:
                total_cost += self.cost_model['name_lookup']
            elif instr.opname == 'LOAD_FAST':
                total_cost += self.cost_model['load_fast']
            elif instr.opname == 'STORE_FAST':
                total_cost += self.cost_model['store_fast']
        
        return total_cost
    
    def _estimate_cost(self, code: types.CodeType, report: BottleneckReport):
        """Estimate total overhead cost."""
        total_cost = 0.0
        for b in report.bottlenecks:
            total_cost += b.get('estimated_cost_ns', 0.0)
        
        # Add baseline bytecode dispatch cost
        total_instructions = sum(report.bytecode_stats.values())
        total_cost += total_instructions * self.cost_model['bytecode_dispatch']
        
        # Add refcount overhead (approximately 2 refcount ops per instruction)
        total_cost += total_instructions * 2 * self.cost_model['refcount_inc']
        
        if self._calibrated:
            total_cost *= self._calibration_factor
        
        report.estimated_overhead_ns = total_cost
    
    def _compute_optimization_potential(self, report: BottleneckReport):
        """
        Compute optimization potential score.
        
        This estimates how much speedup HighPy can achieve based on
        the bottleneck profile:
        - High type dispatch + loops = very optimizable (loop compilation)
        - High attribute access = moderately optimizable (inline caching)
        - High function calls = moderately optimizable (inlining)
        """
        score = 0.0
        
        # Loops with type dispatch are the best optimization targets
        if report.loop_count > 0 and report.type_dispatch_count > 5:
            score += 0.4
        elif report.loop_count > 0:
            score += 0.2
        
        # Type specialization potential
        if report.type_dispatch_count > 10:
            score += 0.25
        elif report.type_dispatch_count > 5:
            score += 0.15
        
        # Attribute access optimization potential
        if report.attribute_lookup_count > 5:
            score += 0.15
        
        # Function call optimization potential
        if report.function_call_count > 3:
            score += 0.1
        
        # Late binding optimization potential
        if report.late_binding_count > 5:
            score += 0.1
        
        report.optimization_potential = min(score, 0.95)


class MemoryOverheadAnalyzer:
    """
    Analyzes Python's per-object memory overhead compared to raw data.
    
    CPython's object model adds significant overhead per object:
    - PyObject_HEAD: 16 bytes (ob_refcnt + ob_type pointer) on 64-bit
    - PyVarObject: 24 bytes (adds ob_size)
    - int: 28+ bytes for small integers
    - float: 24 bytes for 8 bytes of actual data
    - list: 56+ bytes base + 8 bytes per element (pointer)
    - dict: 64+ bytes base + entries
    """
    
    @staticmethod
    def measure_object_overhead() -> Dict[str, Dict[str, Any]]:
        """Measure actual memory overhead of common Python types."""
        results = {}
        
        # Integer
        results['int_small'] = {
            'python_size': sys.getsizeof(42),
            'raw_size': 8,  # int64
            'overhead_ratio': sys.getsizeof(42) / 8,
        }
        
        results['int_large'] = {
            'python_size': sys.getsizeof(2**100),
            'raw_size': 16,  # 128-bit
            'overhead_ratio': sys.getsizeof(2**100) / 16,
        }
        
        # Float  
        results['float'] = {
            'python_size': sys.getsizeof(3.14),
            'raw_size': 8,  # double
            'overhead_ratio': sys.getsizeof(3.14) / 8,
        }
        
        # List of 1000 floats
        lst = [float(i) for i in range(1000)]
        list_size = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)
        results['list_1000_floats'] = {
            'python_size': list_size,
            'raw_size': 8000,  # 1000 doubles
            'overhead_ratio': list_size / 8000,
        }
        
        # Dict
        d = {str(i): float(i) for i in range(100)}
        dict_size = sys.getsizeof(d)
        results['dict_100_entries'] = {
            'python_size': dict_size,
            'raw_size': 100 * 16,  # 100 * (key_ptr + value_ptr)
            'overhead_ratio': dict_size / (100 * 16),
        }
        
        # String
        results['str_100'] = {
            'python_size': sys.getsizeof("a" * 100),
            'raw_size': 100,
            'overhead_ratio': sys.getsizeof("a" * 100) / 100,
        }
        
        # Tuple
        t = tuple(range(100))
        results['tuple_100'] = {
            'python_size': sys.getsizeof(t),
            'raw_size': 800,  # 100 int64s
            'overhead_ratio': sys.getsizeof(t) / 800,
        }
        
        return results
    
    @staticmethod
    def measure_refcount_overhead(iterations: int = 1000000) -> Dict[str, float]:
        """Measure the overhead of reference counting operations."""
        obj = [1, 2, 3]
        
        # Measure assignment (triggers INCREF/DECREF)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            a = obj
            b = obj
            c = obj
            del a
            del b
            del c
        elapsed_refcount = time.perf_counter_ns() - start
        
        # Baseline (no refcount)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            pass
        elapsed_baseline = time.perf_counter_ns() - start
        
        refcount_cost = (elapsed_refcount - elapsed_baseline) / iterations
        
        return {
            'refcount_cost_ns': refcount_cost,
            'per_incref_ns': refcount_cost / 6,  # 3 INCREF + 3 DECREF
        }


class GILContentionAnalyzer:
    """
    Measures Global Interpreter Lock contention overhead.
    
    The GIL prevents true parallel execution of Python threads,
    serializing CPU-bound work. This analyzer quantifies the
    overhead by comparing single-threaded vs multi-threaded
    execution of identical workloads.
    """
    
    @staticmethod
    def measure_gil_overhead(
        workload: Optional[Callable] = None,
        threads: int = 4,
        iterations: int = 100000,
    ) -> Dict[str, Any]:
        """
        Measure GIL contention overhead.
        
        Args:
            workload: CPU-bound function to benchmark (default: arithmetic loop)
            threads: Number of threads
            iterations: Iterations per thread
            
        Returns:
            Dict with single/multi-threaded times and overhead ratio
        """
        if workload is None:
            def workload():
                x = 0
                for i in range(iterations):
                    x += i * i
                return x
        
        # Single-threaded baseline
        start = time.perf_counter_ns()
        for _ in range(threads):
            workload()
        single_threaded_ns = time.perf_counter_ns() - start
        
        # Multi-threaded (GIL-constrained)
        thread_list = []
        start = time.perf_counter_ns()
        for _ in range(threads):
            t = threading.Thread(target=workload)
            thread_list.append(t)
            t.start()
        for t in thread_list:
            t.join()
        multi_threaded_ns = time.perf_counter_ns() - start
        
        return {
            'single_threaded_ns': single_threaded_ns,
            'multi_threaded_ns': multi_threaded_ns,
            'overhead_ratio': multi_threaded_ns / max(single_threaded_ns, 1),
            'expected_ratio_no_gil': 1.0 / threads,
            'gil_penalty': multi_threaded_ns / max(single_threaded_ns, 1) - (1.0 / threads),
            'threads': threads,
        }


class BytecodeComplexityAnalyzer:
    """
    Analyzes the computational complexity profile of Python bytecode.
    
    Maps bytecode patterns to algorithmic complexity classes and
    identifies quadratic/exponential patterns that compound the
    interpreter overhead.
    """
    
    def analyze_complexity(self, func: Callable) -> Dict[str, Any]:
        """Analyze the complexity profile of a function's bytecode."""
        code = func.__code__ if hasattr(func, '__code__') else None
        if code is None:
            return {'error': 'Cannot extract code object'}
        
        instructions = list(dis.get_instructions(code))
        
        # Detect nested loops (potential O(n^2) or worse)
        nesting_depth = self._detect_loop_nesting(instructions)
        
        # Detect recursive calls
        is_recursive = self._detect_recursion(code, instructions)
        
        # Count bytecode per loop body
        loop_bodies = self._extract_loop_bodies(instructions)
        
        return {
            'total_instructions': len(instructions),
            'loop_nesting_depth': nesting_depth,
            'is_recursive': is_recursive,
            'loop_body_sizes': [len(body) for body in loop_bodies],
            'complexity_class': self._infer_complexity(nesting_depth, is_recursive),
            'bytecode_per_iteration': sum(len(body) for body in loop_bodies),
        }
    
    def _detect_loop_nesting(self, instructions: List) -> int:
        """Detect maximum loop nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for instr in instructions:
            if instr.opname in ('GET_ITER', 'FOR_ITER'):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            # Heuristic: backward jumps indicate loop end
            if instr.opname.startswith('JUMP') and instr.argval is not None:
                try:
                    target = instr.argval if isinstance(instr.argval, int) else int(instr.argval)
                    if target < instr.offset:
                        current_depth = max(0, current_depth - 1)
                except (ValueError, TypeError):
                    pass
        
        return max_depth
    
    def _detect_recursion(self, code: types.CodeType, instructions: List) -> bool:
        """Detect if function calls itself recursively."""
        func_name = code.co_name
        
        for i, instr in enumerate(instructions):
            if instr.opname in ('LOAD_GLOBAL', 'LOAD_DEREF', 'LOAD_NAME'):
                if instr.argval == func_name:
                    # Check if followed by a CALL
                    for j in range(i + 1, min(i + 5, len(instructions))):
                        if instructions[j].opname in CALL_OPS:
                            return True
        
        return False
    
    def _extract_loop_bodies(self, instructions: List) -> List[List]:
        """Extract instruction sequences that form loop bodies."""
        bodies = []
        current_body = []
        in_loop = False
        
        for instr in instructions:
            if instr.opname == 'FOR_ITER':
                in_loop = True
                current_body = []
            elif in_loop:
                if instr.opname.startswith('JUMP') and instr.argval is not None:
                    try:
                        target = instr.argval if isinstance(instr.argval, int) else int(instr.argval)
                        if target < instr.offset:
                            bodies.append(current_body)
                            in_loop = False
                            continue
                    except (ValueError, TypeError):
                        pass
                current_body.append(instr)
        
        if in_loop and current_body:
            bodies.append(current_body)
        
        return bodies
    
    def _infer_complexity(self, nesting: int, recursive: bool) -> str:
        """Infer algorithmic complexity class."""
        if recursive:
            return 'O(2^n) or O(n!) [recursive]'
        elif nesting >= 3:
            return f'O(n^{nesting}) [deeply nested]'
        elif nesting == 2:
            return 'O(n^2) [nested loops]'
        elif nesting == 1:
            return 'O(n) [single loop]'
        else:
            return 'O(1) [no loops]'
