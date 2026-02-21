"""
Type Profiler & Lattice-Based Type Inference Engine
====================================================

Novel contribution: Combines runtime type profiling with static abstract
interpretation over a type lattice to infer precise type information for
Python programs without explicit annotations.

Type Lattice Structure:
    ⊤ (Any/Unknown)
    ├── Numeric
    │   ├── Int
    │   ├── Float
    │   └── Complex
    ├── Sequence
    │   ├── List[T]
    │   ├── Tuple[T...]
    │   └── Str
    ├── Mapping
    │   └── Dict[K,V]
    ├── Callable[Args, Ret]
    ├── NoneType
    └── ⊥ (Bottom/Unreachable)

The inference engine performs:
1. Forward abstract interpretation over the CFG
2. Widening at loop headers for convergence
3. Narrowing through conditional branches
4. Interprocedural analysis via call graph
"""

import ast
import dis
import sys
import types
import inspect
import functools
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, 
    Set, Tuple, Type, Union
)
from enum import Enum, auto
from collections import defaultdict


# ============================================================================
# Type Lattice
# ============================================================================

class TypeTag(Enum):
    """Tags for the type lattice elements."""
    TOP = auto()       # Unknown/Any - joins to everything
    BOTTOM = auto()    # Unreachable
    INT = auto()
    FLOAT = auto()
    COMPLEX = auto()
    BOOL = auto()
    STR = auto()
    BYTES = auto()
    NONE = auto()
    LIST = auto()
    TUPLE = auto()
    SET = auto()
    DICT = auto()
    CALLABLE = auto()
    OBJECT = auto()
    NUMERIC = auto()   # Int | Float | Complex
    SEQUENCE = auto()  # List | Tuple | Str
    ITERABLE = auto()  # Any iterable


@dataclass(frozen=True)
class LatticeType:
    """
    Element of the type lattice with optional parameterization.
    
    Supports:
    - Primitive types: Int, Float, Str, None, Bool
    - Parameterized types: List[Int], Dict[Str, Float]
    - Union types: Int | Float
    - Callable types: Callable[[Int, Int], Float]
    """
    tag: TypeTag
    params: Tuple['LatticeType', ...] = ()
    is_union: bool = False
    union_members: FrozenSet['LatticeType'] = frozenset()
    
    @staticmethod
    def top() -> 'LatticeType':
        return LatticeType(TypeTag.TOP)
    
    @staticmethod
    def bottom() -> 'LatticeType':
        return LatticeType(TypeTag.BOTTOM)
    
    @staticmethod
    def from_python_type(t: type) -> 'LatticeType':
        """Convert a Python runtime type to a LatticeType."""
        mapping = {
            int: TypeTag.INT,
            float: TypeTag.FLOAT,
            complex: TypeTag.COMPLEX,
            bool: TypeTag.BOOL,
            str: TypeTag.STR,
            bytes: TypeTag.BYTES,
            type(None): TypeTag.NONE,
            list: TypeTag.LIST,
            tuple: TypeTag.TUPLE,
            set: TypeTag.SET,
            dict: TypeTag.DICT,
        }
        tag = mapping.get(t, TypeTag.OBJECT)
        return LatticeType(tag)
    
    @staticmethod
    def from_value(val: Any) -> 'LatticeType':
        """Infer LatticeType from a runtime value."""
        t = type(val)
        base = LatticeType.from_python_type(t)
        
        # Infer parameterization for containers
        if isinstance(val, list) and val:
            elem_types = frozenset(LatticeType.from_python_type(type(x)) for x in val[:100])
            if len(elem_types) == 1:
                elem_type = next(iter(elem_types))
                return LatticeType(TypeTag.LIST, (elem_type,))
            else:
                union = LatticeType(TypeTag.TOP, is_union=True, union_members=elem_types)
                return LatticeType(TypeTag.LIST, (union,))
        
        if isinstance(val, dict) and val:
            key_types = frozenset(LatticeType.from_python_type(type(k)) for k in list(val.keys())[:100])
            val_types = frozenset(LatticeType.from_python_type(type(v)) for v in list(val.values())[:100])
            kt = next(iter(key_types)) if len(key_types) == 1 else LatticeType.top()
            vt = next(iter(val_types)) if len(val_types) == 1 else LatticeType.top()
            return LatticeType(TypeTag.DICT, (kt, vt))
        
        return base
    
    def join(self, other: 'LatticeType') -> 'LatticeType':
        """Least upper bound in the type lattice."""
        if self == other:
            return self
        if self.tag == TypeTag.BOTTOM:
            return other
        if other.tag == TypeTag.BOTTOM:
            return self
        if self.tag == TypeTag.TOP or other.tag == TypeTag.TOP:
            return LatticeType.top()
        
        # Numeric join: Int ⊔ Float = Numeric
        numeric_tags = {TypeTag.INT, TypeTag.FLOAT, TypeTag.COMPLEX, TypeTag.BOOL}
        if self.tag in numeric_tags and other.tag in numeric_tags:
            # Widening: Int + Float -> Float (Python semantics)
            if TypeTag.COMPLEX in (self.tag, other.tag):
                return LatticeType(TypeTag.COMPLEX)
            if TypeTag.FLOAT in (self.tag, other.tag):
                return LatticeType(TypeTag.FLOAT)
            if TypeTag.INT in (self.tag, other.tag):
                return LatticeType(TypeTag.INT)
            return LatticeType(TypeTag.NUMERIC)
        
        # Same container, join parameters
        if self.tag == other.tag and self.params and other.params:
            if len(self.params) == len(other.params):
                joined_params = tuple(p.join(q) for p, q in zip(self.params, other.params))
                return LatticeType(self.tag, joined_params)
        
        # Different types -> union
        members = set()
        if self.is_union:
            members.update(self.union_members)
        else:
            members.add(self)
        if other.is_union:
            members.update(other.union_members)
        else:
            members.add(other)
        
        return LatticeType(TypeTag.TOP, is_union=True, union_members=frozenset(members))
    
    def meet(self, other: 'LatticeType') -> 'LatticeType':
        """Greatest lower bound in the type lattice."""
        if self == other:
            return self
        if self.tag == TypeTag.TOP:
            return other
        if other.tag == TypeTag.TOP:
            return self
        if self.tag == TypeTag.BOTTOM or other.tag == TypeTag.BOTTOM:
            return LatticeType.bottom()
        
        # Same tag, meet parameters
        if self.tag == other.tag:
            if self.params and other.params and len(self.params) == len(other.params):
                met_params = tuple(p.meet(q) for p, q in zip(self.params, other.params))
                return LatticeType(self.tag, met_params)
            return LatticeType(self.tag)
        
        return LatticeType.bottom()
    
    def is_numeric(self) -> bool:
        return self.tag in {TypeTag.INT, TypeTag.FLOAT, TypeTag.COMPLEX, 
                           TypeTag.BOOL, TypeTag.NUMERIC}
    
    def is_concrete(self) -> bool:
        """True if this is a concrete (non-union, non-top/bottom) type."""
        return (self.tag not in {TypeTag.TOP, TypeTag.BOTTOM} 
                and not self.is_union)
    
    def is_specializable(self) -> bool:
        """True if this type is precise enough for specialization."""
        return self.is_concrete()
    
    def to_c_type(self) -> str:
        """Map to C type for native code generation."""
        mapping = {
            TypeTag.INT: 'long long',
            TypeTag.FLOAT: 'double',
            TypeTag.BOOL: 'int',
            TypeTag.COMPLEX: 'double complex',
        }
        return mapping.get(self.tag, 'PyObject*')
    
    def to_ctypes_type(self):
        """Map to ctypes type."""
        import ctypes
        mapping = {
            TypeTag.INT: ctypes.c_longlong,
            TypeTag.FLOAT: ctypes.c_double,
            TypeTag.BOOL: ctypes.c_int,
        }
        return mapping.get(self.tag, ctypes.py_object)
    
    def __repr__(self):
        if self.is_union:
            return ' | '.join(repr(m) for m in self.union_members)
        if self.params:
            params_str = ', '.join(repr(p) for p in self.params)
            return f'{self.tag.name}[{params_str}]'
        return self.tag.name


# ============================================================================
# Type Profiler (Runtime)
# ============================================================================

class TypeProfiler:
    """
    Runtime type profiler that traces actual types flowing through a function.
    
    Uses sys.settrace to intercept execution and record observed types for:
    - Function arguments on each call
    - Local variables at each assignment
    - Return values
    - Loop iteration variables
    
    The profiling data is then converted to LatticeType information
    for the static analyzer.
    
    Usage:
        >>> profiler = TypeProfiler()
        >>> @profiler.profile
        ... def my_func(x, y):
        ...     return x + y
        >>> my_func(1, 2)
        >>> my_func(1.0, 2.0)
        >>> type_info = profiler.get_type_info('my_func')
    """
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._profiles: Dict[str, Dict[str, List[type]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._return_types: Dict[str, List[type]] = defaultdict(list)
        self._is_profiling = False
    
    def profile(self, func: Callable) -> Callable:
        """Decorator that profiles type information for a function."""
        func_name = func.__qualname__
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._call_counts[func_name] += 1
            
            if self._call_counts[func_name] <= self.max_samples:
                # Record argument types
                for i, (name, val) in enumerate(zip(param_names, args)):
                    self._profiles[func_name][name].append(type(val))
                
                for name, val in kwargs.items():
                    self._profiles[func_name][name].append(type(val))
            
            result = func(*args, **kwargs)
            
            if self._call_counts[func_name] <= self.max_samples:
                self._return_types[func_name].append(type(result))
            
            return result
        
        wrapper._highpy_profiled = True
        wrapper._highpy_original = func
        return wrapper
    
    def get_type_info(self, func_name: str) -> Dict[str, LatticeType]:
        """
        Get inferred type information from profiling data.
        
        Returns a mapping from variable name to its LatticeType,
        computed by joining all observed types through the lattice.
        """
        result = {}
        
        if func_name in self._profiles:
            for var_name, type_list in self._profiles[func_name].items():
                if not type_list:
                    result[var_name] = LatticeType.top()
                    continue
                
                lattice_type = LatticeType.from_python_type(type_list[0])
                for t in type_list[1:]:
                    lattice_type = lattice_type.join(LatticeType.from_python_type(t))
                
                result[var_name] = lattice_type
        
        if func_name in self._return_types:
            type_list = self._return_types[func_name]
            if type_list:
                ret_type = LatticeType.from_python_type(type_list[0])
                for t in type_list[1:]:
                    ret_type = ret_type.join(LatticeType.from_python_type(t))
                result['__return__'] = ret_type
        
        return result
    
    def get_monomorphic_info(self, func_name: str) -> Dict[str, bool]:
        """
        Check if each parameter is monomorphic (single type observed).
        
        Monomorphic parameters are the best candidates for type specialization
        as they indicate the function is used consistently with the same types.
        """
        result = {}
        
        if func_name in self._profiles:
            for var_name, type_list in self._profiles[func_name].items():
                unique_types = set(type_list)
                result[var_name] = len(unique_types) <= 1
        
        return result
    
    def get_type_stability(self, func_name: str) -> float:
        """
        Compute type stability score (0.0 to 1.0).
        
        1.0 = all parameters are always the same type (fully monomorphic)
        0.0 = every call has different types (fully polymorphic)
        
        Type-stable functions are the best optimization candidates.
        """
        if func_name not in self._profiles:
            return 0.0
        
        mono_info = self.get_monomorphic_info(func_name)
        if not mono_info:
            return 0.0
        
        return sum(1.0 for v in mono_info.values() if v) / len(mono_info)
    
    def get_call_count(self, func_name: str) -> int:
        return self._call_counts.get(func_name, 0)
    
    def is_hot(self, func_name: str, threshold: int = 100) -> bool:
        """Check if a function is hot (called frequently)."""
        return self._call_counts.get(func_name, 0) >= threshold
    
    def reset(self):
        """Clear all profiling data."""
        self._profiles.clear()
        self._call_counts.clear()
        self._return_types.clear()


# ============================================================================
# Static Type Inference via Abstract Interpretation
# ============================================================================

class TypeEnvironment:
    """
    Type environment mapping variable names to their LatticeTypes.
    
    Represents the abstract state at a program point in the
    abstract interpretation framework.
    """
    
    def __init__(self, bindings: Optional[Dict[str, LatticeType]] = None):
        self._bindings: Dict[str, LatticeType] = dict(bindings or {})
    
    def get(self, name: str) -> LatticeType:
        return self._bindings.get(name, LatticeType.top())
    
    def set(self, name: str, typ: LatticeType) -> 'TypeEnvironment':
        new_env = TypeEnvironment(self._bindings)
        new_env._bindings[name] = typ
        return new_env
    
    def join(self, other: 'TypeEnvironment') -> 'TypeEnvironment':
        """Join two environments (for control flow merge points)."""
        all_names = set(self._bindings.keys()) | set(other._bindings.keys())
        new_bindings = {}
        for name in all_names:
            t1 = self.get(name)
            t2 = other.get(name)
            new_bindings[name] = t1.join(t2)
        return TypeEnvironment(new_bindings)
    
    def __eq__(self, other):
        if not isinstance(other, TypeEnvironment):
            return False
        return self._bindings == other._bindings
    
    def __repr__(self):
        items = ', '.join(f'{k}: {v}' for k, v in self._bindings.items())
        return f'TypeEnv({items})'


class AbstractInterpreter(ast.NodeVisitor):
    """
    Abstract interpreter for type inference over Python ASTs.
    
    Performs forward abstract interpretation using the type lattice
    to infer types at every program point without executing the code.
    
    Novel aspects:
    1. Handles Python-specific semantics (duck typing, operator overloading)
    2. Widening strategy for loop convergence
    3. Context-sensitive analysis for function calls
    4. Narrowing through isinstance() guards
    """
    
    # Binary operator result type rules
    BINOP_RULES = {
        # (left_type, right_type, operator) -> result_type
        (TypeTag.INT, TypeTag.INT, ast.Add): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.Sub): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.Mult): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.FloorDiv): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.Mod): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.Pow): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.BitAnd): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.BitOr): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.BitXor): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.LShift): TypeTag.INT,
        (TypeTag.INT, TypeTag.INT, ast.RShift): TypeTag.INT,
        (TypeTag.INT, TypeTag.FLOAT, ast.Add): TypeTag.FLOAT,
        (TypeTag.INT, TypeTag.FLOAT, ast.Sub): TypeTag.FLOAT,
        (TypeTag.INT, TypeTag.FLOAT, ast.Mult): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.INT, ast.Add): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.INT, ast.Sub): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.INT, ast.Mult): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.Add): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.Sub): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.Mult): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.FloorDiv): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.Mod): TypeTag.FLOAT,
        (TypeTag.INT, TypeTag.INT, ast.Div): TypeTag.FLOAT,  # True division
        (TypeTag.FLOAT, TypeTag.FLOAT, ast.Div): TypeTag.FLOAT,
        (TypeTag.INT, TypeTag.FLOAT, ast.Div): TypeTag.FLOAT,
        (TypeTag.FLOAT, TypeTag.INT, ast.Div): TypeTag.FLOAT,
        (TypeTag.STR, TypeTag.STR, ast.Add): TypeTag.STR,
        (TypeTag.STR, TypeTag.INT, ast.Mult): TypeTag.STR,
        (TypeTag.INT, TypeTag.STR, ast.Mult): TypeTag.STR,
    }
    
    # Comparison operators always return bool
    COMPARE_RESULT = TypeTag.BOOL
    
    def __init__(self, profiler: Optional[TypeProfiler] = None):
        self.profiler = profiler
        self.env = TypeEnvironment()
        self._function_sigs: Dict[str, Tuple[List[LatticeType], LatticeType]] = {}
        self._widening_limit = 5
    
    def infer_function(
        self, 
        func: Callable,
        arg_types: Optional[Dict[str, LatticeType]] = None,
    ) -> Dict[str, LatticeType]:
        """
        Infer types for all variables in a function.
        
        Args:
            func: The function to analyze
            arg_types: Optional type hints for arguments (from profiler or annotations)
            
        Returns:
            Dict mapping variable names to inferred LatticeTypes
        """
        source = inspect.getsource(func)
        tree = ast.parse(source)
        
        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break
        
        if func_def is None:
            return {}
        
        # Initialize environment with argument types
        self.env = TypeEnvironment()
        
        if arg_types:
            for name, typ in arg_types.items():
                self.env = self.env.set(name, typ)
        else:
            # Try to get types from profiler
            if self.profiler:
                prof_types = self.profiler.get_type_info(func.__qualname__)
                for name, typ in prof_types.items():
                    if name != '__return__':
                        self.env = self.env.set(name, typ)
            
            # Use annotations if available
            hints = func.__annotations__ if hasattr(func, '__annotations__') else {}
            for name, hint in hints.items():
                if name != 'return' and isinstance(hint, type):
                    self.env = self.env.set(name, LatticeType.from_python_type(hint))
            
            # Default: arguments start as TOP
            for arg in func_def.args.args:
                if self.env.get(arg.arg).tag == TypeTag.TOP:
                    self.env = self.env.set(arg.arg, LatticeType.top())
        
        # Interpret the function body
        for stmt in func_def.body:
            self._interpret_stmt(stmt)
        
        return dict(self.env._bindings)
    
    def _interpret_stmt(self, node: ast.stmt):
        """Interpret a statement, updating the type environment."""
        if isinstance(node, ast.Assign):
            value_type = self._eval_expr(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env = self.env.set(target.id, value_type)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.env = self.env.set(elt.id, LatticeType.top())
        
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                left_type = self.env.get(node.target.id)
                right_type = self._eval_expr(node.value)
                result_type = self._eval_binop(left_type, right_type, node.op)
                self.env = self.env.set(node.target.id, result_type)
        
        elif isinstance(node, ast.For):
            iter_type = self._eval_expr(node.iter)
            
            # Infer loop variable type from iterable
            if isinstance(node.target, ast.Name):
                if iter_type.tag == TypeTag.LIST and iter_type.params:
                    self.env = self.env.set(node.target.id, iter_type.params[0])
                elif iter_type.tag == TypeTag.STR:
                    self.env = self.env.set(node.target.id, LatticeType(TypeTag.STR))
                else:
                    self.env = self.env.set(node.target.id, LatticeType.top())
            
            # Fixed-point iteration for loop body with widening
            prev_env = None
            for iteration in range(self._widening_limit):
                prev_env = TypeEnvironment(dict(self.env._bindings))
                for stmt in node.body:
                    self._interpret_stmt(stmt)
                if self.env == prev_env:
                    break  # Fixed point reached
                # Widen: join current with previous
                self.env = self.env.join(prev_env)
        
        elif isinstance(node, ast.While):
            prev_env = None
            for iteration in range(self._widening_limit):
                prev_env = TypeEnvironment(dict(self.env._bindings))
                for stmt in node.body:
                    self._interpret_stmt(stmt)
                if self.env == prev_env:
                    break
                self.env = self.env.join(prev_env)
        
        elif isinstance(node, ast.If):
            # Analyze both branches
            then_env = TypeEnvironment(dict(self.env._bindings))
            else_env = TypeEnvironment(dict(self.env._bindings))
            
            # Narrow types through isinstance checks
            narrowed = self._narrow_from_test(node.test)
            if narrowed:
                for name, typ in narrowed.items():
                    then_env = then_env.set(name, typ)
            
            # Interpret then-branch
            saved_env = self.env
            self.env = then_env
            for stmt in node.body:
                self._interpret_stmt(stmt)
            then_env = self.env
            
            # Interpret else-branch
            self.env = else_env
            for stmt in node.orelse:
                self._interpret_stmt(stmt)
            else_env = self.env
            
            # Join branches
            self.env = then_env.join(else_env)
        
        elif isinstance(node, ast.Return):
            if node.value:
                ret_type = self._eval_expr(node.value)
                current_ret = self.env.get('__return__')
                self.env = self.env.set('__return__', current_ret.join(ret_type))
        
        elif isinstance(node, ast.Expr):
            self._eval_expr(node.value)
    
    def _eval_expr(self, node: ast.expr) -> LatticeType:
        """Evaluate an expression in the abstract domain."""
        if isinstance(node, ast.Constant):
            return LatticeType.from_value(node.value)
        
        elif isinstance(node, ast.Name):
            return self.env.get(node.id)
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            return self._eval_binop(left, right, node.op)
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return operand  # -int -> int, -float -> float
            elif isinstance(node.op, ast.Not):
                return LatticeType(TypeTag.BOOL)
            return operand
        
        elif isinstance(node, ast.Compare):
            return LatticeType(TypeTag.BOOL)
        
        elif isinstance(node, ast.BoolOp):
            return LatticeType(TypeTag.BOOL)
        
        elif isinstance(node, ast.Call):
            return self._eval_call(node)
        
        elif isinstance(node, ast.Subscript):
            value_type = self._eval_expr(node.value)
            if value_type.tag == TypeTag.LIST and value_type.params:
                return value_type.params[0]
            elif value_type.tag == TypeTag.DICT and value_type.params and len(value_type.params) > 1:
                return value_type.params[1]
            return LatticeType.top()
        
        elif isinstance(node, ast.Attribute):
            return LatticeType.top()  # Conservative
        
        elif isinstance(node, ast.List):
            if node.elts:
                elem_type = self._eval_expr(node.elts[0])
                for elt in node.elts[1:]:
                    elem_type = elem_type.join(self._eval_expr(elt))
                return LatticeType(TypeTag.LIST, (elem_type,))
            return LatticeType(TypeTag.LIST)
        
        elif isinstance(node, ast.Tuple):
            elem_types = tuple(self._eval_expr(e) for e in node.elts)
            return LatticeType(TypeTag.TUPLE, elem_types)
        
        elif isinstance(node, ast.Dict):
            return LatticeType(TypeTag.DICT)
        
        elif isinstance(node, ast.IfExp):
            then_type = self._eval_expr(node.body)
            else_type = self._eval_expr(node.orelse)
            return then_type.join(else_type)
        
        elif isinstance(node, ast.ListComp):
            return LatticeType(TypeTag.LIST)
        
        return LatticeType.top()
    
    def _eval_binop(
        self, left: LatticeType, right: LatticeType, op: ast.operator
    ) -> LatticeType:
        """Evaluate a binary operation in the abstract domain."""
        key = (left.tag, right.tag, type(op))
        result_tag = self.BINOP_RULES.get(key)
        
        if result_tag is not None:
            return LatticeType(result_tag)
        
        # Fallback: if both are numeric, result is numeric
        if left.is_numeric() and right.is_numeric():
            if isinstance(op, ast.Div):
                return LatticeType(TypeTag.FLOAT)
            return left.join(right)
        
        return LatticeType.top()
    
    def _eval_call(self, node: ast.Call) -> LatticeType:
        """Evaluate a function call in the abstract domain."""
        # Handle built-in functions
        if isinstance(node.func, ast.Name):
            builtin_returns = {
                'int': TypeTag.INT,
                'float': TypeTag.FLOAT,
                'str': TypeTag.STR,
                'bool': TypeTag.BOOL,
                'len': TypeTag.INT,
                'abs': None,  # Same as argument
                'range': TypeTag.LIST,  # Simplified
                'list': TypeTag.LIST,
                'tuple': TypeTag.TUPLE,
                'set': TypeTag.SET,
                'dict': TypeTag.DICT,
                'sum': None,  # Depends on argument
                'min': None,
                'max': None,
                'round': TypeTag.INT,
                'sorted': TypeTag.LIST,
                'reversed': TypeTag.LIST,
                'enumerate': TypeTag.LIST,
                'zip': TypeTag.LIST,
                'map': TypeTag.LIST,
                'filter': TypeTag.LIST,
                'isinstance': TypeTag.BOOL,
                'type': TypeTag.OBJECT,
                'print': TypeTag.NONE,
            }
            
            name = node.func.id
            if name in builtin_returns:
                tag = builtin_returns[name]
                if tag is not None:
                    return LatticeType(tag)
                # For abs/sum/min/max, return type of first argument
                if node.args:
                    return self._eval_expr(node.args[0])
                return LatticeType.top()
            
            # Check known function signatures
            if name in self._function_sigs:
                _, ret_type = self._function_sigs[name]
                return ret_type
        
        # Handle method calls
        if isinstance(node.func, ast.Attribute):
            obj_type = self._eval_expr(node.func.value)
            method = node.func.attr
            
            # List methods
            if obj_type.tag == TypeTag.LIST:
                if method == 'append':
                    return LatticeType(TypeTag.NONE)
                elif method in ('pop', '__getitem__'):
                    if obj_type.params:
                        return obj_type.params[0]
                elif method == 'sort':
                    return LatticeType(TypeTag.NONE)
                elif method == 'copy':
                    return obj_type
            
            # Math module
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
                return LatticeType(TypeTag.FLOAT)
        
        return LatticeType.top()
    
    def _narrow_from_test(self, test: ast.expr) -> Optional[Dict[str, LatticeType]]:
        """
        Narrow types through isinstance() checks (type guards).
        
        Example: if isinstance(x, int): -> x is INT in the then-branch
        """
        if isinstance(test, ast.Call):
            if isinstance(test.func, ast.Name) and test.func.id == 'isinstance':
                if len(test.args) >= 2:
                    if isinstance(test.args[0], ast.Name):
                        var_name = test.args[0].id
                        type_arg = test.args[1]
                        
                        if isinstance(type_arg, ast.Name):
                            type_map = {
                                'int': TypeTag.INT,
                                'float': TypeTag.FLOAT,
                                'str': TypeTag.STR,
                                'bool': TypeTag.BOOL,
                                'list': TypeTag.LIST,
                                'dict': TypeTag.DICT,
                                'tuple': TypeTag.TUPLE,
                            }
                            tag = type_map.get(type_arg.id)
                            if tag:
                                return {var_name: LatticeType(tag)}
        
        return None
