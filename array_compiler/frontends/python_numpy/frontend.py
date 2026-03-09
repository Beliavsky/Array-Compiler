"""Minimal Python/NumPy frontend.

The first migration milestones cover typed scalar numeric scripts and small
structured return values suitable for option-pricing examples.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ...ir.module import Module
from ...ir.nodes import (
    Assignment,
    Append,
    ArrayTypeRef,
    AugmentedAssignment,
    BinaryOp,
    BinaryOperator,
    BooleanOp,
    Call,
    Compare,
    CompareOperator,
    Constant,
    ExprStatement,
    FieldAccess,
    ForRange,
    Function,
    If,
    IndexAccess,
    Print,
    Program,
    Raise,
    RecordDef,
    RecordLiteral,
    RecordTypeRef,
    Return,
    ScalarType,
    UnaryOp,
    UnaryOperator,
    ValueRef,
)


@dataclass
class PythonNumpyFrontend:
    _function_result_types: dict[str, object] = field(default_factory=dict, init=False)
    _record_defs: dict[str, RecordDef] = field(default_factory=dict, init=False)
    _current_function: str | None = field(default=None, init=False)

    def lower_source(self, source: str, module_name: str = "translated_module") -> Module:
        tree = ast.parse(source)
        self._function_result_types = {}
        self._record_defs = {}
        self._current_function = None

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._register_function_signature(node)

        functions: list[Function] = []
        program = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(self._lower_function(node))
            elif self._is_main_guard(node):
                program = self._lower_main_guard(node)
        exports = [fn.name for fn in functions]
        return Module(
            name=module_name,
            records=list(self._record_defs.values()),
            functions=functions,
            program=program,
            exports=exports,
            library_mode=True,
        )

    def _register_function_signature(self, node: ast.FunctionDef) -> None:
        result_type = self._map_annotation(node.returns, node.name)
        if result_type is not None:
            self._function_result_types[node.name] = result_type

    def _lower_function(self, node: ast.FunctionDef) -> Function:
        self._current_function = node.name
        locals_map: dict[str, object] = {}
        for stmt in node.body:
            self._collect_locals(stmt, locals_map)
        args = [(arg.arg, self._map_annotation(arg.annotation, node.name) or ScalarType.REAL64) for arg in node.args.args]
        arg_names = {name for name, _ in args}
        locals_list = [(name, typ) for name, typ in locals_map.items() if name not in arg_names]
        body = [self._lower_stmt(stmt) for stmt in node.body if not self._is_docstring(stmt)]
        function = Function(
            name=node.name,
            args=args,
            locals=locals_list,
            body=body,
            result_type=self._function_result_types.get(node.name),
        )
        self._current_function = None
        return function

    def _lower_main_guard(self, node: ast.If) -> Program:
        body = [self._lower_stmt(stmt) for stmt in node.body]
        return Program(name="run_main", body=body)

    def _collect_locals(self, stmt: ast.stmt, locals_map: dict[str, object]) -> None:
        if isinstance(stmt, ast.Assign):
            inferred = self._infer_type(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    locals_map.setdefault(target.id, inferred)
                elif isinstance(target, ast.Tuple):
                    if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                        record_type = self._function_result_types.get(stmt.value.func.id)
                        if isinstance(record_type, RecordTypeRef):
                            locals_map.setdefault(self._tuple_temp_name(stmt.value.func.id), record_type)
                    for index, elt in enumerate(target.elts, start=1):
                        if isinstance(elt, ast.Name):
                            locals_map.setdefault(elt.id, self._tuple_item_type(inferred, index))
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            locals_map.setdefault(stmt.target.id, self._infer_type(stmt.value))
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            locals_map.setdefault(self._loop_target_name(stmt.target.id), ScalarType.INTEGER)
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)
        elif isinstance(stmt, ast.If):
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)

    def _lower_stmt(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise NotImplementedError(f"unsupported assignment: {ast.dump(stmt)}")
            target = stmt.targets[0]
            value = self._lower_expr(stmt.value)
            if isinstance(target, ast.Name):
                return Assignment(ValueRef(target.id), value)
            if isinstance(target, ast.Tuple):
                return self._lower_tuple_assignment(target, stmt.value)
            raise NotImplementedError(f"unsupported assignment target: {ast.dump(target)}")
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            op_map = {
                ast.Add: BinaryOperator.ADD,
                ast.Sub: BinaryOperator.SUB,
                ast.Mult: BinaryOperator.MUL,
                ast.Div: BinaryOperator.DIV,
            }
            return AugmentedAssignment(ValueRef(stmt.target.id), op_map[type(stmt.op)], self._lower_expr(stmt.value))
        if isinstance(stmt, ast.Return):
            return Return(self._lower_expr(stmt.value) if stmt.value is not None else None)
        if isinstance(stmt, ast.If):
            return If(
                test=self._lower_expr(stmt.test),
                body=tuple(self._lower_stmt(inner) for inner in stmt.body),
                orelse=tuple(self._lower_stmt(inner) for inner in stmt.orelse),
            )
        if isinstance(stmt, ast.For):
            if not isinstance(stmt.target, ast.Name):
                raise NotImplementedError(f"unsupported loop target: {ast.dump(stmt.target)}")
            if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
                raise NotImplementedError(f"unsupported loop iterator: {ast.dump(stmt.iter)}")
            if len(stmt.iter.args) not in {1, 2, 3}:
                raise NotImplementedError("range supports up to three arguments in the current Python frontend slice")
            start = None
            step = None
            if len(stmt.iter.args) == 1:
                stop = self._lower_expr(stmt.iter.args[0])
            elif len(stmt.iter.args) == 2:
                start = self._lower_expr(stmt.iter.args[0])
                stop = self._lower_expr(stmt.iter.args[1])
            else:
                start = self._lower_expr(stmt.iter.args[0])
                stop = self._lower_expr(stmt.iter.args[1])
                step = self._lower_expr(stmt.iter.args[2])
            return ForRange(
                target=self._loop_target_name(stmt.target.id),
                start=start,
                stop=stop,
                step=step,
                body=tuple(self._lower_stmt(inner) for inner in stmt.body),
            )
        if isinstance(stmt, ast.Raise):
            message = "error"
            if isinstance(stmt.exc, ast.Call) and stmt.exc.args and isinstance(stmt.exc.args[0], ast.Constant):
                message = str(stmt.exc.args[0].value)
            return Raise(message)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "print":
                return Print(self._lower_print_args(stmt.value.args))
            if (
                isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "append"
                and isinstance(stmt.value.func.value, ast.Name)
                and len(stmt.value.args) == 1
            ):
                return Append(ValueRef(stmt.value.func.value.id), self._lower_expr(stmt.value.args[0]))
            return ExprStatement(self._lower_expr(stmt.value))
        raise NotImplementedError(f"unsupported statement: {ast.dump(stmt)}")

    def _lower_tuple_assignment(self, target: ast.Tuple, value: ast.AST):
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            raise NotImplementedError("tuple assignment requires call on right-hand side")
        record_type = self._function_result_types.get(value.func.id)
        if not isinstance(record_type, RecordTypeRef):
            raise NotImplementedError("tuple assignment requires a structured-return function")
        temp_name = self._tuple_temp_name(value.func.id)
        statements: list[object] = [Assignment(ValueRef(temp_name), self._lower_expr(value))]
        for index, elt in enumerate(target.elts, start=1):
            if not isinstance(elt, ast.Name):
                raise NotImplementedError(f"unsupported tuple target: {ast.dump(elt)}")
            statements.append(Assignment(ValueRef(elt.id), FieldAccess(ValueRef(temp_name), f"item{index}")))
        return If(Constant(True), tuple(statements), ())

    def _lower_print_args(self, args: list[ast.AST]) -> tuple[object, ...]:
        values: list[object] = []
        for arg in args:
            if isinstance(arg, ast.JoinedStr):
                values.extend(self._lower_joined_str_parts(arg))
            else:
                values.append(self._lower_expr(arg))
        return tuple(values)

    def _lower_joined_str_parts(self, expr: ast.JoinedStr) -> list[object]:
        values: list[object] = []
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                if part.value:
                    values.append(Constant(part.value))
            elif isinstance(part, ast.FormattedValue):
                values.append(self._lower_expr(part.value))
            else:
                raise NotImplementedError(f"unsupported f-string component: {ast.dump(part)}")
        return values

    def _lower_expr(self, expr: ast.AST):
        if isinstance(expr, ast.Constant):
            return Constant(expr.value)
        if isinstance(expr, ast.Name):
            return ValueRef(expr.id)
        if isinstance(expr, ast.List):
            return Call("ac_array_literal", tuple(self._lower_expr(elt) for elt in expr.elts))
        if isinstance(expr, ast.Dict):
            record_type = self._function_result_types.get(self._current_function or "")
            if not isinstance(record_type, RecordTypeRef):
                raise NotImplementedError("dict literal requires a structured return type")
            fields: list[tuple[str, object]] = []
            typed_fields: list[tuple[str, object]] = []
            for key, value in zip(expr.keys, expr.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise NotImplementedError(f"unsupported dict key: {ast.dump(key)}")
                fields.append((key.value, self._lower_expr(value)))
                typed_fields.append((key.value, self._infer_type(value)))
            self._ensure_record(record_type.name, typed_fields)
            return RecordLiteral(record_type.name, tuple(fields))
        if isinstance(expr, ast.Tuple):
            record_type = self._function_result_types.get(self._current_function or "")
            if not isinstance(record_type, RecordTypeRef):
                raise NotImplementedError("tuple literal requires a structured return type")
            fields = [(f"item{index}", self._lower_expr(value)) for index, value in enumerate(expr.elts, start=1)]
            typed_fields = [(f"item{index}", self._infer_type(value)) for index, value in enumerate(expr.elts, start=1)]
            self._ensure_record(record_type.name, typed_fields)
            return RecordLiteral(record_type.name, tuple(fields))
        if isinstance(expr, ast.Subscript):
            if isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, str):
                return FieldAccess(self._lower_expr(expr.value), expr.slice.value)
            return IndexAccess(self._lower_expr(expr.value), self._lower_expr(expr.slice))
        if isinstance(expr, ast.BinOp):
            op_map = {
                ast.Add: BinaryOperator.ADD,
                ast.Sub: BinaryOperator.SUB,
                ast.Mult: BinaryOperator.MUL,
                ast.Div: BinaryOperator.DIV,
                ast.Pow: BinaryOperator.POW,
            }
            return BinaryOp(self._lower_expr(expr.left), op_map[type(expr.op)], self._lower_expr(expr.right))
        if isinstance(expr, ast.UnaryOp):
            op_map = {
                ast.UAdd: UnaryOperator.PLUS,
                ast.USub: UnaryOperator.MINUS,
                ast.Not: UnaryOperator.NOT,
            }
            return UnaryOp(op_map[type(expr.op)], self._lower_expr(expr.operand))
        if isinstance(expr, ast.Compare):
            op_map = {
                ast.Eq: CompareOperator.EQ,
                ast.NotEq: CompareOperator.NE,
                ast.Lt: CompareOperator.LT,
                ast.LtE: CompareOperator.LE,
                ast.Gt: CompareOperator.GT,
                ast.GtE: CompareOperator.GE,
            }
            if len(expr.ops) > 1:
                left = expr.left
                comparisons: list[object] = []
                for op, comparator in zip(expr.ops, expr.comparators, strict=True):
                    if isinstance(op, ast.NotIn) and isinstance(comparator, ast.Set):
                        comparisons.extend(
                            Compare(self._lower_expr(left), CompareOperator.NE, self._lower_expr(elt))
                            for elt in comparator.elts
                        )
                    else:
                        comparisons.append(Compare(self._lower_expr(left), op_map[type(op)], self._lower_expr(comparator)))
                    left = comparator
                return BooleanOp("and", tuple(comparisons))
            op = expr.ops[0]
            if isinstance(op, ast.NotIn) and isinstance(expr.comparators[0], ast.Set):
                terms = [
                    Compare(self._lower_expr(expr.left), CompareOperator.NE, self._lower_expr(elt))
                    for elt in expr.comparators[0].elts
                ]
                return BooleanOp("and", tuple(terms))
            return Compare(self._lower_expr(expr.left), op_map[type(op)], self._lower_expr(expr.comparators[0]))
        if isinstance(expr, ast.BoolOp):
            op = "and" if isinstance(expr.op, ast.And) else "or"
            return BooleanOp(op, tuple(self._lower_expr(value) for value in expr.values))
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                if expr.func.id in {"abs", "max", "min"}:
                    return Call(expr.func.id, self._lower_call_args(expr))
                return Call(expr.func.id, self._lower_call_args(expr))
            if isinstance(expr.func, ast.Attribute):
                if self._is_strip_lower_chain(expr.func):
                    base = self._lower_expr(expr.func.value.func.value)
                    return Call("ac_lower", (Call("trim", (Call("adjustl", (base,)),)),))
                if expr.func.attr == "strip":
                    base = self._lower_expr(expr.func.value)
                    return Call("trim", (Call("adjustl", (base,)),))
                if expr.func.attr == "gauss":
                    base = self._lower_expr(expr.func.value)
                    return Call("ac_gauss", (base, *self._lower_call_args(expr)))
                chain = self._attr_chain(expr.func)
                if chain == ["random", "Random"]:
                    return Call("ac_random_init", self._lower_call_args(expr))
                if chain[:1] == ["math"]:
                    return Call(chain[1], self._lower_call_args(expr))
            raise NotImplementedError(f"unsupported call: {ast.dump(expr)}")
        raise NotImplementedError(f"unsupported expression: {ast.dump(expr)}")

    def _lower_call_args(self, expr: ast.Call) -> tuple[object, ...]:
        values = [self._lower_expr(arg) for arg in expr.args]
        values.extend(self._lower_expr(keyword.value) for keyword in expr.keywords)
        return tuple(values)

    def _infer_type(self, expr: ast.AST) -> object:
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return ScalarType.LOGICAL
            if isinstance(expr.value, int):
                return ScalarType.INTEGER
            if isinstance(expr.value, float):
                return ScalarType.REAL64
            if isinstance(expr.value, str):
                return ScalarType.STRING
        if isinstance(expr, (ast.Compare, ast.BoolOp)):
            return ScalarType.LOGICAL
        if isinstance(expr, ast.List):
            return ArrayTypeRef(ScalarType.REAL64)
        if isinstance(expr, ast.Subscript):
            return ScalarType.REAL64
        if isinstance(expr, ast.Dict):
            return self._function_result_types.get(self._current_function or "", ScalarType.REAL64)
        if isinstance(expr, ast.Tuple):
            return self._function_result_types.get(self._current_function or "", ScalarType.REAL64)
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Attribute) and self._is_strip_lower_chain(expr.func):
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "strip":
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "gauss":
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute):
                chain = self._attr_chain(expr.func)
                if chain == ["random", "Random"]:
                    return RecordTypeRef("ac_random_state")
                if chain[:1] == ["math"]:
                    return ScalarType.REAL64
            if isinstance(expr.func, ast.Name):
                return self._function_result_types.get(expr.func.id, ScalarType.REAL64)
            return ScalarType.REAL64
        if isinstance(expr, ast.Name):
            return ScalarType.REAL64
        if isinstance(expr, ast.UnaryOp):
            return ScalarType.LOGICAL if isinstance(expr.op, ast.Not) else ScalarType.REAL64
        if isinstance(expr, ast.BinOp):
            return ScalarType.REAL64
        return ScalarType.REAL64

    def _tuple_item_type(self, record_type: object, index: int) -> object:
        if isinstance(record_type, RecordTypeRef):
            record = self._record_defs.get(record_type.name)
            if record is not None and 1 <= index <= len(record.fields):
                return record.fields[index - 1][1]
        return ScalarType.REAL64

    def _map_annotation(self, annotation: ast.AST | None, function_name: str) -> object | None:
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            mapping = {
                "float": ScalarType.REAL64,
                "int": ScalarType.INTEGER,
                "bool": ScalarType.LOGICAL,
                "str": ScalarType.STRING,
            }
            return mapping.get(annotation.id)
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return None
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            if annotation.value.id == "tuple":
                fields = self._tuple_fields_from_annotation(annotation.slice)
                name = f"{function_name}_result"
                return self._ensure_record(name, fields)
            if annotation.value.id == "dict":
                name = f"{function_name}_result"
                return self._ensure_record(name, [])
        return None

    def _tuple_fields_from_annotation(self, node: ast.AST) -> list[tuple[str, object]]:
        if isinstance(node, ast.Tuple):
            return [(f"item{index}", self._map_annotation(elt, self._current_function or "tuple") or ScalarType.REAL64) for index, elt in enumerate(node.elts, start=1)]
        return []

    def _ensure_record(self, name: str, fields: list[tuple[str, object]]) -> RecordTypeRef:
        record = self._record_defs.get(name)
        if record is None:
            record = RecordDef(name=name, fields=list(fields))
            self._record_defs[name] = record
        elif fields and not record.fields:
            record.fields.extend(fields)
        return RecordTypeRef(name)

    def _tuple_temp_name(self, function_name: str) -> str:
        return f"{function_name}_value"

    def _loop_target_name(self, name: str) -> str:
        if name == "_":
            return "ac_loop_index"
        return name

    def _is_docstring(self, stmt: ast.stmt) -> bool:
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    def _is_main_guard(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        )

    def _is_strip_lower_chain(self, node: ast.Attribute) -> bool:
        return (
            node.attr == "lower"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "strip"
        )

    def _attr_chain(self, node: ast.Attribute) -> list[str]:
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return list(reversed(parts))
