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
    ConditionalExpr,
    Compare,
    CompareOperator,
    Constant,
    Continue,
    ExprStatement,
    FieldAccess,
    ForRange,
    Function,
    If,
    IndexAssignment,
    IndexAccess,
    ListComprehension,
    Pass,
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
    _function_arg_defaults: dict[str, list[object | None]] = field(default_factory=dict, init=False)
    _record_defs: dict[str, RecordDef] = field(default_factory=dict, init=False)
    _current_function: str | None = field(default=None, init=False)
    _current_symbols: dict[str, object] = field(default_factory=dict, init=False)

    def lower_source(self, source: str, module_name: str = "translated_module") -> Module:
        if source.startswith("\ufeff"):
            source = source.lstrip("\ufeff")
        tree = ast.parse(source)
        self._function_result_types = {}
        self._function_arg_defaults = {}
        self._record_defs = {}
        self._current_function = None
        self._current_symbols = {}

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
        if result_type is None:
            result_type = self._infer_result_type_from_returns(node)
        if result_type is not None:
            self._function_result_types[node.name] = result_type
        defaults: list[object | None] = [None] * len(node.args.args)
        if node.args.defaults:
            offset = len(node.args.args) - len(node.args.defaults)
            for idx, default in enumerate(node.args.defaults, start=offset):
                defaults[idx] = self._lower_expr(default)
        self._function_arg_defaults[node.name] = defaults

    def _infer_result_type_from_returns(self, node: ast.FunctionDef) -> object | None:
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                if isinstance(stmt.value, ast.Tuple):
                    fields = [(f"item{index}", self._infer_type(value)) for index, value in enumerate(stmt.value.elts, start=1)]
                    return self._ensure_record(f"{node.name}_result", fields)
                if isinstance(stmt.value, ast.Dict):
                    return self._ensure_record(f"{node.name}_result", [])
        return None

    def _lower_function(self, node: ast.FunctionDef) -> Function:
        self._current_function = node.name
        locals_map: dict[str, object] = {}
        arg_types = self._refine_arg_types(node)
        args = [(arg.arg, self._map_annotation(arg.annotation, node.name) or arg_types.get(arg.arg, ScalarType.REAL64)) for arg in node.args.args]
        self._current_symbols = {name: typ for name, typ in args}
        for stmt in node.body:
            self._collect_locals(stmt, locals_map)
        arg_names = {name for name, _ in args}
        locals_list = [(name, typ) for name, typ in locals_map.items() if name not in arg_names]
        self._current_symbols.update(locals_list)
        body = [self._lower_stmt(stmt) for stmt in node.body if not self._is_docstring(stmt)]
        result_type = self._refine_function_result_type(node.name, body)
        function = Function(
            name=node.name,
            args=args,
            locals=locals_list,
            body=body,
            result_type=result_type,
        )
        self._current_function = None
        self._current_symbols = {}
        return function

    def _lower_main_guard(self, node: ast.If) -> Program:
        locals_map: dict[str, object] = {}
        for stmt in node.body:
            self._collect_locals(stmt, locals_map)
        body = [self._lower_stmt(stmt) for stmt in node.body]
        return Program(name="run_main", locals=list(locals_map.items()), body=body)

    def _refine_arg_types(self, node: ast.FunctionDef) -> dict[str, object]:
        arg_types: dict[str, object] = {}
        aliases: dict[str, str] = {}

        for arg, default in zip(node.args.args, [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults), strict=True):
            annotated = self._map_annotation(arg.annotation, node.name)
            if annotated is not None:
                arg_types[arg.arg] = annotated
            elif isinstance(default, ast.Constant) and isinstance(default.value, int):
                arg_types[arg.arg] = ScalarType.INTEGER

        changed = True
        while changed:
            changed = False
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Tuple)
                    and isinstance(stmt.value, ast.Attribute)
                    and stmt.value.attr == "shape"
                    and isinstance(stmt.value.value, ast.Name)
                ):
                    if self._set_array_rank(arg_types, aliases, stmt.value.value.id, len(stmt.targets[0].elts)):
                        changed = True
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    target_name = stmt.targets[0].id
                    if (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Attribute)
                        and self._attr_chain(stmt.value.func) == ["np", "asarray"]
                        and stmt.value.args
                        and isinstance(stmt.value.args[0], ast.Name)
                    ):
                        aliases[target_name] = stmt.value.args[0].id
                if isinstance(stmt, ast.Attribute) and isinstance(stmt.value, ast.Name):
                    if stmt.attr == "ndim":
                        name = stmt.value.id
                        rank = self._rank_from_ndim_compare(stmt, node)
                        if rank is not None:
                            if self._set_array_rank(arg_types, aliases, name, rank):
                                changed = True
                    if stmt.attr == "shape":
                        pass
                if isinstance(stmt, ast.Subscript) and self._is_shape_dim_access(stmt):
                    name = stmt.value.value.id
                    rank = self._const_int_value(stmt.slice) + 1
                    if self._set_array_rank(arg_types, aliases, name, rank):
                        changed = True
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                    chain = self._attr_chain(stmt.func)
                    if chain == ["np", "savetxt"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        if arg_types.get(stmt.args[0].id) != ScalarType.STRING:
                            arg_types[stmt.args[0].id] = ScalarType.STRING
                            changed = True
                    if chain == ["np", "random", "default_rng"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        if arg_types.get(stmt.args[0].id) != ScalarType.INTEGER:
                            arg_types[stmt.args[0].id] = ScalarType.INTEGER
                            changed = True
                    if stmt.func.attr == "choice":
                        size_expr = self._keyword_value(stmt, "size")
                        if isinstance(size_expr, ast.Name) and arg_types.get(size_expr.id) != ScalarType.INTEGER:
                            arg_types[size_expr.id] = ScalarType.INTEGER
                            changed = True
                        if stmt.args and isinstance(stmt.args[0], ast.Name) and stmt.func.value.__class__ is ast.Attribute:
                            pass
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name) and stmt.func.id == "range":
                    for arg in stmt.args:
                        if isinstance(arg, ast.Name) and arg_types.get(arg.id) != ScalarType.INTEGER:
                            arg_types[arg.id] = ScalarType.INTEGER
                            changed = True
        return arg_types

    def _set_array_rank(self, arg_types: dict[str, object], aliases: dict[str, str], name: str, rank: int) -> bool:
        changed = False
        for candidate in {name, aliases.get(name, "")}:
            if not candidate:
                continue
            current = arg_types.get(candidate)
            desired = ArrayTypeRef(ScalarType.REAL64, rank=rank)
            if not isinstance(current, ArrayTypeRef) or current.rank < rank:
                arg_types[candidate] = desired
                changed = True
        return changed

    def _rank_from_ndim_compare(self, attr_node: ast.Attribute, node: ast.FunctionDef) -> int | None:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Compare) and inner.left is attr_node and inner.comparators:
                value = self._const_int_value(inner.comparators[0])
                if value is not None:
                    return value
            if isinstance(inner, ast.Compare) and inner.comparators and inner.comparators[0] is attr_node:
                value = self._const_int_value(inner.left)
                if value is not None:
                    return value
        return None

    def _refine_function_result_type(self, function_name: str, body: list[object]) -> object | None:
        result_type = self._function_result_types.get(function_name)
        if not isinstance(result_type, RecordTypeRef):
            return result_type
        for stmt in body:
            if isinstance(stmt, Return) and isinstance(stmt.value, RecordLiteral):
                fields = [(name, self._infer_ir_type(value)) for name, value in stmt.value.fields]
                self._ensure_record(result_type.name, fields)
                return result_type
        return result_type

    def _infer_ir_type(self, expr: object) -> object:
        if isinstance(expr, ValueRef):
            return self._current_symbols.get(expr.name, ScalarType.REAL64)
        if isinstance(expr, Constant):
            if isinstance(expr.value, int):
                return ScalarType.INTEGER
            if isinstance(expr.value, float):
                return ScalarType.REAL64
            if isinstance(expr.value, str):
                return ScalarType.STRING
        if isinstance(expr, Call):
            if expr.func == "ac_empty2":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_where":
                return ArrayTypeRef(ScalarType.INTEGER)
            if expr.func == "ac_multivariate_normal":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_choice_weighted":
                return ArrayTypeRef(ScalarType.INTEGER)
            if expr.func == "ac_asarray":
                return self._infer_ir_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
        if isinstance(expr, FieldAccess):
            base_type = self._infer_ir_type(expr.value)
            if isinstance(base_type, RecordTypeRef):
                return self._tuple_item_type(base_type, int(expr.field.removeprefix("item")))
        return ScalarType.REAL64

    def _collect_locals(self, stmt: ast.stmt, locals_map: dict[str, object]) -> None:
        if isinstance(stmt, ast.Assign):
            inferred = self._infer_type(stmt.value)
            self._collect_expr_locals(stmt.value, locals_map)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    locals_map.setdefault(target.id, inferred)
                    self._current_symbols.setdefault(target.id, inferred)
                elif isinstance(target, ast.Tuple):
                    if isinstance(stmt.value, ast.Attribute) and stmt.value.attr == "shape":
                        record_type = None
                    elif self._is_slogdet_call(stmt.value):
                        record_type = None
                    else:
                        record_type = self._infer_type(stmt.value)
                    if isinstance(record_type, RecordTypeRef):
                        locals_map.setdefault(self._tuple_temp_name(self._tuple_source_name(stmt.value)), record_type)
                    for index, elt in enumerate(target.elts, start=1):
                        if isinstance(elt, ast.Name):
                            locals_map.setdefault(elt.id, self._tuple_item_type(record_type if isinstance(record_type, RecordTypeRef) else inferred, index))
                            self._current_symbols.setdefault(elt.id, self._tuple_item_type(record_type if isinstance(record_type, RecordTypeRef) else inferred, index))
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            self._collect_expr_locals(stmt.value, locals_map)
            locals_map.setdefault(stmt.target.id, self._infer_type(stmt.value))
            self._current_symbols.setdefault(stmt.target.id, self._infer_type(stmt.value))
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            locals_map.setdefault(self._loop_target_name(stmt.target.id), ScalarType.INTEGER)
            self._current_symbols.setdefault(self._loop_target_name(stmt.target.id), ScalarType.INTEGER)
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Tuple):
            for elt in stmt.target.elts:
                if isinstance(elt, ast.Name):
                    locals_map.setdefault(elt.id, ScalarType.REAL64)
                    self._current_symbols.setdefault(elt.id, ScalarType.REAL64)
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)
        elif isinstance(stmt, ast.If):
            self._collect_expr_locals(stmt.test, locals_map)
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)

    def _collect_expr_locals(self, expr: ast.AST, locals_map: dict[str, object]) -> None:
        for node in ast.walk(expr):
            if isinstance(node, ast.ListComp):
                if len(node.generators) == 1 and isinstance(node.generators[0].target, ast.Name):
                    gen = node.generators[0]
                    if not (isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range"):
                        index_name = f"{gen.target.id}_index"
                        locals_map.setdefault(index_name, ScalarType.INTEGER)
                        self._current_symbols.setdefault(index_name, ScalarType.INTEGER)

    def _lower_stmt(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise NotImplementedError(f"unsupported assignment: {ast.dump(stmt)}")
            target = stmt.targets[0]
            value = self._lower_expr(stmt.value)
            if isinstance(target, ast.Name):
                return Assignment(ValueRef(target.id), value)
            if isinstance(target, ast.Subscript):
                slice_assign = self._lower_slice_assignment(target, value)
                if slice_assign is not None:
                    return slice_assign
                if self._is_row_assignment(target):
                    func_name = "ac_set_rows" if isinstance(self._infer_type(target.slice), ArrayTypeRef) else "ac_set_row"
                    return ExprStatement(Call(func_name, (self._lower_expr(target.value), self._lower_expr(target.slice), value)))
                return IndexAssignment(self._lower_expr(target.value), self._lower_expr(target.slice), value)
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
            if self._is_sys_argv_if(stmt):
                return Pass()
            return If(
                test=self._lower_expr(stmt.test),
                body=tuple(self._lower_stmt(inner) for inner in stmt.body),
                orelse=tuple(self._lower_stmt(inner) for inner in stmt.orelse),
            )
        if isinstance(stmt, ast.For):
            if (
                isinstance(stmt.target, ast.Tuple)
                and len(stmt.target.elts) == 2
                and all(isinstance(elt, ast.Name) for elt in stmt.target.elts)
                and isinstance(stmt.iter, ast.Call)
                and isinstance(stmt.iter.func, ast.Name)
                and stmt.iter.func.id == "enumerate"
                and len(stmt.iter.args) == 1
            ):
                index_name = self._loop_target_name(stmt.target.elts[0].id)
                value_name = stmt.target.elts[1].id
                iterable_expr = self._lower_expr(stmt.iter.args[0])
                body = [Assignment(ValueRef(value_name), IndexAccess(iterable_expr, ValueRef(index_name)))]
                body.extend(self._lower_stmt(inner) for inner in stmt.body)
                return ForRange(
                    target=index_name,
                    start=Constant(0),
                    stop=Call("size", (iterable_expr,)),
                    body=tuple(body),
                )
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
        if isinstance(stmt, ast.Continue):
            return Continue()
        if isinstance(stmt, ast.Pass):
            return Pass()
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
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            statements: list[object] = []
            for index, elt in enumerate(target.elts, start=1):
                if not isinstance(elt, ast.Name):
                    raise NotImplementedError(f"unsupported tuple target: {ast.dump(elt)}")
                statements.append(
                    Assignment(
                        ValueRef(elt.id),
                        Call("ac_shape_dim", (self._lower_expr(value.value), Constant(index))),
                    )
                )
            return If(Constant(True), tuple(statements), ())
        if self._is_slogdet_call(value):
            if len(target.elts) != 2 or not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise NotImplementedError("slogdet tuple assignment requires two simple targets")
            matrix_expr = self._lower_expr(value.args[0])
            statements = [
                Assignment(ValueRef(target.elts[0].id), Call("ac_slogdet_sign", (matrix_expr,))),
                Assignment(ValueRef(target.elts[1].id), Call("ac_slogdet_logabsdet", (matrix_expr,))),
            ]
            return If(Constant(True), tuple(statements), ())
        lowered_value = self._lower_expr(value)
        record_type = self._infer_type(value)
        if not isinstance(record_type, RecordTypeRef):
            raise NotImplementedError("tuple assignment requires a structured-return value")
        temp_name = self._tuple_temp_name(self._tuple_source_name(value))
        statements: list[object] = [Assignment(ValueRef(temp_name), lowered_value)]
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
            elif isinstance(arg, ast.Attribute) and arg.attr == "shape":
                rank = self._infer_array_rank(arg.value)
                values.extend(
                    Call("ac_shape_dim", (self._lower_expr(arg.value), Constant(index)))
                    for index in range(1, rank + 1)
                )
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
            return ValueRef(self._loop_target_name(expr.id) if expr.id == "_" else expr.id)
        if isinstance(expr, ast.List):
            return Call("ac_array_literal", tuple(self._lower_expr(elt) for elt in expr.elts))
        if isinstance(expr, ast.ListComp):
            if len(expr.generators) != 1:
                raise NotImplementedError("only single-generator list comprehensions are supported")
            gen = expr.generators[0]
            if gen.ifs:
                raise NotImplementedError("list comprehension filters are not yet supported")
            if not isinstance(gen.target, ast.Name):
                raise NotImplementedError("only simple list comprehension targets are supported")
            return ListComprehension(gen.target.id, self._lower_expr(gen.iter), self._lower_expr(expr.elt))
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
            if self._is_shape_dim_access(expr):
                return Call("ac_shape_dim", (self._lower_expr(expr.value.value), Constant(self._const_int_value(expr.slice) + 1)))
            if self._is_where_first_index(expr):
                return self._lower_expr(expr.value)
            slice_expr = self._lower_slice_expr(expr)
            if slice_expr is not None:
                return slice_expr
            if self._infer_type(expr.value).__class__ is RecordTypeRef:
                tuple_index = self._const_int_value(expr.slice)
                if tuple_index is not None:
                    return FieldAccess(self._lower_expr(expr.value), f"item{tuple_index + 1}")
            if isinstance(expr.value, ast.Attribute) and expr.value.attr == "shape":
                shape_index = self._const_int_value(expr.slice)
                if shape_index is not None:
                    return Call("ac_shape_dim", (self._lower_expr(expr.value.value), Constant(shape_index + 1)))
            if isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, str):
                return FieldAccess(self._lower_expr(expr.value), expr.slice.value)
            if self._is_row_gather(expr):
                return Call("ac_take_rows", (self._lower_expr(expr.value), self._lower_expr(expr.slice)))
            if self._is_row_access(expr):
                return Call("ac_row", (self._lower_expr(expr.value), self._lower_expr(expr.slice)))
            return IndexAccess(self._lower_expr(expr.value), self._lower_expr(expr.slice))
        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, ast.MatMult):
                return Call("ac_matmul", (self._lower_expr(expr.left), self._lower_expr(expr.right)))
            if isinstance(expr.op, ast.Sub) and self._is_columnwise_binary(expr.left, expr.right):
                return Call("ac_sub_col", (self._lower_expr(expr.left), self._lower_columnwise_arg(expr.right)))
            if isinstance(expr.op, ast.Mult) and self._is_columnwise_binary(expr.left, expr.right):
                return Call("ac_mul_col", (self._lower_expr(expr.left), self._lower_columnwise_arg(expr.right)))
            if isinstance(expr.op, ast.Sub) and self._is_rowwise_binary(expr.left, expr.right):
                return Call("ac_sub_row", (self._lower_expr(expr.left), self._lower_expr(expr.right)))
            if isinstance(expr.op, ast.Mult) and isinstance(expr.left, ast.List) and len(expr.left.elts) == 1:
                return Call("ac_repeat", (self._lower_expr(expr.left.elts[0]), self._lower_expr(expr.right)))
            if isinstance(expr.op, ast.Add) and (
                self._is_list_concat_expr(expr.left) or self._is_list_concat_expr(expr.right)
            ):
                return Call("ac_concat", (self._lower_expr(expr.left), self._lower_expr(expr.right)))
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
        if isinstance(expr, ast.IfExp):
            if self._is_sys_argv_fallback(expr):
                return self._lower_expr(expr.orelse)
            return ConditionalExpr(self._lower_expr(expr.test), self._lower_expr(expr.body), self._lower_expr(expr.orelse))
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                if expr.func.id == "range":
                    return Call("range", self._lower_call_args(expr))
                if expr.func.id == "len":
                    return Call("size", self._lower_call_args(expr))
                if expr.func.id == "int":
                    return Call("int", self._lower_call_args(expr))
                if expr.func.id == "float":
                    return Call("float", self._lower_call_args(expr))
                if expr.func.id == "enumerate":
                    return Call("ac_enumerate", self._lower_call_args(expr))
                if expr.func.id in {"abs", "max", "min"}:
                    return Call(expr.func.id, self._lower_call_args(expr))
                return Call(expr.func.id, self._lower_call_args(expr))
            if isinstance(expr.func, ast.Attribute):
                chain = self._attr_chain(expr.func)
                if self._is_strip_lower_chain(expr.func):
                    base = self._lower_expr(expr.func.value.func.value)
                    return Call("ac_lower", (Call("trim", (Call("adjustl", (base,)),)),))
                if expr.func.attr == "strip":
                    base = self._lower_expr(expr.func.value)
                    return Call("trim", (Call("adjustl", (base,)),))
                if expr.func.attr == "gauss" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    return Call("ac_gauss", (base, *self._lower_call_args(expr)))
                if expr.func.attr == "sum" and chain[:1] != ["np"]:
                    return Call("sum", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "copy" and chain[:1] != ["np"]:
                    return Call("ac_copy", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "reshape" and chain[:1] != ["np"]:
                    return Call("ac_reshape", (self._lower_expr(expr.func.value), *self._lower_call_args(expr)))
                if expr.func.attr == "choice" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    size_expr = self._keyword_value(expr, "size")
                    if size_expr is None:
                        raise NotImplementedError("rng.choice without size is not yet supported")
                    replace_expr = self._keyword_value(expr, "replace")
                    p_expr = self._keyword_value(expr, "p")
                    if p_expr is not None:
                        return Call(
                            "ac_choice_weighted",
                            (
                                base,
                                self._lower_expr(expr.args[0]),
                                self._lower_expr(size_expr),
                                self._lower_expr(p_expr),
                            ),
                        )
                    if replace_expr is not None and isinstance(replace_expr, ast.Constant) and replace_expr.value is False:
                        return Call(
                            "ac_choice_no_replace",
                            (
                                base,
                                self._lower_expr(expr.args[0]),
                                self._lower_expr(size_expr),
                            ),
                        )
                    raise NotImplementedError("only weighted choice and choice without replacement are supported")
                if chain == ["np", "random", "default_rng"]:
                    return Call("ac_random_init", self._lower_call_args(expr))
                if chain == ["np", "random", "multivariate_normal"]:
                    return Call("ac_multivariate_normal", self._lower_call_args(expr))
                if chain == ["np", "savetxt"]:
                    return Call("ac_savetxt", self._lower_call_args(expr))
                if chain == ["np", "loadtxt"]:
                    return Call("ac_loadtxt", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "cov"]:
                    rowvar_expr = self._keyword_value(expr, "rowvar")
                    if isinstance(rowvar_expr, ast.Constant) and rowvar_expr.value is False:
                        return Call("ac_cov_rowvar_false", tuple(self._lower_expr(arg) for arg in expr.args))
                    raise NotImplementedError("only np.cov(..., rowvar=False) is currently supported")
                if chain == ["np", "linalg", "slogdet"]:
                    return Call("ac_slogdet", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "linalg", "inv"]:
                    return Call("ac_inv", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "einsum"]:
                    if (
                        expr.args
                        and isinstance(expr.args[0], ast.Constant)
                        and expr.args[0].value == "ni,ij,nj->n"
                    ):
                        return Call("ac_einsum_ni_ij_nj_to_n", tuple(self._lower_expr(arg) for arg in expr.args[1:]))
                    raise NotImplementedError("only np.einsum('ni,ij,nj->n', ...) is currently supported")
                if chain == ["np", "max"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_max_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("max", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "sum"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_sum_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("sum", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "argsort"]:
                    return Call("ac_argsort", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "ndim"]:
                    return Call("ac_ndim", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "eye"]:
                    return Call("ac_eye", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "atleast_2d"]:
                    return Call("ac_atleast_2d", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "log"]:
                    return Call("log", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "exp"]:
                    return Call("exp", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "array"]:
                    return Call("ac_array", self._lower_call_args(expr))
                if chain == ["np", "asarray"]:
                    return Call("ac_asarray", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "where"]:
                    return Call("ac_where", self._lower_call_args(expr))
                if chain == ["np", "empty"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple) and len(expr.args[0].elts) == 2:
                        return Call("ac_empty2", tuple(self._lower_expr(elt) for elt in expr.args[0].elts))
                    return Call("ac_empty", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "full"]:
                    return Call("ac_full", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["random", "Random"]:
                    return Call("ac_random_init", self._lower_call_args(expr))
                if chain[:1] == ["math"]:
                    return Call(chain[1], self._lower_call_args(expr))
            raise NotImplementedError(f"unsupported call: {ast.dump(expr)}")
        if isinstance(expr, ast.Attribute):
            if self._attr_chain(expr) == ["np", "inf"]:
                return Constant(float("inf"))
            if self._attr_chain(expr) == ["np", "pi"]:
                return Constant(3.141592653589793)
            if self._attr_chain(expr) == ["sys", "argv"]:
                return ValueRef("ac_argv")
            base = self._lower_expr(expr.value)
            if expr.attr == "T":
                return Call("ac_transpose", (base,))
            if expr.attr == "size":
                return Call("size", (base,))
            if expr.attr == "ndim":
                return Call("ac_ndim", (base,))
            if expr.attr == "shape":
                return Call("ac_shape", (base,))
        raise NotImplementedError(f"unsupported expression: {ast.dump(expr)}")

    def _lower_call_args(self, expr: ast.Call) -> tuple[object, ...]:
        values = [self._lower_expr(arg) for arg in expr.args]
        values.extend(self._lower_expr(keyword.value) for keyword in expr.keywords)
        if isinstance(expr.func, ast.Name) and expr.func.id in self._function_arg_defaults:
            defaults = self._function_arg_defaults[expr.func.id]
            for index in range(len(values), len(defaults)):
                default = defaults[index]
                if default is not None:
                    values.append(default)
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
        if isinstance(expr, ast.ListComp):
            return ArrayTypeRef(ScalarType.REAL64)
        if isinstance(expr, ast.Subscript):
            if self._is_where_first_index(expr):
                return ArrayTypeRef(ScalarType.INTEGER)
            if self._is_shape_dim_access(expr):
                return ScalarType.INTEGER
            slice_type = self._infer_slice_type(expr)
            if slice_type is not None:
                return slice_type
            if isinstance(expr.value, ast.Attribute) and expr.value.attr == "shape":
                return ScalarType.INTEGER
            container_type = self._infer_type(expr.value)
            if isinstance(container_type, RecordTypeRef):
                tuple_index = self._const_int_value(expr.slice)
                if tuple_index is not None:
                    return self._tuple_item_type(container_type, tuple_index + 1)
            if isinstance(container_type, ArrayTypeRef):
                if self._is_row_gather(expr):
                    return ArrayTypeRef(container_type.element_type, rank=container_type.rank)
                if container_type.rank > 1 and not isinstance(expr.slice, ast.Tuple):
                    return ArrayTypeRef(container_type.element_type, rank=container_type.rank - 1)
                return container_type.element_type
            return ScalarType.REAL64
        if isinstance(expr, ast.Dict):
            return self._function_result_types.get(self._current_function or "", ScalarType.REAL64)
        if isinstance(expr, ast.Tuple):
            return self._function_result_types.get(self._current_function or "", ScalarType.REAL64)
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id == "len":
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id == "int":
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id == "float":
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id in {"size", "ac_ndim"}:
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_shape":
                return self._shape_record_type(expr.args[0] if expr.args else None)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_shape_dim":
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id in {"min", "max"} and expr.args:
                arg_types = [self._infer_type(arg) for arg in expr.args]
                if all(arg_type == ScalarType.INTEGER for arg_type in arg_types):
                    return ScalarType.INTEGER
                if any(isinstance(arg_type, ArrayTypeRef) for arg_type in arg_types):
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            chain = self._attr_chain(expr.func) if isinstance(expr.func, ast.Attribute) else []
            if isinstance(expr.func, ast.Attribute) and self._is_strip_lower_chain(expr.func):
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "strip":
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "gauss" and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "sum" and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "copy" and chain[:1] != ["np"]:
                return self._infer_type(expr.func.value)
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "reshape" and chain[:1] != ["np"]:
                return ArrayTypeRef(ScalarType.REAL64, rank=self._reshape_rank(expr))
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "choice" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is not None:
                    return ArrayTypeRef(ScalarType.INTEGER)
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Attribute):
                if chain == ["np", "random", "default_rng"]:
                    return RecordTypeRef("ac_random_state")
                if chain == ["np", "random", "multivariate_normal"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "loadtxt"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "cov"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "linalg", "slogdet"]:
                    return self._ensure_record("ac_slogdet_result", [("item1", ScalarType.REAL64), ("item2", ScalarType.REAL64)])
                if chain == ["np", "linalg", "inv"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "einsum"]:
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "max"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "sum"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "argsort"]:
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "ndim"]:
                    return ScalarType.INTEGER
                if chain == ["np", "eye"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "atleast_2d"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "log"] or chain == ["np", "exp"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "array"] or chain == ["np", "asarray"]:
                    if expr.args:
                        return ArrayTypeRef(ScalarType.REAL64, rank=self._infer_array_rank(expr.args[0]))
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "where"]:
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "empty"] or chain == ["np", "full"]:
                    if expr.args:
                        return ArrayTypeRef(ScalarType.REAL64, rank=self._infer_shape_arg_rank(expr.args[0]))
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["random", "Random"]:
                    return RecordTypeRef("ac_random_state")
                if chain[:1] == ["math"]:
                    return ScalarType.REAL64
            if isinstance(expr.func, ast.Name):
                return self._function_result_types.get(expr.func.id, ScalarType.REAL64)
            return ScalarType.REAL64
        if isinstance(expr, ast.Name):
            return self._current_symbols.get(expr.id, ScalarType.REAL64)
        if isinstance(expr, ast.Attribute):
            if self._attr_chain(expr) == ["np", "inf"]:
                return ScalarType.REAL64
            if self._attr_chain(expr) == ["np", "pi"]:
                return ScalarType.REAL64
            if self._attr_chain(expr) == ["sys", "argv"]:
                return ArrayTypeRef(ScalarType.STRING)
            if expr.attr == "T":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.attr in {"size", "ndim"}:
                return ScalarType.INTEGER
            if expr.attr == "shape":
                return self._shape_record_type(expr.value)
        if isinstance(expr, ast.UnaryOp):
            return ScalarType.LOGICAL if isinstance(expr.op, ast.Not) else ScalarType.REAL64
        if isinstance(expr, ast.BinOp):
            left_type = self._infer_type(expr.left)
            right_type = self._infer_type(expr.right)
            if isinstance(expr.op, ast.MatMult):
                return self._infer_matmul_type(left_type, right_type)
            if isinstance(left_type, ArrayTypeRef) or isinstance(right_type, ArrayTypeRef):
                return self._combine_array_types(left_type, right_type)
            left_type = self._infer_type(expr.left)
            right_type = self._infer_type(expr.right)
            if left_type == ScalarType.INTEGER and right_type == ScalarType.INTEGER:
                return ScalarType.INTEGER
            if isinstance(expr.op, ast.Mult) and isinstance(expr.left, ast.List):
                return ArrayTypeRef(ScalarType.REAL64)
            return ScalarType.REAL64
        if isinstance(expr, ast.IfExp):
            return self._infer_type(expr.body)
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
            if annotation.value.id == "list":
                element_type = self._map_annotation(annotation.slice, function_name) or ScalarType.REAL64
                return ArrayTypeRef(element_type)
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
        elif fields:
            if not record.fields:
                record.fields.extend(fields)
            else:
                existing = {field_name: field_type for field_name, field_type in record.fields}
                changed = False
                for field_name, field_type in fields:
                    if existing.get(field_name) != field_type:
                        existing[field_name] = field_type
                        changed = True
                if changed:
                    ordered_names = [field_name for field_name, _ in record.fields]
                    for field_name, _ in fields:
                        if field_name not in ordered_names:
                            ordered_names.append(field_name)
                    record.fields = [(field_name, existing[field_name]) for field_name in ordered_names]
        return RecordTypeRef(name)

    def _tuple_temp_name(self, function_name: str) -> str:
        return f"{function_name}_value"

    def _tuple_source_name(self, expr: ast.AST) -> str:
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                return expr.func.id
            if isinstance(expr.func, ast.Attribute):
                return expr.func.attr
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return "tuple"

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

    def _infer_array_rank(self, expr: ast.AST) -> int:
        if isinstance(expr, ast.List):
            if expr.elts and isinstance(expr.elts[0], ast.List):
                return 2
            return 1
        if isinstance(expr, ast.Name):
            inferred = self._current_symbols.get(expr.id)
            if isinstance(inferred, ArrayTypeRef):
                return inferred.rank
        inferred = self._infer_type(expr)
        if isinstance(inferred, ArrayTypeRef):
            return inferred.rank
        return 1

    def _infer_shape_arg_rank(self, expr: ast.AST) -> int:
        if isinstance(expr, ast.Tuple):
            return len(expr.elts)
        if isinstance(expr, ast.List):
            return len(expr.elts)
        return 1

    def _reshape_rank(self, expr: ast.Call) -> int:
        if len(expr.args) == 1 and isinstance(expr.args[0], ast.Tuple):
            return len(expr.args[0].elts)
        if len(expr.args) >= 1:
            return len(expr.args)
        return 1

    def _shape_record_type(self, expr: ast.AST | None, minimum_rank: int = 1) -> RecordTypeRef:
        rank = self._infer_array_rank(expr) if expr is not None else minimum_rank
        rank = max(rank, minimum_rank)
        name = f"ac_shape_{rank}d"
        fields = [(f"item{index}", ScalarType.INTEGER) for index in range(1, rank + 1)]
        return self._ensure_record(name, fields)

    def _const_int_value(self, expr: ast.AST) -> int | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value
        return None

    def _keyword_value(self, expr: ast.Call, name: str) -> ast.AST | None:
        for keyword in expr.keywords:
            if keyword.arg == name:
                return keyword.value
        return None

    def _is_shape_dim_access(self, expr: ast.Subscript) -> bool:
        return (
            isinstance(expr.value, ast.Attribute)
            and expr.value.attr == "shape"
            and self._const_int_value(expr.slice) is not None
        )

    def _is_where_first_index(self, expr: ast.Subscript) -> bool:
        return (
            isinstance(expr.value, ast.Call)
            and isinstance(expr.value.func, ast.Attribute)
            and self._attr_chain(expr.value.func) == ["np", "where"]
            and self._const_int_value(expr.slice) == 0
        )

    def _is_row_access(self, expr: ast.Subscript) -> bool:
        container_type = self._infer_type(expr.value)
        index_type = self._infer_type(expr.slice)
        return (
            isinstance(container_type, ArrayTypeRef)
            and container_type.rank == 2
            and not isinstance(expr.slice, ast.Tuple)
            and not isinstance(index_type, ArrayTypeRef)
        )

    def _is_row_gather(self, expr: ast.Subscript) -> bool:
        container_type = self._infer_type(expr.value)
        index_type = self._infer_type(expr.slice)
        return (
            isinstance(container_type, ArrayTypeRef)
            and container_type.rank == 2
            and not isinstance(expr.slice, ast.Tuple)
            and isinstance(index_type, ArrayTypeRef)
        )

    def _is_row_assignment(self, target: ast.Subscript) -> bool:
        container_type = self._infer_type(target.value)
        return (
            isinstance(container_type, ArrayTypeRef)
            and container_type.rank == 2
            and not isinstance(target.slice, ast.Tuple)
        )

    def _is_list_concat_expr(self, expr: ast.AST) -> bool:
        if isinstance(expr, (ast.List, ast.ListComp)):
            return True
        return isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add) and (
            self._is_list_concat_expr(expr.left) or self._is_list_concat_expr(expr.right)
        )

    def _is_rowwise_binary(self, left: ast.AST, right: ast.AST) -> bool:
        left_type = self._infer_type(left)
        right_type = self._infer_type(right)
        return (
            isinstance(left_type, ArrayTypeRef)
            and left_type.rank == 2
            and isinstance(right_type, ArrayTypeRef)
            and right_type.rank == 1
        )

    def _is_columnwise_binary(self, left: ast.AST, right: ast.AST) -> bool:
        left_type = self._infer_type(left)
        return (
            isinstance(left_type, ArrayTypeRef)
            and left_type.rank == 2
            and self._extract_columnwise_source(right) is not None
        )

    def _extract_columnwise_source(self, expr: ast.AST) -> ast.AST | None:
        if isinstance(expr, ast.Subscript) and isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
            row_spec, col_spec = expr.slice.elts
            if (
                isinstance(row_spec, ast.Slice)
                and row_spec.lower is None
                and row_spec.upper is None
                and row_spec.step is None
                and isinstance(col_spec, ast.Constant)
                and col_spec.value is None
            ):
                return expr.value
        return None

    def _lower_columnwise_arg(self, expr: ast.AST) -> object:
        source = self._extract_columnwise_source(expr)
        if source is None:
            return self._lower_expr(expr)
        return self._lower_expr(source)

    def _combine_array_types(self, left_type: object, right_type: object) -> object:
        if isinstance(left_type, ArrayTypeRef) and isinstance(right_type, ArrayTypeRef):
            return ArrayTypeRef(ScalarType.REAL64, rank=max(left_type.rank, right_type.rank))
        if isinstance(left_type, ArrayTypeRef):
            return ArrayTypeRef(left_type.element_type, rank=left_type.rank)
        if isinstance(right_type, ArrayTypeRef):
            return ArrayTypeRef(right_type.element_type, rank=right_type.rank)
        return ScalarType.REAL64

    def _infer_matmul_type(self, left_type: object, right_type: object) -> object:
        if isinstance(left_type, ArrayTypeRef) and isinstance(right_type, ArrayTypeRef):
            if left_type.rank == 2 and right_type.rank == 2:
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if left_type.rank == 2 and right_type.rank == 1:
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
            if left_type.rank == 1 and right_type.rank == 2:
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
        return ArrayTypeRef(ScalarType.REAL64, rank=2)

    def _is_sys_argv_fallback(self, expr: ast.IfExp) -> bool:
        return (
            isinstance(expr.test, ast.Compare)
            and isinstance(expr.test.left, ast.Call)
            and isinstance(expr.test.left.func, ast.Name)
            and expr.test.left.func.id == "len"
            and len(expr.test.left.args) == 1
            and self._attr_chain(expr.test.left.args[0]) == ["sys", "argv"]
            and isinstance(expr.body, ast.Subscript)
            and isinstance(expr.body.value, ast.Attribute)
            and self._attr_chain(expr.body.value) == ["sys", "argv"]
            and isinstance(expr.orelse, ast.Constant)
            and isinstance(expr.orelse.value, str)
        )

    def _is_sys_argv_if(self, stmt: ast.If) -> bool:
        return (
            isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Call)
            and isinstance(stmt.test.left.func, ast.Name)
            and stmt.test.left.func.id == "len"
            and len(stmt.test.left.args) == 1
            and self._attr_chain(stmt.test.left.args[0]) == ["sys", "argv"]
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Constant)
            and stmt.test.comparators[0].value == 1
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Gt)
        )

    def _is_slogdet_call(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "linalg", "slogdet"]
            and len(expr.args) == 1
        )

    def _lower_slice_expr(self, expr: ast.Subscript) -> object | None:
        if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
            row_spec, col_spec = expr.slice.elts
            if isinstance(row_spec, ast.Slice) and row_spec.lower is None and row_spec.upper is None and row_spec.step is None:
                if isinstance(col_spec, ast.Constant) and col_spec.value is None:
                    return Call("ac_add_axis", (self._lower_expr(expr.value),))
                if isinstance(col_spec, ast.Name) or isinstance(col_spec, ast.Constant):
                    return Call("ac_column", (self._lower_expr(expr.value), self._lower_expr(col_spec)))
            if (
                isinstance(row_spec, ast.Slice)
                and row_spec.lower is None
                and row_spec.upper is None
                and row_spec.step is None
                and isinstance(col_spec, ast.Constant)
                and col_spec.value is None
            ):
                return Call("ac_add_axis", (self._lower_expr(expr.value),))
        return None

    def _lower_slice_assignment(self, target: ast.Subscript, value: object) -> object | None:
        if isinstance(target.slice, ast.Tuple) and len(target.slice.elts) == 2:
            row_spec, col_spec = target.slice.elts
            if (
                isinstance(row_spec, ast.Slice)
                and row_spec.lower is None
                and row_spec.upper is None
                and row_spec.step is None
                and (isinstance(col_spec, ast.Name) or isinstance(col_spec, ast.Constant))
            ):
                return ExprStatement(Call("ac_set_column", (self._lower_expr(target.value), self._lower_expr(col_spec), value)))
        return None

    def _infer_slice_type(self, expr: ast.Subscript) -> object | None:
        if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
            row_spec, col_spec = expr.slice.elts
            if isinstance(row_spec, ast.Slice) and row_spec.lower is None and row_spec.upper is None and row_spec.step is None:
                if isinstance(col_spec, (ast.Name, ast.Constant)) and not (isinstance(col_spec, ast.Constant) and col_spec.value is None):
                    return ArrayTypeRef(ScalarType.REAL64)
                if isinstance(col_spec, ast.Constant) and col_spec.value is None:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
        return None
