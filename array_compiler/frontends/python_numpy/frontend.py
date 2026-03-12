"""Minimal Python/NumPy frontend.

The first migration milestones cover typed scalar numeric scripts and small
structured return values suitable for option-pricing examples.
"""

from __future__ import annotations

import ast
import copy
import io
import tokenize
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
    Break,
    Call,
    Comment,
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
    While,
)


@dataclass
class PythonNumpyFrontend:
    _function_result_types: dict[str, object] = field(default_factory=dict, init=False)
    _function_arg_defaults: dict[str, list[object | None]] = field(default_factory=dict, init=False)
    _function_arg_types: dict[str, list[object]] = field(default_factory=dict, init=False)
    _function_arg_names: dict[str, list[str]] = field(default_factory=dict, init=False)
    _record_defs: dict[str, RecordDef] = field(default_factory=dict, init=False)
    _current_function: str | None = field(default=None, init=False)
    _current_symbols: dict[str, object] = field(default_factory=dict, init=False)
    _current_dtypes: dict[str, str] = field(default_factory=dict, init=False)
    _current_contiguity: dict[str, tuple[bool, bool]] = field(default_factory=dict, init=False)
    _source_lines: list[str] = field(default_factory=list, init=False)
    _comment_lines: dict[int, str] = field(default_factory=dict, init=False)
    _used_comment_lines: set[int] = field(default_factory=set, init=False)
    _function_nodes: dict[str, ast.FunctionDef] = field(default_factory=dict, init=False)

    def lower_source(self, source: str, module_name: str = "translated_module") -> Module:
        if source.startswith("\ufeff"):
            source = source.lstrip("\ufeff")
        self._source_lines = source.splitlines()
        self._comment_lines = self._extract_comment_lines(source)
        self._used_comment_lines = set()
        tree = ast.parse(source)
        self._function_result_types = {}
        self._function_arg_defaults = {}
        self._function_arg_types = {}
        self._function_arg_names = {}
        self._record_defs = {}
        self._function_nodes = {}
        self._current_function = None
        self._current_symbols = {}
        self._current_dtypes = {}
        self._current_contiguity = {}

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._register_function_signature(node)

        functions: list[Function] = []
        preamble_stmts: list[ast.stmt] = []
        main_guard: ast.If | None = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(self._lower_function(node))
            elif self._is_main_guard(node):
                main_guard = node
            elif not isinstance(node, (ast.Import, ast.ImportFrom)):
                preamble_stmts.append(node)
        program = None
        if preamble_stmts or main_guard is not None:
            program = self._lower_program(preamble_stmts, main_guard)
        exports = [fn.name for fn in functions]
        return Module(
            name=module_name,
            records=list(self._record_defs.values()),
            functions=functions,
            program=program,
            exports=exports,
            library_mode=True,
        )

    def _lower_program(self, preamble_stmts: list[ast.stmt], main_guard: ast.If | None) -> Program:
        locals_map: dict[str, object] = {}
        body: list[object] = []
        leading_comments: list[str] = []
        if preamble_stmts:
            leading_comments = self._leading_comments_after_line(0, preamble_stmts[0].lineno)
        for stmt in preamble_stmts:
            if self._is_docstring(stmt):
                continue
            self._collect_locals(stmt, locals_map)
            body.extend(Comment(text) for text in self._attached_comments_before_stmt(stmt, 0))
            body.extend(self._lower_stmt_list(stmt))
        if main_guard is not None:
            if not leading_comments and main_guard.body:
                leading_comments = self._leading_comments_after_line(main_guard.lineno, main_guard.body[0].lineno)
            for stmt in main_guard.body:
                self._collect_locals(stmt, locals_map)
                body.extend(Comment(text) for text in self._attached_comments_before_stmt(stmt, main_guard.lineno))
                body.extend(self._lower_stmt_list(stmt))
        return Program(name="run_main", locals=list(locals_map.items()), body=body, leading_comments=leading_comments)

    def _register_function_signature(self, node: ast.FunctionDef) -> None:
        self._function_nodes[node.name] = node
        result_type = self._map_annotation(node.returns, node.name)
        if result_type is None:
            result_type = self._infer_result_type_from_returns(node)
        if result_type is not None:
            self._function_result_types[node.name] = result_type
        self._function_arg_names[node.name] = [arg.arg for arg in node.args.args]
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
                return self._infer_type(stmt.value)
        return None

    def _infer_user_call_result_type(self, function_name: str, actual_args: list[ast.AST]) -> object | None:
        node = self._function_nodes.get(function_name)
        if node is None:
            return self._function_result_types.get(function_name)

        saved_symbols = self._current_symbols
        saved_function = self._current_function
        try:
            arg_types = list(self._function_arg_types.get(function_name, []))
            if not arg_types:
                arg_types = [self._map_annotation(arg.annotation, function_name) or ScalarType.REAL64 for arg in node.args.args]
            symbols: dict[str, object] = {}
            for index, arg in enumerate(node.args.args):
                actual_type = self._infer_type(actual_args[index]) if index < len(actual_args) else None
                formal_type = arg_types[index] if index < len(arg_types) else None
                symbols[arg.arg] = self._merge_types(formal_type, actual_type) or formal_type or actual_type or ScalarType.REAL64
            self._current_symbols = symbols
            self._current_function = function_name
            refined: object | None = self._function_result_types.get(function_name)
            for stmt in node.body:
                if self._is_docstring(stmt):
                    continue
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    inferred_value = self._infer_type(stmt.value)
                    self._current_symbols[stmt.targets[0].id] = inferred_value
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    inferred_value = self._map_annotation(stmt.annotation, function_name) or self._infer_type(stmt.value)
                    self._current_symbols[stmt.target.id] = inferred_value
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    refined = self._merge_types(refined, self._infer_type(stmt.value))
            return refined
        finally:
            self._current_symbols = saved_symbols
            self._current_function = saved_function

    def _lower_function(self, node: ast.FunctionDef) -> Function:
        self._current_function = node.name
        locals_map: dict[str, object] = {}
        arg_types = self._refine_arg_types(node)
        args = [(arg.arg, self._map_annotation(arg.annotation, node.name) or arg_types.get(arg.arg, ScalarType.REAL64)) for arg in node.args.args]
        self._function_arg_types[node.name] = [arg_type for _, arg_type in args]
        self._update_function_defaults(node, dict(args))
        self._current_symbols = {name: typ for name, typ in args}
        self._current_dtypes = {}
        self._current_contiguity = {}
        lowered_stmts = self._normalize_stmt_sequence(node.body)
        for stmt in lowered_stmts:
            self._collect_locals(stmt, locals_map)
        arg_names = {name for name, _ in args}
        locals_list = [(name, typ) for name, typ in locals_map.items() if name not in arg_names]
        self._current_symbols.update(locals_list)
        body: list[object] = []
        leading_comments = self._function_leading_comments(node)
        for stmt in lowered_stmts:
            if self._is_docstring(stmt):
                continue
            body.extend(Comment(text) for text in self._attached_comments_before_stmt(stmt, node.lineno))
            body.extend(self._lower_stmt_list(stmt))
        result_type = self._refine_function_result_type(node.name, body)
        if result_type is not None:
            self._function_result_types[node.name] = result_type
        function = Function(
            name=node.name,
            args=args,
            locals=locals_list,
            body=body,
            result_type=result_type,
            leading_comments=leading_comments,
        )
        self._current_function = None
        self._current_symbols = {}
        self._current_dtypes = {}
        self._current_contiguity = {}
        return function

    def _normalize_stmt_sequence(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        normalized: list[ast.stmt] = []
        index = 0
        while index < len(stmts):
            stmt = stmts[index]
            next_stmt = stmts[index + 1] if index + 1 < len(stmts) else None
            collapsed = self._collapse_rank_normalization_pair(stmt, next_stmt)
            if collapsed is not None:
                normalized.append(collapsed)
                index += 2
                continue
            normalized.append(stmt)
            index += 1
        return normalized

    def _collapse_rank_normalization_pair(self, first: ast.stmt, second: ast.stmt | None) -> ast.stmt | None:
        if second is None:
            return None
        if not (
            isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and isinstance(second, ast.Assign)
            and len(second.targets) == 1
            and isinstance(second.targets[0], ast.Name)
            and first.targets[0].id == second.targets[0].id
        ):
            return None
        target_name = first.targets[0].id
        value = second.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "reshape"
            and isinstance(value.func.value, ast.Call)
            and isinstance(value.func.value.func, ast.Attribute)
            and self._attr_chain(value.func.value.func) == ["np", "asarray"]
            and value.func.value.args
            and isinstance(value.func.value.args[0], ast.Name)
            and value.func.value.args[0].id == target_name
            and len(value.args) == 1
            and isinstance(value.args[0], ast.UnaryOp)
            and isinstance(value.args[0].op, ast.USub)
            and isinstance(value.args[0].operand, ast.Constant)
            and value.args[0].operand.value == 1
        ):
            return None
        replacement = copy.deepcopy(second)
        assert isinstance(replacement, ast.Assign)
        reshape_call = replacement.value
        assert isinstance(reshape_call, ast.Call)
        asarray_call = reshape_call.func.value
        assert isinstance(asarray_call, ast.Call)
        if isinstance(first.value, ast.Call) and isinstance(first.value.func, ast.Attribute):
            if self._attr_chain(first.value.func) == ["np", "loadtxt"]:
                replacement.value = ast.Call(
                    func=ast.Name(id="ac_loadtxt_1d", ctx=ast.Load()),
                    args=[copy.deepcopy(arg) for arg in first.value.args],
                    keywords=[],
                )
                return replacement
        asarray_call.args[0] = copy.deepcopy(first.value)
        return replacement

    def _update_function_defaults(self, node: ast.FunctionDef, arg_types: dict[str, object]) -> None:
        defaults: list[object | None] = [None] * len(node.args.args)
        if node.args.defaults:
            offset = len(node.args.args) - len(node.args.defaults)
            for idx, default in enumerate(node.args.defaults, start=offset):
                arg_name = node.args.args[idx].arg
                if isinstance(default, ast.Constant) and default.value is None:
                    arg_type = arg_types.get(arg_name)
                    if isinstance(arg_type, ArrayTypeRef):
                        defaults[idx] = Call("ac_zeros", (Constant(0),))
                        continue
                    if arg_type == ScalarType.INTEGER:
                        defaults[idx] = Constant(-1)
                        continue
                    if isinstance(arg_type, RecordTypeRef) and arg_type.name == "ac_random_state":
                        defaults[idx] = Call("ac_random_init", ())
                        continue
                defaults[idx] = self._lower_expr(default)
        self._function_arg_defaults[node.name] = defaults

    def _lower_main_guard(self, node: ast.If) -> Program:
        locals_map: dict[str, object] = {}
        for stmt in node.body:
            self._collect_locals(stmt, locals_map)
        body: list[object] = []
        leading_comments = self._leading_comments_after_line(node.lineno, node.body[0].lineno) if node.body else []
        for stmt in node.body:
            body.extend(Comment(text) for text in self._attached_comments_before_stmt(stmt, node.lineno))
            body.extend(self._lower_stmt_list(stmt))
        return Program(name="run_main", locals=list(locals_map.items()), body=body, leading_comments=leading_comments)

    def _extract_comment_lines(self, source: str) -> dict[int, str]:
        comments: dict[int, str] = {}
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                text = token.string[1:].strip()
                comments[token.start[0]] = text
        return comments

    def _line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self._source_lines):
            return self._source_lines[lineno - 1]
        return ""

    def _leading_comments_after_line(self, start_line: int, first_stmt_line: int) -> list[str]:
        comments: list[str] = []
        line = start_line + 1
        while line < first_stmt_line:
            stripped = self._line_text(line).strip()
            if not stripped:
                line += 1
                continue
            if line in self._comment_lines:
                comments.append(self._comment_lines[line])
                self._used_comment_lines.add(line)
                line += 1
                continue
            break
        return comments

    def _function_leading_comments(self, node: ast.FunctionDef) -> list[str]:
        first_stmt_line = node.body[0].lineno if node.body else node.lineno + 1
        comments = self._leading_comments_after_line(node.lineno, first_stmt_line)
        docstring_lines = self._function_docstring_lines(node)
        if docstring_lines:
            comments.extend(docstring_lines)
        return comments

    def _function_docstring_lines(self, node: ast.FunctionDef) -> list[str]:
        if not node.body:
            return []
        first_stmt = node.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            return [line.rstrip() for line in first_stmt.value.value.strip().splitlines() if line.strip()]
        return []

    def _attached_comments_before_stmt(self, stmt: ast.stmt, lower_bound_line: int) -> list[str]:
        line = stmt.lineno - 1
        comments: list[str] = []
        while line > lower_bound_line:
            stripped = self._line_text(line).strip()
            if not stripped:
                line -= 1
                continue
            if line in self._comment_lines and line not in self._used_comment_lines:
                comments.append(self._comment_lines[line])
                line -= 1
                continue
            break
        comments.reverse()
        self._used_comment_lines.update(
            stmt.lineno - offset
            for offset in range(1, stmt.lineno - line)
            if (stmt.lineno - offset) in self._comment_lines
        )
        return comments

    def _refine_arg_types(self, node: ast.FunctionDef) -> dict[str, object]:
        arg_types: dict[str, object] = {}
        aliases: dict[str, str] = {}
        integer_like_names: set[str] = set()
        local_types: dict[str, object] = {}
        sequence_like_names: set[str] = set()

        for arg, default in zip(node.args.args, [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults), strict=True):
            annotated = self._map_annotation(arg.annotation, node.name)
            if annotated is not None:
                arg_types[arg.arg] = annotated
            elif isinstance(default, ast.Constant) and isinstance(default.value, int):
                arg_types[arg.arg] = ScalarType.INTEGER
            elif isinstance(default, ast.Constant) and isinstance(default.value, float):
                arg_types[arg.arg] = ScalarType.REAL64

        for stmt in ast.walk(node):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "print"
            ):
                for arg in stmt.value.args:
                    if isinstance(arg, ast.Name) and arg.id in {item.arg for item in node.args.args}:
                        if arg_types.get(arg.id) != ScalarType.STRING:
                            arg_types[arg.id] = ScalarType.STRING
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name) and stmt.func.id == "len" and stmt.args and isinstance(stmt.args[0], ast.Name):
                    self._merge_arg_type(arg_types, stmt.args[0].id, ArrayTypeRef(ScalarType.REAL64, 1))
                    sequence_like_names.add(stmt.args[0].id)
                if isinstance(stmt.func, ast.Name) and stmt.func.id == "range":
                    for arg in stmt.args:
                        self._collect_integer_like_names(arg, integer_like_names)
                if isinstance(stmt.func, ast.Attribute):
                    chain = self._attr_chain(stmt.func)
                    if chain == ["np", "asarray"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        self._merge_arg_type(arg_types, stmt.args[0].id, ArrayTypeRef(ScalarType.REAL64, 1))
                        sequence_like_names.add(stmt.args[0].id)
                    if chain == ["np", "linspace"] and len(stmt.args) >= 3:
                        self._collect_integer_like_names(stmt.args[2], integer_like_names)
                    if chain == ["np", "loadtxt"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        arg_types[stmt.args[0].id] = ScalarType.STRING
                    if stmt.func.attr == "normal" and chain[:1] != ["np"]:
                        size_expr = self._keyword_value(stmt, "size")
                        if size_expr is not None:
                            self._collect_integer_like_names(size_expr, integer_like_names)
                    if chain in (["np", "zeros"], ["np", "empty"], ["np", "full"]) and stmt.args:
                        self._collect_integer_like_names(stmt.args[0], integer_like_names)
                        if isinstance(stmt.args[0], ast.Tuple):
                            for elt in stmt.args[0].elts:
                                self._collect_integer_like_names(elt, integer_like_names)
            if isinstance(stmt, ast.Subscript):
                if isinstance(stmt.value, ast.Name) and stmt.value.id in {item.arg for item in node.args.args}:
                    if isinstance(stmt.slice, ast.Slice):
                        self._merge_arg_type(arg_types, stmt.value.id, ArrayTypeRef(ScalarType.REAL64, 1))
                        sequence_like_names.add(stmt.value.id)
                    elif not isinstance(stmt.slice, ast.Tuple):
                        self._merge_arg_type(arg_types, stmt.value.id, ArrayTypeRef(ScalarType.REAL64, 1))
                        sequence_like_names.add(stmt.value.id)
                if isinstance(stmt.slice, ast.Name):
                    integer_like_names.add(stmt.slice.id)
                elif isinstance(stmt.slice, ast.Slice):
                    for part in (stmt.slice.lower, stmt.slice.upper, stmt.slice.step):
                        if part is not None:
                            self._collect_integer_like_names(part, integer_like_names)
                elif isinstance(stmt.slice, ast.Tuple):
                    for elt in stmt.slice.elts:
                        if isinstance(elt, ast.Name):
                            integer_like_names.add(elt.id)
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                try:
                    inferred_local = self._infer_type(stmt.value)
                except Exception:
                    inferred_local = None
                if inferred_local is not None:
                    local_types[stmt.targets[0].id] = inferred_local
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Tuple)
                and isinstance(stmt.value, ast.Name)
                and stmt.value.id in {item.arg for item in node.args.args}
            ):
                if self._merge_arg_type(arg_types, stmt.value.id, ArrayTypeRef(ScalarType.REAL64, 1)):
                    sequence_like_names.add(stmt.value.id)
            if (
                isinstance(stmt, ast.For)
                and isinstance(stmt.iter, ast.Name)
                and stmt.iter.id in {item.arg for item in node.args.args}
            ):
                if self._merge_arg_type(arg_types, stmt.iter.id, ArrayTypeRef(ScalarType.REAL64, 2)):
                    pass
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if (
                    isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and self._attr_chain(stmt.value.func) == ["np", "random", "default_rng"]
                    and stmt.targets[0].id in {item.arg for item in node.args.args}
                ):
                    arg_types[stmt.targets[0].id] = RecordTypeRef("ac_random_state")
            if isinstance(stmt, ast.Compare):
                names: list[str] = []
                if isinstance(stmt.left, ast.Name):
                    names.append(stmt.left.id)
                for comparator in stmt.comparators:
                    if isinstance(comparator, ast.Name):
                        names.append(comparator.id)
                compare_nodes = [stmt.left, *stmt.comparators]
                if any(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in compare_nodes) or any(
                    isinstance(item, (ast.Tuple, ast.List, ast.Set))
                    and any(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in item.elts)
                    for item in compare_nodes
                ):
                    for name in names:
                        if name in {item.arg for item in node.args.args}:
                            arg_types[name] = ScalarType.STRING

        for arg in node.args.args:
            if arg.arg in integer_like_names and arg_types.get(arg.arg) != ScalarType.STRING:
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
                        source_name = stmt.value.args[0].id
                        aliases[target_name] = source_name
                        source_type = arg_types.get(source_name)
                        target_type = arg_types.get(target_name)
                        if isinstance(source_type, ArrayTypeRef):
                            changed |= self._merge_arg_type(arg_types, target_name, source_type)
                        if isinstance(target_type, ArrayTypeRef):
                            changed |= self._merge_arg_type(arg_types, source_name, target_type)
                    if (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Attribute)
                        and self._attr_chain(stmt.value.func) == ["np", "random", "default_rng"]
                    ):
                        if arg_types.get(target_name) != RecordTypeRef("ac_random_state"):
                            arg_types[target_name] = RecordTypeRef("ac_random_state")
                            changed = True
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
                    if chain == ["np", "loadtxt"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        if arg_types.get(stmt.args[0].id) != ScalarType.STRING:
                            arg_types[stmt.args[0].id] = ScalarType.STRING
                            changed = True
                    if chain == ["np", "random", "default_rng"] and stmt.args and isinstance(stmt.args[0], ast.Name):
                        if arg_types.get(stmt.args[0].id) != ScalarType.INTEGER:
                            arg_types[stmt.args[0].id] = ScalarType.INTEGER
                            changed = True
                    if tuple(chain) in {
                        ("np", "linalg", "cholesky"),
                        ("np", "linalg", "det"),
                        ("np", "linalg", "eig"),
                        ("np", "linalg", "inv"),
                        ("np", "linalg", "svd"),
                    } and stmt.args and isinstance(stmt.args[0], ast.Name):
                        if self._set_array_rank(arg_types, aliases, stmt.args[0].id, 2):
                            changed = True
                    if chain == ["np", "linalg", "solve"] and stmt.args:
                        if isinstance(stmt.args[0], ast.Name):
                            if self._set_array_rank(arg_types, aliases, stmt.args[0].id, 2):
                                changed = True
                        if len(stmt.args) > 1 and isinstance(stmt.args[1], ast.Name):
                            if self._set_array_rank(arg_types, aliases, stmt.args[1].id, 1):
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
                if isinstance(stmt, ast.BinOp):
                    left_type = self._infer_type(stmt.left)
                    right_type = self._infer_type(stmt.right)
                    if isinstance(stmt.left, ast.Name) and stmt.left.id in local_types:
                        left_type = local_types[stmt.left.id]
                    if isinstance(stmt.right, ast.Name) and stmt.right.id in local_types:
                        right_type = local_types[stmt.right.id]
                    if isinstance(stmt.left, ast.Name) and isinstance(right_type, ArrayTypeRef):
                        current_type = arg_types.get(stmt.left.id)
                        if current_type is None or isinstance(current_type, ArrayTypeRef):
                            changed |= self._merge_arg_type(arg_types, stmt.left.id, ArrayTypeRef(right_type.element_type, right_type.rank))
                    if isinstance(stmt.right, ast.Name) and isinstance(left_type, ArrayTypeRef):
                        current_type = arg_types.get(stmt.right.id)
                        if current_type is None or isinstance(current_type, ArrayTypeRef):
                            changed |= self._merge_arg_type(arg_types, stmt.right.id, ArrayTypeRef(left_type.element_type, left_type.rank))
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    if stmt.targets[0].id in integer_like_names:
                        changed |= self._mark_integer_names(arg_types, stmt.value)
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name) and stmt.func.id in self._function_arg_types:
                    expected_types = self._function_arg_types[stmt.func.id]
                    for actual_arg, expected_type in zip(stmt.args, expected_types, strict=False):
                        if isinstance(actual_arg, ast.Name):
                            changed |= self._merge_arg_type(arg_types, actual_arg.id, expected_type)
                    arg_names = self._function_arg_names.get(stmt.func.id, [])
                    expected_by_name = {
                        name: expected_type
                        for name, expected_type in zip(arg_names, expected_types, strict=False)
                    }
                    for keyword in stmt.keywords:
                        if keyword.arg is None or not isinstance(keyword.value, ast.Name):
                            continue
                        expected_type = expected_by_name.get(keyword.arg)
                        if expected_type is not None:
                            changed |= self._merge_arg_type(arg_types, keyword.value.id, expected_type)
        return arg_types

    def _mark_integer_names(self, arg_types: dict[str, object], expr: ast.AST) -> bool:
        changed = False
        if isinstance(expr, ast.Name):
            changed |= self._merge_arg_type(arg_types, expr.id, ScalarType.INTEGER)
        elif isinstance(expr, ast.BinOp):
            changed |= self._mark_integer_names(arg_types, expr.left)
            changed |= self._mark_integer_names(arg_types, expr.right)
        elif isinstance(expr, ast.UnaryOp):
            changed |= self._mark_integer_names(arg_types, expr.operand)
        elif isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id in {"len", "int", "range", "size"}:
                for arg in expr.args:
                    changed |= self._mark_integer_names(arg_types, arg)
            if isinstance(expr.func, ast.Name) and expr.func.id == "abs" and expr.args:
                changed |= self._mark_integer_names(arg_types, expr.args[0])
        return changed

    def _collect_integer_like_names(self, expr: ast.AST, names: set[str]) -> None:
        if isinstance(expr, ast.Name):
            names.add(expr.id)
            return
        if isinstance(expr, ast.BinOp):
            self._collect_integer_like_names(expr.left, names)
            self._collect_integer_like_names(expr.right, names)
            return
        if isinstance(expr, ast.UnaryOp):
            self._collect_integer_like_names(expr.operand, names)
            return
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in {"len", "int", "size", "abs", "max", "min"}:
            for arg in expr.args:
                self._collect_integer_like_names(arg, names)

    def _merge_arg_type(self, arg_types: dict[str, object], name: str, new_type: object) -> bool:
        current = arg_types.get(name)
        merged = self._merge_types(current, new_type)
        if merged != current:
            arg_types[name] = merged
            return True
        return False

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
            refined = result_type
            for stmt in body:
                if isinstance(stmt, Return) and stmt.value is not None:
                    refined = self._merge_types(refined, self._infer_ir_type(stmt.value))
            return refined
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
            if expr.func in self._function_result_types:
                return self._function_result_types[expr.func]
            if expr.func == "ac_array_literal":
                if not expr.args:
                    return ArrayTypeRef(ScalarType.REAL64)
                first_type = self._infer_ir_type(expr.args[0])
                if isinstance(first_type, ArrayTypeRef):
                    return ArrayTypeRef(first_type.element_type, rank=first_type.rank + 1)
                element_type = first_type if first_type is not None else ScalarType.REAL64
                return ArrayTypeRef(element_type)
            if expr.func == "ac_array":
                if not expr.args:
                    return ArrayTypeRef(ScalarType.REAL64)
                return self._infer_ir_type(expr.args[0])
            if expr.func == "ac_empty":
                return ArrayTypeRef(ScalarType.REAL64)
            if expr.func == "ac_empty2":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_full":
                fill_type = self._infer_ir_type(expr.args[-1]) if expr.args else ScalarType.REAL64
                element_type = fill_type.element_type if isinstance(fill_type, ArrayTypeRef) else fill_type
                rank = 2 if len(expr.args) == 3 else 1
                return ArrayTypeRef(element_type, rank=rank)
            if expr.func == "ac_where":
                return ArrayTypeRef(ScalarType.INTEGER)
            if expr.func == "ac_multivariate_normal":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_choice_weighted":
                return ArrayTypeRef(ScalarType.INTEGER)
            if expr.func == "ac_asarray":
                return self._infer_ir_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
            if expr.func == "ac_roots":
                return ArrayTypeRef(ScalarType.COMPLEX128)
            if expr.func == "ac_arange_int":
                return ArrayTypeRef(ScalarType.INTEGER)
            if expr.func in {"ac_zeros", "ac_arange", "ac_round", "ac_slice"}:
                return ArrayTypeRef(ScalarType.REAL64)
            if expr.func in {"ac_zeros2", "ac_column_stack"}:
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_normal_vec":
                return ArrayTypeRef(ScalarType.REAL64)
            if expr.func == "ac_dot":
                return ScalarType.REAL64
            if expr.func == "ac_mean":
                return ScalarType.REAL64
            if expr.func in {"ac_sum_axis", "ac_mean_axis", "ac_min_axis", "ac_max_axis"}:
                return ArrayTypeRef(ScalarType.REAL64)
            if expr.func == "ac_solve_linear":
                return ArrayTypeRef(ScalarType.REAL64)
            if expr.func == "ac_cholesky":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "ac_file_exists":
                return ScalarType.LOGICAL
            if expr.func == "ac_parse_int":
                return ScalarType.INTEGER
            if expr.func == "ac_parse_real":
                return ScalarType.REAL64
            if expr.func == "ac_wall_time":
                return ScalarType.REAL64
            if expr.func in {"trim", "adjustl", "ac_lower", "ac_format_default", "ac_format_fixed", "ac_format_scientific", "ac_format_int"}:
                return ScalarType.STRING
            if expr.func == "str":
                return ScalarType.STRING
            if expr.func == "int":
                return ScalarType.INTEGER
            if expr.func == "float":
                return ScalarType.REAL64
            if expr.func == "ac_slice2":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.func == "abs" and expr.args:
                arg_type = self._infer_ir_type(expr.args[0])
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(ScalarType.REAL64, rank=arg_type.rank)
                return ScalarType.REAL64
            if expr.func == "all":
                return ScalarType.LOGICAL
        if isinstance(expr, BinaryOp):
            left_type = self._infer_ir_type(expr.left)
            right_type = self._infer_ir_type(expr.right)
            if isinstance(left_type, ArrayTypeRef) or isinstance(right_type, ArrayTypeRef):
                return self._combine_array_types(left_type, right_type)
            if left_type == ScalarType.INTEGER and right_type == ScalarType.INTEGER:
                return ScalarType.INTEGER
            return ScalarType.REAL64
        if isinstance(expr, UnaryOp):
            return ScalarType.LOGICAL if expr.op == UnaryOperator.NOT else self._infer_ir_type(expr.operand)
        if isinstance(expr, IndexAccess):
            container_type = self._infer_ir_type(expr.value)
            if isinstance(container_type, ArrayTypeRef):
                if container_type.rank <= 1:
                    return container_type.element_type
                return ArrayTypeRef(container_type.element_type, rank=container_type.rank - 1)
        if isinstance(expr, ConditionalExpr):
            return self._merge_types(self._infer_ir_type(expr.if_true), self._infer_ir_type(expr.if_false))
        if isinstance(expr, FieldAccess):
            base_type = self._infer_ir_type(expr.value)
            if isinstance(base_type, RecordTypeRef):
                if expr.field.startswith("item"):
                    return self._tuple_item_type(base_type, int(expr.field.removeprefix("item")))
                record = self._record_defs.get(base_type.name)
                if record is not None:
                    for field_name, field_type in record.fields:
                        if field_name == expr.field:
                            return field_type
        return ScalarType.REAL64

    def _collect_locals(self, stmt: ast.stmt, locals_map: dict[str, object]) -> None:
        if isinstance(stmt, ast.Assign):
            inferred = self._infer_type(stmt.value)
            self._collect_expr_locals(stmt.value, locals_map)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    current = locals_map.get(target.id)
                    if (
                        isinstance(current, ArrayTypeRef)
                        and isinstance(inferred, ArrayTypeRef)
                        and current.rank != inferred.rank
                    ):
                        merged = inferred
                    else:
                        merged = self._merge_types(current, inferred)
                    locals_map[target.id] = merged
                    self._current_symbols[target.id] = merged
                elif isinstance(target, ast.Tuple):
                    if isinstance(stmt.value, ast.Attribute) and stmt.value.attr == "shape":
                        record_type = None
                    elif self._is_eig_call(stmt.value):
                        record_type = self._ensure_record(
                            "ac_eig_result",
                            [("item1", ArrayTypeRef(ScalarType.REAL64)), ("item2", ArrayTypeRef(ScalarType.REAL64, rank=2))],
                        )
                    elif self._is_svd_call(stmt.value):
                        record_type = self._ensure_record(
                            "ac_svd_result",
                            [("item1", ArrayTypeRef(ScalarType.REAL64, rank=2)), ("item2", ArrayTypeRef(ScalarType.REAL64)), ("item3", ArrayTypeRef(ScalarType.REAL64, rank=2))],
                        )
                    elif self._is_slogdet_call(stmt.value):
                        record_type = None
                    else:
                        record_type = self._infer_type(stmt.value)
                    if isinstance(record_type, RecordTypeRef):
                        temp_name = self._tuple_temp_name(self._tuple_source_name(stmt.value))
                        locals_map[temp_name] = self._merge_types(locals_map.get(temp_name), record_type)
                    for index, elt in enumerate(target.elts, start=1):
                        if isinstance(elt, ast.Name):
                            elt_type = self._tuple_item_type(record_type if isinstance(record_type, RecordTypeRef) else inferred, index)
                            merged = self._merge_types(locals_map.get(elt.id), elt_type)
                            locals_map[elt.id] = merged
                            self._current_symbols[elt.id] = merged
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            annotated = self._map_annotation(stmt.annotation, self._current_function or "module")
            inferred = annotated or self._infer_type(stmt.value)
            self._collect_expr_locals(stmt.value, locals_map)
            current = locals_map.get(stmt.target.id)
            if (
                isinstance(current, ArrayTypeRef)
                and isinstance(inferred, ArrayTypeRef)
                and current.rank != inferred.rank
            ):
                merged = inferred
            else:
                merged = self._merge_types(current, inferred)
            locals_map[stmt.target.id] = merged
            self._current_symbols[stmt.target.id] = merged
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            self._collect_expr_locals(stmt.value, locals_map)
            inferred = self._infer_type(stmt.value)
            merged = self._merge_types(locals_map.get(stmt.target.id), inferred)
            locals_map[stmt.target.id] = merged
            self._current_symbols[stmt.target.id] = merged
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range":
                loop_name = self._loop_target_name(stmt.target.id)
                locals_map[loop_name] = ScalarType.INTEGER
                self._current_symbols[loop_name] = ScalarType.INTEGER
            elif isinstance(stmt.iter, ast.Tuple):
                elt_types = [self._infer_type(elt) for elt in stmt.iter.elts]
                target_type = elt_types[0] if elt_types else ScalarType.REAL64
                locals_map[stmt.target.id] = self._merge_types(locals_map.get(stmt.target.id), target_type)
                self._current_symbols[stmt.target.id] = locals_map[stmt.target.id]
            else:
                iter_type = self._infer_type(stmt.iter)
                if isinstance(iter_type, ArrayTypeRef):
                    target_type = (
                        ArrayTypeRef(iter_type.element_type, rank=iter_type.rank - 1)
                        if iter_type.rank > 1
                        else iter_type.element_type
                    )
                    locals_map[stmt.target.id] = self._merge_types(locals_map.get(stmt.target.id), target_type)
                    self._current_symbols[stmt.target.id] = locals_map[stmt.target.id]
                    loop_name = self._iter_loop_index_name(stmt.target.id)
                    locals_map[loop_name] = ScalarType.INTEGER
                    self._current_symbols[loop_name] = ScalarType.INTEGER
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Tuple):
            iter_type = self._infer_type(stmt.iter)
            tuple_item_type = ScalarType.REAL64
            if isinstance(iter_type, ArrayTypeRef):
                tuple_item_type = (
                    ArrayTypeRef(iter_type.element_type, rank=iter_type.rank - 1)
                    if iter_type.rank > 2
                    else iter_type.element_type
                )
                loop_name = self._iter_loop_index_name(self._tuple_source_name(stmt.iter))
                locals_map[loop_name] = ScalarType.INTEGER
                self._current_symbols[loop_name] = ScalarType.INTEGER
                if iter_type.rank > 1:
                    temp_name = self._tuple_temp_name(self._tuple_source_name(stmt.iter))
                    temp_type = ArrayTypeRef(iter_type.element_type, rank=iter_type.rank - 1)
                    locals_map[temp_name] = self._merge_types(locals_map.get(temp_name), temp_type)
                    self._current_symbols[temp_name] = locals_map[temp_name]
            for elt in stmt.target.elts:
                if isinstance(elt, ast.Name):
                    locals_map[elt.id] = self._merge_types(locals_map.get(elt.id), tuple_item_type)
                    self._current_symbols[elt.id] = locals_map[elt.id]
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
        elif isinstance(stmt, ast.While):
            self._collect_expr_locals(stmt.test, locals_map)
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
        elif isinstance(stmt, ast.Try):
            for inner in stmt.body:
                self._collect_locals(inner, locals_map)
            for handler in stmt.handlers:
                for inner in handler.body:
                    self._collect_locals(inner, locals_map)
            for inner in stmt.orelse:
                self._collect_locals(inner, locals_map)
            for inner in stmt.finalbody:
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
            if self._is_identity_asarray_assignment(stmt):
                return Pass()
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                target_type = self._current_symbols.get(target.id)
                if isinstance(target_type, ArrayTypeRef) and target_type.rank == 1 and target_type.element_type == ScalarType.REAL64:
                    return Assignment(ValueRef(target.id), Call("ac_zeros", (Constant(0),)))
            if isinstance(target, ast.Tuple):
                return self._lower_tuple_assignment(target, stmt.value)
            value = self._lower_expr(stmt.value)
            if isinstance(target, ast.Name):
                dtype_name = self._infer_dtype_name(stmt.value)
                if dtype_name is not None:
                    self._current_dtypes[target.id] = dtype_name
                contiguity = self._infer_contiguity(stmt.value)
                if contiguity is not None:
                    self._current_contiguity[target.id] = contiguity
                return Assignment(ValueRef(target.id), value)
            if isinstance(target, ast.Subscript):
                slice_assign = self._lower_slice_assignment(target, value)
                if slice_assign is not None:
                    return slice_assign
                if self._is_pairwise_item2_access(target):
                    row_index, col_index = target.slice.elts
                    return ExprStatement(
                        Call(
                            "ac_set_pick2",
                            (
                                self._lower_expr(target.value),
                                self._lower_index_expr(row_index),
                                self._lower_index_expr(col_index),
                                value,
                            ),
                        )
                    )
                if self._is_item2_access(target):
                    row_index, col_index = target.slice.elts
                    return ExprStatement(
                        Call(
                            "ac_set_item2",
                            (
                                self._lower_expr(target.value),
                                self._lower_index_expr(row_index),
                                self._lower_index_expr(col_index),
                                value,
                            ),
                        )
                    )
                if self._is_row_assignment(target):
                    func_name = "ac_set_rows" if isinstance(self._infer_type(target.slice), ArrayTypeRef) else "ac_set_row"
                    return ExprStatement(Call(func_name, (self._lower_expr(target.value), self._lower_index_expr(target.slice), value)))
                return IndexAssignment(self._lower_expr(target.value), self._lower_index_expr(target.slice), value)
            raise NotImplementedError(f"unsupported assignment target: {ast.dump(target)}")
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name) or stmt.value is None:
                raise NotImplementedError(f"unsupported annotated assignment: {ast.dump(stmt)}")
            annotated = self._map_annotation(stmt.annotation, self._current_function or "module")
            if annotated is not None:
                merged = self._merge_types(self._current_symbols.get(stmt.target.id), annotated)
                self._current_symbols[stmt.target.id] = merged
            value = self._lower_expr(stmt.value)
            dtype_name = self._infer_dtype_name(stmt.value)
            if dtype_name is not None:
                self._current_dtypes[stmt.target.id] = dtype_name
            contiguity = self._infer_contiguity(stmt.value)
            if contiguity is not None:
                self._current_contiguity[stmt.target.id] = contiguity
            return Assignment(ValueRef(stmt.target.id), value)
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            op_map = {
                ast.Add: BinaryOperator.ADD,
                ast.Sub: BinaryOperator.SUB,
                ast.Mult: BinaryOperator.MUL,
                ast.Div: BinaryOperator.DIV,
            }
            return AugmentedAssignment(ValueRef(stmt.target.id), op_map[type(stmt.op)], self._lower_expr(stmt.value))
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Subscript):
            op_map = {
                ast.Add: BinaryOperator.ADD,
                ast.Sub: BinaryOperator.SUB,
                ast.Mult: BinaryOperator.MUL,
                ast.Div: BinaryOperator.DIV,
            }
            value = self._lower_expr(stmt.value)
            if self._is_item2_access(stmt.target):
                row_index, col_index = stmt.target.slice.elts
                target_value = Call("ac_item2", (self._lower_expr(stmt.target.value), self._lower_index_expr(row_index), self._lower_index_expr(col_index)))
                return ExprStatement(
                    Call(
                        "ac_set_item2",
                        (
                            self._lower_expr(stmt.target.value),
                            self._lower_index_expr(row_index),
                            self._lower_index_expr(col_index),
                            BinaryOp(target_value, op_map[type(stmt.op)], value),
                        ),
                    )
                )
            return IndexAssignment(
                self._lower_expr(stmt.target.value),
                self._lower_index_expr(stmt.target.slice),
                BinaryOp(IndexAccess(self._lower_expr(stmt.target.value), self._lower_index_expr(stmt.target.slice)), op_map[type(stmt.op)], value),
            )
        if isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Tuple):
                result_type = self._function_result_types.get(self._current_function or "")
                if isinstance(result_type, RecordTypeRef):
                    fields = [(f"item{index}", self._lower_expr(value)) for index, value in enumerate(stmt.value.elts, start=1)]
                    typed_fields = [(f"item{index}", self._infer_type(value)) for index, value in enumerate(stmt.value.elts, start=1)]
                    self._ensure_record(result_type.name, typed_fields)
                    return Return(RecordLiteral(result_type.name, tuple(fields)))
            return Return(self._lower_expr(stmt.value) if stmt.value is not None else None)
        if isinstance(stmt, ast.If):
            if self._is_sys_argv_if(stmt):
                return Pass()
            sentinel_if = self._lower_optional_none_if(stmt)
            if sentinel_if is not None:
                return sentinel_if
            return If(
                test=self._lower_expr(stmt.test),
                body=tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner)),
                orelse=tuple(lowered for inner in stmt.orelse for lowered in self._lower_stmt_list(inner)),
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
                body.extend(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner))
                return ForRange(
                    target=index_name,
                    start=Constant(0),
                    stop=Call("size", (iterable_expr,)),
                    body=tuple(body),
                )
            if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
                if isinstance(stmt.iter, ast.Tuple):
                    iterable_expr = Call("ac_array_literal", tuple(self._lower_expr(elt) for elt in stmt.iter.elts))
                else:
                    iterable_expr = self._lower_expr(stmt.iter)
                target_name = stmt.target.id if isinstance(stmt.target, ast.Name) else self._tuple_source_name(stmt.iter)
                loop_name = self._iter_loop_index_name(target_name)
                body: list[object]
                iterable_type = self._infer_type(stmt.iter)
                if isinstance(iterable_type, ArrayTypeRef) and iterable_type.rank > 1:
                    body = []
                    if isinstance(stmt.target, ast.Name):
                        row_expr = Call("ac_row", (iterable_expr, ValueRef(loop_name)))
                        body.append(Assignment(ValueRef(stmt.target.id), row_expr))
                    elif isinstance(stmt.target, ast.Tuple):
                        for index, elt in enumerate(stmt.target.elts):
                            if not isinstance(elt, ast.Name):
                                raise NotImplementedError(f"unsupported loop target: {ast.dump(elt)}")
                            body.append(
                                Assignment(
                                    ValueRef(elt.id),
                                    Call("ac_item2", (iterable_expr, ValueRef(loop_name), Constant(index))),
                                )
                            )
                    else:
                        raise NotImplementedError(f"unsupported loop target: {ast.dump(stmt.target)}")
                    stop_expr = Call("ac_shape_dim", (iterable_expr, Constant(1)))
                else:
                    if isinstance(stmt.target, ast.Name):
                        body = [Assignment(ValueRef(stmt.target.id), IndexAccess(iterable_expr, ValueRef(loop_name)))]
                    elif isinstance(stmt.target, ast.Tuple):
                        body = []
                        for index, elt in enumerate(stmt.target.elts):
                            if not isinstance(elt, ast.Name):
                                raise NotImplementedError(f"unsupported loop target: {ast.dump(elt)}")
                            body.append(Assignment(ValueRef(elt.id), IndexAccess(IndexAccess(iterable_expr, ValueRef(loop_name)), Constant(index))))
                    else:
                        raise NotImplementedError(f"unsupported loop target: {ast.dump(stmt.target)}")
                    stop_expr = Call("size", (iterable_expr,))
                body.extend(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner))
                return ForRange(
                    target=loop_name,
                    start=Constant(0),
                    stop=stop_expr,
                    body=tuple(body),
                )
            if not isinstance(stmt.target, ast.Name):
                raise NotImplementedError(f"unsupported loop target: {ast.dump(stmt.target)}")
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
                body=tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner)),
            )
        if isinstance(stmt, ast.While):
            return While(
                test=self._lower_expr(stmt.test),
                body=tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner)),
            )
        if isinstance(stmt, ast.Try):
            narrowed = self._lower_narrow_try_stmt(stmt)
            if narrowed is not None:
                return narrowed
            raise NotImplementedError(f"unsupported statement: {ast.dump(stmt)}")
        if isinstance(stmt, ast.Raise):
            message = "error"
            if isinstance(stmt.exc, ast.Call) and stmt.exc.args and isinstance(stmt.exc.args[0], ast.Constant):
                message = str(stmt.exc.args[0].value)
            return Raise(message)
        if isinstance(stmt, ast.Continue):
            return Continue()
        if isinstance(stmt, ast.Break):
            return Break()
        if isinstance(stmt, ast.Pass):
            return Pass()
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if (
                isinstance(stmt.value.func, ast.Attribute)
                and self._attr_chain(stmt.value.func) == ["np", "set_printoptions"]
            ):
                return ExprStatement(self._lower_numpy_set_printoptions(stmt.value))
            if (
                isinstance(stmt.value.func, ast.Attribute)
                and self._attr_chain(stmt.value.func) == ["np", "put"]
                and len(stmt.value.args) == 3
            ):
                return ExprStatement(Call("ac_put", tuple(self._lower_expr(arg) for arg in stmt.value.args)))
            if (
                isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "shuffle"
                and self._attr_chain(stmt.value.func)[:1] != ["np"]
                and len(stmt.value.args) == 1
            ):
                return ExprStatement(Call("ac_shuffle", (self._lower_expr(stmt.value.func.value), self._lower_expr(stmt.value.args[0]))))
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

    def _lower_stmt_list(self, stmt: ast.stmt) -> tuple[object, ...]:
        if isinstance(stmt, ast.For) and not (
            isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range"
        ) and isinstance(stmt.iter, ast.Tuple):
            if not isinstance(stmt.target, ast.Name):
                raise NotImplementedError(f"unsupported loop target: {ast.dump(stmt.target)}")
            lowered: list[object] = []
            for elt in stmt.iter.elts:
                lowered.append(Assignment(ValueRef(stmt.target.id), self._lower_expr(elt)))
                for inner in stmt.body:
                    lowered.extend(self._lower_stmt_list(inner))
            return tuple(lowered)
        narrowed_list = self._lower_narrow_try_stmt_list(stmt)
        if narrowed_list is not None:
            return narrowed_list
        return (self._lower_stmt(stmt),)

    def _lower_narrow_try_stmt(self, stmt: ast.Try) -> object | None:
        lowered_list = self._lower_narrow_try_stmt_list(stmt)
        if lowered_list is not None and len(lowered_list) == 1:
            return lowered_list[0]
        if stmt.orelse or stmt.finalbody or len(stmt.body) != 1 or len(stmt.handlers) != 1:
            return None
        body_stmt = stmt.body[0]
        handler = stmt.handlers[0]
        if (
            not isinstance(body_stmt, ast.Assign)
            or len(body_stmt.targets) != 1
            or not isinstance(body_stmt.targets[0], ast.Name)
            or not isinstance(handler.type, ast.Attribute)
            or self._attr_chain(handler.type) != ["np", "linalg", "LinAlgError"]
            or len(handler.body) != 1
            or not isinstance(handler.body[0], ast.Assign)
            or len(handler.body[0].targets) != 1
            or not isinstance(handler.body[0].targets[0], ast.Name)
            or handler.body[0].targets[0].id != body_stmt.targets[0].id
        ):
            return None
        try_call = body_stmt.value
        except_value = handler.body[0].value
        if not (
            isinstance(try_call, ast.Call)
            and self._attr_chain(try_call.func) == ["np", "linalg", "solve"]
            and isinstance(except_value, ast.Subscript)
            and isinstance(except_value.slice, ast.Constant)
            and except_value.slice.value == 0
            and isinstance(except_value.value, ast.Call)
            and self._attr_chain(except_value.value.func) == ["np", "linalg", "lstsq"]
        ):
            return None
        return Assignment(
            ValueRef(body_stmt.targets[0].id),
            Call("ac_solve_linear_fallback", tuple(self._lower_expr(arg) for arg in try_call.args)),
        )

    def _lower_narrow_try_stmt_list(self, stmt: ast.stmt) -> tuple[object, ...] | None:
        if not isinstance(stmt, ast.Try):
            return None
        if stmt.orelse or stmt.finalbody or len(stmt.handlers) != 1:
            return None
        handler = stmt.handlers[0]
        if (
            isinstance(handler.type, ast.Name)
            and handler.type.id == "OSError"
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Assign)
            and len(stmt.body[0].targets) == 1
            and isinstance(stmt.body[0].targets[0], ast.Name)
            and isinstance(stmt.body[0].value, ast.Call)
            and isinstance(stmt.body[0].value.func, ast.Attribute)
            and self._attr_chain(stmt.body[0].value.func) == ["np", "loadtxt"]
            and stmt.body[0].value.args
            and len(handler.body) == 1
            and isinstance(handler.body[0], ast.Return)
        ):
            except_value = handler.body[0].value
            if isinstance(except_value, ast.Constant) and except_value.value is None:
                lowered_except = Return(None)
            elif (
                isinstance(except_value, ast.Call)
                and isinstance(except_value.func, ast.Attribute)
                and self._attr_chain(except_value.func) == ["np", "empty"]
            ):
                lowered_except = Return(self._lower_expr(except_value))
            else:
                lowered_except = None
            if lowered_except is None:
                return None
            return (
                If(
                    test=Call("ac_file_exists", (self._lower_expr(stmt.body[0].value.args[0]),)),
                    body=(
                        Assignment(
                            ValueRef(stmt.body[0].targets[0].id),
                            self._lower_expr(stmt.body[0].value),
                        ),
                    ),
                    orelse=(lowered_except,),
                ),
            )
        if (
            isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
            and len(handler.body) == 1
            and isinstance(handler.body[0], ast.Return)
            and handler.body[0].value is not None
        ):
            return tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner))
        return None

    def _lower_optional_none_if(self, stmt: ast.If) -> If | None:
        test = stmt.test
        if (
            not isinstance(test, ast.Compare)
            or len(test.ops) != 1
            or len(test.comparators) != 1
            or not isinstance(test.left, ast.Name)
            or not isinstance(test.comparators[0], ast.Constant)
            or test.comparators[0].value is not None
            or not isinstance(test.ops[0], (ast.Is, ast.IsNot))
        ):
            return None
        left_type = self._current_symbols.get(test.left.id)
        sentinel: object | None = None
        if left_type == ScalarType.INTEGER:
            sentinel = Constant(-1)
        elif isinstance(left_type, RecordTypeRef) and left_type.name == "ac_random_state":
            return If(
                test=Constant(isinstance(test.ops[0], ast.IsNot)),
                body=tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner)),
                orelse=tuple(lowered for inner in stmt.orelse for lowered in self._lower_stmt_list(inner)),
            )
        if sentinel is None:
            return None
        compare_op = CompareOperator.NE if isinstance(test.ops[0], ast.IsNot) else CompareOperator.EQ
        return If(
            test=Compare(ValueRef(test.left.id), compare_op, sentinel),
            body=tuple(lowered for inner in stmt.body for lowered in self._lower_stmt_list(inner)),
            orelse=tuple(lowered for inner in stmt.orelse for lowered in self._lower_stmt_list(inner)),
        )

    def _lower_numpy_set_printoptions(self, expr: ast.Call) -> object:
        precision = self._keyword_value(expr, "precision")
        suppress = self._keyword_value(expr, "suppress")
        linewidth = self._keyword_value(expr, "linewidth")
        return Call(
            "ac_set_printoptions",
            (
                self._lower_expr(precision) if precision is not None else Constant(-1),
                self._lower_expr(suppress) if suppress is not None else Constant(False),
                self._lower_expr(linewidth) if linewidth is not None else Constant(-1),
                Constant(precision is not None),
                Constant(suppress is not None),
                Constant(linewidth is not None),
            ),
        )

    def _lower_tuple_assignment(self, target: ast.Tuple, value: ast.AST):
        if self._is_eig_call(value):
            if len(target.elts) != 2 or not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise NotImplementedError("eig tuple assignment requires two simple targets")
            return ExprStatement(
                Call("ac_eig", (self._lower_expr(value.args[0]), ValueRef(target.elts[0].id), ValueRef(target.elts[1].id)))
            )
        if self._is_svd_call(value):
            if len(target.elts) != 3 or not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise NotImplementedError("svd tuple assignment requires three simple targets")
            return ExprStatement(
                Call(
                    "ac_svd",
                    (
                        self._lower_expr(value.args[0]),
                        ValueRef(target.elts[0].id),
                        ValueRef(target.elts[1].id),
                        ValueRef(target.elts[2].id),
                    ),
                )
            )
        if self._is_meshgrid_call(value):
            if len(target.elts) != 2 or not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise NotImplementedError("meshgrid tuple assignment requires two simple targets")
            x_expr = self._lower_expr(value.args[0])
            y_expr = self._lower_expr(value.args[1])
            statements = [
                Assignment(ValueRef(target.elts[0].id), Call("spread", (x_expr, Constant(1), Call("size", (y_expr,))))),
                Assignment(ValueRef(target.elts[1].id), Call("spread", (y_expr, Constant(2), Call("size", (x_expr,))))),
            ]
            return If(Constant(True), tuple(statements), ())
        if self._is_histogram_call(value):
            if len(target.elts) != 2 or not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise NotImplementedError("histogram tuple assignment requires two simple targets")
            bins_expr = self._lower_histogram_bins(value)
            x_expr = self._lower_expr(value.args[0])
            statements = [
                Assignment(ValueRef(target.elts[0].id), Call("ac_histogram_counts", (x_expr, bins_expr))),
                Assignment(ValueRef(target.elts[1].id), Call("ac_histogram_edges", (bins_expr,))),
            ]
            return If(Constant(True), tuple(statements), ())
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
        if isinstance(record_type, ArrayTypeRef) and record_type.rank == 1:
            statements = []
            for index, elt in enumerate(target.elts):
                if not isinstance(elt, ast.Name):
                    raise NotImplementedError(f"unsupported tuple target: {ast.dump(elt)}")
                statements.append(Assignment(ValueRef(elt.id), IndexAccess(lowered_value, Constant(index))))
            return If(Constant(True), tuple(statements), ())
        if not isinstance(record_type, RecordTypeRef):
            statements = []
            for index, elt in enumerate(target.elts):
                if not isinstance(elt, ast.Name):
                    raise NotImplementedError(f"unsupported tuple target: {ast.dump(elt)}")
                statements.append(Assignment(ValueRef(elt.id), IndexAccess(lowered_value, Constant(index))))
            return If(Constant(True), tuple(statements), ())
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
                values.append(self._lower_formatted_value(part))
            else:
                raise NotImplementedError(f"unsupported f-string component: {ast.dump(part)}")
        return values

    def _lower_formatted_value(self, expr: ast.FormattedValue) -> object:
        spec = self._format_spec_string(expr.format_spec)
        value = self._lower_expr(expr.value)
        if spec is None:
            return Call("ac_format_default", (value,))
        fixed = self._parse_float_format_spec(spec, "f")
        if fixed is not None:
            width, decimals = fixed
            if width is None:
                return Call("ac_format_fixed", (value, Constant(decimals)))
            return Call("ac_format_fixed", (value, Constant(decimals), Constant(width)))
        scientific = self._parse_float_format_spec(spec, "e")
        if scientific is not None:
            width, decimals = scientific
            if width is None:
                return Call("ac_format_scientific", (value, Constant(decimals)))
            return Call("ac_format_scientific", (value, Constant(decimals), Constant(width)))
        if spec.endswith("d") and spec[:-1].isdigit():
            return Call("ac_format_int", (value, Constant(int(spec[:-1]))))
        return value

    def _parse_float_format_spec(self, spec: str, kind: str) -> tuple[int | None, int] | None:
        if not spec.endswith(kind):
            return None
        core = spec[:-1]
        if core.startswith(".") and core[1:].isdigit():
            return None, int(core[1:])
        if "." not in core:
            return None
        width_text, decimals_text = core.split(".", 1)
        if width_text.isdigit() and decimals_text.isdigit():
            return int(width_text), int(decimals_text)
        return None

    def _format_spec_string(self, expr: ast.AST | None) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.JoinedStr) and all(isinstance(value, ast.Constant) and isinstance(value.value, str) for value in expr.values):
            return "".join(value.value for value in expr.values)
        return None

    def _lower_expr(self, expr: ast.AST):
        if isinstance(expr, ast.Constant):
            return Constant(expr.value)
        if isinstance(expr, ast.Name):
            return ValueRef(self._loop_target_name(expr.id) if expr.id == "_" else expr.id)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "ac_loadtxt_1d":
            return Call("ac_loadtxt_1d", tuple(self._lower_expr(arg) for arg in expr.args))
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
            if expr.elts and all(self._infer_type(value) == ScalarType.INTEGER for value in expr.elts):
                record_type = self._shape_record_type(expr, minimum_rank=len(expr.elts))
                fields = [(f"item{index}", self._lower_expr(value)) for index, value in enumerate(expr.elts, start=1)]
                return RecordLiteral(record_type.name, tuple(fields))
            tuple_type = self._infer_type(expr)
            if isinstance(tuple_type, ArrayTypeRef) and tuple_type.rank == 1:
                return Call("ac_array_literal", tuple(self._lower_expr(value) for value in expr.elts))
            record_type = self._function_result_types.get(self._current_function or "")
            if isinstance(record_type, RecordTypeRef):
                fields = [(f"item{index}", self._lower_expr(value)) for index, value in enumerate(expr.elts, start=1)]
                typed_fields = [(f"item{index}", self._infer_type(value)) for index, value in enumerate(expr.elts, start=1)]
                self._ensure_record(record_type.name, typed_fields)
                return RecordLiteral(record_type.name, tuple(fields))
            return Call("ac_array_literal", tuple(self._lower_expr(value) for value in expr.elts))
        if isinstance(expr, ast.Subscript):
            if self._is_np_r_concat(expr):
                return self._lower_np_r_concat(expr)
            if self._is_boolean_mask_access(expr):
                return Call("ac_mask", (self._lower_expr(expr.value), self._lower_expr(expr.slice)))
            if self._is_item3_access(expr):
                i0, i1, i2 = expr.slice.elts
                return Call(
                    "ac_item3",
                    (
                        self._lower_expr(expr.value),
                        self._lower_index_expr(i0),
                        self._lower_index_expr(i1),
                        self._lower_index_expr(i2),
                    ),
                )
            if self._is_pairwise_item2_access(expr):
                row_index, col_index = expr.slice.elts
                return Call("ac_pick2", (self._lower_expr(expr.value), self._lower_index_expr(row_index), self._lower_index_expr(col_index)))
            if self._is_item2_access(expr):
                row_index, col_index = expr.slice.elts
                return Call("ac_item2", (self._lower_expr(expr.value), self._lower_index_expr(row_index), self._lower_index_expr(col_index)))
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
                return Call("ac_take_rows", (self._lower_expr(expr.value), self._lower_index_expr(expr.slice)))
            if self._is_row_access(expr):
                return Call("ac_row", (self._lower_expr(expr.value), self._lower_index_expr(expr.slice)))
            return IndexAccess(self._lower_expr(expr.value), self._lower_index_expr(expr.slice))
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
                ast.Mod: BinaryOperator.MOD,
                ast.BitAnd: BinaryOperator.AND,
                ast.BitOr: BinaryOperator.OR,
            }
            return BinaryOp(self._lower_expr(expr.left), op_map[type(expr.op)], self._lower_expr(expr.right))
        if isinstance(expr, ast.UnaryOp):
            op_map = {
                ast.UAdd: UnaryOperator.PLUS,
                ast.USub: UnaryOperator.MINUS,
                ast.Not: UnaryOperator.NOT,
                ast.Invert: UnaryOperator.NOT,
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
                ast.Is: CompareOperator.EQ,
                ast.IsNot: CompareOperator.NE,
            }
            if (
                len(expr.ops) == 1
                and isinstance(expr.comparators[0], ast.Constant)
                and expr.comparators[0].value is None
                and isinstance(expr.ops[0], (ast.Is, ast.IsNot))
                and isinstance(self._infer_type(expr.left), ArrayTypeRef)
            ):
                return Compare(
                    Call("size", (self._lower_expr(expr.left),)),
                    CompareOperator.GT if isinstance(expr.ops[0], ast.IsNot) else CompareOperator.EQ,
                    Constant(0),
                )
            if (
                len(expr.ops) == 1
                and isinstance(expr.comparators[0], ast.Constant)
                and expr.comparators[0].value is None
                and isinstance(expr.ops[0], (ast.Is, ast.IsNot))
                and isinstance(self._infer_type(expr.left), RecordTypeRef)
                and self._infer_type(expr.left).name == "ac_random_state"
            ):
                return Constant(isinstance(expr.ops[0], ast.IsNot))
            if (
                len(expr.ops) == 1
                and isinstance(expr.comparators[0], ast.Constant)
                and expr.comparators[0].value is None
                and isinstance(expr.ops[0], (ast.Is, ast.IsNot))
                and self._infer_type(expr.left) == ScalarType.INTEGER
            ):
                return Compare(
                    self._lower_expr(expr.left),
                    CompareOperator.NE if isinstance(expr.ops[0], ast.IsNot) else CompareOperator.EQ,
                    Constant(-1),
                )
            if len(expr.ops) > 1:
                left = expr.left
                comparisons: list[object] = []
                for op, comparator in zip(expr.ops, expr.comparators, strict=True):
                    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                        compare_op = CompareOperator.EQ if isinstance(op, ast.In) else CompareOperator.NE
                        join_op = "or" if isinstance(op, ast.In) else "and"
                        terms = tuple(
                            Compare(self._lower_expr(left), compare_op, self._lower_expr(elt))
                            for elt in comparator.elts
                        )
                        comparisons.append(BooleanOp(join_op, terms))
                    else:
                        comparisons.append(Compare(self._lower_expr(left), op_map[type(op)], self._lower_expr(comparator)))
                    left = comparator
                return BooleanOp("and", tuple(comparisons))
            op = expr.ops[0]
            if isinstance(op, (ast.In, ast.NotIn)) and isinstance(expr.comparators[0], (ast.Set, ast.Tuple, ast.List)):
                compare_op = CompareOperator.EQ if isinstance(op, ast.In) else CompareOperator.NE
                join_op = "or" if isinstance(op, ast.In) else "and"
                terms = [
                    Compare(self._lower_expr(expr.left), compare_op, self._lower_expr(elt))
                    for elt in expr.comparators[0].elts
                ]
                return BooleanOp(join_op, tuple(terms))
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
                    if expr.args and self._infer_type(expr.args[0]) == ScalarType.STRING:
                        return Call("len", self._lower_call_args(expr))
                    return Call("size", self._lower_call_args(expr))
                if expr.func.id == "int":
                    if expr.args and self._infer_type(expr.args[0]) == ScalarType.STRING:
                        return Call("ac_parse_int", self._lower_call_args(expr))
                    return Call("int", self._lower_call_args(expr))
                if expr.func.id == "float":
                    if expr.args and self._infer_type(expr.args[0]) == ScalarType.STRING:
                        return Call("ac_parse_real", self._lower_call_args(expr))
                    return Call("float", self._lower_call_args(expr))
                if expr.func.id == "enumerate":
                    return Call("ac_enumerate", self._lower_call_args(expr))
                if expr.func.id in {"abs", "max", "min", "all"}:
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
                if expr.func.attr == "rstrip":
                    base = self._lower_expr(expr.func.value)
                    return Call("trim", (base,))
                if expr.func.attr == "gauss" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    return Call("ac_gauss", (base, *self._lower_call_args(expr)))
                if expr.func.attr == "normal" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    size_expr = self._keyword_value(expr, "size")
                    if size_expr is not None:
                        loc_ast = self._keyword_value(expr, "loc")
                        scale_ast = self._keyword_value(expr, "scale")
                        loc_expr = self._lower_expr(loc_ast) if loc_ast is not None else Constant(0.0)
                        scale_expr = self._lower_expr(scale_ast) if scale_ast is not None else Constant(1.0)
                        return Call("ac_normal_vec", (base, loc_expr, scale_expr, self._lower_expr(size_expr)))
                    return Call("ac_gauss", (base, *self._lower_call_args(expr)))
                if expr.func.attr == "standard_normal" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    if expr.args:
                        return Call("ac_normal_vec", (base, Constant(0.0), Constant(1.0), self._lower_expr(expr.args[0])))
                    size_ast = self._keyword_value(expr, "size")
                    if size_ast is not None:
                        return Call("ac_normal_vec", (base, Constant(0.0), Constant(1.0), self._lower_expr(size_ast)))
                    return Call("ac_gauss", (base, Constant(0.0), Constant(1.0)))
                if expr.func.attr == "uniform" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    size_ast = self._keyword_value(expr, "size")
                    if size_ast is not None:
                        low_ast = self._keyword_value(expr, "low")
                        high_ast = self._keyword_value(expr, "high")
                        low_expr = self._lower_expr(low_ast) if low_ast is not None else Constant(0.0)
                        high_expr = self._lower_expr(high_ast) if high_ast is not None else Constant(1.0)
                        if isinstance(size_ast, ast.Tuple) and len(size_ast.elts) == 2:
                            n_expr = self._lower_expr(size_ast.elts[0])
                            m_expr = self._lower_expr(size_ast.elts[1])
                            return Call(
                                "ac_reshape",
                                (
                                    Call(
                                        "ac_uniform_vec",
                                        (base, low_expr, high_expr, BinaryOp(n_expr, BinaryOperator.MUL, m_expr)),
                                    ),
                                    n_expr,
                                    m_expr,
                                ),
                            )
                        return Call("ac_uniform_vec", (base, low_expr, high_expr, self._lower_expr(size_ast)))
                if expr.func.attr == "integers" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    size_ast = self._keyword_value(expr, "size")
                    if size_ast is None or len(expr.args) < 2:
                        raise NotImplementedError("rng.integers currently requires low, high, and size")
                    return Call("ac_random_integers", (base, self._lower_expr(expr.args[0]), self._lower_expr(expr.args[1]), self._lower_expr(size_ast)))
                if expr.func.attr == "sum" and chain[:1] != ["np"]:
                    return Call("sum", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "mean" and chain[:1] != ["np"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    keepdims_expr = self._keyword_value(expr, "keepdims")
                    if axis_expr is not None:
                        lowered = Call("ac_mean_axis", (self._lower_expr(expr.func.value), self._lower_expr(axis_expr)))
                        if isinstance(keepdims_expr, ast.Constant) and keepdims_expr.value is True:
                            return Call("ac_add_axis", (lowered,))
                        return lowered
                    return Call("ac_mean", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "min" and chain[:1] != ["np"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_min_axis", (self._lower_expr(expr.func.value), self._lower_expr(axis_expr)))
                    return Call("minval", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "max" and chain[:1] != ["np"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_max_axis", (self._lower_expr(expr.func.value), self._lower_expr(axis_expr)))
                    return Call("maxval", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "var" and chain[:1] != ["np"]:
                    ddof_expr = self._keyword_value(expr, "ddof")
                    if ddof_expr is None:
                        ddof_expr = Constant(0)
                    return Call("ac_var", (self._lower_expr(expr.func.value), self._lower_expr(ddof_expr)))
                if expr.func.attr == "std" and chain[:1] != ["np"]:
                    ddof_expr = self._keyword_value(expr, "ddof")
                    if ddof_expr is None:
                        ddof_expr = Constant(0)
                    return Call("ac_std", (self._lower_expr(expr.func.value), self._lower_expr(ddof_expr)))
                if expr.func.attr == "copy" and chain[:1] != ["np"]:
                    return Call("ac_copy", (self._lower_expr(expr.func.value),))
                if expr.func.attr == "astype" and chain[:1] != ["np"]:
                    dtype_name = None
                    if expr.args:
                        dtype_name = self._numpy_dtype_name(expr.args[0]) if isinstance(expr.args[0], ast.Attribute) else None
                        if isinstance(expr.args[0], ast.Name):
                            dtype_name = expr.args[0].id
                    if dtype_name in {"int32", "int64", "int"}:
                        return Call("ac_astype_int", (self._lower_expr(expr.func.value),))
                    if dtype_name in {"float32", "float64", "float"}:
                        return Call("ac_astype_float", (self._lower_expr(expr.func.value),))
                    return Call("ac_asarray", (self._lower_expr(expr.func.value),))
                if expr.func.attr in {"ravel", "flatten"} and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    return Call("ac_reshape", (base, Call("size", (base,))))
                if expr.func.attr == "reshape" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    reshape_args = [self._lower_expr(arg) for arg in expr.args]
                    if len(expr.args) == 1 and isinstance(expr.args[0], ast.Tuple):
                        reshape_args = [self._lower_expr(elt) for elt in expr.args[0].elts]
                    order_ast = self._keyword_value(expr, "order")
                    if order_ast is not None:
                        reshape_args.append(self._lower_expr(order_ast))
                    return Call("ac_reshape", (base, *reshape_args))
                if expr.func.attr == "transpose" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    if expr.args:
                        return Call("ac_transpose_perm", (base, *tuple(self._lower_expr(arg) for arg in expr.args)))
                    return Call("transpose", (base,))
                if expr.func.attr == "dot" and chain[:1] != ["np"]:
                    base = self._lower_expr(expr.func.value)
                    left_type = self._infer_type(expr.func.value)
                    right_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    if isinstance(left_type, ArrayTypeRef) and isinstance(right_type, ArrayTypeRef):
                        if left_type.rank == 1 and right_type.rank == 1:
                            return Call("ac_dot", (base, self._lower_expr(expr.args[0])))
                        return Call("ac_matmul", (base, self._lower_expr(expr.args[0])))
                    return Call("ac_dot", (base, self._lower_expr(expr.args[0])))
                if expr.func.attr == "trace" and chain[:1] != ["np"]:
                    return Call("ac_trace", (self._lower_expr(expr.func.value),))
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
                    if replace_expr is not None and isinstance(replace_expr, ast.Constant) and replace_expr.value is True and expr.args:
                        return Call(
                            "ac_choice_replace",
                            (
                                base,
                                self._lower_expr(expr.args[0]),
                                self._lower_expr(size_expr),
                            ),
                        )
                    raise NotImplementedError("only weighted choice and explicit replace/no-replace choice are supported")
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
                    if (
                        rowvar_expr is None
                        and len(expr.args) == 1
                        and isinstance(expr.args[0], ast.Attribute)
                        and expr.args[0].attr == "T"
                    ):
                        return Call("ac_cov_rowvar_false", (self._lower_expr(expr.args[0].value),))
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
                    if (
                        expr.args
                        and isinstance(expr.args[0], ast.Constant)
                        and expr.args[0].value == "ii->"
                    ):
                        return Call("ac_trace", (self._lower_expr(expr.args[1]),))
                    raise NotImplementedError("only np.einsum('ni,ij,nj->n', ...) is currently supported")
                if chain == ["np", "max"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_max_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("maxval", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "min"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_min_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("minval", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "sum"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_sum_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("sum", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "mean"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    keepdims_expr = self._keyword_value(expr, "keepdims")
                    if axis_expr is not None:
                        lowered = Call("ac_mean_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                        if isinstance(keepdims_expr, ast.Constant) and keepdims_expr.value is True:
                            return Call("ac_add_axis", (lowered,))
                        return lowered
                    return Call("ac_mean", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "var"]:
                    ddof_expr = self._keyword_value(expr, "ddof")
                    if ddof_expr is None:
                        ddof_expr = ast.Constant(value=0)
                    return Call("ac_var", (self._lower_expr(expr.args[0]), self._lower_expr(ddof_expr)))
                if chain == ["np", "std"]:
                    ddof_expr = self._keyword_value(expr, "ddof")
                    if ddof_expr is None:
                        ddof_expr = ast.Constant(value=0)
                    return Call("ac_std", (self._lower_expr(expr.args[0]), self._lower_expr(ddof_expr)))
                if chain == ["np", "argmin"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_argmin_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("ac_argmin", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "argmax"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_argmax_axis", (self._lower_expr(expr.args[0]), self._lower_expr(axis_expr)))
                    return Call("ac_argmax", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "trace"]:
                    return Call("ac_trace", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "argsort"]:
                    return Call("ac_argsort", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "sort"]:
                    return Call("ac_sort", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "unique"]:
                    return Call("ac_unique", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "bincount"]:
                    return Call("ac_bincount", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "searchsorted"]:
                    side_expr = self._keyword_value(expr, "side")
                    side_text = "left"
                    if isinstance(side_expr, ast.Constant) and isinstance(side_expr.value, str):
                        side_text = side_expr.value
                    helper = "ac_searchsorted_left" if side_text == "left" else "ac_searchsorted_right"
                    return Call(helper, tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "outer"]:
                    return Call("ac_outer", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "kron"]:
                    return Call("ac_kron", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "meshgrid"]:
                    raise NotImplementedError("np.meshgrid is only supported in tuple assignment")
                if chain == ["np", "ndim"]:
                    return Call("ac_ndim", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "eye"]:
                    return Call("ac_eye", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "diag"]:
                    args = tuple(self._lower_expr(arg) for arg in expr.args)
                    if len(expr.args) == 1:
                        return Call("ac_diag", args)
                    return Call("ac_diag_k", args)
                if chain == ["np", "atleast_2d"]:
                    return Call("ac_atleast_2d", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "log"]:
                    return Call("log", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "exp"]:
                    return Call("exp", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "sqrt"]:
                    return Call("sqrt", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "sign"]:
                    return Call("ac_sign", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "floor"]:
                    return Call("ac_floor", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "ceil"]:
                    return Call("ac_ceil", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "abs"]:
                    return Call("abs", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "isnan"]:
                    return Call("ac_isnan", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "isfinite"]:
                    return Call("ac_isfinite", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "nansum"]:
                    return Call("ac_nansum", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "all"]:
                    return Call("all", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "dot"]:
                    return Call("ac_dot", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "array"]:
                    if expr.args and isinstance(expr.args[0], ast.List) and not expr.args[0].elts:
                        return Call("ac_zeros", (Constant(0),))
                    if expr.args:
                        return Call("ac_array", (self._lower_expr(expr.args[0]),))
                    return Call("ac_array", ())
                if chain == ["np", "asarray"]:
                    if expr.args:
                        return Call("ac_asarray", (self._lower_expr(expr.args[0]),))
                    return Call("ac_asarray", ())
                if chain == ["np", "roots"]:
                    return Call("ac_roots", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "zeros"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple) and len(expr.args[0].elts) == 2:
                        return Call("ac_zeros2", tuple(self._lower_expr(elt) for elt in expr.args[0].elts))
                    return Call("ac_zeros", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "ones"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple) and len(expr.args[0].elts) == 2:
                        return Call("ac_full", (*tuple(self._lower_expr(elt) for elt in expr.args[0].elts), Constant(1.0)))
                    if expr.args:
                        return Call("ac_full", (self._lower_expr(expr.args[0]), Constant(1.0)))
                    return Call("ac_full", (Constant(0), Constant(1.0)))
                if chain == ["np", "where"]:
                    if len(expr.args) == 3:
                        return Call("ac_where_select", tuple(self._lower_expr(arg) for arg in expr.args))
                    return Call("ac_where", self._lower_call_args(expr))
                if chain == ["np", "any"]:
                    return Call("ac_any", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "empty"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple) and len(expr.args[0].elts) == 2:
                        return Call("ac_empty2", tuple(self._lower_expr(elt) for elt in expr.args[0].elts))
                    return Call("ac_empty", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "full"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple) and len(expr.args[0].elts) == 2:
                        return Call("ac_full", (*tuple(self._lower_expr(elt) for elt in expr.args[0].elts), self._lower_expr(expr.args[1])))
                    return Call("ac_full", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "arange"]:
                    if expr.args and all(self._infer_type(arg) == ScalarType.INTEGER for arg in expr.args):
                        if len(expr.args) == 1:
                            return Call("ac_arange_int", (Constant(0), self._lower_expr(expr.args[0])))
                        if len(expr.args) == 2:
                            return Call("ac_arange_int", tuple(self._lower_expr(arg) for arg in expr.args))
                    if len(expr.args) == 1:
                        return Call("ac_arange", (Constant(0), self._lower_expr(expr.args[0])))
                    return Call("ac_arange", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "linspace"]:
                    return Call("ac_linspace", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "column_stack"]:
                    if expr.args and isinstance(expr.args[0], ast.Tuple):
                        return Call("ac_column_stack", tuple(self._lower_expr(elt) for elt in expr.args[0].elts))
                    return Call("ac_column_stack", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "concatenate"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    axis_value = self._const_int_value(axis_expr) if axis_expr is not None else 0
                    arrays = expr.args[0].elts if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) else expr.args
                    helper = "ac_concatenate_axis0" if axis_value == 0 else "ac_concatenate_axis1"
                    return Call(helper, tuple(self._lower_expr(arg) for arg in arrays))
                if chain == ["np", "hstack"]:
                    arrays = expr.args[0].elts if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) else expr.args
                    return Call("ac_hstack", tuple(self._lower_expr(arg) for arg in arrays))
                if chain == ["np", "vstack"]:
                    arrays = expr.args[0].elts if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) else expr.args
                    return Call("ac_vstack", tuple(self._lower_expr(arg) for arg in arrays))
                if chain == ["np", "stack"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    axis_value = self._const_int_value(axis_expr) if axis_expr is not None else 0
                    arrays = expr.args[0].elts if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) else expr.args
                    helper = "ac_stack_axis0" if axis_value == 0 else "ac_stack_axis2"
                    return Call(helper, tuple(self._lower_expr(arg) for arg in arrays))
                if chain == ["np", "round"]:
                    return Call("ac_round", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "clip"]:
                    return Call("ac_clip", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "repeat"]:
                    axis_expr = self._keyword_value(expr, "axis")
                    if axis_expr is not None:
                        return Call("ac_repeat_axis", (self._lower_expr(expr.args[0]), self._lower_expr(expr.args[1]), self._lower_expr(axis_expr)))
                    return Call("ac_repeat", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "tile"]:
                    return Call("ac_tile", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "take"]:
                    return Call("ac_take", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "broadcast_to"]:
                    if len(expr.args) != 2 or not isinstance(expr.args[1], ast.Tuple) or len(expr.args[1].elts) != 2:
                        raise NotImplementedError("np.broadcast_to currently requires a 2D target shape")
                    return Call("spread", (self._lower_expr(expr.args[0]), Constant(1), self._lower_expr(expr.args[1].elts[0])))
                if chain == ["np", "histogram"]:
                    raise NotImplementedError("np.histogram is only supported in tuple assignment")
                if chain == ["np", "pad"]:
                    if len(expr.args) < 2 or not isinstance(expr.args[1], ast.Tuple) or len(expr.args[1].elts) != 2:
                        raise NotImplementedError("np.pad currently requires a 2-tuple pad width")
                    const_expr = self._keyword_value(expr, "constant_values")
                    constant_value = self._lower_expr(const_expr) if const_expr is not None else Constant(0)
                    return Call(
                        "ac_pad",
                        (
                            self._lower_expr(expr.args[0]),
                            self._lower_expr(expr.args[1].elts[0]),
                            self._lower_expr(expr.args[1].elts[1]),
                            constant_value,
                        ),
                    )
                if chain == ["np", "roll"]:
                    return Call("ac_roll", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "flip"]:
                    return Call("ac_reverse", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "triu"]:
                    return Call("ac_triu", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "tril"]:
                    return Call("ac_tril", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "cumsum"]:
                    return Call("ac_cumsum", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "cumprod"]:
                    return Call("ac_cumprod", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "diff"]:
                    return Call("ac_diff", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "gradient"]:
                    return Call("ac_gradient", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "swapaxes"]:
                    return Call("ac_swapaxes", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "add", "reduceat"]:
                    return Call("ac_reduceat_add", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "ascontiguousarray"] or chain == ["np", "asfortranarray"]:
                    return Call("ac_copy", (self._lower_expr(expr.args[0]),))
                if chain == ["np", "linalg", "det"]:
                    return Call("ac_det", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "linalg", "cholesky"]:
                    return Call("ac_cholesky", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "linalg", "eig"]:
                    raise NotImplementedError("np.linalg.eig is only supported in tuple assignment")
                if chain == ["np", "linalg", "svd"]:
                    raise NotImplementedError("np.linalg.svd is only supported in tuple assignment")
                if chain == ["np", "linalg", "solve"]:
                    return Call("ac_solve_linear", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["np", "linalg", "norm"]:
                    return Call("ac_norm", tuple(self._lower_expr(arg) for arg in expr.args))
                if chain == ["random", "Random"]:
                    return Call("ac_random_init", self._lower_call_args(expr))
                if chain[:1] == ["math"]:
                    return Call(chain[1], self._lower_call_args(expr))
            raise NotImplementedError(f"unsupported call: {ast.dump(expr)}")
        if isinstance(expr, ast.Attribute):
            dtype_name = self._numpy_dtype_name(expr)
            if dtype_name is not None:
                return Constant(dtype_name)
            flag_value = self._infer_contiguity_flag(expr)
            if flag_value is not None:
                return Constant(flag_value)
            if self._attr_chain(expr) == ["np", "inf"]:
                return Constant(float("inf"))
            if self._attr_chain(expr) == ["np", "nan"]:
                return Call("ac_nan", ())
            if self._attr_chain(expr) == ["np", "pi"]:
                return Constant(3.141592653589793)
            if self._attr_chain(expr) == ["sys", "argv"]:
                return ValueRef("ac_argv")
            if expr.attr == "dtype" and isinstance(expr.value, ast.Name):
                dtype_name = self._current_dtypes.get(expr.value.id)
                if dtype_name is not None:
                    return Constant(dtype_name)
            base = self._lower_expr(expr.value)
            if expr.attr == "T":
                return Call("transpose", (base,))
            if expr.attr == "size":
                return Call("size", (base,))
            if expr.attr == "ndim":
                return Call("ac_ndim", (base,))
            if expr.attr == "shape":
                return Call("ac_shape", (base,))
        raise NotImplementedError(f"unsupported expression: {ast.dump(expr)}")

    def _lower_call_args(self, expr: ast.Call) -> tuple[object, ...]:
        if isinstance(expr.func, ast.Name) and expr.func.id in self._function_arg_names:
            arg_names = self._function_arg_names[expr.func.id]
            defaults = self._function_arg_defaults.get(expr.func.id, [None] * len(arg_names))
            values: list[object | None] = [None] * len(arg_names)
            for index, arg in enumerate(expr.args):
                if index < len(values):
                    values[index] = self._lower_expr(arg)
            for keyword in expr.keywords:
                if keyword.arg in arg_names:
                    values[arg_names.index(keyword.arg)] = self._lower_expr(keyword.value)
                else:
                    values.append(self._lower_expr(keyword.value))
            for index, default in enumerate(defaults):
                if index < len(values) and values[index] is None:
                    values[index] = default
            return tuple(value for value in values if value is not None)
        values = [self._lower_expr(arg) for arg in expr.args]
        values.extend(self._lower_expr(keyword.value) for keyword in expr.keywords)
        if isinstance(expr.func, ast.Name) and expr.func.id in self._function_arg_defaults:
            defaults = self._function_arg_defaults[expr.func.id]
            for index in range(len(values), len(defaults)):
                default = defaults[index]
                if default is not None:
                    values.append(default)
        return tuple(values)

    def _lower_index_expr(self, expr: ast.AST) -> object:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            chain = self._attr_chain(expr.func)
            if chain == ["np", "arange"]:
                return Call("ac_arange_int", tuple(self._lower_expr(arg) for arg in expr.args))
        if isinstance(expr, ast.BinOp):
            op_map = {
                ast.Add: BinaryOperator.ADD,
                ast.Sub: BinaryOperator.SUB,
                ast.Mult: BinaryOperator.MUL,
                ast.Div: BinaryOperator.DIV,
                ast.Pow: BinaryOperator.POW,
                ast.Mod: BinaryOperator.MOD,
                ast.BitAnd: BinaryOperator.AND,
                ast.BitOr: BinaryOperator.OR,
            }
            return BinaryOp(self._lower_index_expr(expr.left), op_map[type(expr.op)], self._lower_index_expr(expr.right))
        if isinstance(expr, ast.UnaryOp):
            op_map = {
                ast.USub: UnaryOperator.MINUS,
                ast.UAdd: UnaryOperator.PLUS,
                ast.Not: UnaryOperator.NOT,
                ast.Invert: UnaryOperator.NOT,
            }
            return UnaryOp(op_map[type(expr.op)], self._lower_index_expr(expr.operand))
        return self._lower_expr(expr)

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
        if isinstance(expr, ast.Compare):
            left_type = self._infer_type(expr.left)
            comparator_types = [self._infer_type(comp) for comp in expr.comparators]
            array_types = [typ for typ in [left_type, *comparator_types] if isinstance(typ, ArrayTypeRef)]
            if array_types:
                return ArrayTypeRef(ScalarType.LOGICAL, rank=max(typ.rank for typ in array_types))
            return ScalarType.LOGICAL
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.MatMult):
            left_type = self._infer_type(expr.left)
            right_type = self._infer_type(expr.right)
            if isinstance(left_type, ArrayTypeRef) and isinstance(right_type, ArrayTypeRef):
                if left_type.rank == 1 and right_type.rank == 1:
                    return ScalarType.REAL64
                if (left_type.rank == 1 and right_type.rank == 2) or (left_type.rank == 2 and right_type.rank == 1):
                    return ArrayTypeRef(ScalarType.REAL64, rank=1)
                if left_type.rank == 2 and right_type.rank == 2:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
            return ScalarType.REAL64
        if isinstance(expr, ast.BoolOp):
            value_types = [self._infer_type(value) for value in expr.values]
            array_types = [typ for typ in value_types if isinstance(typ, ArrayTypeRef)]
            if array_types:
                return ArrayTypeRef(ScalarType.LOGICAL, rank=max(typ.rank for typ in array_types))
            return ScalarType.LOGICAL
        if isinstance(expr, ast.List):
            return ArrayTypeRef(ScalarType.REAL64)
        if isinstance(expr, ast.ListComp):
            return ArrayTypeRef(ScalarType.REAL64)
        if isinstance(expr, ast.Subscript):
            if self._is_np_r_concat(expr):
                return ArrayTypeRef(ScalarType.REAL64)
            if self._is_pairwise_item2_access(expr):
                container_type = self._infer_type(expr.value)
                if isinstance(container_type, ArrayTypeRef):
                    return ArrayTypeRef(container_type.element_type, rank=1)
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
            if self._is_item2_access(expr):
                container_type = self._infer_type(expr.value)
                if isinstance(container_type, ArrayTypeRef):
                    return container_type.element_type
                return ScalarType.REAL64
            if self._is_item3_access(expr):
                container_type = self._infer_type(expr.value)
                if isinstance(container_type, ArrayTypeRef):
                    return container_type.element_type
                return ScalarType.REAL64
            if self._is_boolean_mask_access(expr):
                container_type = self._infer_type(expr.value)
                if isinstance(container_type, ArrayTypeRef):
                    return ArrayTypeRef(container_type.element_type, rank=1)
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
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
                if isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, str):
                    record = self._record_defs.get(container_type.name)
                    if record is not None:
                        for field_name, field_type in record.fields:
                            if field_name == expr.slice.value:
                                return field_type
            if isinstance(container_type, ArrayTypeRef):
                if self._is_row_gather(expr):
                    return ArrayTypeRef(container_type.element_type, rank=container_type.rank)
                index_type = self._infer_type(expr.slice)
                if isinstance(index_type, ArrayTypeRef):
                    return ArrayTypeRef(container_type.element_type, rank=container_type.rank)
                if container_type.rank > 1 and not isinstance(expr.slice, ast.Tuple):
                    return ArrayTypeRef(container_type.element_type, rank=container_type.rank - 1)
                return container_type.element_type
            return ScalarType.REAL64
        if isinstance(expr, ast.Dict):
            return self._function_result_types.get(self._current_function or "", ScalarType.REAL64)
        if isinstance(expr, ast.Tuple):
            if expr.elts and all(self._infer_type(value) == ScalarType.INTEGER for value in expr.elts):
                return self._shape_record_type(expr, minimum_rank=len(expr.elts))
            elt_types = [self._infer_type(value) for value in expr.elts]
            if elt_types and all(isinstance(elt_type, ScalarType) for elt_type in elt_types) and all(
                elt_type == elt_types[0] for elt_type in elt_types
            ):
                return ArrayTypeRef(elt_types[0], rank=1)
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
            if isinstance(expr.func, ast.Name) and expr.func.id == "abs" and expr.args:
                arg_type = self._infer_type(expr.args[0])
                if arg_type == ScalarType.INTEGER:
                    return ScalarType.INTEGER
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(ScalarType.REAL64, rank=arg_type.rank)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_shape":
                return self._shape_record_type(expr.args[0] if expr.args else None)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_shape_dim":
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id in {"min", "max", "minval", "maxval"} and expr.args:
                arg_types = [self._infer_type(arg) for arg in expr.args]
                if all(arg_type == ScalarType.INTEGER for arg_type in arg_types):
                    return ScalarType.INTEGER
                if any(isinstance(arg_type, ArrayTypeRef) for arg_type in arg_types):
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_argmin", "ac_argmax"}:
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_var", "ac_std"}:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_trace":
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_outer":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_kron":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_sum_axis", "ac_mean_axis", "ac_min_axis", "ac_max_axis"}:
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_concatenate_axis0", "ac_concatenate_axis1", "ac_hstack", "ac_vstack"}:
                first_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64, rank=2)
                if isinstance(first_type, ArrayTypeRef):
                    return ArrayTypeRef(first_type.element_type, rank=2)
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_stack_axis0", "ac_stack_axis2"}:
                first_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64, rank=2)
                element_type = first_type.element_type if isinstance(first_type, ArrayTypeRef) else ScalarType.REAL64
                return ArrayTypeRef(element_type, rank=3)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_cumsum", "ac_cumprod", "ac_diff", "ac_gradient"}:
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_take", "ac_pad", "ac_roll"}:
                arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                return arg_type if isinstance(arg_type, ArrayTypeRef) else ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id in {"ac_floor", "ac_ceil"}:
                arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                return arg_type
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_isnan":
                arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(ScalarType.LOGICAL, rank=arg_type.rank)
                return ScalarType.LOGICAL
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_isfinite":
                arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(ScalarType.LOGICAL, rank=arg_type.rank)
                return ScalarType.LOGICAL
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_nansum":
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_random_integers":
                return ArrayTypeRef(ScalarType.INTEGER)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_choice_replace":
                return ArrayTypeRef(ScalarType.INTEGER)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_diag":
                arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                if isinstance(arg_type, ArrayTypeRef) and arg_type.rank == 1:
                    return ArrayTypeRef(arg_type.element_type, rank=2)
                if isinstance(arg_type, ArrayTypeRef) and arg_type.rank == 2:
                    return ArrayTypeRef(arg_type.element_type, rank=1)
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_reduceat_add":
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_histogram_counts":
                return ArrayTypeRef(ScalarType.INTEGER)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_histogram_edges":
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_det":
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_loadtxt_1d":
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_loadtxt":
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_reshape":
                if len(expr.args) == 2:
                    return ArrayTypeRef(ScalarType.REAL64)
                if len(expr.args) == 3 and all(self._infer_type(arg) == ScalarType.INTEGER for arg in expr.args[1:]):
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if len(expr.args) == 4 and all(self._infer_type(arg) == ScalarType.INTEGER for arg in expr.args[1:4]):
                    return ArrayTypeRef(ScalarType.REAL64, rank=3)
                base_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                if isinstance(base_type, ArrayTypeRef):
                    return base_type
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Name) and expr.func.id == "spread" and expr.args:
                arg_type = self._infer_type(expr.args[0]) if isinstance(expr.args[0], ast.AST) else ScalarType.REAL64
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(arg_type.element_type, rank=arg_type.rank + 1)
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(expr.func, ast.Name) and expr.func.id == "transpose" and expr.args:
                arg_type = self._infer_type(expr.args[0])
                if isinstance(arg_type, ArrayTypeRef):
                    return ArrayTypeRef(arg_type.element_type, rank=arg_type.rank)
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            chain = self._attr_chain(expr.func) if isinstance(expr.func, ast.Attribute) else []
            if chain in (["np", "array"], ["np", "asarray"]):
                element_type = self._infer_numpy_array_element_type(expr)
                if expr.args:
                    return ArrayTypeRef(element_type, rank=self._infer_array_rank(expr.args[0]))
                return ArrayTypeRef(element_type)
            if chain == ["np", "linalg", "solve"]:
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
            if chain == ["np", "linalg", "norm"]:
                return ScalarType.REAL64
            if chain == ["np", "linalg", "det"]:
                return ScalarType.REAL64
            if chain == ["np", "linalg", "cholesky"]:
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if chain in (["np", "zeros"], ["np", "ones"], ["np", "empty"], ["np", "full"]):
                element_type = self._infer_numpy_array_element_type(expr)
                if expr.args:
                    return ArrayTypeRef(element_type, rank=self._infer_shape_arg_rank(expr.args[0]))
                return ArrayTypeRef(element_type)
            if chain == ["np", "arange"]:
                if expr.args and all(self._infer_type(arg) == ScalarType.INTEGER for arg in expr.args):
                    return ArrayTypeRef(ScalarType.INTEGER)
                return ArrayTypeRef(ScalarType.REAL64)
            if chain == ["np", "linspace"]:
                return ArrayTypeRef(ScalarType.REAL64)
            if chain == ["np", "clip"]:
                arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                return arg_type
            if isinstance(expr.func, ast.Attribute) and self._is_strip_lower_chain(expr.func):
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "strip":
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "rstrip":
                return ScalarType.STRING
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "gauss" and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "normal" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is not None:
                    if isinstance(size_expr, ast.Tuple) and len(size_expr.elts) == 2:
                        return ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "standard_normal" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is None and expr.args:
                    size_expr = expr.args[0]
                if size_expr is not None:
                    if isinstance(size_expr, ast.Tuple) and len(size_expr.elts) == 2:
                        return ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "uniform" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is not None:
                    if isinstance(size_expr, ast.Tuple) and len(size_expr.elts) == 2:
                        return ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "integers" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is not None:
                    if isinstance(size_expr, ast.Tuple) and len(size_expr.elts) == 2:
                        return ArrayTypeRef(ScalarType.INTEGER, rank=2)
                    return ArrayTypeRef(ScalarType.INTEGER)
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "sum" and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "mean" and chain[:1] != ["np"]:
                axis_expr = self._keyword_value(expr, "axis")
                keepdims_expr = self._keyword_value(expr, "keepdims")
                if axis_expr is not None:
                    if isinstance(keepdims_expr, ast.Constant) and keepdims_expr.value is True:
                        return ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr in {"min", "max"} and chain[:1] != ["np"]:
                axis_expr = self._keyword_value(expr, "axis")
                if axis_expr is not None:
                    return ArrayTypeRef(ScalarType.REAL64)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr in {"var", "std"} and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "copy" and chain[:1] != ["np"]:
                return self._infer_type(expr.func.value)
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "astype" and chain[:1] != ["np"]:
                if expr.args:
                    dtype_name = self._numpy_dtype_name(expr.args[0]) if isinstance(expr.args[0], ast.Attribute) else None
                    if isinstance(expr.args[0], ast.Name) and expr.args[0].id == "int":
                        dtype_name = "int"
                    if isinstance(expr.args[0], ast.Name) and expr.args[0].id == "float":
                        dtype_name = "float"
                    if dtype_name in {"int32", "int64", "int"}:
                        value_type = self._infer_type(expr.func.value)
                        if isinstance(value_type, ArrayTypeRef):
                            return ArrayTypeRef(ScalarType.INTEGER, rank=value_type.rank)
                        return ScalarType.INTEGER
                    if dtype_name in {"float32", "float64", "float"}:
                        value_type = self._infer_type(expr.func.value)
                        if isinstance(value_type, ArrayTypeRef):
                            return ArrayTypeRef(ScalarType.REAL64, rank=value_type.rank)
                        return ScalarType.REAL64
                return self._infer_type(expr.func.value)
            if isinstance(expr.func, ast.Attribute) and expr.func.attr in {"ravel", "flatten"} and chain[:1] != ["np"]:
                value_type = self._infer_type(expr.func.value)
                if isinstance(value_type, ArrayTypeRef):
                    return ArrayTypeRef(value_type.element_type, rank=1)
                return ArrayTypeRef(ScalarType.REAL64)
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "reshape" and chain[:1] != ["np"]:
                value_type = self._infer_type(expr.func.value)
                element_type = value_type.element_type if isinstance(value_type, ArrayTypeRef) else ScalarType.REAL64
                return ArrayTypeRef(element_type, rank=self._reshape_rank(expr))
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "transpose" and chain[:1] != ["np"]:
                base_type = self._infer_type(expr.func.value)
                if isinstance(base_type, ArrayTypeRef):
                    return base_type
                return ArrayTypeRef(ScalarType.REAL64, rank=max(2, len(expr.args)))
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "dot" and chain[:1] != ["np"]:
                base_type = self._infer_type(expr.func.value)
                other_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                if isinstance(base_type, ArrayTypeRef) and isinstance(other_type, ArrayTypeRef):
                    if base_type.rank == 1 and other_type.rank == 1:
                        return ScalarType.REAL64
                    if base_type.rank == 2 and other_type.rank == 2:
                        return ArrayTypeRef(ScalarType.REAL64, rank=2)
                    if base_type.rank == 2 and other_type.rank == 1:
                        return ArrayTypeRef(ScalarType.REAL64, rank=1)
                    if base_type.rank == 1 and other_type.rank == 2:
                        return ArrayTypeRef(ScalarType.REAL64, rank=1)
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "trace" and chain[:1] != ["np"]:
                return ScalarType.REAL64
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "choice" and chain[:1] != ["np"]:
                size_expr = self._keyword_value(expr, "size")
                if size_expr is not None:
                    return ArrayTypeRef(ScalarType.INTEGER)
                return ScalarType.INTEGER
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "shuffle" and chain[:1] != ["np"]:
                return ScalarType.NONE
            if isinstance(expr.func, ast.Attribute):
                if chain == ["np", "random", "default_rng"]:
                    return RecordTypeRef("ac_random_state")
                if chain == ["np", "random", "multivariate_normal"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "loadtxt"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "cov"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "meshgrid"]:
                    return self._ensure_record(
                        "ac_meshgrid_result",
                        [("item1", ArrayTypeRef(ScalarType.REAL64, rank=2)), ("item2", ArrayTypeRef(ScalarType.REAL64, rank=2))],
                    )
                if chain == ["np", "histogram"]:
                    return self._ensure_record(
                        "ac_histogram_result",
                        [("item1", ArrayTypeRef(ScalarType.INTEGER)), ("item2", ArrayTypeRef(ScalarType.REAL64))],
                    )
                if chain == ["np", "broadcast_to"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "add", "reduceat"]:
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "ascontiguousarray"] or chain == ["np", "asfortranarray"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    return arg_type
                if chain == ["np", "linalg", "det"]:
                    return ScalarType.REAL64
                if chain == ["np", "linalg", "eig"]:
                    return self._ensure_record(
                        "ac_eig_result",
                        [("item1", ArrayTypeRef(ScalarType.REAL64)), ("item2", ArrayTypeRef(ScalarType.REAL64, rank=2))],
                    )
                if chain == ["np", "linalg", "svd"]:
                    return self._ensure_record(
                        "ac_svd_result",
                        [("item1", ArrayTypeRef(ScalarType.REAL64, rank=2)), ("item2", ArrayTypeRef(ScalarType.REAL64)), ("item3", ArrayTypeRef(ScalarType.REAL64, rank=2))],
                    )
                if chain == ["np", "linalg", "slogdet"]:
                    return self._ensure_record("ac_slogdet_result", [("item1", ScalarType.REAL64), ("item2", ScalarType.REAL64)])
                if chain == ["np", "linalg", "inv"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "linalg", "cholesky"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "einsum"]:
                    if expr.args and isinstance(expr.args[0], ast.Constant) and expr.args[0].value == "ii->":
                        return ScalarType.REAL64
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "max"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "min"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "sum"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "mean"]:
                    keepdims_expr = self._keyword_value(expr, "keepdims")
                    if self._keyword_value(expr, "axis") is not None:
                        if isinstance(keepdims_expr, ast.Constant) and keepdims_expr.value is True:
                            return ArrayTypeRef(ScalarType.REAL64, rank=2)
                        return ArrayTypeRef(ScalarType.REAL64)
                    return ScalarType.REAL64
                if chain == ["np", "argmin"] or chain == ["np", "argmax"]:
                    if self._keyword_value(expr, "axis") is not None:
                        return ArrayTypeRef(ScalarType.INTEGER)
                    return ScalarType.INTEGER
                if chain == ["np", "cumsum"] or chain == ["np", "cumprod"] or chain == ["np", "diff"] or chain == ["np", "gradient"]:
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "argsort"]:
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "sort"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    return arg_type
                if chain == ["np", "unique"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(arg_type.element_type)
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "bincount"]:
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "searchsorted"]:
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "ndim"]:
                    return ScalarType.INTEGER
                if chain == ["np", "eye"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "diag"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    if isinstance(arg_type, ArrayTypeRef) and arg_type.rank == 1:
                        return ArrayTypeRef(arg_type.element_type, rank=2)
                    if isinstance(arg_type, ArrayTypeRef) and arg_type.rank == 2:
                        return ArrayTypeRef(arg_type.element_type, rank=1)
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "atleast_2d"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "log"] or chain == ["np", "exp"] or chain == ["np", "sqrt"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "floor"] or chain == ["np", "ceil"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "sign"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(ScalarType.REAL64, rank=arg_type.rank)
                    return ScalarType.REAL64
                if chain == ["np", "abs"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(ScalarType.REAL64, rank=arg_type.rank)
                    return ScalarType.REAL64
                if chain == ["np", "all"]:
                    return ScalarType.LOGICAL
                if chain == ["np", "any"]:
                    return ScalarType.LOGICAL
                if chain == ["np", "isnan"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(ScalarType.LOGICAL, rank=arg_type.rank)
                    return ScalarType.LOGICAL
                if chain == ["np", "isfinite"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(ScalarType.LOGICAL, rank=arg_type.rank)
                    return ScalarType.LOGICAL
                if chain == ["np", "nansum"]:
                    return ScalarType.REAL64
                if chain == ["np", "copy"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "clip"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "repeat"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    return arg_type
                if chain == ["np", "tile"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    return arg_type
                if chain == ["np", "take"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                    if isinstance(arg_type, ArrayTypeRef):
                        return ArrayTypeRef(arg_type.element_type, rank=1)
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "pad"] or chain == ["np", "roll"] or chain == ["np", "flip"]:
                    return self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "triu"] or chain == ["np", "tril"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return arg_type
                if chain == ["np", "dot"]:
                    left_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    right_type = self._infer_type(expr.args[1]) if len(expr.args) > 1 else ScalarType.REAL64
                    if isinstance(left_type, ArrayTypeRef) and isinstance(right_type, ArrayTypeRef):
                        if left_type.rank == 1 and right_type.rank == 1:
                            return ScalarType.REAL64
                        if left_type.rank == 2 and right_type.rank == 2:
                            return ArrayTypeRef(ScalarType.REAL64, rank=2)
                        if left_type.rank == 2 and right_type.rank == 1:
                            return ArrayTypeRef(ScalarType.REAL64, rank=1)
                        if left_type.rank == 1 and right_type.rank == 2:
                            return ArrayTypeRef(ScalarType.REAL64, rank=1)
                    return ScalarType.REAL64
                if chain == ["np", "trace"]:
                    return ScalarType.REAL64
                if chain == ["np", "outer"] or chain == ["np", "kron"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "array"] or chain == ["np", "asarray"]:
                    element_type = self._infer_numpy_array_element_type(expr)
                    if expr.args:
                        return ArrayTypeRef(element_type, rank=self._infer_array_rank(expr.args[0]))
                    return ArrayTypeRef(element_type)
                if chain == ["np", "roots"]:
                    return ArrayTypeRef(ScalarType.COMPLEX128)
                if chain == ["np", "zeros"]:
                    element_type = self._infer_numpy_array_element_type(expr)
                    if expr.args:
                        return ArrayTypeRef(element_type, rank=self._infer_shape_arg_rank(expr.args[0]))
                    return ArrayTypeRef(element_type)
                if chain == ["np", "ones"]:
                    element_type = self._infer_numpy_array_element_type(expr)
                    if expr.args:
                        return ArrayTypeRef(element_type, rank=self._infer_shape_arg_rank(expr.args[0]))
                    return ArrayTypeRef(element_type)
                if chain == ["np", "where"]:
                    if len(expr.args) == 3:
                        true_type = self._infer_type(expr.args[1])
                        false_type = self._infer_type(expr.args[2])
                        if isinstance(true_type, ArrayTypeRef):
                            return true_type
                        if isinstance(false_type, ArrayTypeRef):
                            return false_type
                        return true_type
                    return ArrayTypeRef(ScalarType.INTEGER)
                if chain == ["np", "empty"] or chain == ["np", "full"]:
                    element_type = self._infer_numpy_array_element_type(expr)
                    if expr.args:
                        return ArrayTypeRef(element_type, rank=self._infer_shape_arg_rank(expr.args[0]))
                    return ArrayTypeRef(element_type)
                if chain == ["np", "arange"]:
                    if expr.args and all(self._infer_type(arg) == ScalarType.INTEGER for arg in expr.args):
                        return ArrayTypeRef(ScalarType.INTEGER)
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "linspace"]:
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "column_stack"]:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "concatenate"] or chain == ["np", "hstack"] or chain == ["np", "vstack"]:
                    first_type = self._infer_type(expr.args[0].elts[0]) if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) and expr.args[0].elts else ArrayTypeRef(ScalarType.REAL64, rank=2)
                    if isinstance(first_type, ArrayTypeRef):
                        return ArrayTypeRef(first_type.element_type, rank=2)
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
                if chain == ["np", "stack"]:
                    first_type = self._infer_type(expr.args[0].elts[0]) if expr.args and isinstance(expr.args[0], (ast.List, ast.Tuple)) and expr.args[0].elts else ArrayTypeRef(ScalarType.REAL64, rank=2)
                    element_type = first_type.element_type if isinstance(first_type, ArrayTypeRef) else ScalarType.REAL64
                    return ArrayTypeRef(element_type, rank=3)
                if chain == ["np", "round"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ScalarType.REAL64
                    return arg_type
                if chain == ["np", "swapaxes"]:
                    arg_type = self._infer_type(expr.args[0]) if expr.args else ArrayTypeRef(ScalarType.REAL64, rank=2)
                    return arg_type
                if chain == ["np", "linalg", "solve"]:
                    return ArrayTypeRef(ScalarType.REAL64)
                if chain == ["np", "linalg", "norm"]:
                    return ScalarType.REAL64
                if chain == ["random", "Random"]:
                    return RecordTypeRef("ac_random_state")
                if chain[:1] == ["math"]:
                    return ScalarType.REAL64
            if isinstance(expr.func, ast.Name) and expr.func.id == "ac_loadtxt_1d":
                return ArrayTypeRef(ScalarType.REAL64, rank=1)
            if isinstance(expr.func, ast.Name):
                return self._infer_user_call_result_type(expr.func.id, list(expr.args)) or self._function_result_types.get(expr.func.id, ScalarType.REAL64)
            return ScalarType.REAL64
        if isinstance(expr, ast.Name):
            return self._current_symbols.get(expr.id, ScalarType.REAL64)
        if isinstance(expr, ast.Attribute):
            if self._numpy_dtype_name(expr) is not None:
                return ScalarType.STRING
            if self._infer_contiguity_flag(expr) is not None:
                return ScalarType.LOGICAL
            if self._attr_chain(expr) == ["np", "inf"]:
                return ScalarType.REAL64
            if self._attr_chain(expr) == ["np", "nan"]:
                return ScalarType.REAL64
            if self._attr_chain(expr) == ["np", "pi"]:
                return ScalarType.REAL64
            if self._attr_chain(expr) == ["sys", "argv"]:
                return ArrayTypeRef(ScalarType.STRING)
            if expr.attr == "T":
                base_type = self._infer_type(expr.value)
                if isinstance(base_type, ArrayTypeRef):
                    return base_type
                return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if expr.attr in {"size", "ndim"}:
                return ScalarType.INTEGER
            if expr.attr == "shape":
                return self._shape_record_type(expr.value)
            if expr.attr == "dtype":
                return ScalarType.STRING
        if isinstance(expr, ast.UnaryOp):
            if isinstance(expr.op, ast.Not):
                return ScalarType.LOGICAL
            operand_type = self._infer_type(expr.operand)
            if isinstance(operand_type, ArrayTypeRef):
                return operand_type
            if operand_type == ScalarType.INTEGER:
                return ScalarType.INTEGER
            return ScalarType.REAL64
        if isinstance(expr, ast.BinOp):
            left_type = self._infer_type(expr.left)
            right_type = self._infer_type(expr.right)
            if isinstance(expr.op, ast.MatMult):
                return self._infer_matmul_type(left_type, right_type)
            if isinstance(expr.op, ast.Add) and (left_type == ScalarType.STRING or right_type == ScalarType.STRING):
                return ScalarType.STRING
            if isinstance(left_type, ArrayTypeRef) or isinstance(right_type, ArrayTypeRef):
                if isinstance(expr.op, (ast.BitAnd, ast.BitOr)):
                    return self._combine_array_types(
                        ArrayTypeRef(ScalarType.LOGICAL, getattr(left_type, "rank", 1)) if not isinstance(left_type, ArrayTypeRef) else left_type,
                        ArrayTypeRef(ScalarType.LOGICAL, getattr(right_type, "rank", 1)) if not isinstance(right_type, ArrayTypeRef) else right_type,
                    )
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
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval")
            except SyntaxError:
                return None
            return self._map_annotation(parsed.body, function_name)
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
            if annotation.value.id == "Final":
                return self._map_annotation(annotation.slice, function_name)
            if annotation.value.id in {"Array1D", "Array2D", "Array3D"}:
                rank = {"Array1D": 1, "Array2D": 2, "Array3D": 3}[annotation.value.id]
                element_type = self._map_annotation(annotation.slice, function_name) or ScalarType.REAL64
                if isinstance(element_type, ArrayTypeRef):
                    element_type = element_type.element_type
                return ArrayTypeRef(element_type, rank=rank)
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

    def _iter_loop_index_name(self, name: str) -> str:
        return f"{name}_index"

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
            if expr.elts:
                first_type = self._infer_type(expr.elts[0])
                if isinstance(first_type, ArrayTypeRef):
                    return first_type.rank + 1
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
        if isinstance(expr, (ast.Tuple, ast.List)):
            rank = len(expr.elts)
        else:
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

    def _numpy_dtype_name(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name) and expr.id in {"int", "float"}:
            return expr.id
        if not isinstance(expr, ast.Attribute):
            return None
        chain = self._attr_chain(expr)
        mapping = {
            ("np", "int32"): "int32",
            ("np", "int64"): "int64",
            ("np", "float32"): "float32",
            ("np", "float64"): "float64",
        }
        return mapping.get(tuple(chain))

    def _infer_dtype_name(self, expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            chain = self._attr_chain(expr.func)
            if chain == ["np", "array"] or chain == ["np", "asarray"]:
                dtype_expr = self._keyword_value(expr, "dtype")
                return self._numpy_dtype_name(dtype_expr) if dtype_expr is not None else None
            if chain == ["np", "ones"] or chain == ["np", "zeros"] or chain == ["np", "full"]:
                dtype_expr = self._keyword_value(expr, "dtype")
                return self._numpy_dtype_name(dtype_expr) if dtype_expr is not None else None
            if expr.func.attr == "astype" and chain[:1] != ["np"] and expr.args:
                return self._numpy_dtype_name(expr.args[0])
        if isinstance(expr, ast.BinOp):
            return self._infer_dtype_name(expr.left) or self._infer_dtype_name(expr.right)
        if isinstance(expr, ast.Name):
            return self._current_dtypes.get(expr.id)
        return None

    def _infer_numpy_array_element_type(self, expr: ast.Call) -> object:
        dtype_name = self._infer_dtype_name(expr)
        if dtype_name in {"int32", "int64", "int"}:
            return ScalarType.INTEGER
        if dtype_name in {"float32", "float64", "float"}:
            return ScalarType.REAL64
        if (
            isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "full"]
            and len(expr.args) >= 2
        ):
            return self._infer_literal_array_element_type(expr.args[1]) or self._infer_type(expr.args[1]) or ScalarType.REAL64
        if expr.args:
            return self._infer_literal_array_element_type(expr.args[0]) or ScalarType.REAL64
        return ScalarType.REAL64

    def _infer_literal_array_element_type(self, expr: ast.AST) -> object | None:
        if isinstance(expr, ast.List):
            element_types = [self._infer_literal_array_element_type(elt) for elt in expr.elts]
            element_types = [elt for elt in element_types if elt is not None]
            if element_types and all(elt == ScalarType.STRING for elt in element_types):
                return ScalarType.STRING
            if element_types and all(elt == ScalarType.INTEGER for elt in element_types):
                return ScalarType.INTEGER
            if element_types:
                return ScalarType.REAL64
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return ScalarType.LOGICAL
            if isinstance(expr.value, str):
                return ScalarType.STRING
            if isinstance(expr.value, int):
                return ScalarType.INTEGER
            if isinstance(expr.value, float):
                return ScalarType.REAL64
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
            if left_type.element_type == right_type.element_type:
                element_type = left_type.element_type
            elif (
                left_type.element_type == ScalarType.COMPLEX128
                or right_type.element_type == ScalarType.COMPLEX128
            ):
                element_type = ScalarType.COMPLEX128
            elif left_type.element_type == ScalarType.REAL64 or right_type.element_type == ScalarType.REAL64:
                element_type = ScalarType.REAL64
            elif left_type.element_type == ScalarType.INTEGER and right_type.element_type == ScalarType.INTEGER:
                element_type = ScalarType.INTEGER
            elif left_type.element_type == ScalarType.LOGICAL and right_type.element_type == ScalarType.LOGICAL:
                element_type = ScalarType.LOGICAL
            else:
                element_type = ScalarType.REAL64
            return ArrayTypeRef(element_type, rank=max(left_type.rank, right_type.rank))
        if isinstance(left_type, ArrayTypeRef):
            return ArrayTypeRef(left_type.element_type, rank=left_type.rank)
        if isinstance(right_type, ArrayTypeRef):
            return ArrayTypeRef(right_type.element_type, rank=right_type.rank)
        return ScalarType.REAL64

    def _merge_types(self, current: object | None, new_type: object | None) -> object | None:
        if current is None:
            return new_type
        if new_type is None or current == new_type:
            return current
        if isinstance(current, ArrayTypeRef) or isinstance(new_type, ArrayTypeRef):
            if isinstance(current, ArrayTypeRef) and isinstance(new_type, ArrayTypeRef):
                return self._combine_array_types(current, new_type)
            return new_type if isinstance(new_type, ArrayTypeRef) else current
        if isinstance(current, RecordTypeRef) or isinstance(new_type, RecordTypeRef):
            return new_type if isinstance(new_type, RecordTypeRef) else current
        if current == ScalarType.REAL64 and new_type == ScalarType.INTEGER:
            return ScalarType.REAL64
        if current == ScalarType.INTEGER and new_type == ScalarType.REAL64:
            return ScalarType.REAL64
        return new_type

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
            and isinstance(expr.test.left.args[0], ast.Attribute)
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
            and isinstance(stmt.test.left.args[0], ast.Attribute)
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

    def _is_meshgrid_call(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "meshgrid"]
            and len(expr.args) >= 2
        )

    def _is_histogram_call(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "histogram"]
            and len(expr.args) >= 1
        )

    def _is_eig_call(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "linalg", "eig"]
            and len(expr.args) == 1
        )

    def _is_svd_call(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "linalg", "svd"]
            and len(expr.args) == 1
        )

    def _lower_histogram_bins(self, expr: ast.Call) -> object:
        bins_expr = self._keyword_value(expr, "bins")
        if bins_expr is None:
            raise NotImplementedError("np.histogram currently requires explicit bins=")
        return self._lower_expr(bins_expr)

    def _infer_contiguity_flag(self, expr: ast.Attribute) -> bool | None:
        if expr.attr not in {"c_contiguous", "f_contiguous"}:
            return None
        if not isinstance(expr.value, ast.Attribute) or expr.value.attr != "flags":
            return None
        contiguity = self._infer_contiguity(expr.value.value)
        if contiguity is None:
            return None
        return contiguity[0] if expr.attr == "c_contiguous" else contiguity[1]

    def _infer_contiguity(self, expr: ast.AST) -> tuple[bool, bool] | None:
        if isinstance(expr, ast.Name):
            return self._current_contiguity.get(expr.id)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            chain = self._attr_chain(expr.func)
            if chain == ["np", "array"] or chain == ["np", "zeros"] or chain == ["np", "ones"] or chain == ["np", "full"] or chain == ["np", "empty"]:
                return (True, False)
            if chain == ["np", "arange"] or chain == ["np", "linspace"]:
                return (True, True)
            if chain == ["np", "ascontiguousarray"]:
                return (True, False)
            if chain == ["np", "asfortranarray"]:
                return (False, True)
            if expr.func.attr == "reshape" and chain[:1] != ["np"]:
                order_ast = self._keyword_value(expr, "order")
                if isinstance(order_ast, ast.Constant) and order_ast.value == "F":
                    return (False, True)
                return (True, False)
            if expr.func.attr == "transpose" and chain[:1] != ["np"]:
                base = self._infer_contiguity(expr.func.value)
                if base is None:
                    return None
                return (base[1], base[0])
        if isinstance(expr, ast.Attribute) and expr.attr == "T":
            base = self._infer_contiguity(expr.value)
            if base is None:
                return None
            return (base[1], base[0])
        return None

    def _is_identity_asarray_assignment(self, stmt: ast.Assign) -> bool:
        return (
            len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and self._attr_chain(stmt.value.func) == ["np", "asarray"]
            and stmt.value.args
            and isinstance(stmt.value.args[0], ast.Name)
            and stmt.targets[0].id == stmt.value.args[0].id
        )

    def _lower_slice_expr(self, expr: ast.Subscript) -> object | None:
        container_type = self._infer_type(expr.value)
        if (
            isinstance(expr.slice, ast.Slice)
            and expr.slice.lower is None
            and expr.slice.upper is None
            and isinstance(expr.slice.step, ast.UnaryOp)
            and isinstance(expr.slice.step.op, ast.USub)
            and isinstance(expr.slice.step.operand, ast.Constant)
            and expr.slice.step.operand.value == 1
        ):
            return Call("ac_reverse", (self._lower_expr(expr.value),))
        if isinstance(expr.slice, ast.Slice) and isinstance(container_type, ArrayTypeRef) and container_type.rank == 2:
            base = self._lower_expr(expr.value)
            row_start = self._lower_expr(expr.slice.lower) if expr.slice.lower is not None else Constant(0)
            row_stop = self._lower_expr(expr.slice.upper) if expr.slice.upper is not None else Call("ac_shape_dim", (base, Constant(1)))
            row_step = self._lower_expr(expr.slice.step) if expr.slice.step is not None else Constant(1)
            return Call(
                "ac_slice2",
                (
                    base,
                    row_start,
                    row_stop,
                    row_step,
                    Constant(0),
                    Call("ac_shape_dim", (base, Constant(2))),
                    Constant(1),
                ),
            )
        if self._is_1d_slice(expr):
            args = [
                self._lower_expr(expr.value),
                self._lower_expr(expr.slice.lower) if expr.slice.lower is not None else Constant(0),
                self._lower_expr(expr.slice.upper) if expr.slice.upper is not None else Call("size", (self._lower_expr(expr.value),)),
            ]
            if expr.slice.step is not None:
                args.append(self._lower_expr(expr.slice.step))
                return Call("ac_slice_step", tuple(args))
            return Call("ac_slice", tuple(args))
        if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
            row_spec, col_spec = expr.slice.elts
            if isinstance(row_spec, ast.Slice) and isinstance(col_spec, ast.Slice):
                base = self._lower_expr(expr.value)
                row_start = self._lower_expr(row_spec.lower) if row_spec.lower is not None else Constant(0)
                row_stop = self._lower_expr(row_spec.upper) if row_spec.upper is not None else Call("ac_shape_dim", (base, Constant(1)))
                row_step = self._lower_expr(row_spec.step) if row_spec.step is not None else Constant(1)
                col_start = self._lower_expr(col_spec.lower) if col_spec.lower is not None else Constant(0)
                col_stop = self._lower_expr(col_spec.upper) if col_spec.upper is not None else Call("ac_shape_dim", (base, Constant(2)))
                col_step = self._lower_expr(col_spec.step) if col_spec.step is not None else Constant(1)
                return Call("ac_slice2", (base, row_start, row_stop, row_step, col_start, col_stop, col_step))
            if isinstance(row_spec, ast.Constant) and row_spec.value is None:
                if isinstance(col_spec, ast.Slice) and col_spec.lower is None and col_spec.upper is None and col_spec.step is None:
                    return Call("ac_row_axis", (Call("ac_asarray", (self._lower_expr(expr.value),)),))
            if isinstance(row_spec, ast.Slice) and row_spec.lower is None and row_spec.upper is None and row_spec.step is None:
                if isinstance(col_spec, ast.Constant) and col_spec.value is None:
                    return Call("ac_add_axis", (Call("ac_asarray", (self._lower_expr(expr.value),)),))
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
        if self._is_1d_slice(target) and target.slice.step is None:
            func_name = "ac_set_slice" if isinstance(self._infer_ir_type(value), ArrayTypeRef) else "ac_fill_slice"
            return ExprStatement(
                Call(
                    func_name,
                    (
                        self._lower_expr(target.value),
                        self._lower_expr(target.slice.lower) if target.slice.lower is not None else Constant(0),
                        self._lower_expr(target.slice.upper) if target.slice.upper is not None else Call("size", (self._lower_expr(target.value),)),
                        value,
                    ),
                )
            )
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
        if isinstance(expr.slice, ast.Slice):
            container_type = self._infer_type(expr.value)
            if isinstance(container_type, ArrayTypeRef) and container_type.rank == 2:
                return ArrayTypeRef(container_type.element_type, rank=2)
        if self._is_1d_slice(expr):
            container_type = self._infer_type(expr.value)
            if isinstance(container_type, ArrayTypeRef):
                return ArrayTypeRef(container_type.element_type, rank=1)
        if (
            isinstance(expr.slice, ast.Slice)
            and expr.slice.lower is None
            and expr.slice.upper is None
            and isinstance(expr.slice.step, ast.UnaryOp)
            and isinstance(expr.slice.step.op, ast.USub)
            and isinstance(expr.slice.step.operand, ast.Constant)
            and expr.slice.step.operand.value == 1
        ):
            container_type = self._infer_type(expr.value)
            if isinstance(container_type, ArrayTypeRef):
                return container_type
        if isinstance(expr.slice, ast.Tuple) and len(expr.slice.elts) == 2:
            row_spec, col_spec = expr.slice.elts
            if isinstance(row_spec, ast.Slice) and isinstance(col_spec, ast.Slice):
                container_type = self._infer_type(expr.value)
                if isinstance(container_type, ArrayTypeRef):
                    return ArrayTypeRef(container_type.element_type, rank=2)
            if isinstance(row_spec, ast.Constant) and row_spec.value is None:
                if isinstance(col_spec, ast.Slice) and col_spec.lower is None and col_spec.upper is None and col_spec.step is None:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
            if isinstance(row_spec, ast.Slice) and row_spec.lower is None and row_spec.upper is None and row_spec.step is None:
                if isinstance(col_spec, (ast.Name, ast.Constant)) and not (isinstance(col_spec, ast.Constant) and col_spec.value is None):
                    return ArrayTypeRef(ScalarType.REAL64)
                if isinstance(col_spec, ast.Constant) and col_spec.value is None:
                    return ArrayTypeRef(ScalarType.REAL64, rank=2)
        return None

    def _is_1d_slice(self, expr: ast.Subscript) -> bool:
        return (
            isinstance(expr.slice, ast.Slice)
            and (
                not isinstance(self._infer_type(expr.value), ArrayTypeRef)
                or self._infer_type(expr.value).rank == 1
            )
        )

    def _is_item2_access(self, expr: ast.Subscript) -> bool:
        return (
            isinstance(expr.slice, ast.Tuple)
            and len(expr.slice.elts) == 2
            and not any(isinstance(elt, ast.Slice) for elt in expr.slice.elts)
            and not self._is_pairwise_item2_access(expr)
            and isinstance(self._infer_type(expr.value), ArrayTypeRef)
            and self._infer_type(expr.value).rank == 2
        )

    def _is_pairwise_item2_access(self, expr: ast.Subscript) -> bool:
        if not (
            isinstance(expr.slice, ast.Tuple)
            and len(expr.slice.elts) == 2
            and not any(isinstance(elt, ast.Slice) for elt in expr.slice.elts)
            and isinstance(self._infer_type(expr.value), ArrayTypeRef)
            and self._infer_type(expr.value).rank == 2
        ):
            return False
        row_type = self._infer_type(expr.slice.elts[0])
        col_type = self._infer_type(expr.slice.elts[1])
        return isinstance(row_type, ArrayTypeRef) and isinstance(col_type, ArrayTypeRef)

    def _is_item3_access(self, expr: ast.Subscript) -> bool:
        return (
            isinstance(expr.slice, ast.Tuple)
            and len(expr.slice.elts) == 3
            and not any(isinstance(elt, ast.Slice) for elt in expr.slice.elts)
            and isinstance(self._infer_type(expr.value), ArrayTypeRef)
            and self._infer_type(expr.value).rank == 3
        )

    def _is_np_r_concat(self, expr: ast.Subscript) -> bool:
        return isinstance(expr.value, ast.Attribute) and self._attr_chain(expr.value) == ["np", "r_"]

    def _is_boolean_mask_access(self, expr: ast.Subscript) -> bool:
        return (
            not isinstance(expr.slice, ast.Tuple)
            and isinstance(self._infer_type(expr.value), ArrayTypeRef)
            and isinstance(self._infer_type(expr.slice), ArrayTypeRef)
            and self._infer_type(expr.slice).element_type == ScalarType.LOGICAL
        )

    def _lower_np_r_concat(self, expr: ast.Subscript) -> object:
        items: list[ast.AST]
        if isinstance(expr.slice, ast.Tuple):
            items = list(expr.slice.elts)
        else:
            items = [expr.slice]
        if not items:
            raise NotImplementedError("np.r_[] with no items is not supported")
        lowered = [self._lower_expr(item) for item in items]
        current = lowered[0]
        for item in lowered[1:]:
            current = Call("ac_r_concat", (current, item))
        return current


