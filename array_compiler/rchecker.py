from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass

from array_compiler.frontends.r.frontend import AssignStmt
from array_compiler.frontends.r.frontend import BinaryExpr
from array_compiler.frontends.r.frontend import BoolExpr
from array_compiler.frontends.r.frontend import CallExpr
from array_compiler.frontends.r.frontend import ExprStmt
from array_compiler.frontends.r.frontend import FieldExpr
from array_compiler.frontends.r.frontend import ForStmt
from array_compiler.frontends.r.frontend import FunctionDefStmt
from array_compiler.frontends.r.frontend import IfExpr
from array_compiler.frontends.r.frontend import IfStmt
from array_compiler.frontends.r.frontend import IndexExpr
from array_compiler.frontends.r.frontend import NameExpr
from array_compiler.frontends.r.frontend import NumberExpr
from array_compiler.frontends.r.frontend import Param
from array_compiler.frontends.r.frontend import RSubsetParser
from array_compiler.frontends.r.frontend import ReturnStmt
from array_compiler.frontends.r.frontend import StringExpr
from array_compiler.frontends.r.frontend import UnaryExpr
from array_compiler.frontends.r.frontend import WhileStmt


_INTEGER_COUNT_FUNCS = {
    "runif",
    "rnorm",
    "rexp",
    "rpois",
    "rbinom",
    "rgamma",
    "rbeta",
    "rt",
    "rchisq",
    "rf",
    "rweibull",
    "rgeom",
    "rnbinom",
    "rcauchy",
    "rlogis",
    "rlnorm",
}
_INTEGER_ARG0_FUNCS = {"seq_len", "numeric", "integer", "logical", "character", "set.seed"}
_BUILTIN_SIGNATURES = {
    "character": ["length"],
    "integer": ["length"],
    "logical": ["length"],
    "matrix": ["data", "nrow", "ncol", "byrow", "dimnames"],
    "numeric": ["length"],
    "rep": ["x", "times", "length.out", "each"],
    "rbinom": ["n", "size", "prob"],
    "rchisq": ["n", "df", "ncp"],
    "rexp": ["n", "rate"],
    "rf": ["n", "df1", "df2", "ncp"],
    "rgamma": ["n", "shape", "rate", "scale"],
    "rgeom": ["n", "prob"],
    "rlnorm": ["n", "meanlog", "sdlog"],
    "rlogis": ["n", "location", "scale"],
    "rnbinom": ["n", "size", "prob", "mu"],
    "rnorm": ["n", "mean", "sd"],
    "rpois": ["n", "lambda"],
    "rt": ["n", "df", "ncp"],
    "runif": ["n", "min", "max"],
    "rweibull": ["n", "shape", "scale"],
    "sample": ["x", "size", "replace", "prob"],
    "seq": ["from", "to", "by", "length.out", "along.with"],
    "seq_len": ["length.out"],
    "set.seed": ["seed"],
}
_VECTORIZED_BINARY_OPS = {"+", "-", "*", "/", "^", "%%", "%/%", "<", "<=", ">", ">=", "==", "!=", "&", "|"}


@dataclass(frozen=True)
class DiagnosticInfo:
    level: str
    line: int
    message: str


@dataclass(frozen=True)
class _Binding:
    name: str
    expr: object
    origin: str
    inferred_type: str | None = None


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    replacement: str


class RChecker:
    def check_source(self, source: str) -> list[DiagnosticInfo]:
        diagnostics, _edits = self._analyze(source)
        return diagnostics

    def fix_source(self, source: str, *, fix_type_changes: bool = False) -> tuple[str, list[DiagnosticInfo]]:
        diagnostics, edits = self._analyze(source)
        if fix_type_changes:
            parser = RSubsetParser(source)
            program = parser.parse_program()
            function_sigs = self._collect_function_signatures(program)
            used_names = self._collect_used_names(program)
            type_change_edits = self._collect_type_change_fix_edits(program, {}, function_sigs, used_names)
            for key, edit in type_change_edits.items():
                edits.setdefault(key, edit)
        if not edits:
            return source, diagnostics
        fixed = source
        for edit in sorted(edits.values(), key=lambda item: item.start, reverse=True):
            fixed = fixed[: edit.start] + edit.replacement + fixed[edit.end :]
        return fixed, diagnostics

    def format_human(self, diagnostics: list[DiagnosticInfo]) -> str:
        if not diagnostics:
            return "No issues found.\n"
        return "".join(f"{diag.level}: line {diag.line}: {diag.message}\n" for diag in diagnostics)

    def format_json(self, diagnostics: list[DiagnosticInfo]) -> str:
        return json.dumps([asdict(diag) for diag in diagnostics], indent=2) + "\n"

    def _analyze(self, source: str) -> tuple[list[DiagnosticInfo], dict[tuple[int, int], _Edit]]:
        parser = RSubsetParser(source)
        program = parser.parse_program()
        diagnostics: list[DiagnosticInfo] = []
        edits: dict[tuple[int, int], _Edit] = {}
        function_sigs = self._collect_function_signatures(program)
        self._visit_block(program, {}, function_sigs, diagnostics, edits)
        return self._dedupe(diagnostics), edits

    def _collect_function_signatures(self, statements: list[object] | tuple[object, ...]) -> dict[str, list[str]]:
        signatures = dict(_BUILTIN_SIGNATURES)
        for stmt in statements:
            if isinstance(stmt, FunctionDefStmt):
                signatures[stmt.name] = [param.name for param in stmt.params]
        return signatures

    def _visit_block(
        self,
        statements: list[object] | tuple[object, ...],
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> dict[str, _Binding | None]:
        env = dict(bindings)
        for stmt in statements:
            self._visit_stmt(stmt, env, function_sigs, diagnostics, edits)
        return env

    def _collect_type_change_fix_edits(
        self,
        statements: list[object] | tuple[object, ...],
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        used_names: set[str],
    ) -> dict[tuple[int, int], _Edit]:
        env = dict(bindings)
        edits: dict[tuple[int, int], _Edit] = {}
        for index, stmt in enumerate(statements):
            if isinstance(stmt, AssignStmt) and isinstance(stmt.target, NameExpr):
                name = stmt.target.name
                previous = env.get(name)
                new_binding = self._binding_from_expr(name, stmt.value, env, origin="assignment")
                if (
                    previous is not None
                    and new_binding is not None
                    and previous.inferred_type is not None
                    and new_binding.inferred_type is not None
                    and previous.inferred_type != new_binding.inferred_type
                ):
                    new_name = self._fresh_type_change_name(name, new_binding.inferred_type, used_names)
                    tail = list(statements[index + 1 :])
                    if self._tail_is_flat_renamable(tail, name, new_name):
                        edits[(stmt.target.start, stmt.target.end)] = _Edit(stmt.target.start, stmt.target.end, new_name)
                        used_names.add(new_name)
                        for rename_expr in self._collect_name_occurrences(tail, name):
                            edits[(rename_expr.start, rename_expr.end)] = _Edit(rename_expr.start, rename_expr.end, new_name)
                env[name] = new_binding
                continue
            if isinstance(stmt, FunctionDefStmt):
                inner = dict(bindings)
                for param in stmt.params:
                    if param.default is not None:
                        continue
                    inner[param.name] = None
                inner_sigs = dict(function_sigs)
                inner_sigs[stmt.name] = [param.name for param in stmt.params]
                nested_used = set(used_names)
                edits.update(self._collect_type_change_fix_edits(stmt.body, inner, inner_sigs, nested_used))
                continue
            if isinstance(stmt, IfStmt):
                nested_used = set(used_names)
                edits.update(self._collect_type_change_fix_edits(stmt.body, dict(env), function_sigs, nested_used))
                edits.update(self._collect_type_change_fix_edits(stmt.orelse, dict(env), function_sigs, nested_used))
                continue
            if isinstance(stmt, ForStmt):
                nested_used = set(used_names)
                inner = dict(env)
                inner[stmt.target] = None
                edits.update(self._collect_type_change_fix_edits(stmt.body, inner, function_sigs, nested_used))
                continue
            if isinstance(stmt, WhileStmt):
                nested_used = set(used_names)
                edits.update(self._collect_type_change_fix_edits(stmt.body, dict(env), function_sigs, nested_used))
        return edits

    def _visit_stmt(
        self,
        stmt: object,
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if isinstance(stmt, AssignStmt):
            self._visit_lvalue(stmt.target, bindings, function_sigs, diagnostics, edits)
            self._visit_expr(stmt.value, bindings, function_sigs, diagnostics, edits)
            if isinstance(stmt.target, NameExpr):
                new_binding = self._binding_from_expr(stmt.target.name, stmt.value, bindings, origin="assignment")
                previous = bindings.get(stmt.target.name)
                self._warn_type_change(stmt.target.name, previous, new_binding, stmt.target.line, diagnostics)
                bindings[stmt.target.name] = new_binding
            return
        if isinstance(stmt, ExprStmt):
            self._visit_expr(stmt.value, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(stmt, ReturnStmt):
            self._visit_expr(stmt.value, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(stmt, IfStmt):
            self._require_logical(stmt.test, "if condition", bindings, diagnostics, edits)
            incoming = dict(bindings)
            body_env = self._visit_block(stmt.body, dict(bindings), function_sigs, diagnostics, edits)
            orelse_env = self._visit_block(stmt.orelse, dict(bindings), function_sigs, diagnostics, edits)
            merged = self._merge_branch_bindings(incoming, body_env, orelse_env)
            bindings.clear()
            bindings.update(merged)
            return
        if isinstance(stmt, ForStmt):
            self._visit_expr(stmt.iterable, bindings, function_sigs, diagnostics, edits)
            inner = dict(bindings)
            inner[stmt.target] = None
            self._visit_block(stmt.body, inner, function_sigs, diagnostics, edits)
            return
        if isinstance(stmt, WhileStmt):
            self._require_logical(stmt.test, "while condition", bindings, diagnostics, edits)
            self._visit_block(stmt.body, dict(bindings), function_sigs, diagnostics, edits)
            return
        if isinstance(stmt, FunctionDefStmt):
            inner = dict(bindings)
            for param in stmt.params:
                if param.default is not None:
                    self._visit_expr(param.default, bindings, function_sigs, diagnostics, edits)
                inner[param.name] = self._binding_from_param(param)
            inner_sigs = dict(function_sigs)
            inner_sigs[stmt.name] = [param.name for param in stmt.params]
            self._visit_block(stmt.body, inner, inner_sigs, diagnostics, edits)

    def _visit_lvalue(
        self,
        expr: object,
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if isinstance(expr, IndexExpr):
            self._visit_expr(expr.value, bindings, function_sigs, diagnostics, edits)
            for index in expr.indices:
                if index is not None and not self._is_logical_index(index):
                    self._require_integer(index, "subscript", bindings, diagnostics, edits)
            return
        if isinstance(expr, FieldExpr):
            self._visit_expr(expr.value, bindings, function_sigs, diagnostics, edits)

    def _visit_expr(
        self,
        expr: object,
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if isinstance(expr, BinaryExpr):
            if expr.op == ":":
                self._require_integer(expr.left, "range bound", bindings, diagnostics, edits)
                self._require_integer(expr.right, "range bound", bindings, diagnostics, edits)
            else:
                if expr.op in {"&&", "||", "&", "|"}:
                    self._require_logical(expr.left, f"{expr.op} operand", bindings, diagnostics, edits)
                    self._require_logical(expr.right, f"{expr.op} operand", bindings, diagnostics, edits)
                if expr.op in _VECTORIZED_BINARY_OPS:
                    self._warn_incompatible_recycling(expr, bindings, diagnostics)
                self._visit_expr(expr.left, bindings, function_sigs, diagnostics, edits)
                self._visit_expr(expr.right, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(expr, UnaryExpr):
            if expr.op == "!":
                self._require_logical(expr.operand, "! operand", bindings, diagnostics, edits)
            self._visit_expr(expr.operand, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(expr, IfExpr):
            self._require_logical(expr.test, "if expression condition", bindings, diagnostics, edits)
            self._visit_expr(expr.body, bindings, function_sigs, diagnostics, edits)
            self._visit_expr(expr.orelse, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(expr, CallExpr):
            self._check_call(expr, bindings, function_sigs, diagnostics, edits)
            for arg in expr.args:
                self._visit_expr(arg.value, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(expr, IndexExpr):
            self._visit_expr(expr.value, bindings, function_sigs, diagnostics, edits)
            for index in expr.indices:
                if index is not None and not self._is_logical_index(index):
                    self._require_integer(index, "subscript", bindings, diagnostics, edits)
            return
        if isinstance(expr, FieldExpr):
            self._visit_expr(expr.value, bindings, function_sigs, diagnostics, edits)
            return
        if isinstance(expr, NameExpr):
            self._check_name(expr, diagnostics, edits)
            return
        if isinstance(expr, (NumberExpr, StringExpr, BoolExpr)):
            return

    def _check_call(
        self,
        expr: CallExpr,
        bindings: dict[str, _Binding | None],
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if not isinstance(expr.func, NameExpr):
            return
        name = expr.func.name
        positional = [arg.value for arg in expr.args if arg.name is None]
        keyword_map = {arg.name: arg.value for arg in expr.args if arg.name is not None}

        if name in _INTEGER_ARG0_FUNCS and positional:
            self._require_integer(positional[0], f"{name}() argument", bindings, diagnostics, edits)
        if name in _INTEGER_COUNT_FUNCS and positional:
            self._require_integer(positional[0], f"{name}() count", bindings, diagnostics, edits)
        if name == "sample":
            size_arg = keyword_map.get("size")
            if size_arg is None and len(positional) >= 2:
                size_arg = positional[1]
            if size_arg is not None:
                self._require_integer(size_arg, "sample() size", bindings, diagnostics, edits)
        if name == "matrix":
            if "nrow" in keyword_map:
                self._require_integer(keyword_map["nrow"], "matrix() nrow", bindings, diagnostics, edits)
            if "ncol" in keyword_map:
                self._require_integer(keyword_map["ncol"], "matrix() ncol", bindings, diagnostics, edits)
            if len(positional) >= 2:
                self._require_integer(positional[1], "matrix() nrow", bindings, diagnostics, edits)
            if len(positional) >= 3:
                self._require_integer(positional[2], "matrix() ncol", bindings, diagnostics, edits)
        if name == "rep":
            times_arg = keyword_map.get("times")
            each_arg = keyword_map.get("each")
            length_out_arg = keyword_map.get("length.out")
            if times_arg is None and len(positional) >= 2:
                times_arg = positional[1]
            for arg, context in (
                (times_arg, "rep() times"),
                (each_arg, "rep() each"),
                (length_out_arg, "rep() length.out"),
            ):
                if arg is not None:
                    self._require_integer(arg, context, bindings, diagnostics, edits)
        if name == "seq":
            length_out_arg = keyword_map.get("length.out")
            if length_out_arg is not None:
                self._require_integer(length_out_arg, "seq() length.out", bindings, diagnostics, edits)
        self._warn_partial_argument_matching(expr, function_sigs, diagnostics)

    def _warn_partial_argument_matching(
        self,
        expr: CallExpr,
        function_sigs: dict[str, list[str]],
        diagnostics: list[DiagnosticInfo],
    ) -> None:
        if not isinstance(expr.func, NameExpr):
            return
        signature = function_sigs.get(expr.func.name)
        if signature is None:
            return
        for arg in expr.args:
            if arg.name is None or arg.name in signature:
                continue
            matches = [param for param in signature if param.startswith(arg.name)]
            if len(matches) == 1:
                diagnostics.append(
                    DiagnosticInfo(
                        level="warning",
                        line=self._expr_line(arg.value),
                        message=f"partial argument name {arg.name!r} in {expr.func.name}(); prefer {matches[0]!r}",
                    )
                )

    def _require_integer(
        self,
        expr: object,
        context: str,
        bindings: dict[str, _Binding | None],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if isinstance(expr, NumberExpr):
            self._warn_literal(expr, context, diagnostics, edits)
            return
        if isinstance(expr, UnaryExpr) and expr.op in {"+", "-"} and isinstance(expr.operand, NumberExpr):
            self._warn_literal(expr.operand, context, diagnostics, edits)
            return
        if isinstance(expr, NameExpr):
            self._check_name(expr, diagnostics, edits)
            binding = bindings.get(expr.name)
            if binding is not None:
                self._warn_binding(binding, context, diagnostics, edits)
            return
        if isinstance(expr, BinaryExpr) and expr.op == ":":
            self._require_integer(expr.left, context, bindings, diagnostics, edits)
            self._require_integer(expr.right, context, bindings, diagnostics, edits)
            return
        if isinstance(expr, IndexExpr):
            for index in expr.indices:
                if index is not None and not self._is_logical_index(index):
                    self._require_integer(index, context, bindings, diagnostics, edits)
            return

    def _require_logical(
        self,
        expr: object,
        context: str,
        bindings: dict[str, _Binding | None],
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if isinstance(expr, BoolExpr):
            return
        if isinstance(expr, NameExpr):
            self._check_name(expr, diagnostics, edits)
            if expr.name in {"TRUE", "FALSE", "T", "F"}:
                return
            if expr.name == "NA":
                diagnostics.append(
                    DiagnosticInfo(
                        level="warning",
                        line=expr.line,
                        message=f"bare NA is used in {context}; prefer a typed NA variant when the type matters",
                    )
                )
                return
            binding = bindings.get(expr.name)
            if binding is not None and self._binding_is_numeric(binding):
                diagnostics.append(
                    DiagnosticInfo(
                        level="warning",
                        line=self._binding_line(binding),
                        message=f"{binding.name} looks numeric but is used as a logical in {context}",
                    )
                )
            return
        if isinstance(expr, NumberExpr):
            diagnostics.append(
                DiagnosticInfo(
                    level="warning",
                    line=expr.line,
                    message=f"numeric literal {expr.text} is used as a logical in {context}; prefer TRUE/FALSE or an explicit comparison",
                )
            )
            return
        if isinstance(expr, UnaryExpr) and expr.op in {"+", "-"} and isinstance(expr.operand, NumberExpr):
            diagnostics.append(
                DiagnosticInfo(
                    level="warning",
                    line=expr.operand.line,
                    message=f"numeric literal {expr.op}{expr.operand.text} is used as a logical in {context}; prefer TRUE/FALSE or an explicit comparison",
                )
            )
            return

    def _warn_literal(
        self,
        expr: NumberExpr,
        context: str,
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if not expr.integer_like or expr.explicit_integer:
            return
        diagnostics.append(
            DiagnosticInfo(
                level="warning",
                line=expr.line,
                message=f"double literal {expr.text} is used as an integer in {context}; prefer {expr.text}L",
            )
        )
        edits[(expr.start, expr.end)] = _Edit(expr.start, expr.end, f"{expr.text}L")

    def _warn_binding(
        self,
        binding: _Binding,
        context: str,
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        expr = binding.expr
        if isinstance(expr, NumberExpr) and expr.integer_like and not expr.explicit_integer:
            diagnostics.append(
                DiagnosticInfo(
                    level="warning",
                    line=expr.line,
                    message=(
                        f"{binding.name} is assigned double literal {expr.text} "
                        f"but used as an integer in {context}; prefer {expr.text}L"
                    ),
                )
            )
            edits[(expr.start, expr.end)] = _Edit(expr.start, expr.end, f"{expr.text}L")

    def _check_name(
        self,
        expr: NameExpr,
        diagnostics: list[DiagnosticInfo],
        edits: dict[tuple[int, int], _Edit],
    ) -> None:
        if expr.name == "T":
            diagnostics.append(DiagnosticInfo(level="warning", line=expr.line, message="prefer TRUE over T"))
            edits[(expr.start, expr.end)] = _Edit(expr.start, expr.end, "TRUE")
        elif expr.name == "F":
            diagnostics.append(DiagnosticInfo(level="warning", line=expr.line, message="prefer FALSE over F"))
            edits[(expr.start, expr.end)] = _Edit(expr.start, expr.end, "FALSE")
        elif expr.name == "NA":
            diagnostics.append(
                DiagnosticInfo(
                    level="warning",
                    line=expr.line,
                    message="bare NA is untyped; prefer NA_integer_, NA_real_, or NA_character_ when the type matters",
                )
            )

    def _warn_incompatible_recycling(
        self,
        expr: BinaryExpr,
        bindings: dict[str, _Binding | None],
        diagnostics: list[DiagnosticInfo],
    ) -> None:
        left_len = self._expr_length(expr.left, bindings)
        right_len = self._expr_length(expr.right, bindings)
        if left_len is None or right_len is None:
            return
        if left_len <= 1 or right_len <= 1 or left_len == right_len:
            return
        if left_len % right_len == 0 or right_len % left_len == 0:
            return
        diagnostics.append(
            DiagnosticInfo(
                level="warning",
                line=self._expr_line(expr.left),
                message=f"vector lengths {left_len} and {right_len} are incompatible for recycling with {expr.op!r}",
            )
        )

    def _expr_length(self, expr: object, bindings: dict[str, _Binding | None]) -> int | None:
        if isinstance(expr, (NumberExpr, StringExpr, BoolExpr)):
            return 1
        if isinstance(expr, NameExpr):
            binding = bindings.get(expr.name)
            if binding is None:
                return None
            return self._expr_length(binding.expr, bindings)
        if isinstance(expr, UnaryExpr):
            return self._expr_length(expr.operand, bindings)
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            if expr.func.name == "c":
                total = 0
                for arg in expr.args:
                    arg_len = self._expr_length(arg.value, bindings)
                    if arg_len is None:
                        return None
                    total += arg_len
                return total
            if expr.func.name == "rep":
                if not expr.args:
                    return None
                base_len = self._expr_length(expr.args[0].value, bindings)
                if base_len is None:
                    return None
                times_value: object | None = None
                for arg in expr.args[1:]:
                    if arg.name in {None, "times"}:
                        times_value = arg.value
                        if arg.name == "times":
                            break
                if isinstance(times_value, NumberExpr) and times_value.integer_like:
                    return base_len * int(times_value.text)
        return None

    def _binding_is_numeric(self, binding: _Binding) -> bool:
        return binding.inferred_type in {"scalar_integer", "scalar_double", "vector_integer", "vector_double", "matrix_integer", "matrix_double"}

    def _binding_line(self, binding: _Binding) -> int:
        if isinstance(binding.expr, NumberExpr):
            return binding.expr.line
        if isinstance(binding.expr, NameExpr):
            return binding.expr.line
        return 0

    def _binding_from_expr(
        self,
        name: str,
        expr: object,
        bindings: dict[str, _Binding | None],
        *,
        origin: str,
    ) -> _Binding | None:
        inferred_type = self._infer_expr_type(expr, bindings)
        if inferred_type is None and not isinstance(expr, (NumberExpr, NameExpr, CallExpr, StringExpr, BoolExpr, BinaryExpr, UnaryExpr, IndexExpr, IfExpr)):
            return None
        return _Binding(name=name, expr=expr, origin=origin, inferred_type=inferred_type)

    def _binding_from_param(self, param: Param) -> _Binding | None:
        if isinstance(param.default, (NumberExpr, NameExpr, CallExpr, StringExpr, BoolExpr, BinaryExpr, UnaryExpr, IndexExpr, IfExpr)):
            return _Binding(
                name=param.name,
                expr=param.default,
                origin="default",
                inferred_type=self._infer_expr_type(param.default, {}),
            )
        return None

    def _warn_type_change(
        self,
        name: str,
        previous: _Binding | None,
        current: _Binding | None,
        line: int,
        diagnostics: list[DiagnosticInfo],
    ) -> None:
        if previous is None or current is None:
            return
        if previous.inferred_type is None or current.inferred_type is None:
            return
        if previous.inferred_type == current.inferred_type:
            return
        diagnostics.append(
            DiagnosticInfo(
                level="warning",
                line=line,
                message=f"{name} changes type: {previous.inferred_type} -> {current.inferred_type}",
            )
        )

    def _merge_branch_bindings(
        self,
        incoming: dict[str, _Binding | None],
        body_env: dict[str, _Binding | None],
        orelse_env: dict[str, _Binding | None],
    ) -> dict[str, _Binding | None]:
        merged = dict(incoming)
        for name in set(body_env) | set(orelse_env):
            body_binding = body_env.get(name, incoming.get(name))
            orelse_binding = orelse_env.get(name, incoming.get(name))
            if body_binding is None or orelse_binding is None:
                merged[name] = None
                continue
            if body_binding.inferred_type == orelse_binding.inferred_type:
                merged[name] = body_binding
            else:
                merged[name] = None
        return merged

    def _fresh_type_change_name(self, name: str, inferred_type: str, used_names: set[str]) -> str:
        suffix = inferred_type
        if "_" in suffix:
            suffix = suffix.split("_", 1)[1]
        candidate = f"{name}_{suffix}"
        counter = 2
        while candidate in used_names:
            candidate = f"{name}_{suffix}{counter}"
            counter += 1
        return candidate

    def _infer_expr_type(self, expr: object, bindings: dict[str, _Binding | None]) -> str | None:
        if isinstance(expr, NumberExpr):
            return "scalar_integer" if expr.explicit_integer else "scalar_double"
        if isinstance(expr, StringExpr):
            return "scalar_character"
        if isinstance(expr, BoolExpr):
            return "scalar_logical"
        if isinstance(expr, NameExpr):
            binding = bindings.get(expr.name)
            return binding.inferred_type if binding is not None else None
        if isinstance(expr, UnaryExpr):
            if expr.op == "!":
                operand_type = self._infer_expr_type(expr.operand, bindings)
                return self._logical_result_type(operand_type)
            return self._infer_expr_type(expr.operand, bindings)
        if isinstance(expr, IfExpr):
            return self._merge_types(
                self._infer_expr_type(expr.body, bindings),
                self._infer_expr_type(expr.orelse, bindings),
            )
        if isinstance(expr, BinaryExpr):
            if expr.op == ":":
                return "vector_double"
            left_type = self._infer_expr_type(expr.left, bindings)
            right_type = self._infer_expr_type(expr.right, bindings)
            if expr.op in {"<", "<=", ">", ">=", "==", "!=", "&", "|", "&&", "||"}:
                return self._logical_merge(left_type, right_type)
            if expr.op in {"+", "-", "*", "/", "^", "%%", "%/%"}:
                return self._numeric_merge(left_type, right_type)
            return self._merge_types(left_type, right_type)
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            name = expr.func.name
            if name == "c":
                return self._infer_c_type(expr, bindings)
            if name in {"numeric", "runif", "rnorm", "rexp", "rpois", "rbinom", "rgamma", "rbeta", "rt", "rchisq", "rf", "rweibull", "rgeom", "rnbinom", "rcauchy", "rlogis", "rlnorm", "seq"}:
                return "vector_double"
            if name == "seq_len":
                return "vector_integer"
            if name in {"integer"}:
                return "vector_integer"
            if name in {"logical"}:
                return "vector_logical"
            if name in {"character"}:
                return "vector_character"
            if name in {"as.character"}:
                return "vector_character"
            if name in {"as.integer"}:
                return "vector_integer"
            if name in {"as.numeric", "as.double"}:
                return "vector_double"
            if name in {"as.logical"}:
                return "vector_logical"
            if name == "matrix":
                base = self._infer_expr_type(expr.args[0].value, bindings) if expr.args else None
                element = self._element_type(base) or "double"
                return f"matrix_{element}"
            if name == "rep":
                if not expr.args:
                    return None
                base = self._infer_expr_type(expr.args[0].value, bindings)
                element = self._element_type(base)
                return f"vector_{element}" if element is not None else None
            if name == "sample":
                if not expr.args:
                    return None
                base = self._infer_expr_type(expr.args[0].value, bindings)
                element = self._element_type(base)
                return f"vector_{element}" if element is not None else None
        if isinstance(expr, IndexExpr):
            base_type = self._infer_expr_type(expr.value, bindings)
            if base_type is None:
                return None
            element = self._element_type(base_type)
            if element is None:
                return None
            if base_type.startswith("matrix_") and len(expr.indices) == 2 and all(index is not None for index in expr.indices):
                return f"scalar_{element}"
            if base_type.startswith("matrix_"):
                return f"vector_{element}"
            return f"scalar_{element}"
        return None

    def _infer_c_type(self, expr: CallExpr, bindings: dict[str, _Binding | None]) -> str | None:
        element: str | None = None
        for arg in expr.args:
            arg_type = self._infer_expr_type(arg.value, bindings)
            if arg_type is None:
                return None
            arg_element = self._element_type(arg_type)
            element = self._merge_elements(element, arg_element)
        return f"vector_{element}" if element is not None else None

    def _element_type(self, inferred_type: str | None) -> str | None:
        if inferred_type is None:
            return None
        if "_" not in inferred_type:
            return None
        return inferred_type.split("_", 1)[1]

    def _merge_types(self, left: str | None, right: str | None) -> str | None:
        if left is None:
            return right
        if right is None or left == right:
            return left
        left_prefix = left.split("_", 1)[0]
        right_prefix = right.split("_", 1)[0]
        if left_prefix != right_prefix:
            return None
        merged_element = self._merge_elements(self._element_type(left), self._element_type(right))
        return f"{left_prefix}_{merged_element}" if merged_element is not None else None

    def _merge_elements(self, left: str | None, right: str | None) -> str | None:
        if left is None:
            return right
        if right is None or left == right:
            return left
        if "character" in {left, right}:
            return "character"
        if "double" in {left, right} and "integer" in {left, right}:
            return "double"
        if "logical" in {left, right} and {"integer", "double"} & {left, right}:
            return "double"
        return None

    def _numeric_merge(self, left: str | None, right: str | None) -> str | None:
        prefix = "scalar"
        for candidate in (left, right):
            if candidate is not None and candidate.startswith("matrix_"):
                prefix = "matrix"
                break
            if candidate is not None and candidate.startswith("vector_"):
                prefix = "vector"
        merged_element = self._merge_elements(self._element_type(left), self._element_type(right))
        if merged_element is None:
            merged_element = "double"
        return f"{prefix}_{merged_element}"

    def _logical_merge(self, left: str | None, right: str | None) -> str:
        prefix = "scalar"
        for candidate in (left, right):
            if candidate is not None and candidate.startswith("matrix_"):
                prefix = "matrix"
                break
            if candidate is not None and candidate.startswith("vector_"):
                prefix = "vector"
        return f"{prefix}_logical"

    def _logical_result_type(self, operand_type: str | None) -> str | None:
        if operand_type is None:
            return None
        prefix = operand_type.split("_", 1)[0]
        return f"{prefix}_logical"

    def _tail_is_flat_renamable(self, statements: list[object], name: str, new_name: str) -> bool:
        for stmt in statements:
            if isinstance(stmt, (IfStmt, ForStmt, WhileStmt, FunctionDefStmt)):
                return False
            if self._stmt_writes_name(stmt, name) or self._stmt_writes_name(stmt, new_name):
                return False
        return True

    def _stmt_writes_name(self, stmt: object, name: str) -> bool:
        if isinstance(stmt, AssignStmt):
            return self._target_writes_name(stmt.target, name)
        return False

    def _target_writes_name(self, target: object, name: str) -> bool:
        if isinstance(target, NameExpr):
            return target.name == name
        if isinstance(target, IndexExpr):
            return self._expr_contains_name(target.value, name)
        if isinstance(target, FieldExpr):
            return self._expr_contains_name(target.value, name)
        return False

    def _expr_contains_name(self, expr: object, name: str) -> bool:
        if isinstance(expr, NameExpr):
            return expr.name == name
        if isinstance(expr, UnaryExpr):
            return self._expr_contains_name(expr.operand, name)
        if isinstance(expr, BinaryExpr):
            return self._expr_contains_name(expr.left, name) or self._expr_contains_name(expr.right, name)
        if isinstance(expr, IfExpr):
            return (
                self._expr_contains_name(expr.test, name)
                or self._expr_contains_name(expr.body, name)
                or self._expr_contains_name(expr.orelse, name)
            )
        if isinstance(expr, CallExpr):
            return any(self._expr_contains_name(arg.value, name) for arg in expr.args)
        if isinstance(expr, IndexExpr):
            return self._expr_contains_name(expr.value, name) or any(
                index is not None and self._expr_contains_name(index, name) for index in expr.indices
            )
        if isinstance(expr, FieldExpr):
            return self._expr_contains_name(expr.value, name)
        return False

    def _collect_name_occurrences(self, statements: list[object], name: str) -> list[NameExpr]:
        occurrences: list[NameExpr] = []
        for stmt in statements:
            occurrences.extend(self._name_occurrences_in_stmt(stmt, name))
        return occurrences

    def _name_occurrences_in_stmt(self, stmt: object, name: str) -> list[NameExpr]:
        if isinstance(stmt, AssignStmt):
            return self._name_occurrences_in_expr(stmt.value, name)
        if isinstance(stmt, ExprStmt):
            return self._name_occurrences_in_expr(stmt.value, name)
        if isinstance(stmt, ReturnStmt):
            return self._name_occurrences_in_expr(stmt.value, name)
        return []

    def _name_occurrences_in_expr(self, expr: object, name: str) -> list[NameExpr]:
        if isinstance(expr, NameExpr):
            return [expr] if expr.name == name else []
        if isinstance(expr, UnaryExpr):
            return self._name_occurrences_in_expr(expr.operand, name)
        if isinstance(expr, BinaryExpr):
            return self._name_occurrences_in_expr(expr.left, name) + self._name_occurrences_in_expr(expr.right, name)
        if isinstance(expr, IfExpr):
            return (
                self._name_occurrences_in_expr(expr.test, name)
                + self._name_occurrences_in_expr(expr.body, name)
                + self._name_occurrences_in_expr(expr.orelse, name)
            )
        if isinstance(expr, CallExpr):
            found: list[NameExpr] = []
            for arg in expr.args:
                found.extend(self._name_occurrences_in_expr(arg.value, name))
            return found
        if isinstance(expr, IndexExpr):
            found = self._name_occurrences_in_expr(expr.value, name)
            for index in expr.indices:
                if index is not None:
                    found.extend(self._name_occurrences_in_expr(index, name))
            return found
        if isinstance(expr, FieldExpr):
            return self._name_occurrences_in_expr(expr.value, name)
        return []

    def _collect_used_names(self, statements: list[object] | tuple[object, ...]) -> set[str]:
        names: set[str] = set()
        for stmt in statements:
            names.update(self._used_names_in_stmt(stmt))
        return names

    def _used_names_in_stmt(self, stmt: object) -> set[str]:
        names: set[str] = set()
        if isinstance(stmt, AssignStmt):
            names.update(self._used_names_in_target(stmt.target))
            names.update(self._used_names_in_expr(stmt.value))
        elif isinstance(stmt, ExprStmt):
            names.update(self._used_names_in_expr(stmt.value))
        elif isinstance(stmt, ReturnStmt):
            names.update(self._used_names_in_expr(stmt.value))
        elif isinstance(stmt, IfStmt):
            names.update(self._used_names_in_expr(stmt.test))
            names.update(self._collect_used_names(stmt.body))
            names.update(self._collect_used_names(stmt.orelse))
        elif isinstance(stmt, ForStmt):
            names.add(stmt.target)
            names.update(self._used_names_in_expr(stmt.iterable))
            names.update(self._collect_used_names(stmt.body))
        elif isinstance(stmt, WhileStmt):
            names.update(self._used_names_in_expr(stmt.test))
            names.update(self._collect_used_names(stmt.body))
        elif isinstance(stmt, FunctionDefStmt):
            names.add(stmt.name)
            for param in stmt.params:
                names.add(param.name)
                if param.default is not None:
                    names.update(self._used_names_in_expr(param.default))
            names.update(self._collect_used_names(stmt.body))
        return names

    def _used_names_in_target(self, target: object) -> set[str]:
        if isinstance(target, NameExpr):
            return {target.name}
        if isinstance(target, IndexExpr):
            names = self._used_names_in_expr(target.value)
            for index in target.indices:
                if index is not None:
                    names.update(self._used_names_in_expr(index))
            return names
        if isinstance(target, FieldExpr):
            return self._used_names_in_expr(target.value)
        return set()

    def _used_names_in_expr(self, expr: object) -> set[str]:
        if isinstance(expr, NameExpr):
            return {expr.name}
        if isinstance(expr, UnaryExpr):
            return self._used_names_in_expr(expr.operand)
        if isinstance(expr, BinaryExpr):
            return self._used_names_in_expr(expr.left) | self._used_names_in_expr(expr.right)
        if isinstance(expr, IfExpr):
            return (
                self._used_names_in_expr(expr.test)
                | self._used_names_in_expr(expr.body)
                | self._used_names_in_expr(expr.orelse)
            )
        if isinstance(expr, CallExpr):
            names: set[str] = set()
            if isinstance(expr.func, NameExpr):
                names.add(expr.func.name)
            for arg in expr.args:
                names.update(self._used_names_in_expr(arg.value))
            return names
        if isinstance(expr, IndexExpr):
            names = self._used_names_in_expr(expr.value)
            for index in expr.indices:
                if index is not None:
                    names.update(self._used_names_in_expr(index))
            return names
        if isinstance(expr, FieldExpr):
            return self._used_names_in_expr(expr.value)
        return set()

    def _expr_line(self, expr: object) -> int:
        if isinstance(expr, NumberExpr):
            return expr.line
        if isinstance(expr, NameExpr):
            return expr.line
        if isinstance(expr, UnaryExpr):
            return self._expr_line(expr.operand)
        if isinstance(expr, BinaryExpr):
            return self._expr_line(expr.left)
        if isinstance(expr, CallExpr):
            if expr.args:
                return self._expr_line(expr.args[0].value)
        return 0

    def _is_logical_index(self, expr: object) -> bool:
        if isinstance(expr, BoolExpr):
            return True
        if isinstance(expr, UnaryExpr) and expr.op == "!":
            return True
        if isinstance(expr, BinaryExpr) and expr.op in {"<", "<=", ">", ">=", "==", "!=", "&", "|", "&&", "||"}:
            return True
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr) and expr.func.name == "is.na":
            return True
        return False

    def _dedupe(self, diagnostics: list[DiagnosticInfo]) -> list[DiagnosticInfo]:
        seen: set[tuple[str, int, str]] = set()
        unique: list[DiagnosticInfo] = []
        for diag in diagnostics:
            key = (diag.level, diag.line, diag.message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(diag)
        return unique
