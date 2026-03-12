from __future__ import annotations

import ast
import json
from dataclasses import asdict
from dataclasses import dataclass

from array_compiler.annotator import PythonAnnotator
from array_compiler.annotator import WarningInfo


@dataclass(frozen=True)
class DiagnosticInfo:
    level: str
    line: int
    message: str


class PythonChecker:
    def check_source(
        self,
        source: str,
        *,
        check_intent: bool = False,
        fortran_preflight: bool = False,
    ) -> list[DiagnosticInfo]:
        diagnostics: list[DiagnosticInfo] = []
        annotator = PythonAnnotator()
        annotator._source = source
        annotator._known_name_types = {}
        annotator._known_function_arg_types = {}
        tree = ast.parse(source)

        assigned_types: dict[str, object] = {}
        assigned_lines: dict[tuple[str, str], list[int]] = {}
        assignment_infos: dict[int, object] = {}
        assignment_keys: dict[int, tuple[str, str]] = {}
        collect_warnings: list[WarningInfo] = []
        annotator._collect_assignments(
            tree.body,
            assigned_types,
            assigned_lines,
            assignment_infos,
            assignment_keys,
            collect_warnings,
            known={},
            scope="<module>",
        )
        diagnostics.extend(
            DiagnosticInfo(level="warning", line=warning.line, message=warning.message)
            for warning in collect_warnings
        )
        diagnostics.extend(self._collect_simple_type_change_warnings(tree.body, annotator, known={}))

        annotator._known_name_types = {name: info.text for name, info in assigned_types.items()}
        annotator._known_function_arg_types = {
            node.name: {
                arg.arg: ast.unparse(arg.annotation)
                for arg in node.args.args
                if arg.annotation is not None
            }
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                declared = self._normalize_annotation(ast.unparse(node.annotation))
                inferred_info = annotator._infer_annotation(node.value)
                if inferred_info is not None and not self._annotation_accepts(declared, inferred_info.text):
                    diagnostics.append(
                        DiagnosticInfo(
                            level="error",
                            line=node.lineno,
                            message=f"{node.target.id} is annotated {declared} but assigned {inferred_info.text}",
                        )
                    )

        for scope_key, lines in assigned_lines.items():
            if len(lines) <= 1:
                continue
            scope, name = scope_key
            source_lines = source.splitlines()
            final_declared = any(self._line_has_final_annotation(source_lines[line - 1], name) for line in lines)
            if not final_declared:
                continue
            for line in lines[1:]:
                diagnostics.append(
                    DiagnosticInfo(
                        level="error",
                        line=line,
                        message=f"{name} is annotated Final but rebound",
                    )
                )

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            inferred_args, arg_warnings = annotator._infer_arg_annotations(node)
            diagnostics.extend(
                DiagnosticInfo(level="warning", line=warning.line, message=warning.message)
                for warning in arg_warnings
            )
            for arg in node.args.args:
                if arg.annotation is None:
                    continue
                declared = self._normalize_annotation(ast.unparse(arg.annotation))
                inferred = inferred_args.get(arg.arg)
                if inferred is not None and not self._annotation_accepts(declared, inferred):
                    diagnostics.append(
                        DiagnosticInfo(
                            level="error",
                            line=arg.lineno,
                            message=f"parameter {arg.arg} is annotated {declared} but used as {inferred}",
                        )
                    )
            if node.returns is not None:
                inferred_return = self._infer_explicit_return_type(annotator, node)
                declared_return = self._normalize_annotation(ast.unparse(node.returns))
                if inferred_return is not None and not self._annotation_accepts(declared_return, inferred_return):
                    diagnostics.append(
                        DiagnosticInfo(
                            level="error",
                            line=node.lineno,
                            message=f"{node.name} is annotated to return {declared_return} but returns {inferred_return}",
                        )
                    )
            else:
                _ret, return_warning = annotator._infer_return_annotation(node, inferred_args)
                if return_warning is not None:
                    diagnostics.append(
                        DiagnosticInfo(level="warning", line=return_warning.line, message=return_warning.message)
                    )
            if check_intent:
                diagnostics.extend(
                    DiagnosticInfo(level="note", line=warning.line, message=warning.message)
                    for warning in annotator._infer_parameter_intents(node)
                )
            if fortran_preflight:
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg not in inferred_args:
                        diagnostics.append(
                            DiagnosticInfo(
                                level="warning",
                                line=arg.lineno,
                                message=f"parameter {arg.arg} remains untyped for translation",
                            )
                        )

            annotator._known_function_arg_types[node.name] = {
                arg.arg: (ast.unparse(arg.annotation) if arg.annotation is not None else inferred_args.get(arg.arg))
                for arg in node.args.args
                if arg.annotation is not None or arg.arg in inferred_args
            }

        return self._dedupe(diagnostics)

    def fix_source(self, source: str, *, fix_type_changes: bool = False) -> tuple[str, list[DiagnosticInfo]]:
        diagnostics = self.check_source(source)
        if not fix_type_changes:
            return source, diagnostics
        tree = ast.parse(source)
        annotator = PythonAnnotator()
        annotator._source = source
        annotator._known_name_types = {}
        annotator._known_function_arg_types = {}
        used_names = self._collect_used_names(tree.body)
        edits = self._collect_type_change_fix_edits(tree.body, {}, annotator, used_names)
        if not edits:
            return source, diagnostics
        fixed = source
        for start, end, replacement in sorted(edits.values(), key=lambda item: item[0], reverse=True):
            fixed = fixed[:start] + replacement + fixed[end:]
        return fixed, diagnostics

    def format_human(self, diagnostics: list[DiagnosticInfo]) -> str:
        if not diagnostics:
            return "No issues found.\n"
        return "".join(f"{diag.level}: line {diag.line}: {diag.message}\n" for diag in diagnostics)

    def format_json(self, diagnostics: list[DiagnosticInfo]) -> str:
        return json.dumps([asdict(diag) for diag in diagnostics], indent=2) + "\n"

    def _infer_explicit_return_type(self, annotator: PythonAnnotator, node: ast.FunctionDef) -> str | None:
        inferred_args, _warnings = annotator._infer_arg_annotations(node)
        known_name_types = dict(annotator._known_name_types)
        known_name_types.update(inferred_args)
        previous_known = annotator._known_name_types
        annotator._known_name_types = known_name_types
        return_infos = [annotator._infer_annotation(stmt.value) for stmt in ast.walk(node) if isinstance(stmt, ast.Return) and stmt.value is not None]
        annotator._known_name_types = previous_known
        return_infos = [info for info in return_infos if info is not None]
        if not return_infos:
            return None
        first = return_infos[0].text
        if any(info.text != first for info in return_infos[1:]):
            return None
        return first

    def _normalize_annotation(self, text: str) -> str:
        if text.startswith("Final[") and text.endswith("]"):
            return text[6:-1]
        return text

    def _annotation_accepts(self, declared: str, inferred: str) -> bool:
        if declared == inferred:
            return True
        if "| None" in declared:
            parts = {part.strip() for part in declared.split("|")}
            return inferred in parts
        return False

    def _line_has_final_annotation(self, line: str, name: str) -> bool:
        stripped = line.strip()
        return stripped.startswith(f"{name}: Final[")

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

    def _collect_type_change_fix_edits(
        self,
        stmts: list[ast.stmt],
        known_types: dict[str, str],
        annotator: PythonAnnotator,
        used_names: set[str],
    ) -> dict[tuple[int, int], tuple[int, int, str]]:
        edits: dict[tuple[int, int], tuple[int, int, str]] = {}
        env = dict(known_types)
        previous_known = annotator._known_name_types
        annotator._known_name_types = dict(env)
        try:
            for index, stmt in enumerate(stmts):
                if isinstance(stmt, ast.FunctionDef):
                    function_known = {
                        arg.arg: ast.unparse(arg.annotation)
                        for arg in stmt.args.args
                        if arg.annotation is not None
                    }
                    nested_used = set(used_names)
                    edits.update(self._collect_type_change_fix_edits(stmt.body, function_known, annotator, nested_used))
                    continue
                target_name, target_node, value = self._assignment_target_info(stmt)
                if target_name is None or target_node is None or value is None:
                    continue
                new_type = self._infer_expression_type(value, env, annotator)
                previous_type = env.get(target_name)
                if previous_type is not None and new_type is not None and previous_type != new_type:
                    new_name = self._fresh_type_change_name(target_name, new_type, used_names)
                    tail = stmts[index + 1 :]
                    if self._tail_is_flat_renamable(tail, target_name, new_name):
                        edits[(target_node.lineno, target_node.col_offset)] = (
                            self._node_start_offset(source=annotator._source, node=target_node),
                            self._node_end_offset(source=annotator._source, node=target_node),
                            new_name,
                        )
                        used_names.add(new_name)
                        for name_node in self._collect_load_name_occurrences(tail, target_name):
                            edits[(name_node.lineno, name_node.col_offset)] = (
                                self._node_start_offset(source=annotator._source, node=name_node),
                                self._node_end_offset(source=annotator._source, node=name_node),
                                new_name,
                            )
                if new_type is not None:
                    env[target_name] = new_type
                    annotator._known_name_types[target_name] = new_type
        finally:
            annotator._known_name_types = previous_known
        return edits

    def _collect_simple_type_change_warnings(
        self,
        stmts: list[ast.stmt],
        annotator: PythonAnnotator,
        *,
        known: dict[str, str],
    ) -> list[DiagnosticInfo]:
        diagnostics: list[DiagnosticInfo] = []
        env = dict(known)
        previous_known = annotator._known_name_types
        annotator._known_name_types = dict(env)
        try:
            for stmt in stmts:
                if isinstance(stmt, ast.FunctionDef):
                    function_known = {
                        arg.arg: ast.unparse(arg.annotation)
                        for arg in stmt.args.args
                        if arg.annotation is not None
                    }
                    diagnostics.extend(self._collect_simple_type_change_warnings(stmt.body, annotator, known=function_known))
                    continue
                target_name, _target_node, value = self._assignment_target_info(stmt)
                if target_name is not None and value is not None:
                    new_type = self._infer_expression_type(value, env, annotator)
                    previous_type = env.get(target_name)
                    if previous_type is not None and new_type is not None and previous_type != new_type:
                        diagnostics.append(
                            DiagnosticInfo(
                                level="warning",
                                line=stmt.lineno,
                                message=f"{target_name} has unstable type: {previous_type} -> {new_type}",
                            )
                        )
                    if new_type is not None:
                        env[target_name] = new_type
                        annotator._known_name_types[target_name] = new_type
                    continue
                if isinstance(stmt, ast.If):
                    diagnostics.extend(self._collect_simple_type_change_warnings(stmt.body, annotator, known=dict(env)))
                    diagnostics.extend(self._collect_simple_type_change_warnings(stmt.orelse, annotator, known=dict(env)))
                elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                    for attr in ("body", "orelse", "finalbody"):
                        diagnostics.extend(self._collect_simple_type_change_warnings(getattr(stmt, attr, []), annotator, known=dict(env)))
                    if isinstance(stmt, ast.Try):
                        for handler in stmt.handlers:
                            diagnostics.extend(self._collect_simple_type_change_warnings(handler.body, annotator, known=dict(env)))
        finally:
            annotator._known_name_types = previous_known
        return diagnostics

    def _infer_expression_type(
        self,
        expr: ast.AST,
        env: dict[str, str],
        annotator: PythonAnnotator,
    ) -> str | None:
        previous_known = annotator._known_name_types
        annotator._known_name_types = dict(previous_known)
        annotator._known_name_types.update(env)
        try:
            info = annotator._infer_annotation(expr)
        finally:
            annotator._known_name_types = previous_known
        if info is not None:
            return info.text
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            mapping = {
                "str": "str",
                "int": "int",
                "float": "float",
                "bool": "bool",
            }
            return mapping.get(expr.func.id)
        return None

    def _assignment_target_info(self, stmt: ast.stmt) -> tuple[str | None, ast.Name | None, ast.AST | None]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id, stmt.targets[0], stmt.value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            return stmt.target.id, stmt.target, stmt.value
        return None, None, None

    def _tail_is_flat_renamable(self, stmts: list[ast.stmt], old_name: str, new_name: str) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)):
                return False
            if self._stmt_writes_or_mutates_name(stmt, old_name) or self._stmt_writes_or_mutates_name(stmt, new_name):
                return False
        return True

    def _stmt_writes_or_mutates_name(self, stmt: ast.stmt, name: str) -> bool:
        if isinstance(stmt, ast.Assign):
            return any(self._target_writes_or_mutates_name(target, name) for target in stmt.targets)
        if isinstance(stmt, ast.AnnAssign):
            return self._target_writes_or_mutates_name(stmt.target, name)
        if isinstance(stmt, ast.AugAssign):
            return self._target_writes_or_mutates_name(stmt.target, name)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.value.id == name and call.func.attr in {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse", "fill", "resize", "shuffle"}:
                    return True
        return False

    def _target_writes_or_mutates_name(self, target: ast.expr, name: str) -> bool:
        if isinstance(target, ast.Name):
            return target.id == name
        if isinstance(target, ast.Subscript):
            return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target.value))
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._target_writes_or_mutates_name(elt, name) for elt in target.elts)
        if isinstance(target, ast.Attribute):
            return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target.value))
        return False

    def _collect_load_name_occurrences(self, stmts: list[ast.stmt], name: str) -> list[ast.Name]:
        matches: list[ast.Name] = []
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name:
                    matches.append(node)
        return matches

    def _collect_used_names(self, stmts: list[ast.stmt]) -> set[str]:
        used: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                used.add(stmt.name)
                used.update(arg.arg for arg in stmt.args.args)
                used.update(arg.arg for arg in stmt.args.kwonlyargs)
                if stmt.args.vararg is not None:
                    used.add(stmt.args.vararg.arg)
                if stmt.args.kwarg is not None:
                    used.add(stmt.args.kwarg.arg)
                used.update(self._collect_used_names(stmt.body))
                continue
            if isinstance(stmt, ast.ClassDef):
                used.add(stmt.name)
                used.update(self._collect_used_names(stmt.body))
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name):
                    used.add(node.id)
        return used

    def _fresh_type_change_name(self, name: str, annotation_text: str, used_names: set[str]) -> str:
        suffix = "".join(ch.lower() if ch.isalnum() else "_" for ch in annotation_text)
        while "__" in suffix:
            suffix = suffix.replace("__", "_")
        suffix = suffix.strip("_") or "value"
        candidate = f"{name}_{suffix}"
        counter = 2
        while candidate in used_names:
            candidate = f"{name}_{suffix}{counter}"
            counter += 1
        return candidate

    def _node_start_offset(self, *, source: str, node: ast.AST) -> int:
        lines = source.splitlines(keepends=True)
        return sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset

    def _node_end_offset(self, *, source: str, node: ast.AST) -> int:
        lines = source.splitlines(keepends=True)
        end_lineno = getattr(node, "end_lineno", node.lineno)
        end_col = getattr(node, "end_col_offset", node.col_offset)
        return sum(len(line) for line in lines[: end_lineno - 1]) + end_col
