from __future__ import annotations

import ast
from dataclasses import dataclass


TYPE_IMPORT = "from typing import Final"
ARRAY_IMPORT = "from array_compiler.annotations import Array1D, Array2D, Array3D"
GENERATED_COMMENT_PREFIX = "xpyannotate:"
READONLY_ARRAY_COMMENT = f"{GENERATED_COMMENT_PREFIX} readonly-array"


@dataclass(frozen=True)
class AnnotationInfo:
    text: str
    is_final: bool = False


@dataclass(frozen=True)
class WarningInfo:
    line: int
    message: str


class PythonAnnotator:
    def annotate_source(self, source: str, *, infer_intent: bool = True) -> tuple[str, list[WarningInfo]]:
        self._source = source
        self._known_name_types: dict[str, str] = {}
        self._known_function_arg_types: dict[str, dict[str, str]] = {}
        tree = ast.parse(source)
        lines = source.splitlines()
        replacements: dict[int, tuple[int, list[str]]] = {}
        warnings: list[WarningInfo] = []
        assigned_types: dict[str, AnnotationInfo] = {}
        assigned_lines: dict[tuple[str, str], list[int]] = {}
        assignment_infos: dict[int, AnnotationInfo] = {}
        assignment_keys: dict[int, tuple[str, str]] = {}
        self._collect_assignments(
            tree.body,
            assigned_types,
            assigned_lines,
            assignment_infos,
            assignment_keys,
            warnings,
            known={},
            scope="<module>",
        )
        mutated_names_by_scope = self._collect_mutated_names_by_scope(tree.body)
        self._known_name_types = {name: info.text for name, info in assigned_types.items()}
        self._known_function_arg_types = {
            node.name: {
                arg.arg: ast.unparse(arg.annotation)
                for arg in node.args.args
                if arg.annotation is not None
            }
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            inferred_args, arg_warnings = self._infer_arg_annotations(node)
            warnings.extend(arg_warnings)
            inferred_return, return_warning = self._infer_return_annotation(node, inferred_args)
            if return_warning is not None:
                warnings.append(return_warning)
            function_comment = None
            if infer_intent:
                function_comment = self._format_intent_comment(node)
            if not inferred_args and inferred_return is None:
                if function_comment is None:
                    continue
            signature_end_lineno = self._signature_end_lineno(node)
            replacement_end, replacement_lines = self._rewrite_function_signature(
                lines[node.lineno - 1:signature_end_lineno],
                node,
                inferred_args=inferred_args,
                inferred_return=inferred_return,
            )
            if function_comment is not None:
                replacement_lines = [function_comment, *replacement_lines]
            replacements[node.lineno] = (replacement_end, replacement_lines)
            self._known_function_arg_types[node.name] = {
                arg.arg: (ast.unparse(arg.annotation) if arg.annotation is not None else inferred_args.get(arg.arg))
                for arg in node.args.args
                if arg.annotation is not None or arg.arg in inferred_args
            }

        for node in ast.walk(tree):
            target: str | None = None
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    continue
                if node.lineno != node.end_lineno:
                    continue
                target = node.targets[0].id
            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name) or node.value is None:
                    continue
                if node.lineno != node.end_lineno:
                    continue
                target = node.target.id
            else:
                continue
            info = assignment_infos.get(node.lineno)
            if info is None or target is None:
                continue
            assignment_key = assignment_keys.get(node.lineno)
            is_final = assignment_key is not None and len(assigned_lines.get(assignment_key, [])) == 1
            scope = assignment_key[0] if assignment_key is not None else "<module>"
            generated_comment = None
            if (
                is_final
                and self._is_array_annotation_text(info.text)
                and target not in mutated_names_by_scope.get(scope, set())
            ):
                generated_comment = READONLY_ARRAY_COMMENT
            line = lines[node.lineno - 1]
            replacements[node.lineno] = (
                node.lineno,
                [
                    self._rewrite_assignment_line(
                        line,
                        target=target,
                        annotation_text=info.text,
                        is_final=is_final,
                        generated_comment=generated_comment,
                    )
                ],
            )

        rewritten_lines = list(lines)
        for lineno in sorted(replacements.keys(), reverse=True):
            end_lineno, new_lines = replacements[lineno]
            rewritten_lines[lineno - 1:end_lineno] = new_lines

        needs_final = any(len(line_nos) == 1 for line_nos in assigned_lines.values())
        needs_array_import = any(
            any(prefix in info.text for prefix in ("Array1D", "Array2D", "Array3D"))
            for info in assigned_types.values()
        )
        rewritten_lines = self._ensure_imports(rewritten_lines, needs_final=needs_final, needs_array_import=needs_array_import)
        return "\n".join(rewritten_lines) + ("\n" if source.endswith("\n") else ""), warnings

    def _infer_parameter_intents(self, node: ast.FunctionDef) -> list[WarningInfo]:
        intent_map = self._infer_parameter_intent_map(node)
        return [WarningInfo(line=arg.lineno, message=f"parameter {arg.arg} intent={intent_map[arg.arg]}") for arg in node.args.args]

    def _infer_parameter_intent_map(self, node: ast.FunctionDef) -> dict[str, str]:
        params = {arg.arg for arg in node.args.args}
        if not params:
            return {}
        rebound: set[str] = set()
        mutated: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    self._classify_target(target, params, rebound, mutated)
            elif isinstance(child, ast.AnnAssign):
                self._classify_target(child.target, params, rebound, mutated)
            elif isinstance(child, ast.AugAssign):
                self._classify_target(child.target, params, rebound, mutated)
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name) and child.target.id in params:
                    rebound.add(child.target.id)
            elif isinstance(child, ast.With):
                for item in child.items:
                    if isinstance(item.optional_vars, ast.Name) and item.optional_vars.id in params:
                        rebound.add(item.optional_vars.id)
            elif isinstance(child, ast.Call):
                self._classify_mutating_call(child, params, mutated)
        intent_map: dict[str, str] = {}
        for arg in node.args.args:
            intent = "in"
            if arg.arg in mutated:
                intent = "mutated"
            elif arg.arg in rebound:
                intent = "rebound"
            intent_map[arg.arg] = intent
        return intent_map

    def _format_intent_comment(self, node: ast.FunctionDef) -> str | None:
        intent_map = self._infer_parameter_intent_map(node)
        if not intent_map:
            return None
        label_map = {
            "in": "in",
            "mutated": "inout",
            "rebound": "local-rebind",
        }
        pieces = [f"{arg.arg}={label_map[intent_map[arg.arg]]}" for arg in node.args.args]
        indent = self._source.splitlines()[node.lineno - 1][: len(self._source.splitlines()[node.lineno - 1]) - len(self._source.splitlines()[node.lineno - 1].lstrip())]
        return f"{indent}# {GENERATED_COMMENT_PREFIX} intent " + ", ".join(pieces)

    def _classify_target(self, target: ast.expr, params: set[str], rebound: set[str], mutated: set[str]) -> None:
        if isinstance(target, ast.Name) and target.id in params:
            rebound.add(target.id)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in params:
            mutated.add(target.value.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._classify_target(elt, params, rebound, mutated)

    def _classify_mutating_call(self, call: ast.Call, params: set[str], mutated: set[str]) -> None:
        if not isinstance(call.func, ast.Attribute):
            return
        if not isinstance(call.func.value, ast.Name):
            return
        if call.func.value.id not in params:
            return
        if call.func.attr in {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse", "fill", "resize", "shuffle"}:
            mutated.add(call.func.value.id)

    def _collect_mutated_names_by_scope(self, stmts: list[ast.stmt], scope: str = "<module>") -> dict[str, set[str]]:
        mutated_by_scope: dict[str, set[str]] = {scope: set()}
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                child_mutations = self._collect_mutated_names_by_scope(stmt.body, stmt.name)
                for child_scope, names in child_mutations.items():
                    mutated_by_scope.setdefault(child_scope, set()).update(names)
                continue
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._collect_mutated_target_names(target, mutated_by_scope[scope])
            elif isinstance(stmt, ast.AnnAssign):
                self._collect_mutated_target_names(stmt.target, mutated_by_scope[scope])
            elif isinstance(stmt, ast.AugAssign):
                self._collect_mutated_target_names(stmt.target, mutated_by_scope[scope])
            elif isinstance(stmt, ast.Call):
                self._collect_mutating_call_names(stmt, mutated_by_scope[scope])
            for child_scope_stmts in self._nested_stmt_lists(stmt):
                child_mutations = self._collect_mutated_names_by_scope(child_scope_stmts, scope)
                mutated_by_scope[scope].update(child_mutations.get(scope, set()))
        return mutated_by_scope

    def _nested_stmt_lists(self, stmt: ast.stmt) -> list[list[ast.stmt]]:
        nested: list[list[ast.stmt]] = []
        for attr in ("body", "orelse", "finalbody"):
            value = getattr(stmt, attr, None)
            if isinstance(value, list):
                nested.append(value)
        if isinstance(stmt, ast.Try):
            nested.extend(handler.body for handler in stmt.handlers)
        return nested

    def _collect_mutated_target_names(self, target: ast.expr, mutated: set[str]) -> None:
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            mutated.add(target.value.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_mutated_target_names(elt, mutated)

    def _collect_mutating_call_names(self, call: ast.Call, mutated: set[str]) -> None:
        if not isinstance(call.func, ast.Attribute):
            return
        if not isinstance(call.func.value, ast.Name):
            return
        if call.func.attr in {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse", "fill", "resize", "shuffle"}:
            mutated.add(call.func.value.id)

    def _collect_assignments(
        self,
        stmts: list[ast.stmt],
        assigned_types: dict[str, AnnotationInfo],
        assigned_lines: dict[tuple[str, str], list[int]],
        assignment_infos: dict[int, AnnotationInfo],
        assignment_keys: dict[int, tuple[str, str]],
        warnings: list[WarningInfo],
        *,
        known: dict[str, str],
        scope: str,
    ) -> None:
        previous_known = self._known_name_types
        self._known_name_types = dict(previous_known)
        self._known_name_types.update(known)
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target = stmt.targets[0].id
                info = self._infer_annotation(stmt.value)
                if info is not None:
                    assignment_infos[stmt.lineno] = info
                    assignment_key = (scope, target)
                    assignment_keys[stmt.lineno] = assignment_key
                    assigned_lines.setdefault(assignment_key, []).append(stmt.lineno)
                    previous = assigned_types.get(target)
                    if previous is not None and previous.text != info.text:
                        warnings.append(
                            WarningInfo(
                                line=stmt.lineno,
                                message=f"{target} has unstable type: {previous.text} -> {info.text}",
                            )
                        )
                    assigned_types[target] = info
                    self._known_name_types[target] = info.text
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                target = stmt.target.id
                info = self._infer_annotation(stmt.value)
                if info is not None:
                    assignment_infos[stmt.lineno] = info
                    assignment_key = (scope, target)
                    assignment_keys[stmt.lineno] = assignment_key
                    assigned_lines.setdefault(assignment_key, []).append(stmt.lineno)
                    previous = assigned_types.get(target)
                    if previous is not None and previous.text != info.text:
                        warnings.append(
                            WarningInfo(
                                line=stmt.lineno,
                                message=f"{target} has unstable type: {previous.text} -> {info.text}",
                            )
                        )
                    assigned_types[target] = info
                    self._known_name_types[target] = info.text
            elif isinstance(stmt, ast.FunctionDef):
                function_known = {
                    arg.arg: ast.unparse(arg.annotation)
                    for arg in stmt.args.args
                    if arg.annotation is not None
                }
                self._collect_assignments(
                    stmt.body,
                    assigned_types,
                    assigned_lines,
                    assignment_infos,
                    assignment_keys,
                    warnings,
                    known=function_known,
                    scope=stmt.name,
                )
            elif isinstance(stmt, ast.If):
                self._collect_assignments(
                    stmt.body,
                    assigned_types,
                    assigned_lines,
                    assignment_infos,
                    assignment_keys,
                    warnings,
                    known=dict(self._known_name_types),
                    scope=scope,
                )
                self._collect_assignments(
                    stmt.orelse,
                    assigned_types,
                    assigned_lines,
                    assignment_infos,
                    assignment_keys,
                    warnings,
                    known=dict(self._known_name_types),
                    scope=scope,
                )
            elif isinstance(stmt, (ast.For, ast.While, ast.With, ast.Try)):
                for inner in getattr(stmt, "body", []):
                    self._collect_assignments(
                        [inner],
                        assigned_types,
                        assigned_lines,
                        assignment_infos,
                        assignment_keys,
                        warnings,
                        known=dict(self._known_name_types),
                        scope=scope,
                    )
                for inner in getattr(stmt, "orelse", []):
                    self._collect_assignments(
                        [inner],
                        assigned_types,
                        assigned_lines,
                        assignment_infos,
                        assignment_keys,
                        warnings,
                        known=dict(self._known_name_types),
                        scope=scope,
                    )
                for inner in getattr(stmt, "finalbody", []):
                    self._collect_assignments(
                        [inner],
                        assigned_types,
                        assigned_lines,
                        assignment_infos,
                        assignment_keys,
                        warnings,
                        known=dict(self._known_name_types),
                        scope=scope,
                    )
                for handler in getattr(stmt, "handlers", []):
                    self._collect_assignments(
                        handler.body,
                        assigned_types,
                        assigned_lines,
                        assignment_infos,
                        assignment_keys,
                        warnings,
                        known=dict(self._known_name_types),
                        scope=scope,
                    )
        self._known_name_types = previous_known

    def _signature_end_lineno(self, node: ast.FunctionDef) -> int:
        if node.body:
            return node.body[0].lineno - 1
        return node.end_lineno or node.lineno

    def _rewrite_function_signature(
        self,
        signature_lines: list[str],
        node: ast.FunctionDef,
        *,
        inferred_args: dict[str, str],
        inferred_return: str | None,
    ) -> tuple[int, list[str]]:
        signature_line = signature_lines[0]
        indent = signature_line[: len(signature_line) - len(signature_line.lstrip())]
        defaults = node.args.defaults
        default_offset = len(node.args.args) - len(defaults)
        parts: list[str] = []
        for index, arg in enumerate(node.args.args):
            annotation_text = ast.unparse(arg.annotation) if arg.annotation is not None else inferred_args.get(arg.arg)
            part = arg.arg if annotation_text is None else f"{arg.arg}: {annotation_text}"
            if index >= default_offset:
                default_expr = defaults[index - default_offset]
                default_text = ast.get_source_segment(self._source, default_expr) or ast.unparse(default_expr)
                if default_text == "None" and annotation_text is not None and "None" not in annotation_text:
                    part = f"{arg.arg}: {annotation_text} | None"
                part += f" = {default_text}"
            parts.append(part)
        if node.args.vararg is not None:
            part = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation is not None:
                part = f"{part}: {ast.unparse(node.args.vararg.annotation)}"
            parts.append(part)
        for kwonly_arg, default_expr in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
            annotation_text = ast.unparse(kwonly_arg.annotation) if kwonly_arg.annotation is not None else inferred_args.get(kwonly_arg.arg)
            part = kwonly_arg.arg if annotation_text is None else f"{kwonly_arg.arg}: {annotation_text}"
            if default_expr is not None:
                default_text = ast.get_source_segment(self._source, default_expr) or ast.unparse(default_expr)
                part += f" = {default_text}"
            parts.append(part)
        if node.args.kwarg is not None:
            part = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation is not None:
                part = f"{part}: {ast.unparse(node.args.kwarg.annotation)}"
            parts.append(part)
        return_text = ast.unparse(node.returns) if node.returns is not None else inferred_return
        line = f"{indent}def {node.name}({', '.join(parts)})"
        if return_text is not None:
            line += f" -> {return_text}"
        line += ":"
        return self._signature_end_lineno(node), [line]

    def _infer_return_annotation(self, node: ast.FunctionDef, inferred_args: dict[str, str]) -> tuple[str | None, WarningInfo | None]:
        if node.returns is not None:
            return None, None
        known_name_types = dict(self._known_name_types)
        known_name_types.update(inferred_args)
        previous_known = self._known_name_types
        self._known_name_types = known_name_types
        return_infos = [self._infer_annotation(stmt.value) for stmt in ast.walk(node) if isinstance(stmt, ast.Return) and stmt.value is not None]
        self._known_name_types = previous_known
        return_infos = [info for info in return_infos if info is not None]
        if not return_infos:
            return None, None
        first = return_infos[0]
        if any(info.text != first.text for info in return_infos[1:]):
            return None, WarningInfo(line=node.lineno, message=f"{node.name} has unstable return type")
        return first.text, None

    def _infer_arg_annotations(self, node: ast.FunctionDef) -> tuple[dict[str, str], list[WarningInfo]]:
        params = {arg.arg for arg in node.args.args}
        inferred: dict[str, str] = {}
        warnings: list[WarningInfo] = []

        def merge(name: str, annotation: str, line: int) -> None:
            previous = inferred.get(name)
            if previous is None:
                inferred[name] = annotation
                return
            if previous != annotation:
                warnings.append(WarningInfo(line=line, message=f"{name} has unstable argument type: {previous} -> {annotation}"))

        defaults = node.args.defaults
        default_offset = len(node.args.args) - len(defaults)
        for index, arg in enumerate(node.args.args):
            if arg.annotation is not None:
                continue
            if index >= default_offset:
                default_info = self._infer_annotation(defaults[index - default_offset])
                if default_info is not None:
                    merge(arg.arg, default_info.text, arg.lineno)

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and child.targets[0].id in params:
                    assigned_info = self._infer_annotation(child.value)
                    if assigned_info is not None:
                        merge(child.targets[0].id, assigned_info.text, child.lineno)
            if isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name) and child.target.id in params and child.value is not None:
                    assigned_info = self._infer_annotation(child.value)
                    if assigned_info is not None:
                        merge(child.target.id, assigned_info.text, child.lineno)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "range":
                for arg in child.args:
                    if isinstance(arg, ast.Name) and arg.id in params:
                        merge(arg.id, "int", child.lineno)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                callee_arg_types = self._known_function_arg_types.get(child.func.id, {})
                callee_formals = list(callee_arg_types.values())
                for actual, formal in zip(child.args, callee_formals, strict=False):
                    if isinstance(actual, ast.Name) and actual.id in params:
                        merge(actual.id, formal, child.lineno)
                for keyword in child.keywords:
                    if keyword.arg in callee_arg_types and isinstance(keyword.value, ast.Name) and keyword.value.id in params:
                        merge(keyword.value.id, callee_arg_types[keyword.arg], child.lineno)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                chain = self._attr_chain(child.func)
                if chain == ["np", "random", "default_rng"]:
                    for arg in child.args:
                        if isinstance(arg, ast.Name) and arg.id in params:
                            merge(arg.id, "int", child.lineno)
                if chain[:1] == ["np"] and child.args:
                    if child.func.attr in {"zeros", "ones", "full", "empty"} and isinstance(child.args[0], ast.Tuple):
                        for elt in child.args[0].elts:
                            if isinstance(elt, ast.Name) and elt.id in params:
                                merge(elt.id, "int", child.lineno)
                    if child.func.attr in {"zeros", "ones", "full", "empty"} and not isinstance(child.args[0], ast.Tuple):
                        for name in self._names_in_expr(child.args[0]):
                            if name in params:
                                merge(name, "int", child.lineno)
                if child.func.attr in {"reshape", "ravel", "flatten", "astype", "sum", "mean", "min", "max", "var", "std", "transpose", "copy"}:
                    if isinstance(child.func.value, ast.Name) and child.func.value.id in params:
                        merge(child.func.value.id, "Array1D[float]", child.lineno)
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id in params:
                if child.attr in {"shape", "ndim", "dtype"}:
                    merge(child.value.id, "Array1D[float]", child.lineno)
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id in params:
                rank = 2 if isinstance(child.slice, ast.Tuple) else 1
                merge(child.value.id, f"Array{rank}D[float]", child.lineno)
            if isinstance(child, ast.Compare):
                for side in [child.left, *child.comparators]:
                    if isinstance(side, ast.Name) and side.id in params:
                        others = [child.left, *child.comparators]
                        for other in others:
                            if other is side:
                                continue
                            other_info = self._infer_annotation(other)
                            if other_info is not None and other_info.text in {"float", "int", "str"}:
                                merge(side.id, other_info.text, child.lineno)

        stable = {name: annotation for name, annotation in inferred.items() if not any(w.message.startswith(f"{name} has unstable argument type") for w in warnings)}
        return stable, warnings

    def _names_in_expr(self, expr: ast.AST) -> set[str]:
        return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}

    def _ensure_imports(self, lines: list[str], *, needs_final: bool, needs_array_import: bool) -> list[str]:
        if not needs_final and not needs_array_import:
            return lines
        existing = set(lines)
        insert_at = 0
        while insert_at < len(lines) and (lines[insert_at].startswith("from __future__") or lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")):
            insert_at += 1
        imports_to_add: list[str] = []
        if needs_final and TYPE_IMPORT not in existing:
            imports_to_add.append(TYPE_IMPORT)
        if needs_array_import and ARRAY_IMPORT not in existing:
            imports_to_add.append(ARRAY_IMPORT)
        if not imports_to_add:
            return lines
        return lines[:insert_at] + imports_to_add + lines[insert_at:]

    def _has_existing_annotation(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return False
        left = stripped.split("=", 1)[0]
        return ":" in left

    def _rewrite_assignment_line(
        self,
        line: str,
        *,
        target: str,
        annotation_text: str,
        is_final: bool,
        generated_comment: str | None = None,
    ) -> str:
        before, sep, after = line.partition("=")
        if not sep:
            return line
        stripped_before = before.strip()
        if ":" in stripped_before:
            name_part, _, _existing = stripped_before.partition(":")
            indent = before[: len(before) - len(before.lstrip())]
            type_text = f"Final[{annotation_text}]" if is_final else annotation_text
            rewritten = f"{indent}{name_part.strip()}: {type_text} = {after.lstrip()}"
            return self._append_generated_comment(rewritten, generated_comment)
        annotation = f"{target}: Final[{annotation_text}] = " if is_final else f"{target}: {annotation_text} = "
        rewritten = line.replace(f"{target} = ", annotation, 1)
        return self._append_generated_comment(rewritten, generated_comment)

    def _append_generated_comment(self, line: str, generated_comment: str | None) -> str:
        if generated_comment is None or generated_comment in line:
            return line
        return f"{line}  # {generated_comment}"

    def _is_array_annotation_text(self, text: str) -> bool:
        return text.startswith("Array1D[") or text.startswith("Array2D[") or text.startswith("Array3D[")

    def _infer_annotation(self, expr: ast.AST) -> AnnotationInfo | None:
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return AnnotationInfo("bool")
            if isinstance(expr.value, int):
                return AnnotationInfo("int")
            if isinstance(expr.value, float):
                return AnnotationInfo("float")
            if isinstance(expr.value, str):
                return AnnotationInfo("str")
            return None
        if isinstance(expr, ast.Name):
            known = getattr(self, "_known_name_types", {}).get(expr.id)
            if known is not None:
                return AnnotationInfo(known)
            return None
        if isinstance(expr, ast.List):
            element_type = self._infer_list_element_type(expr)
            if element_type is None:
                return None
            rank = 2 if expr.elts and isinstance(expr.elts[0], ast.List) else 1
            return AnnotationInfo(f"Array{rank}D[{element_type}]")
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            chain = self._attr_chain(expr.func)
            if chain in (
                ["np", "array"],
                ["np", "asarray"],
                ["np", "zeros"],
                ["np", "ones"],
                ["np", "full"],
                ["np", "empty"],
                ["np", "arange"],
                ["np", "linspace"],
            ):
                rank = self._infer_numpy_rank(expr)
                dtype = self._infer_numpy_dtype(expr) or "float"
                return AnnotationInfo(f"Array{rank}D[{dtype}]")
            if expr.func.attr == "astype":
                dtype = self._numpy_dtype_name(expr.args[0]) if expr.args else "float"
                rank = 1
                return AnnotationInfo(f"Array{rank}D[{dtype}]")
            if expr.func.attr == "mean":
                return AnnotationInfo("float")
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            if expr.func.id == "len":
                return AnnotationInfo("int")
            if expr.func.id in {"max", "min"} and expr.args:
                arg_infos = [self._infer_annotation(arg) for arg in expr.args]
                arg_infos = [info for info in arg_infos if info is not None]
                if arg_infos and all(info.text == "int" for info in arg_infos):
                    return AnnotationInfo("int")
                if arg_infos and all(info.text in {"int", "float"} for info in arg_infos):
                    if any(info.text == "float" for info in arg_infos):
                        return AnnotationInfo("float")
                    return AnnotationInfo("int")
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and self._attr_chain(expr.func) == ["np", "dot"]
        ):
            return AnnotationInfo("float")
        if isinstance(expr, ast.UnaryOp):
            return self._infer_annotation(expr.operand)
        if isinstance(expr, ast.BinOp):
            left = self._infer_annotation(expr.left)
            right = self._infer_annotation(expr.right)
            if left is not None and left.text.startswith("Array"):
                return left
            if right is not None and right.text.startswith("Array"):
                return right
            if left is not None and right is not None:
                if left.text == "float" or right.text == "float":
                    return AnnotationInfo("float")
                if left.text == right.text:
                    return left
        if isinstance(expr, ast.Compare):
            return AnnotationInfo("Array1D[bool]")
        return None

    def _infer_list_element_type(self, expr: ast.List) -> str | None:
        if not expr.elts:
            return "float"
        first = expr.elts[0]
        if isinstance(first, ast.List):
            return self._infer_list_element_type(first)
        inferred = self._infer_annotation(first)
        if inferred is None:
            return None
        return inferred.text

    def _infer_numpy_rank(self, expr: ast.Call) -> int:
        if isinstance(expr.func, ast.Attribute) and expr.func.attr in {"arange", "linspace"}:
            return 1
        if not expr.args:
            return 1
        first = expr.args[0]
        if isinstance(first, ast.Tuple):
            return len(first.elts)
        if isinstance(first, ast.List) and first.elts and isinstance(first.elts[0], ast.List):
            return 2
        if isinstance(first, ast.List):
            return 1
        return 1

    def _infer_numpy_dtype(self, expr: ast.Call) -> str | None:
        for keyword in expr.keywords:
            if keyword.arg == "dtype":
                return self._numpy_dtype_name(keyword.value)
        return None

    def _numpy_dtype_name(self, expr: ast.AST) -> str | None:
        if not isinstance(expr, ast.Attribute):
            return None
        mapping = {
            ("np", "int32"): "int",
            ("np", "int64"): "int",
            ("np", "float32"): "float",
            ("np", "float64"): "float",
            ("np", "bool_"): "bool",
        }
        return mapping.get(tuple(self._attr_chain(expr)))

    def _attr_chain(self, node: ast.Attribute) -> list[str]:
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return list(reversed(parts))
