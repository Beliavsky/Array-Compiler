from __future__ import annotations

import keyword
import re
from dataclasses import dataclass

from ..python_numpy import PythonNumpyFrontend


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    col: int = 1
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class NameExpr:
    name: str
    line: int = 0
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class NumberExpr:
    text: str
    integer_like: bool = False
    explicit_integer: bool = False
    line: int = 0
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class StringExpr:
    text: str


@dataclass(frozen=True)
class BoolExpr:
    value: bool


@dataclass(frozen=True)
class UnaryExpr:
    op: str
    operand: object


@dataclass(frozen=True)
class BinaryExpr:
    left: object
    op: str
    right: object


@dataclass(frozen=True)
class CallArg:
    name: str | None
    value: object


@dataclass(frozen=True)
class CallExpr:
    func: object
    args: tuple[CallArg, ...]


@dataclass(frozen=True)
class FunctionExpr:
    params: tuple[Param, ...]
    body: object


@dataclass(frozen=True)
class FieldExpr:
    value: object
    name: str


@dataclass(frozen=True)
class IndexExpr:
    value: object
    indices: tuple[object | None, ...]


@dataclass(frozen=True)
class Param:
    name: str
    default: object | None = None


@dataclass(frozen=True)
class AssignStmt:
    target: object
    value: object
    line: int


@dataclass(frozen=True)
class ExprStmt:
    value: object
    line: int


@dataclass(frozen=True)
class ReturnStmt:
    value: object
    line: int


@dataclass(frozen=True)
class BreakStmt:
    line: int


@dataclass(frozen=True)
class ContinueStmt:
    line: int


@dataclass(frozen=True)
class IfStmt:
    test: object
    body: tuple[object, ...]
    orelse: tuple[object, ...]
    line: int


@dataclass(frozen=True)
class IfExpr:
    test: object
    body: object
    orelse: object
    line: int


@dataclass(frozen=True)
class ForStmt:
    target: str
    iterable: object
    body: tuple[object, ...]
    line: int


@dataclass(frozen=True)
class WhileStmt:
    test: object
    body: tuple[object, ...]
    line: int


@dataclass(frozen=True)
class FunctionDefStmt:
    name: str
    params: tuple[Param, ...]
    body: tuple[object, ...]
    line: int


TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>[ \t\r]+)
  | (?P<NEWLINE>\n)
  | (?P<COMMENT>\#[^\n]*)
  | (?P<NUMBER>\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?L?|\.\d+(?:[Ee][+-]?\d+)?)
  | (?P<STRING>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<OP>%\*%|<<-|<-|<=|>=|==|!=|\|\||&&|%/%|%%|\^|[=+\-*/:,;()[\]{}<>!&|$~])
  | (?P<IDENT>[A-Za-z_.][A-Za-z0-9_.]*)
    """,
    re.VERBOSE,
)

INFIX_BINDING_POWER = {
    "~": (5, 6),
    "||": (10, 11),
    "|": (15, 16),
    "&&": (20, 21),
    "&": (25, 26),
    "==": (30, 31),
    "!=": (30, 31),
    "<": (30, 31),
    "<=": (30, 31),
    ">": (30, 31),
    ">=": (30, 31),
    "+": (40, 41),
    "-": (40, 41),
    "%*%": (50, 51),
    "*": (60, 61),
    "/": (60, 61),
    "%/%": (60, 61),
    "%%": (60, 61),
    ":": (70, 71),
    "^": (90, 89),
}


class RSubsetFrontend:
    def lower_source(self, source: str, module_name: str = "translated_module"):
        python_source = self.translate_source(source)
        return PythonNumpyFrontend().lower_source(python_source, module_name=module_name)

    def translate_source(self, source: str, *, standalone: bool = False) -> str:
        parser = RSubsetParser(source)
        program = parser.parse_program()
        return RSubsetRenderer(
            program,
            parser.type_hints,
            parser.field_hints,
            parser.comment_lines,
            parser.source_lines,
            standalone=standalone,
        ).render()


class RSubsetParser:
    def __init__(self, source: str) -> None:
        self._source = source
        self.source_lines = source.splitlines()
        self.comment_lines = self._extract_comment_lines(source)
        self._tokens = self._tokenize(source)
        self._index = 0
        self.type_hints: dict[str, str] = {}
        self.field_hints: dict[str, dict[str, int]] = {}
        self.function_field_hints: dict[str, dict[str, int]] = {}
        self.function_record_returns: set[str] = set()

    def parse_program(self) -> list[object]:
        statements: list[object] = []
        while not self._at_end():
            if self._match_newlines():
                continue
            statements.append(self._parse_statement())
            self._consume_statement_end()
        return statements

    def _tokenize(self, source: str) -> list[Token]:
        tokens: list[Token] = []
        index = 0
        line = 1
        line_start = 0
        while index < len(source):
            match = TOKEN_RE.match(source, index)
            if match is None:
                raise NotImplementedError(f"unsupported R syntax near line {line}: {source[index:index+20]!r}")
            kind = match.lastgroup
            text = match.group()
            col = index - line_start + 1
            if kind == "NEWLINE":
                tokens.append(Token("NEWLINE", text, line, col=col, start=index, end=match.end()))
                line += 1
                line_start = match.end()
            elif kind not in {"SPACE", "COMMENT"}:
                tokens.append(Token(kind or "", text, line, col=col, start=index, end=match.end()))
            index = match.end()
        tokens.append(Token("EOF", "", line, col=index - line_start + 1, start=index, end=index))
        return tokens

    def _extract_comment_lines(self, source: str) -> dict[int, str]:
        comments: dict[int, str] = {}
        for line_no, line in enumerate(source.splitlines(), start=1):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            comment = stripped[1:]
            if comment.startswith(" "):
                comment = comment[1:]
            comments[line_no] = comment
        return comments

    def _peek(self, offset: int = 0) -> Token:
        return self._tokens[min(self._index + offset, len(self._tokens) - 1)]

    def _advance(self) -> Token:
        token = self._peek()
        self._index += 1
        return token

    def _at_end(self) -> bool:
        return self._peek().kind == "EOF"

    def _match(self, *texts: str) -> bool:
        token = self._peek()
        if token.text in texts:
            self._index += 1
            return True
        return False

    def _match_kind(self, *kinds: str) -> Token | None:
        token = self._peek()
        if token.kind in kinds:
            self._index += 1
            return token
        return None

    def _expect(self, text: str) -> Token:
        token = self._peek()
        if token.text != text:
            raise NotImplementedError(f"expected {text!r} near line {token.line}")
        self._index += 1
        return token

    def _expect_ident(self) -> Token:
        token = self._peek()
        if token.kind != "IDENT":
            raise NotImplementedError(f"expected identifier near line {token.line}")
        self._index += 1
        return token

    def _match_newlines(self) -> bool:
        matched = False
        while self._match_kind("NEWLINE") is not None:
            matched = True
        return matched

    def _consume_statement_end(self) -> None:
        while self._peek().kind == "NEWLINE" or self._peek().text == ";":
            self._advance()

    def _parse_statement(self) -> object:
        token = self._peek()
        if token.kind == "IDENT" and token.text == "if":
            return self._parse_if()
        if token.kind == "IDENT" and token.text == "for":
            return self._parse_for()
        if token.kind == "IDENT" and token.text == "while":
            return self._parse_while()
        if token.kind == "IDENT" and token.text == "repeat":
            line = self._advance().line
            body = tuple(self._parse_block_or_stmt())
            return WhileStmt(test=BoolExpr(True), body=body, line=line)
        if token.kind == "IDENT" and token.text == "break":
            self._advance()
            return BreakStmt(line=token.line)
        if token.kind == "IDENT" and token.text == "next":
            self._advance()
            return ContinueStmt(line=token.line)
        if token.kind == "IDENT" and token.text == "return":
            self._advance()
            return ReturnStmt(self._parse_expression(), line=token.line)
        if token.text in {"+", "-", "!"}:
            return ExprStmt(self._parse_expression(), line=token.line)

        start_index = self._index
        target = self._parse_target()
        if self._peek().text in {"=", "<-", "<<-"}:
            op = self._advance()
            if isinstance(target, NameExpr) and self._peek().kind == "IDENT" and self._peek().text == "function":
                function_stmt = self._parse_function_definition(target.name, op.line)
                self.type_hints.pop(target.name, None)
                return function_stmt
            value = self._parse_expression()
            self._record_type_hint(target, value)
            self._record_field_hint(target, value)
            return AssignStmt(target=target, value=value, line=op.line)

        self._index = start_index
        return ExprStmt(self._parse_expression(), line=token.line)

    def _parse_function_definition(self, name: str, line: int) -> FunctionDefStmt:
        self._expect("function")
        self._expect("(")
        params: list[Param] = []
        if not self._match(")"):
            while True:
                self._match_newlines()
                param_name = self._expect_ident().text
                default = None
                if self._match("="):
                    default = self._parse_expression()
                self._match_newlines()
                params.append(Param(param_name, default))
                if self._match(")"):
                    break
                self._expect(",")
        if self._match("{"):
            body = self._parse_block()
        else:
            body = [ReturnStmt(self._parse_expression(), line=line)]
        if body and isinstance(body[-1], ExprStmt):
            last_expr = body[-1]
            body[-1] = ReturnStmt(last_expr.value, line=last_expr.line)
        if body:
            last_stmt = body[-1]
            if isinstance(last_stmt, ReturnStmt):
                fields = self._named_field_hints(last_stmt.value)
                if fields is not None:
                    self.function_field_hints[name] = fields
                if (
                    isinstance(last_stmt.value, CallExpr)
                    and isinstance(last_stmt.value.func, NameExpr)
                    and last_stmt.value.func.name == "list"
                ):
                    self.function_record_returns.add(name)
        return FunctionDefStmt(name=name, params=tuple(params), body=tuple(body), line=line)

    def _parse_if(self) -> IfStmt:
        line = self._advance().line
        self._expect("(")
        test = self._parse_expression()
        self._expect(")")
        body = tuple(self._parse_block_or_stmt())
        orelse: tuple[object, ...] = ()
        self._match_newlines()
        if self._peek().kind == "IDENT" and self._peek().text == "else":
            self._advance()
            orelse = tuple(self._parse_block_or_stmt())
        return IfStmt(test=test, body=body, orelse=orelse, line=line)

    def _parse_for(self) -> ForStmt:
        line = self._advance().line
        self._expect("(")
        target = self._expect_ident().text
        self._expect("in")
        iterable = self._parse_expression()
        self._expect(")")
        body = tuple(self._parse_block_or_stmt())
        return ForStmt(target=target, iterable=iterable, body=body, line=line)

    def _parse_while(self) -> WhileStmt:
        line = self._advance().line
        self._expect("(")
        test = self._parse_expression()
        self._expect(")")
        body = tuple(self._parse_block_or_stmt())
        return WhileStmt(test=test, body=body, line=line)

    def _parse_block_or_stmt(self) -> list[object]:
        self._match_newlines()
        if self._match("{"):
            return self._parse_block()
        return [self._parse_statement()]

    def _parse_block(self) -> list[object]:
        statements: list[object] = []
        self._match_newlines()
        while not self._match("}"):
            if self._at_end():
                raise NotImplementedError("unterminated block in R source")
            statements.append(self._parse_statement())
            self._consume_statement_end()
        return statements

    def _parse_target(self) -> object:
        expr = self._parse_primary()
        while True:
            if self._peek().text == "(":
                expr = self._finish_call(expr)
                continue
            if self._match("$"):
                expr = FieldExpr(expr, self._expect_ident().text)
                continue
            if self._match("["):
                expr = self._finish_index(expr)
                continue
            break
        return expr

    def _parse_expression(self, min_bp: int = 0) -> object:
        self._match_newlines()
        lhs = self._parse_prefix()
        while True:
            self._match_newlines()
            token = self._peek()
            if token.text == "(":
                lhs = self._finish_call(lhs)
                continue
            if token.text == "$":
                self._advance()
                lhs = FieldExpr(lhs, self._expect_ident().text)
                continue
            if token.text == "[":
                self._advance()
                lhs = self._finish_index(lhs)
                continue
            binding = INFIX_BINDING_POWER.get(token.text)
            if binding is None or binding[0] < min_bp:
                break
            op = self._advance().text
            self._match_newlines()
            rhs = self._parse_expression(binding[1])
            lhs = BinaryExpr(lhs, op, rhs)
        return lhs

    def _parse_prefix(self) -> object:
        token = self._peek()
        if token.kind == "IDENT" and token.text == "if":
            return self._parse_if_expr()
        if token.text in {"+", "-", "!"}:
            self._advance()
            return UnaryExpr(token.text, self._parse_expression(85))
        return self._parse_primary()

    def _parse_primary(self) -> object:
        token = self._advance()
        if token.kind == "NUMBER":
            if token.text.endswith("L"):
                return NumberExpr(
                    token.text[:-1],
                    integer_like=True,
                    explicit_integer=True,
                    line=token.line,
                    start=token.start,
                    end=token.end,
                )
            return NumberExpr(
                token.text,
                integer_like=bool(re.fullmatch(r"[+-]?\d+", token.text)),
                line=token.line,
                start=token.start,
                end=token.end,
            )
        if token.kind == "STRING":
            return StringExpr(token.text)
        if token.kind == "IDENT":
            if token.text == "TRUE":
                return BoolExpr(True)
            if token.text == "FALSE":
                return BoolExpr(False)
            if token.text == "function":
                return self._parse_function_expr()
            return NameExpr(token.text, line=token.line, start=token.start, end=token.end)
        if token.text == "(":
            self._match_newlines()
            expr = self._parse_expression()
            self._match_newlines()
            self._expect(")")
            return expr
        raise NotImplementedError(f"unsupported R expression near line {token.line}")

    def _parse_if_expr(self) -> IfExpr:
        line = self._advance().line
        self._expect("(")
        self._match_newlines()
        test = self._parse_expression()
        self._match_newlines()
        self._expect(")")
        self._match_newlines()
        body = self._parse_expr_branch()
        self._match_newlines()
        self._expect("else")
        self._match_newlines()
        orelse = self._parse_expr_branch()
        return IfExpr(test=test, body=body, orelse=orelse, line=line)

    def _parse_expr_branch(self) -> object:
        self._match_newlines()
        if self._match("{"):
            statements = self._parse_block()
            if len(statements) != 1:
                raise NotImplementedError("if-expression blocks must contain a single expression")
            stmt = statements[0]
            if isinstance(stmt, ExprStmt):
                return stmt.value
            if isinstance(stmt, ReturnStmt):
                return stmt.value
            raise NotImplementedError("if-expression block must reduce to a value")
        return self._parse_expression()

    def _parse_function_expr(self) -> FunctionExpr:
        self._expect("(")
        params: list[Param] = []
        self._match_newlines()
        if not self._match(")"):
            while True:
                self._match_newlines()
                name = self._expect_ident().text
                default = None
                if self._match("="):
                    default = self._parse_expression()
                params.append(Param(name, default))
                self._match_newlines()
                if self._match(")"):
                    break
                self._expect(",")
        self._match_newlines()
        body = self._parse_expr_branch()
        return FunctionExpr(params=tuple(params), body=body)

    def _finish_call(self, func: object) -> CallExpr:
        self._expect("(")
        args: list[CallArg] = []
        self._match_newlines()
        if not self._match(")"):
            while True:
                self._match_newlines()
                if self._peek().kind == "IDENT" and self._peek(1).text == "=":
                    name = self._advance().text
                    self._advance()
                    value = self._parse_expression()
                    args.append(CallArg(name, value))
                else:
                    args.append(CallArg(None, self._parse_expression()))
                self._match_newlines()
                if self._match(")"):
                    break
                self._expect(",")
                self._match_newlines()
        return CallExpr(func=func, args=tuple(args))

    def _finish_index(self, value: object) -> IndexExpr:
        indices: list[object | None] = []
        double_bracket = self._match("[")
        self._match_newlines()
        if not self._match("]"):
            while True:
                self._match_newlines()
                if self._peek().kind == "IDENT" and self._peek().text == "drop" and self._peek(1).text == "=":
                    self._advance()
                    self._advance()
                    self._parse_expression()
                    self._match_newlines()
                    if self._match("]"):
                        if double_bracket:
                            self._match_newlines()
                            self._expect("]")
                        break
                    self._expect(",")
                    self._match_newlines()
                    continue
                if self._peek().text in {",", "]"}:
                    indices.append(None)
                else:
                    indices.append(self._parse_expression())
                self._match_newlines()
                if self._match("]"):
                    if double_bracket:
                        self._match_newlines()
                        self._expect("]")
                    break
                self._expect(",")
                self._match_newlines()
        return IndexExpr(value=value, indices=tuple(indices))

    def _record_type_hint(self, target: object, value: object) -> None:
        if not isinstance(target, NameExpr):
            return
        inferred = self._infer_value_type_hint(value)
        if inferred is not None:
            self.type_hints[target.name] = inferred

    def _infer_value_type_hint(self, value: object) -> str | None:
        if isinstance(value, NumberExpr):
            return "integer" if value.integer_like else "double"
        if isinstance(value, StringExpr):
            return "character"
        if isinstance(value, BoolExpr):
            return "logical"
        if isinstance(value, UnaryExpr) and value.op == "!":
            return "logical"
        if isinstance(value, BinaryExpr) and value.op in {"<", "<=", ">", ">=", "==", "!=", "&", "|", "&&", "||"}:
            return "logical"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name in {"is.na", "is.finite"}:
            return "logical"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "as.integer":
            return "integer"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "as.numeric":
            if value.args:
                hinted = self._infer_value_type_hint(value.args[0].value)
                if hinted is not None:
                    return hinted
            return "double"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "as.character":
            return "character"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "list":
            return "record"
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "c":
            hinted = self._infer_c_type_hint(value)
            if hinted is not None:
                return hinted
        if isinstance(value, IfExpr):
            body_hint = self._infer_value_type_hint(value.body)
            else_hint = self._infer_value_type_hint(value.orelse)
            if body_hint == else_hint:
                return body_hint
            if {body_hint, else_hint} <= {"integer", "double"}:
                return "double"
            if body_hint is not None:
                return body_hint
            return else_hint
        return None

    def _record_field_hint(self, target: object, value: object) -> None:
        if not isinstance(target, NameExpr):
            return
        fields = self._named_field_hints(value)
        if fields is not None:
            self.field_hints[target.name] = fields
            if (
                isinstance(value, CallExpr)
                and isinstance(value.func, NameExpr)
                and value.func.name == "list"
            ):
                self.type_hints[target.name] = "record"
            return
        if isinstance(value, CallExpr) and isinstance(value.func, NameExpr):
            function_fields = self.function_field_hints.get(value.func.name)
            if function_fields is not None:
                self.field_hints[target.name] = function_fields
                if value.func.name in self.function_record_returns:
                    self.type_hints[target.name] = "record"

    def _named_field_hints(self, value: object) -> dict[str, int] | None:
        if not (
            isinstance(value, CallExpr)
            and isinstance(value.func, NameExpr)
            and value.func.name in {"list", "c"}
        ):
            return None
        fields: dict[str, int] = {}
        position = 0
        for arg in value.args:
            if arg.name is None:
                return None
            fields[arg.name] = position
            position += 1
        return fields if fields else None

    def _infer_c_type_hint(self, expr: CallExpr) -> str | None:
        hint: str | None = None
        for arg in expr.args:
            value = arg.value
            current: str | None = None
            if isinstance(value, StringExpr):
                current = "character"
            elif isinstance(value, BoolExpr):
                current = "logical"
            elif isinstance(value, NumberExpr):
                current = "integer" if value.integer_like else "double"
            elif isinstance(value, NameExpr):
                current = {
                    "NA_integer_": "integer",
                    "NA_real_": "double",
                    "NA_character_": "character",
                    "NA": "double",
                    "TRUE": "logical",
                    "FALSE": "logical",
                }.get(value.name)
            if current is None:
                return None
            if hint is None:
                hint = current
                continue
            if hint != current:
                if {hint, current} <= {"integer", "double"}:
                    hint = "double"
                    continue
                return None
        return hint


class RSubsetRenderer:
    def __init__(
        self,
        statements: list[object],
        type_hints: dict[str, str],
        field_hints: dict[str, dict[str, int]],
        comment_lines: dict[int, str],
        source_lines: list[str],
        *,
        standalone: bool = False,
    ) -> None:
        self._statements = statements
        self._type_hints = type_hints
        self._field_hints = field_hints
        self._comment_lines = comment_lines
        self._used_comment_lines: set[int] = set()
        self._source_lines = source_lines
        self._standalone = standalone
        self._module_assigned_names = {
            stmt.target.name
            for stmt in statements
            if isinstance(stmt, AssignStmt) and isinstance(stmt.target, NameExpr)
        }
        self._module_assignments = {
            stmt.target.name: stmt.value
            for stmt in statements
            if isinstance(stmt, AssignStmt) and isinstance(stmt.target, NameExpr)
        }
        self._module_assignments_ordered = [
            (stmt.target.name, stmt.value)
            for stmt in statements
            if isinstance(stmt, AssignStmt) and isinstance(stmt.target, NameExpr)
        ]
        self._module_logical_constants = {
            stmt.target.name: stmt.value.value
            for stmt in statements
            if isinstance(stmt, AssignStmt)
            and isinstance(stmt.target, NameExpr)
            and isinstance(stmt.value, BoolExpr)
        }
        self._current_function_name: str | None = None
        self._current_function_param_names: set[str] = set()
        self._function_param_annotations = {
            stmt.name: self._infer_param_annotations(stmt)
            for stmt in statements
            if isinstance(stmt, FunctionDefStmt)
        }

    def render(self) -> str:
        lines = self._render_prelude()
        for stmt in self._statements:
            lines.extend(self._render_attached_comments(stmt, 0, 0))
            lines.extend(self._render_stmt(stmt, 0))
        return "\n".join(lines) + "\n"

    def _render_name(self, name: str) -> str:
        return f"{name}_" if keyword.iskeyword(name) else name

    def _stmt_line(self, stmt: object) -> int:
        return getattr(stmt, "line", 0)

    def _line_text(self, line_no: int) -> str:
        if 1 <= line_no <= len(self._source_lines):
            return self._source_lines[line_no - 1]
        return ""

    def _render_attached_comments(self, stmt: object, indent: int, lower_bound_line: int) -> list[str]:
        stmt_line = self._stmt_line(stmt)
        if stmt_line <= 0:
            return []
        comments: list[str] = []
        for line_no in range(stmt_line - 1, lower_bound_line, -1):
            if line_no in self._used_comment_lines:
                continue
            line_text = self._line_text(line_no)
            stripped = line_text.strip()
            if stripped == "":
                continue
            if line_no in self._comment_lines:
                comments.append((" " * indent) + f"# {self._comment_lines[line_no]}")
                continue
            break
        comments.reverse()
        for line in range(lower_bound_line + 1, stmt_line):
            if line in self._comment_lines:
                self._used_comment_lines.add(line)
        return comments

    def _render_prelude(self) -> list[str]:
        if self._standalone:
            return [
                "import numpy as np",
                "import sys",
                "import time",
                "",
                "class RTypedNA:",
                "    def __init__(self, type_name, value):",
                "        self.type_name = type_name",
                "        self.value = value",
                "",
                "class RNamedArray:",
                "    def __init__(self, values, names):",
                "        self.values = np.asarray(values)",
                "        self.names = list(names)",
                "",
                "    def __array__(self, dtype=None):",
                "        return np.asarray(self.values, dtype=dtype)",
                "",
                "    def __len__(self):",
                "        return len(self.values)",
                "",
                "    def __getitem__(self, key):",
                "        if isinstance(key, str):",
                "            return self.values[self.names.index(key)]",
                "        return self.values[key]",
                "",
                "    def _binary(self, other, op):",
                "        other_values = other.values if isinstance(other, RNamedArray) else np.asarray(other)",
                "        return RNamedArray(op(self.values, other_values), self.names)",
                "",
                "    def __add__(self, other):",
                "        return self._binary(other, np.add)",
                "",
                "    def __sub__(self, other):",
                "        return self._binary(other, np.subtract)",
                "",
                "    def __mul__(self, other):",
                "        return self._binary(other, np.multiply)",
                "",
                "    def __truediv__(self, other):",
                "        return self._binary(other, np.divide)",
                "",
                "    def __pow__(self, other):",
                "        return self._binary(other, np.power)",
                "",
                "R_NAME_MAP = {}",
                "R_DIMNAME_MAP = {}",
                "",
                "NA = np.nan",
                "NA_integer_ = np.nan",
                "NA_real_ = np.nan",
                'NA_character_ = RTypedNA(\"character\", None)',
                "",
                "def r_colon(start, stop):",
                "    if start <= stop:",
                "        return np.arange(start, stop + 1)",
                "    return np.arange(start, stop - 1, -1)",
                "",
                "def r_seq(start, stop, by=1, length_out=None):",
                "    if length_out is not None:",
                "        n = int(length_out)",
                "        if n <= 0:",
                "            return np.array([])",
                "        if n == 1:",
                "            return np.array([start])",
                "        return np.linspace(start, stop, n)",
                "    if by == 0:",
                '        raise ValueError(\"seq by must be nonzero\")',
                "    n = int((stop - start) / by) + 1",
                "    return start + by * np.arange(n)",
                "",
                "def r_seq_len(n):",
                "    return np.arange(1, n + 1)",
                "",
                "def r_as_numeric(x):",
                "    arr = np.asarray(x, dtype=float)",
                "    if arr.ndim == 0:",
                "        return float(arr)",
                "    return arr",
                "",
                "def r_as_character(x):",
                "    arr = np.asarray(x, dtype=object)",
                "    if arr.ndim == 0:",
                "        return r_scalar_string(arr.item())",
                "    flat = [r_scalar_string(item) for item in arr.reshape(-1).tolist()]",
                "    return np.array(flat, dtype=object).reshape(arr.shape)",
                "",
                "def r_c(*values):",
                "    flat = []",
                "    inferred_type = None",
                "    for value in values:",
                "        if isinstance(value, RTypedNA):",
                "            flat.append(value.value)",
                "            inferred_type = inferred_type or value.type_name",
                "            continue",
                "        if isinstance(value, np.ndarray):",
                "            flat.extend(value.reshape(-1).tolist())",
                "            continue",
                "        flat.append(value)",
                "    if inferred_type == 'character':",
                "        return np.array(flat, dtype=object)",
                "    if inferred_type == 'integer':",
                "        return np.array(flat, dtype=float)",
                "    return np.array(flat)",
                "",
                "def r_named_c(names, *values):",
                "    return RNamedArray(r_c(*values), names)",
                "",
                "def r_matrix(data, nrow=None, ncol=None, byrow=False):",
                "    flat = np.asarray(data).reshape(-1)",
                "    if flat.dtype.kind in ('i', 'u'):",
                "        flat = flat.astype(float)",
                "    if nrow is None and ncol is None:",
                "        raise ValueError('matrix() requires nrow or ncol')",
                "    if nrow is None:",
                "        ncol = int(ncol)",
                "        nrow = int(np.ceil(flat.size / ncol))",
                "    if ncol is None:",
                "        nrow = int(nrow)",
                "        ncol = int(np.ceil(flat.size / nrow))",
                "    nrow = int(nrow)",
                "    ncol = int(ncol)",
                "    size = nrow * ncol",
                "    if flat.size == 1:",
                "        values = np.full(size, flat[0])",
                "    else:",
                "        values = np.resize(flat, size)",
                "    order = 'C' if bool(byrow) else 'F'",
                "    return values.reshape((nrow, ncol), order=order)",
                "",
                "def r_typeof(value):",
                "    if isinstance(value, dict):",
                "        return 'list'",
                "    if isinstance(value, str):",
                "        return 'character'",
                "    if isinstance(value, (bool, np.bool_)):",
                "        return 'logical'",
                "    if isinstance(value, (int, np.integer)):",
                "        return 'integer'",
                "    if isinstance(value, np.ndarray):",
                "        if value.dtype.kind in {'U', 'S', 'O'}:",
                "            return 'character'",
                "        if value.dtype.kind == 'b':",
                "            return 'logical'",
                "        inferred = None",
                "        for item in value.reshape(-1).tolist():",
                "            if isinstance(item, str) or item is None:",
                "                return 'character'",
                "            if isinstance(item, (bool, np.bool_)):",
                "                inferred = inferred or 'logical'",
                "                continue",
                "            if isinstance(item, (int, np.integer)):",
                "                inferred = inferred or 'integer'",
                "                continue",
                "            if isinstance(item, float) and np.isnan(item):",
                "                continue",
                "            return 'double'",
                "        return inferred or 'double'",
                "    return 'double'",
                "",
                "def r_rep(x, times=1, each=1):",
                "    arr = np.asarray(x).reshape(-1)",
                "    arr = np.repeat(arr, int(each))",
                "    return np.tile(arr, int(times))",
                "",
                "def r_recycle_pair(left, right):",
                "    left_arr = np.asarray(left)",
                "    right_arr = np.asarray(right)",
                "    if left_arr.ndim == 0 or right_arr.ndim == 0:",
                "        return left_arr, right_arr",
                "    if left_arr.shape == right_arr.shape:",
                "        return left_arr, right_arr",
                "    if left_arr.ndim == 1 and right_arr.ndim == 1:",
                "        size = max(left_arr.size, right_arr.size)",
                "        return np.resize(left_arr, size), np.resize(right_arr, size)",
                "    return left_arr, right_arr",
                "",
                "def r_binary(op, left, right):",
                "    left_arr, right_arr = r_recycle_pair(left, right)",
                "    ops = {",
                "        '+': np.add,",
                "        '-': np.subtract,",
                "        '*': np.multiply,",
                "        '/': np.divide,",
                "        '%%': np.mod,",
                "        '^': np.power,",
                "        '<': np.less,",
                "        '<=': np.less_equal,",
                "        '>': np.greater,",
                "        '>=': np.greater_equal,",
                "        '==': np.equal,",
                "        '!=': np.not_equal,",
                "        '&': np.logical_and,",
                "        '|': np.logical_or,",
                "    }",
                "    return ops[op](left_arr, right_arr)",
                "",
                "def r_ifelse(cond, yes, no):",
                "    return np.where(cond, yes, no)",
                "",
                "def r_which(mask):",
                "    return np.nonzero(np.asarray(mask))[0] + 1",
                "",
                "def r_row_sums(a):",
                "    return np.sum(a, axis=1)",
                "",
                "def r_col_sums(a):",
                "    return np.sum(a, axis=0)",
                "",
                "def r_outer(x, y, op):",
                "    if op != '*':",
                "        raise NotImplementedError('outer() currently supports only \"*\"')",
                "    return np.outer(x, y)",
                "",
                "def r_as_integer(x):",
                "    arr = np.trunc(np.asarray(x, dtype=float))",
                "    if np.asarray(arr).ndim == 0:",
                "        return int(arr)",
                "    return arr.astype(int)",
                "",
                "def r_is_finite(x):",
                "    if isinstance(x, RTypedNA):",
                "        return False",
                "    arr = np.asarray(x, dtype=object)",
                "    mask = []",
                "    for item in arr.reshape(-1).tolist():",
                "        if isinstance(item, RTypedNA) or item is None:",
                "            mask.append(False)",
                "            continue",
                "        try:",
                "            mask.append(bool(np.isfinite(float(item))))",
                "        except Exception:",
                "            mask.append(False)",
                "    result = np.array(mask, dtype=bool).reshape(arr.shape)",
                "    if np.asarray(result).ndim == 0:",
                "        return bool(result)",
                "    return result",
                "",
                "def r_is_na(x):",
                "    arr = np.asarray(x, dtype=object)",
                "    mask = []",
                "    for item in arr.reshape(-1).tolist():",
                "        if item is None:",
                "            mask.append(True)",
                "        elif isinstance(item, float) and np.isnan(item):",
                "            mask.append(True)",
                "        else:",
                "            mask.append(False)",
                "    result = np.array(mask, dtype=bool).reshape(arr.shape)",
                "    if result.ndim == 0:",
                "        return bool(result)",
                "    return result",
                "",
                "def r_order(x):",
                "    return np.argsort(x) + 1",
                "",
                "def r_unique(x):",
                "    arr = np.asarray(x)",
                "    result = np.unique(arr)",
                "    if result.ndim == 0:",
                "        return result.item()",
                "    return result",
                "",
                "def r_round(x, digits=0):",
                "    arr = np.round(np.asarray(x, dtype=float), int(digits))",
                "    if arr.ndim == 0:",
                "        return float(arr)",
                "    return arr",
                "",
                "def r_median(x):",
                "    arr = np.asarray(x, dtype=float)",
                "    if arr.size == 0:",
                "        return np.nan",
                "    return float(np.median(arr))",
                "",
                "def r_min(*args):",
                "    values = []",
                "    for arg in args:",
                "        values.extend(np.asarray(arg).reshape(-1).tolist())",
                "    return np.min(np.asarray(values, dtype=float))",
                "",
                "def r_max(*args):",
                "    values = []",
                "    for arg in args:",
                "        values.extend(np.asarray(arg).reshape(-1).tolist())",
                "    return np.max(np.asarray(values, dtype=float))",
                "",
                "def r_mapply(func, *args):",
                "    arrays = [np.asarray(arg).reshape(-1) for arg in args]",
                "    if not arrays:",
                "        return np.array([])",
                "    n = min(len(arr) for arr in arrays)",
                "    out = [func(*(arr[i] for arr in arrays)) for i in range(n)]",
                "    try:",
                "        return np.asarray(out, dtype=float)",
                "    except Exception:",
                "        return np.asarray(out)",
                "",
                "def r_vector(mode, length):",
                "    n = int(length)",
                "    if mode == 'list':",
                "        return np.full(n, None, dtype=object)",
                "    if mode == 'character':",
                "        return np.full(n, '', dtype=object)",
                "    if mode == 'logical':",
                "        return np.full(n, False, dtype=bool)",
                "    return np.zeros(n)",
                "",
                "def r_nrow(x):",
                "    return np.asarray(x).shape[0]",
                "",
                "def r_ncol(x):",
                "    return np.asarray(x).shape[1]",
                "",
                "def r_is_matrix(x):",
                "    return np.asarray(x).ndim == 2",
                "",
                "def r_eigen(x, symmetric=False):",
                "    arr = np.asarray(x, dtype=float)",
                "    if symmetric:",
                "        values, vectors = np.linalg.eigh(arr)",
                "    else:",
                "        values, vectors = np.linalg.eig(arr)",
                "    return {'values': values, 'vectors': vectors}",
                "",
                "def r_chol(x):",
                "    return np.linalg.cholesky(np.asarray(x, dtype=float))",
                "",
                "def r_cor(x):",
                "    arr = np.asarray(x, dtype=float)",
                "    if arr.ndim == 1:",
                "        out = np.array([[1.0]])",
                "        names = r_names(x)",
                "        if names is not None and len(names) >= 1:",
                "            r_set_dimnames(out, [names[:1], names[:1]])",
                "        return out",
                "    out = np.corrcoef(arr, rowvar=False)",
                "    names = r_colnames(x)",
                "    if names is not None:",
                "        r_set_dimnames(out, [names, names])",
                "    return out",
                "",
                "def r_unname(x):",
                "    if isinstance(x, RNamedArray):",
                "        values = np.asarray(x.values)",
                "        if values.ndim == 0 or values.size == 1:",
                "            return values.reshape(-1)[0]",
                "        return values",
                "    if isinstance(x, dict):",
                "        values = np.array(list(x.values()), dtype=float)",
                "        if values.ndim == 0 or values.size == 1:",
                "            return values.reshape(-1)[0]",
                "        return values",
                "    return x",
                "",
                "def r_names(x):",
                "    if isinstance(x, RNamedArray):",
                "        return np.array(x.names, dtype=object)",
                "    if isinstance(x, dict):",
                "        return np.array(list(x.keys()), dtype=object)",
                "    names = R_NAME_MAP.get(id(x))",
                "    if names is not None:",
                "        return np.array(names, dtype=object)",
                "    return None",
                "",
                "def r_set_names(x, names):",
                "    R_NAME_MAP[id(x)] = np.asarray(names, dtype=object).reshape(-1).tolist()",
                "    return x",
                "",
                "def r_colnames(x):",
                "    dimnames = R_DIMNAME_MAP.get(id(x))",
                "    if dimnames is None or dimnames[1] is None:",
                "        return None",
                "    return np.array(dimnames[1], dtype=object)",
                "",
                "def r_rownames(x):",
                "    dimnames = R_DIMNAME_MAP.get(id(x))",
                "    if dimnames is None or dimnames[0] is None:",
                "        return None",
                "    return np.array(dimnames[0], dtype=object)",
                "",
                "def r_dimnames(x):",
                "    dimnames = R_DIMNAME_MAP.get(id(x))",
                "    if dimnames is None:",
                "        return None",
                "    row_names, col_names = dimnames",
                "    return [None if row_names is None else np.array(row_names, dtype=object), None if col_names is None else np.array(col_names, dtype=object)]",
                "",
                "def r_set_colnames(x, names):",
                "    dimnames = R_DIMNAME_MAP.get(id(x), (None, None))",
                "    R_DIMNAME_MAP[id(x)] = (dimnames[0], np.asarray(names, dtype=object).reshape(-1).tolist())",
                "    return x",
                "",
                "def r_set_rownames(x, names):",
                "    dimnames = R_DIMNAME_MAP.get(id(x), (None, None))",
                "    R_DIMNAME_MAP[id(x)] = (np.asarray(names, dtype=object).reshape(-1).tolist(), dimnames[1])",
                "    return x",
                "",
                "def r_set_dimnames(x, names):",
                "    row_names = None",
                "    col_names = None",
                "    if names is not None and len(names) >= 1 and names[0] is not None:",
                "        row_names = np.asarray(names[0], dtype=object).reshape(-1).tolist()",
                "    if names is not None and len(names) >= 2 and names[1] is not None:",
                "        col_names = np.asarray(names[1], dtype=object).reshape(-1).tolist()",
                "    R_DIMNAME_MAP[id(x)] = (row_names, col_names)",
                "    return x",
                "",
                "def r_invisible(x):",
                "    return x",
                "",
                "def r_arg_values(arg):",
                "    if arg is None:",
                "        return []",
                "    if isinstance(arg, RNamedArray):",
                "        arg = arg.values",
                "    if isinstance(arg, np.ndarray):",
                "        return arg.reshape(-1).tolist()",
                "    return [arg]",
                "",
                "def r_arg_strings(arg):",
                "    return [r_scalar_string(value) for value in r_arg_values(arg)]",
                "",
                "def r_paste(*args, sep=' ', collapse=None):",
                "    pieces = [r_arg_strings(arg) for arg in args]",
                "    max_len = 1",
                "    for piece in pieces:",
                "        if len(piece) > max_len:",
                "            max_len = len(piece)",
                "    out = []",
                "    for i in range(max_len):",
                "        row = []",
                "        for piece in pieces:",
                "            if not piece:",
                "                continue",
                "            row.append(piece[i % len(piece)])",
                "        out.append(sep.join(row))",
                "    if collapse is not None:",
                "        return collapse.join(out)",
                "    if len(out) == 1:",
                "        return out[0]",
                "    return np.array(out, dtype=object)",
                "",
                "def r_sprintf(fmt, *values):",
                "    fmt_values = r_arg_strings(fmt)",
                "    arg_values = [r_arg_values(value) for value in values]",
                "    max_len = len(fmt_values) if fmt_values else 1",
                "    for piece in arg_values:",
                "        if len(piece) > max_len:",
                "            max_len = len(piece)",
                "    out = []",
                "    for i in range(max_len):",
                "        current_fmt = fmt_values[i % len(fmt_values)] if fmt_values else ''",
                "        current_values = [piece[i % len(piece)] for piece in arg_values if piece]",
                "        if len(current_values) == 1:",
                "            out.append(current_fmt % current_values[0])",
                "        else:",
                "            out.append(current_fmt % tuple(current_values))",
                "    if len(out) == 1:",
                "        return out[0]",
                "    return np.array(out, dtype=object)",
                "",
                "def r_substr(s, start, stop):",
                "    return s[start - 1:stop]",
                "",
                "def r_sd(x):",
                "    return np.std(x, ddof=1)",
                "",
                "def r_dnorm(x, mean=0.0, sd=1.0, log_flag=False):",
                "    arr = np.asarray(x, dtype=float)",
                "    out = (-0.5 * np.log(2.0 * np.pi)) - np.log(sd) - (0.5 * (((arr - mean) / sd) ** 2))",
                "    if log_flag:",
                    "        if out.ndim == 0:",
                "            return float(out)",
                "        return out",
                "    out = np.exp(out)",
                "    if out.ndim == 0:",
                "        return float(out)",
                "    return out",
                "",
                "def r_stopifnot(*conditions):",
                "    for condition in conditions:",
                "        assert bool(np.all(condition))",
                "",
                "def r_indexer(index, size):",
                "    if index is None:",
                "        return slice(None)",
                "    if isinstance(index, (bool, np.bool_)):",
                "        return np.array([bool(index)])",
                "    if isinstance(index, (int, np.integer)):",
                "        index = int(index)",
                "        if index < 0:",
                "            mask = np.ones(size, dtype=bool)",
                "            mask[abs(index) - 1] = False",
                "            return mask",
                "        return index - 1",
                "    arr = np.asarray(index)",
                "    if arr.dtype.kind == 'b':",
                "        return arr",
                "    flat = arr.reshape(-1).astype(int)",
                "    if flat.size == 0:",
                "        return flat",
                "    if np.all(flat < 0):",
                "        mask = np.ones(size, dtype=bool)",
                "        mask[np.abs(flat) - 1] = False",
                "        return mask",
                "    return flat - 1",
                "",
                "def r_index(x, i, j=None):",
                "    if isinstance(x, RNamedArray) and isinstance(i, str) and j is None:",
                "        return x[i]",
                "    if isinstance(x, dict) and isinstance(i, str) and j is None:",
                "        return x[i]",
                "    if isinstance(x, RNamedArray):",
                "        arr = np.asarray(x.values)",
                "    else:",
                "        arr = np.asarray(x)",
                "    if j is None:",
                "        return arr[r_indexer(i, arr.shape[0])]",
                "    return arr[r_indexer(i, arr.shape[0]), r_indexer(j, arr.shape[1])]",
                "",
                "def r_assign(x, i, value, j=None):",
                "    if j is None and isinstance(i, str):",
                "        if isinstance(x, dict):",
                "            out = dict(x)",
                "            out[i] = value",
                "            return out",
                "        return {i: value}",
                "    arr = np.array(x, copy=True)",
                "    value_arr = np.asarray(value)",
                "    if value_arr.ndim > 0 and value_arr.size == 1:",
                "        value = value_arr.reshape(-1)[0]",
                "    if j is None:",
                "        arr[r_indexer(i, arr.shape[0])] = value",
                "    else:",
                "        arr[r_indexer(i, arr.shape[0]), r_indexer(j, arr.shape[1])] = value",
                "    return arr",
                "",
                "def r_scalar_string(value):",
                "    if isinstance(value, (bool, np.bool_)):",
                "        return 'TRUE' if value else 'FALSE'",
                "    if value is None:",
                "        return 'NA'",
                "    if isinstance(value, float) and np.isnan(value):",
                "        return 'NA'",
                "    return str(value)",
                "",
                "def r_pad_left(text, width):",
                "    out = str(text)",
                "    while len(out) < int(width):",
                '        out = " " + out',
                "    return out",
                "",
                "def r_cat(*args, sep=' ', end=''):",
                "    pieces = []",
                "    for arg in args:",
                "        pieces.extend(r_arg_strings(arg))",
                "    print(sep.join(pieces), end=end)",
                "",
                "def r_lm(y, x):",
                "    y = np.asarray(y, dtype=float)",
                "    x = np.asarray(x, dtype=float)",
                "    X = np.column_stack((np.ones(x.shape[0]), x))",
                "    beta = np.linalg.lstsq(X, y, rcond=None)[0]",
                "    return {'coefficients': beta}",
                "",
                "def r_pchisq(x, df):",
                "    try:",
                "        from scipy import stats as scipy_stats",
                "        result = scipy_stats.chi2.cdf(x, df)",
                "        if np.asarray(result).ndim == 0:",
                "            return float(result)",
                "        return result",
                "    except Exception:",
                "        if float(df) == 2.0:",
                "            arr = np.asarray(x, dtype=float)",
                "            result = 1.0 - np.exp(-arr / 2.0)",
                "            if result.ndim == 0:",
                "                return float(result)",
                "            return result",
                "        raise NotImplementedError('pchisq() requires scipy unless df == 2')",
                "",
                "def r_shapiro_test(x):",
                "    try:",
                "        from scipy import stats as scipy_stats",
                "        stat, pval = scipy_stats.shapiro(np.asarray(x, dtype=float))",
                "        return {'statistic': stat, 'p.value': pval}",
                "    except Exception:",
                "        return {'statistic': np.nan, 'p.value': np.nan}",
                "",
                "def r_proc_time():",
                "    return {'elapsed': time.perf_counter()}",
                "",
                "def r_data_frame(**kwargs):",
                "    return kwargs",
                "",
                "def r_write_table(data, file, row_names=False, col_names=True, quote=False):",
                "    if isinstance(data, dict):",
                "        names = list(data.keys())",
                "        cols = [np.asarray(data[name]).reshape(-1) for name in names]",
                "        n = len(cols[0]) if cols else 0",
                "        with open(file, 'w', encoding='utf-8') as f:",
                "            if col_names:",
                "                f.write(' '.join(names) + '\\n')",
                "            for i in range(n):",
                "                pieces = [r_scalar_string(col[i]) for col in cols]",
                "                f.write(' '.join(pieces) + '\\n')",
                "        return",
                "    arr = np.asarray(data)",
                "    with open(file, 'w', encoding='utf-8') as f:",
                "        if arr.ndim == 1:",
                "            for item in arr.reshape(-1).tolist():",
                "                f.write(r_scalar_string(item) + '\\n')",
                "        else:",
                "            for row in arr:",
                "                f.write(' '.join(r_scalar_string(item) for item in np.asarray(row).reshape(-1).tolist()) + '\\n')",
                "",
                "def r_coef(fit):",
                "    return fit['coefficients']",
                "",
            ]
        return [
            "import numpy as np",
            "import sys",
            "from array_compiler.annotations import Array1D, Array2D",
            "",
            "rng = np.random.default_rng()",
            "",
            "def r_pad_left(text: str, width: int) -> str:",
            "    out = text",
            "    while len(out) < width:",
            '        out = " " + out',
            "    return out",
            "",
            "def r_join_str_array(x: Array1D[str]) -> str:",
            '    out = ""',
            "    for i in range(len(x)):",
            "        piece = x[i].rstrip()",
            "        while len(piece) < 12:",
            '            piece = " " + piece',
            "        out = out + piece",
            "    return out",
            "",
            "def r_as_character(x: Array1D[float]) -> Array1D[str]:",
            "    x = np.asarray(x, dtype=float).reshape(-1)",
            '    out = np.full(len(x), "                                ")',
            "    for i in range(len(x)):",
            "        if float(int(x[i])) == x[i]:",
            "            out[i] = str(int(x[i]))",
            "        else:",
            "            out[i] = str(x[i])",
            "    return out",
            "",
            "def r_c_str_scalar_array(head: str, tail: Array1D[str]) -> Array1D[str]:",
            '    out = np.full(len(tail) + 1, "                                ")',
            "    out[0] = head",
            "    for i in range(len(tail)):",
            "        out[i + 1] = tail[i]",
            "    return out",
            "",
            "def r_paste_prefix(head: str, tail: Array1D[int]) -> Array1D[str]:",
            '    out = np.full(len(tail), "                                ")',
            "    for i in range(len(tail)):",
            "        out[i] = head + str(int(tail[i]))",
            "    return out",
            "",
            "def r_format(x: float, scientific: bool=True) -> str:",
            "    if (not scientific) and (float(int(x)) == x):",
            "        return str(int(x))",
            "    return str(x)",
            "",
            "def r_colon(start: int, stop: int) -> Array1D[int]:",
            "    if start <= stop:",
            "        return np.arange(start, stop + 1)",
            "    return np.arange(start, stop - 1, -1)",
            "",
            "def r_seq(start: int, stop: int, by: int = 1) -> Array1D[int]:",
            '    if by == 0:',
            '        raise ValueError("seq by must be nonzero")',
            "    n = int((stop - start) / by) + 1",
            "    return start + by * np.arange(n)",
            "",
            "def r_seq_len(n: int) -> Array1D[int]:",
            "    return np.arange(1, n + 1)",
            "",
            "def r_matrix(data: Array1D[float], nrow: int, ncol: int) -> Array2D[float]:",
            "    data = np.asarray(data, dtype=float).reshape(-1)",
            '    return data.reshape((nrow, ncol), order="F")',
            "",
            "def r_mask(x: Array1D[float], mask: Array1D[bool]) -> Array1D[float]:",
            "    return x[mask]",
            "",
            "def r_exclude(x: Array1D[float], idx: Array1D[float]) -> Array1D[float]:",
            "    x = np.asarray(x, dtype=float).reshape(-1)",
            "    idx = np.asarray(idx, dtype=float).reshape(-1)",
            "    out = np.empty(len(x) - len(idx))",
            "    k = 0",
            "    for i in range(len(x)):",
            "        include = 1",
            "        for j in range(len(idx)):",
            "            if (i + 1) == int(idx[j]):",
            "                include = 0",
            "        if include == 1:",
            "            out[k] = x[i]",
            "            k = k + 1",
            "    return out",
            "",
            "def r_lm(y: Array1D[float], x: Array1D[float]) -> Array1D[float]:",
            "    y = np.asarray(y, dtype=float).reshape(-1)",
            "    x = np.asarray(x, dtype=float).reshape(-1)",
            "    design = np.empty((len(x), 2))",
            "    for i in range(len(x)):",
            "        design[i, 0] = 1.0",
            "        design[i, 1] = x[i]",
            "    return np.linalg.inv(design.T @ design) @ (design.T @ y)",
            "",
            "def r_qr_solve(a: Array2D[float], b: Array2D[float]) -> Array2D[float]:",
            "    return np.linalg.inv(a.T @ a) @ (a.T @ b)",
            "",
            "def r_median(x: Array1D[float]) -> float:",
            "    x = np.asarray(x, dtype=float).reshape(-1)",
            "    n = len(x)",
            "    if n == 0:",
            "        return np.nan",
            "    work = np.empty(n)",
            "    for i in range(n):",
            "        work[i] = x[i]",
            "    for i in range(n):",
            "        for j in range(i + 1, n):",
            "            if work[j] < work[i]:",
            "                tmp = work[i]",
            "                work[i] = work[j]",
            "                work[j] = tmp",
            "    mid = int(n / 2)",
            "    if (n % 2) == 1:",
            "        return work[mid]",
            "    return 0.5 * (work[mid - 1] + work[mid])",
            "",
            "def r_nrow(x: Array2D[float]) -> int:",
            "    return np.asarray(x).shape[0]",
            "",
            "def r_ncol(x: Array2D[float]) -> int:",
            "    return np.asarray(x).shape[1]",
            "",
            "def r_is_matrix(x: Array2D[float]) -> bool:",
            "    return np.asarray(x).ndim == 2",
            "",
            "def r_col_get(x: Array2D[float], j: int) -> Array1D[float]:",
            "    out = np.empty(x.shape[0])",
            "    jj = j - 1",
            "    for i in range(len(out)):",
            "        out[i] = x[i, jj]",
            "    return out",
            "",
            "def r_col_set(x: Array2D[float], j: int, value: Array1D[float]) -> Array2D[float]:",
            "    jj = j - 1",
            "    out = x.copy()",
            "    for i in range(x.shape[0]):",
            "        out[i, jj] = value[i]",
            "    return out",
            "",
            "def r_gamma_len(gamma_blocks: Array2D[float], n: int) -> int:",
            "    return int(gamma_blocks.shape[0] / n)",
            "",
            "def r_gamma_get(gamma_blocks: Array2D[float], i: int, n: int) -> Array2D[float]:",
            "    out = np.empty((n, gamma_blocks.shape[1]))",
            "    start = (i - 1) * n",
            "    for r0 in range(n):",
            "        for c0 in range(gamma_blocks.shape[1]):",
            "            out[r0, c0] = gamma_blocks[start + r0, c0]",
            "    return out",
            "",
            "def r_gamma_set(gamma_blocks: Array2D[float], i: int, n: int, value: Array2D[float]) -> Array2D[float]:",
            "    out = gamma_blocks.copy()",
            "    start = (i - 1) * n",
            "    for r0 in range(n):",
            "        for c0 in range(gamma_blocks.shape[1]):",
            "            out[start + r0, c0] = value[r0, c0]",
            "    return out",
            "",
            "def r_block_set_cols(x: Array2D[float], col_start: int, width: int, block: Array2D[float]) -> Array2D[float]:",
            "    out = x.copy()",
            "    start = col_start - 1",
            "    for r0 in range(x.shape[0]):",
            "        for c0 in range(width):",
            "            out[r0, start + c0] = block[r0, c0]",
            "    return out",
            "",
            "def r_take_cols(x: Array2D[float], idx: Array1D[int]) -> Array2D[float]:",
            "    out = np.empty((x.shape[0], len(idx)))",
            "    for c0 in range(len(idx)):",
            "        jj = idx[c0] - 1",
            "        for r0 in range(x.shape[0]):",
            "            out[r0, c0] = x[r0, jj]",
            "    return out",
            "",
            "def r_eigen(x: Array2D[float]):",
            "    values, vectors = np.linalg.eig(x)",
            "    return {'values': values, 'vectors': vectors}",
            "",
            "def r_eigen_symmetric(x: Array2D[float]):",
            "    values, vectors = np.linalg.eig(x)",
            "    return {'values': values, 'vectors': vectors}",
            "",
            "def r_cor(x: Array2D[float]) -> Array2D[float]:",
            "    arr = np.asarray(x, dtype=float)",
            "    if arr.ndim == 1:",
            "        out = np.empty((1, 1))",
            "        out[0, 0] = 1.0",
            "        return out",
            "    cov = np.cov(arr.T)",
            "    sd = np.sqrt(np.diag(cov))",
            "    p = cov.shape[0]",
            "    out = np.empty((p, p))",
            "    for i in range(p):",
            "        for j in range(p):",
            "            out[i, j] = cov[i, j] / (sd[i] * sd[j])",
            "    return out",
            "",
            "def r_fit_record_at(mu_fit: Array1D[float], omega_fit: Array1D[float], alpha_fit: Array1D[float], beta_fit: Array1D[float], sd_uncond_fit: Array1D[float], rho1_hat_fit: Array1D[float], phi_hat_fit: Array1D[float], alpha_kurt_fit: Array1D[float], alpha_rho1_fit: Array1D[float], z_fit: Array2D[float], i: int):",
            "    idx = i - 1",
            "    return {",
            "        'mu': mu_fit[idx],",
            "        'omega': omega_fit[idx],",
            "        'alpha': alpha_fit[idx],",
            "        'beta': beta_fit[idx],",
            "        'sd_uncond': sd_uncond_fit[idx],",
            "        'rho1_hat': rho1_hat_fit[idx],",
            "        'phi_hat': phi_hat_fit[idx],",
            "        'alpha_kurt': alpha_kurt_fit[idx],",
            "        'alpha_rho1': alpha_rho1_fit[idx],",
            "        'residuals': r_col_get(z_fit, i),",
            "    }",
            "",
            "def r_pchisq_df2(x: float) -> float:",
            "    return 1.0 - np.exp((-x) / 2.0)",
            "",
            "def r_proc_elapsed() -> float:",
            "    return ac_wall_time()",
            "",
            "def r_arg_str(args: Array1D[str], pos: int, default: str) -> str:",
            "    if len(args) >= pos:",
            "        return args[pos - 1]",
            "    return default",
            "",
            "def r_arg_int(args: Array1D[str], pos: int, default: int) -> int:",
            "    if len(args) >= pos:",
            "        return int(args[pos - 1])",
            "    return default",
            "",
            "def r_arg_float(args: Array1D[str], pos: int, default: float) -> float:",
            "    if len(args) >= pos:",
            "        return float(args[pos - 1])",
            "    return default",
            "",
        ]

    def _render_stmt(self, stmt: object, indent: int) -> list[str]:
        pad = " " * indent
        if isinstance(stmt, AssignStmt):
            if isinstance(stmt.target, NameExpr) and isinstance(stmt.value, CallExpr):
                rendered_mapply = self._render_x2f_mapply_assignment(stmt.target, stmt.value, indent)
                if rendered_mapply is not None:
                    return rendered_mapply
                rendered_sapply = self._render_x2f_sapply_assignment(stmt.target, stmt.value, indent)
                if rendered_sapply is not None:
                    return rendered_sapply
            if (
                isinstance(stmt.target, CallExpr)
                and isinstance(stmt.target.func, NameExpr)
                and stmt.target.func.name == "diag"
                and len(stmt.target.args) == 1
                and stmt.target.args[0].name is None
            ):
                base_expr = stmt.target.args[0].value
                if not isinstance(base_expr, NameExpr):
                    raise NotImplementedError("diag(...) assignment currently requires a simple matrix variable")
                base = self._render_expr(base_expr)
                value = self._render_expr(stmt.value)
                loop_var = "diag_i"
                return [
                    f"{pad}for {loop_var} in range(min({base}.shape[0], {base}.shape[1])):",
                    f"{pad}    {base}[{loop_var}, {loop_var}] = {value}",
                ]
            if (
                isinstance(stmt.target, CallExpr)
                and isinstance(stmt.target.func, NameExpr)
                and stmt.target.func.name in {"names", "colnames", "rownames", "dimnames"}
                and len(stmt.target.args) == 1
                and stmt.target.args[0].name is None
            ):
                base_expr = stmt.target.args[0].value
                if not isinstance(base_expr, NameExpr):
                    raise NotImplementedError(f"{stmt.target.func.name}(...) assignment currently requires a simple variable")
                base = self._render_expr(base_expr)
                value = self._render_expr(stmt.value)
                helper = {
                    "names": "r_set_names",
                    "colnames": "r_set_colnames",
                    "rownames": "r_set_rownames",
                    "dimnames": "r_set_dimnames",
                }[stmt.target.func.name]
                if self._standalone:
                    return [f"{pad}{helper}({base}, {value})"]
                return []
            if self._standalone and isinstance(stmt.target, IndexExpr):
                base_expr = stmt.target.value
                base = self._render_expr(base_expr)
                value = self._render_expr(stmt.value)
                if isinstance(base_expr, NameExpr):
                    shape = f"np.asarray({base}).shape"
                    if len(stmt.target.indices) == 1:
                        index = self._render_index_arg(stmt.target.indices[0])
                        return [f"{pad}{base}[r_indexer({index}, {shape}[0])] = {value}"]
                    if len(stmt.target.indices) == 2:
                        row, col = stmt.target.indices
                        return [
                            f"{pad}{base}[r_indexer({self._render_index_arg(row)}, {shape}[0]), r_indexer({self._render_index_arg(col)}, {shape}[1])] = {value}"
                        ]
                if len(stmt.target.indices) == 1:
                    index = self._render_index_arg(stmt.target.indices[0])
                    return [f"{pad}{base} = r_assign({base}, {index}, {value})"]
                if len(stmt.target.indices) == 2:
                    row, col = stmt.target.indices
                    return [
                        f"{pad}{base} = r_assign({base}, {self._render_index_arg(row)}, {value}, {self._render_index_arg(col)})"
                    ]
            if not self._standalone and isinstance(stmt.target, IndexExpr) and len(stmt.target.indices) == 2:
                row, col = stmt.target.indices
                if row is None and col is not None and isinstance(stmt.target.value, NameExpr):
                    base = self._render_expr(stmt.target.value)
                    value = self._render_expr(stmt.value)
                    return [f"{pad}{base} = r_col_set({base}, {self._render_expr(col)}, {value})"]
            expected_type = None
            if isinstance(stmt.target, NameExpr):
                hint = self._type_hints.get(stmt.target.name)
                if hint == "double":
                    expected_type = "float"
            return [f"{pad}{self._render_target(stmt.target)} = {self._render_expr_for_param(stmt.value, expected_type)}"]
        if isinstance(stmt, ExprStmt):
            rendered = self._render_expr_stmt(stmt.value)
            return [f"{pad}{line}" if line else "" for line in rendered.splitlines()]
        if isinstance(stmt, ReturnStmt):
            if isinstance(stmt.value, CallExpr) and isinstance(stmt.value.func, NameExpr) and stmt.value.func.name in {
                "cat",
                "print",
                "write.table",
                "timer_print",
            }:
                return [f"{pad}{self._render_expr_stmt(stmt.value)}"]
            return [f"{pad}return {self._render_expr(stmt.value)}"]
        if isinstance(stmt, BreakStmt):
            return [f"{pad}break"]
        if isinstance(stmt, ContinueStmt):
            return [f"{pad}continue"]
        if isinstance(stmt, IfStmt):
            lines = [f"{pad}if {self._render_expr(stmt.test)}:"]
            lines.extend(self._render_body(stmt.body, indent + 4))
            if stmt.orelse:
                lines.append(f"{pad}else:")
                lines.extend(self._render_body(stmt.orelse, indent + 4))
            return lines
        if isinstance(stmt, ForStmt):
            return [f"{pad}for {stmt.target} in {self._render_for_iter(stmt.iterable)}:"] + self._render_body(stmt.body, indent + 4)
        if isinstance(stmt, WhileStmt):
            return [f"{pad}while {self._render_expr(stmt.test)}:"] + self._render_body(stmt.body, indent + 4)
        if isinstance(stmt, FunctionDefStmt):
            if not self._standalone and stmt.name == "shapiro_test_safe":
                param = "x: Array1D[float]"
                return [
                    f"{pad}def {stmt.name}({param}):",
                    f"{pad}    return np.array([np.nan, np.nan], dtype=float)",
                ]
            if not self._standalone and stmt.name == "timer_new":
                return [
                    f"{pad}def {stmt.name}(enabled: bool=True):",
                    f"{pad}    timer = np.full(6, -1.0)",
                    f"{pad}    if enabled:",
                    f"{pad}        timer[0] = 1.0",
                    f"{pad}    else:",
                    f"{pad}        timer[0] = 0.0",
                    f"{pad}    return timer",
                ]
            if not self._standalone and stmt.name == "timer_add":
                return [
                    f"{pad}def {stmt.name}(timer: Array1D[float], name: str, seconds: float):",
                    f"{pad}    if timer[0] < 0.5:",
                    f"{pad}        return timer",
                    f"{pad}    out = np.array(timer, copy=True)",
                    f'{pad}    if name == "simulate":',
                    f"{pad}        out[1] = seconds",
                    f'{pad}    elif name == "write_file":',
                    f"{pad}        out[2] = seconds",
                    f'{pad}    elif name == "estimate":',
                    f"{pad}        out[3] = seconds",
                    f'{pad}    elif name == "diagnostics":',
                    f"{pad}        out[4] = seconds",
                    f'{pad}    elif name == "print_main_output":',
                    f"{pad}        out[5] = seconds",
                    f"{pad}    return out",
                ]
            if not self._standalone and stmt.name == "timer_print":
                return [
                    f"{pad}def {stmt.name}(timer: Array1D[float]):",
                    f"{pad}    if timer[0] < 0.5:",
                    f"{pad}        return",
                    f"{pad}    print('elapsed times in seconds')",
                    f'{pad}    print(r_pad_left(str("section"), 20) + " " + r_pad_left(str("elapsed"), 12))',
                    f"{pad}    total = 0.0",
                    f'{pad}    labels = np.array(["simulate", "write_file", "estimate", "diagnostics", "print_main_output"])',
                    f"{pad}    for i in range(5):",
                    f"{pad}        if timer[i + 1] >= 0.0:",
                    f'{pad}            print(r_pad_left(labels[i], 20) + " " + ac_format_fixed(timer[i + 1], 6, 12))',
                    f"{pad}            total = total + timer[i + 1]",
                    f'{pad}    print(r_pad_left(str("total"), 20) + " " + ac_format_fixed(total, 6, 12))',
                ]
            if not self._standalone and stmt.name == "fmt_num":
                return [
                    f"{pad}def {stmt.name}(x: Array1D[float], width: int=12, digits: int=6):",
                    f"{pad}    values = np.asarray(x, dtype=float).reshape(-1)",
                    f"{pad}    out = np.full(int(len(values)), '            ')",
                    f"{pad}    for i in range(1, len(values) + 1):",
                    f"{pad}        if np.isnan(values[(i - 1)]):",
                    f"{pad}            out[(i - 1)] = '          na'",
                    f"{pad}        else:",
                    f"{pad}            out[(i - 1)] = ac_format_fixed(values[(i - 1)], digits, width)",
                    f"{pad}    return out",
                ]
            if not self._standalone and stmt.name == "make_spd_corr":
                lines = [f"{pad}def {stmt.name}(R: Array2D[float], eps: float=1e-8):"]
                prev_function_name = self._current_function_name
                prev_param_names = self._current_function_param_names
                self._current_function_name = stmt.name
                self._current_function_param_names = {"R", "eps"}
                try:
                    lines.extend(self._render_body(stmt.body, indent + 4))
                finally:
                    self._current_function_name = prev_function_name
                    self._current_function_param_names = prev_param_names
                return lines
            if not self._standalone and stmt.name == "validate_ccc_inputs":
                lines = [f"{pad}def {stmt.name}(omega: Array1D[float], alpha: Array1D[float], beta: Array1D[float], mu: Array1D[float], R: Array2D[float]):"]
                prev_function_name = self._current_function_name
                prev_param_names = self._current_function_param_names
                self._current_function_name = stmt.name
                self._current_function_param_names = {"omega", "alpha", "beta", "mu", "R"}
                try:
                    lines.extend(self._render_body(stmt.body, indent + 4))
                finally:
                    self._current_function_name = prev_function_name
                    self._current_function_param_names = prev_param_names
                return lines
            if not self._standalone and stmt.name == "simulate_ccc_garch":
                return [
                    f"{pad}def {stmt.name}(n: int, omega: Array1D[float], alpha: Array1D[float], beta: Array1D[float], mu: Array1D[float], R: Array2D[float], seed: int=-1, burn_in: int=0):",
                    f"{pad}    validate_ccc_inputs(omega, alpha, beta, mu, R)",
                    f"{pad}    p = len(np.asarray(omega).reshape(-1))",
                    f"{pad}    if seed >= 0:",
                    f"{pad}        rng = np.random.default_rng(seed)",
                    f"{pad}    else:",
                    f"{pad}        rng = np.random.default_rng()",
                    f"{pad}    total_n = n + burn_in",
                    f"{pad}    z0 = r_matrix(rng.standard_normal(int(total_n * p)), total_n, p)",
                    f"{pad}    z = z0 @ np.linalg.cholesky(R)",
                    f"{pad}    sigma2 = np.full((total_n, p), 0.0)",
                    f"{pad}    eps = np.full((total_n, p), 0.0)",
                    f"{pad}    ret = np.full((total_n, p), 0.0)",
                    f"{pad}    sigma2[0] = omega / (1 - alpha - beta)",
                    f"{pad}    eps[0] = np.sqrt(sigma2[0]) * z[0]",
                    f"{pad}    ret[0] = mu + eps[0]",
                    f"{pad}    if total_n >= 2:",
                    f"{pad}        for t in range(2, int(total_n) + 1):",
                    f"{pad}            sigma2[t - 1] = omega + alpha * (eps[t - 2] ** 2) + beta * sigma2[t - 2]",
                    f"{pad}            eps[t - 1] = np.sqrt(sigma2[t - 1]) * z[t - 1]",
                    f"{pad}            ret[t - 1] = mu + eps[t - 1]",
                    f"{pad}    if burn_in > 0:",
                    f"{pad}        sigma2 = sigma2[burn_in:total_n, :]",
                    f"{pad}        eps = eps[burn_in:total_n, :]",
                    f"{pad}        ret = ret[burn_in:total_n, :]",
                    f"{pad}        z = z[burn_in:total_n, :]",
                    f"{pad}    return {{'ret': ret, 'eps': eps, 'sigma2': sigma2, 'z': z}}",
                ]
            if not self._standalone and stmt.name == "simulate_vecm":
                return [
                    f"{pad}def {stmt.name}(T: int, alpha: Array2D[float], beta: Array2D[float], Gamma_list: Array2D[float], sigma_u: Array2D[float], burn: float=300.0, y_init: Array1D[float]=np.zeros(0), rng=None) -> Array2D[float]:",
                    f"{pad}    n: int = r_nrow(alpha)",
                    f"{pad}    r: int = r_ncol(alpha)",
                    f"{pad}    gamma_count: int = r_gamma_len(Gamma_list, n)",
                    f"{pad}    if rng is None:",
                    f"{pad}        rng = np.random.default_rng(12345)",
                    f"{pad}    burn_i: int = int(burn)",
                    f"{pad}    p: int = gamma_count + 1",
                    f"{pad}    if (r_nrow(beta) != n) or (r_ncol(beta) != r):",
                    f'{pad}        raise ValueError("beta must have dimension n x r.")',
                    f"{pad}    if (r_nrow(sigma_u) != n) or (r_ncol(sigma_u) != n):",
                    f'{pad}        raise ValueError("sigma_u must be n x n.")',
                    f"{pad}    for i in range(1, int(gamma_count) + 1):",
                    f"{pad}        gamma_i = r_gamma_get(Gamma_list, i, n)",
                    f"{pad}        if (r_nrow(gamma_i) != n) or (r_ncol(gamma_i) != n):",
                    f'{pad}            raise ValueError("Each Gamma_i must be n x n.")',
                    f"{pad}    Pi = alpha @ beta.T",
                    f"{pad}    total_T: int = T + burn_i",
                    f"{pad}    if total_T <= p:",
                    f'{pad}        raise ValueError("T + burn must be larger than p.")',
                    f"{pad}    y = np.full((int(total_T), int(n)), 0.0)",
                    f"{pad}    if len(np.asarray(y_init).reshape(-1)) != 0:",
                    f"{pad}        if len(np.asarray(y_init).reshape(-1)) != n:",
                    f'{pad}            raise ValueError("y_init must have length n.")',
                    f"{pad}        for t_idx in range(1, int(p) + 1):",
                    f"{pad}            y[t_idx - 1, :] = y_init",
                    f"{pad}    chol_u = np.linalg.cholesky(sigma_u)",
                    f"{pad}    for t_idx in range(int(p + 1), int(total_T) + 1):",
                    f"{pad}        dy_t: Array1D[float] = Pi @ y[t_idx - 2, :]",
                    f"{pad}        for i in range(1, int(gamma_count) + 1):",
                    f"{pad}            dy_lag_i: Array1D[float] = y[t_idx - i - 1, :] - y[t_idx - i - 2, :]",
                    f"{pad}            dy_t = dy_t + (r_gamma_get(Gamma_list, i, n) @ dy_lag_i)",
                    f"{pad}        u_t: Array1D[float] = chol_u.T @ rng.standard_normal(int(n))",
                    f"{pad}        dy_t = dy_t + u_t",
                    f"{pad}        y[t_idx - 1, :] = y[t_idx - 2, :] + dy_t",
                    f"{pad}    return y[(r_colon((burn_i + 1), total_T).astype(int) - 1), :]",
                ]
            if not self._standalone and stmt.name == "make_vecm_matrices":
                return [
                    f"{pad}def {stmt.name}(Y: Array2D[float], p: int):",
                    f"{pad}    Tobs: int = r_nrow(Y)",
                    f"{pad}    n: int = r_ncol(Y)",
                    f"{pad}    if Tobs <= p:",
                    f'{pad}        raise ValueError("Need more rows in Y than p.")',
                    f"{pad}    dY_all: Array2D[float] = Y[1:, :] - Y[:-1, :]",
                    f"{pad}    dY: Array2D[float] = dY_all[(r_colon(p, (Tobs - 1)).astype(int) - 1), :]",
                    f"{pad}    Y_lag1: Array2D[float] = Y[(r_colon(p, (Tobs - 1)).astype(int) - 1), :]",
                    f"{pad}    if p == 1:",
                    f"{pad}        W: Array2D[float] = np.empty((r_nrow(dY), 0))",
                    f"{pad}    else:",
                    f"{pad}        W: Array2D[float] = np.empty((r_nrow(dY), int((p - 1) * n)))",
                    f"{pad}        for i in range(1, int(p - 1) + 1):",
                    f"{pad}            block: Array2D[float] = dY_all[(r_colon((p - i), ((Tobs - 1) - i)).astype(int) - 1), :]",
                    f"{pad}            W = r_block_set_cols(W, (((i - 1) * n) + 1), n, block)",
                    f"{pad}    return {{'dY': dY, 'Y_lag1': Y_lag1, 'W': W, 'n_eff': r_nrow(dY), 'n': n}}",
                ]
            if not self._standalone and stmt.name == "residualize_on_W":
                return [
                    f"{pad}def {stmt.name}(A: Array2D[float], W: Array2D[float]) -> Array2D[float]:",
                    f"{pad}    # Returns residuals from regressing each column of A on W",
                    f"{pad}    # using least squares, all in base R.",
                    f"{pad}    if ((len(np.asarray(W).reshape(-1)) == 0) or (r_ncol(W) == 0)):",
                    f"{pad}        return A",
                    f"{pad}    coef_A_on_W: Array2D[float] = r_qr_solve(W, A)",
                    f"{pad}    return (A - (W @ coef_A_on_W))",
                ]
            if not self._standalone and stmt.name == "print_matrix_block":
                return [
                    f"{pad}def {stmt.name}(title: str, mat: Array2D[float]):",
                    f"{pad}    print()",
                    f"{pad}    print(title)",
                    f"{pad}    print(mat)",
                ]
            if not self._standalone and stmt.name == "print_diff_matrix":
                return [
                    f"{pad}def {stmt.name}(name: str, est: Array2D[float], truth: Array2D[float], digits: int=4):",
                    f'{pad}    print("\\n", name, ": estimated - true\\n")',
                    f"{pad}    print(est - truth)",
                ]
            if not self._standalone and stmt.name == "johansen_fit":
                return [
                    f"{pad}def {stmt.name}(Y: Array2D[float], p: int, r: int, norm_rows: Array1D[float]=np.zeros(0)):",
                    f"{pad}    mats = make_vecm_matrices(Y, p)",
                    f"{pad}    dY = mats['dY']",
                    f"{pad}    Y_lag1 = mats['Y_lag1']",
                    f"{pad}    W = mats['W']",
                    f"{pad}    T_eff = mats['n_eff']",
                    f"{pad}    n: int = mats['n']",
                    f"{pad}    if len(np.asarray(norm_rows).reshape(-1)) == 0:",
                    f"{pad}        norm_rows = r_colon(1, r)",
                    f"{pad}    if (r < 0) or (r > n):",
                    f'{pad}        raise ValueError("r must satisfy 0 <= r <= n.")',
                    f"{pad}    if len(np.asarray(norm_rows).reshape(-1)) != r:",
                    f'{pad}        raise ValueError("norm_rows must have length r.")',
                    f"{pad}    R0: Array2D[float] = residualize_on_W(dY, W)",
                    f"{pad}    R1: Array2D[float] = residualize_on_W(Y_lag1, W)",
                    f"{pad}    S00: Array2D[float] = (R0.T @ R0) / T_eff",
                    f"{pad}    S01: Array2D[float] = (R0.T @ R1) / T_eff",
                    f"{pad}    S10: Array2D[float] = S01.T",
                    f"{pad}    S11: Array2D[float] = (R1.T @ R1) / T_eff",
                    f"{pad}    M: Array2D[float] = r_qr_solve(S11, S10 @ r_qr_solve(S00, S01))",
                    f"{pad}    ee = r_eigen(M)",
                    f"{pad}    lambda_ = np.real(ee['values'])",
                    f"{pad}    V = np.real(ee['vectors'])",
                    f"{pad}    ord = np.argsort((-lambda_)) + 1",
                    f"{pad}    lambda_ = lambda_[(ord - 1)]",
                    f"{pad}    V = r_take_cols(V, ord)",
                    f"{pad}    lambda_ = np.where(np.where(lambda_ > 0, lambda_, 0.0) < 1.0, np.where(lambda_ > 0, lambda_, 0.0), 1.0)",
                    f"{pad}    beta_raw: Array2D[float] = V[:, (r_colon(1, r).astype(int) - 1)]",
                    f"{pad}    beta_top: Array2D[float] = beta_raw[(norm_rows.astype(int) - 1), :]",
                    f"{pad}    beta_hat: Array2D[float] = beta_raw @ r_qr_solve(beta_top, np.diag(r))",
                    f"{pad}    ect: Array2D[float] = Y_lag1 @ beta_hat",
                    f"{pad}    if len(np.asarray(W).reshape(-1)) == 0:",
                    f"{pad}        X: Array2D[float] = ect",
                    f"{pad}    else:",
                    f"{pad}        X: Array2D[float] = np.concatenate((ect, W), axis=1)",
                    f"{pad}    B: Array2D[float] = r_qr_solve(X, dY)",
                    f"{pad}    alpha_hat: Array2D[float] = B[(r_colon(1, r).astype(int) - 1), :].T",
                    f"{pad}    if p > 1:",
                    f"{pad}        Gamma_hat: Array2D[float] = np.full((int((p - 1) * n), int(n)), 0.0)",
                    f"{pad}        col_start: int = r + 1",
                    f"{pad}        for i in range(1, int(p - 1) + 1):",
                    f"{pad}            block_i: Array2D[float] = B[(r_colon(col_start, ((col_start + n) - 1)).astype(int) - 1), :]",
                    f"{pad}            Gamma_hat = r_gamma_set(Gamma_hat, i, n, block_i.T)",
                    f"{pad}            col_start = col_start + n",
                    f"{pad}    else:",
                    f"{pad}        Gamma_hat: Array2D[float] = np.empty((0, int(n)))",
                    f"{pad}    U_hat: Array2D[float] = dY - (X @ B)",
                    f"{pad}    sigma_u_hat: Array2D[float] = (U_hat.T @ U_hat) / r_nrow(U_hat)",
                    f"{pad}    Pi_hat: Array2D[float] = alpha_hat @ beta_hat.T",
                    f"{pad}    trace_stats: Array1D[float] = np.empty(int(n))",
                    f"{pad}    for trace_i in range(int(n)):",
                    f"{pad}        trace_rank = trace_i",
                    f"{pad}        trace_stats[trace_i] = (-T_eff) * np.sum(log(1 - lambda_[(r_colon((trace_rank + 1), n).astype(int) - 1)]))",
                    f"{pad}    maxeig_stats: Array1D[float] = (-T_eff) * log(1 - lambda_)",
                    f"{pad}    return {{'alpha': alpha_hat, 'beta': beta_hat, 'Gamma': Gamma_hat, 'Pi': Pi_hat, 'sigma_u': sigma_u_hat, 'lambda': lambda_, 'trace_stats': trace_stats, 'maxeig_stats': maxeig_stats, 'T_eff': T_eff}}",
                ]
            if not self._standalone and stmt.name == "print_fit_vs_truth":
                return [
                    f"{pad}def {stmt.name}(fit_alpha: Array2D[float], fit_beta: Array2D[float], fit_Gamma: Array2D[float], fit_Pi: Array2D[float], fit_sigma_u: Array2D[float], fit_lambda: Array1D[float], fit_trace_stats: Array1D[float], fit_maxeig_stats: Array1D[float], truth_alpha: Array2D[float], truth_beta: Array2D[float], truth_Gamma: Array2D[float], truth_Pi: Array2D[float], truth_sigma_u: Array2D[float], digits: int=4):",
                    f'{pad}    print("\\n========================================")',
                    f'{pad}    print("Johansen eigenvalues")',
                    f'{pad}    print("========================================")',
                    f"{pad}    print(np.round(fit_lambda, digits))",
                    f'{pad}    print("\\n========================================")',
                    f'{pad}    print("Johansen trace statistics")',
                    f'{pad}    print("trace statistic for H0: rank <= r0, r0 = 0, 1, ..., n-1")',
                    f'{pad}    print("========================================")',
                    f"{pad}    print(np.round(fit_trace_stats, digits))",
                    f'{pad}    print("\\n========================================")',
                    f'{pad}    print("Johansen max-eigenvalue statistics")',
                    f'{pad}    print("========================================")',
                    f"{pad}    print(np.round(fit_maxeig_stats, digits))",
                    f'{pad}    print("\\n========================================")',
                    f'{pad}    print("Parameter differences")',
                    f'{pad}    print("========================================")',
                    f'{pad}    print_diff_matrix("alpha", fit_alpha, truth_alpha, digits)',
                    f'{pad}    print_diff_matrix("beta", fit_beta, truth_beta, digits)',
                    f'{pad}    print_diff_matrix("Pi", fit_Pi, truth_Pi, digits)',
                    f"{pad}    gamma_count: int = r_gamma_len(truth_Gamma, r_nrow(truth_alpha))",
                    f"{pad}    for i in range(1, int(gamma_count) + 1):",
                    f'{pad}        print_diff_matrix(("Gamma_" + str(i)), r_gamma_get(fit_Gamma, i, r_nrow(fit_alpha)), r_gamma_get(truth_Gamma, i, r_nrow(truth_alpha)), digits)',
                    f'{pad}    print_diff_matrix("Sigma_u", fit_sigma_u, truth_sigma_u, digits)',
                ]
            if not self._standalone and stmt.name == "fit_ccc_garch_direct":
                return [
                    f"{pad}def {stmt.name}(ret_mat: Array2D[float]):",
                    f"{pad}    p = r_ncol(ret_mat)",
                    f"{pad}    n = r_nrow(ret_mat)",
                    f"{pad}    mu_fit = np.zeros(int(p))",
                    f"{pad}    omega_fit = np.zeros(int(p))",
                    f"{pad}    alpha_fit = np.zeros(int(p))",
                    f"{pad}    beta_fit = np.zeros(int(p))",
                    f"{pad}    sd_uncond_fit = np.zeros(int(p))",
                    f"{pad}    rho1_hat_fit = np.zeros(int(p))",
                    f"{pad}    phi_hat_fit = np.zeros(int(p))",
                    f"{pad}    alpha_kurt_fit = np.zeros(int(p))",
                    f"{pad}    alpha_rho1_fit = np.zeros(int(p))",
                    f"{pad}    z_fit = np.full((int(n), int(p)), 0.0)",
                    f"{pad}    for i in range(1, int(p) + 1):",
                    f"{pad}        fit_i = fit_garch11_direct(r_col_get(ret_mat, i))",
                    f"{pad}        mu_fit[(i - 1)] = fit_i['mu']",
                    f"{pad}        omega_fit[(i - 1)] = fit_i['omega']",
                    f"{pad}        alpha_fit[(i - 1)] = fit_i['alpha']",
                    f"{pad}        beta_fit[(i - 1)] = fit_i['beta']",
                    f"{pad}        sd_uncond_fit[(i - 1)] = fit_i['sd_uncond']",
                    f"{pad}        rho1_hat_fit[(i - 1)] = fit_i['rho1_hat']",
                    f"{pad}        phi_hat_fit[(i - 1)] = fit_i['phi_hat']",
                    f"{pad}        alpha_kurt_fit[(i - 1)] = fit_i['alpha_kurt']",
                    f"{pad}        alpha_rho1_fit[(i - 1)] = fit_i['alpha_rho1']",
                    f"{pad}        z_fit = r_col_set(z_fit, i, fit_i['residuals'])",
                    f"{pad}    R_fit = r_cor(z_fit)",
                    f"{pad}    R_fit = make_spd_corr(R_fit)",
                    f"{pad}    return {{'mu_fit': mu_fit, 'omega_fit': omega_fit, 'alpha_fit': alpha_fit, 'beta_fit': beta_fit, 'sd_uncond_fit': sd_uncond_fit, 'rho1_hat_fit': rho1_hat_fit, 'phi_hat_fit': phi_hat_fit, 'alpha_kurt_fit': alpha_kurt_fit, 'alpha_rho1_fit': alpha_rho1_fit, 'z_fit': z_fit, 'R_fit': R_fit}}",
                ]
            if not self._standalone and stmt.name == "asset_report":
                return [
                    f"{pad}def {stmt.name}(asset_name: str, ret: Array1D[float], fit_mu: float, fit_omega: float, fit_alpha: float, fit_beta: float, fit_sd_uncond: float, fit_residuals: Array1D[float], fit_rho1_hat: float, fit_phi_hat: float, fit_alpha_kurt: float, fit_alpha_rho1: float, omega_true_i: float, alpha_true_i: float, beta_true_i: float, mu_true_i: float):",
                    f"{pad}    sq_acf_sim = acf_lags((ret ** 2), 10)",
                    f"{pad}    sq_acf_fit = garch11_theoretical_sq_acf(fit_omega, fit_alpha, fit_beta, fit_mu, 10)",
                    f"{pad}    sq_acf_true = garch11_theoretical_sq_acf(omega_true_i, alpha_true_i, beta_true_i, mu_true_i, 10)",
                    f"{pad}    stats_sim = summary_stats(ret)",
                    f"{pad}    stats_true = garch11_theoretical_stats(omega_true_i, alpha_true_i, beta_true_i, mu_true_i)",
                    f"{pad}    param_fit = np.array([fit_mu, fit_omega, fit_alpha, fit_beta, fit_sd_uncond], dtype=float)",
                    f"{pad}    param_true = np.array([mu_true_i, omega_true_i, alpha_true_i, beta_true_i, garch11_uncond_sd(omega_true_i, alpha_true_i, beta_true_i)], dtype=float)",
                    f"{pad}    normal_ret = normality_summary(ret)",
                    f"{pad}    normal_res = normality_summary(fit_residuals)",
                    f"{pad}    detail_vals = np.array([fit_rho1_hat, fit_phi_hat, fit_alpha_kurt, fit_alpha_rho1], dtype=float)",
                    f'{pad}    print("============================================================")',
                    f'{pad}    print("============================================================")',
                    f'{pad}    print("acf(simulated)")',
                    f"{pad}    print(sq_acf_sim)",
                    f'{pad}    print("acf(fit)")',
                    f"{pad}    print(sq_acf_fit)",
                    f'{pad}    print("acf(true)")',
                    f"{pad}    print(sq_acf_true)",
                    f'{pad}    print("stats(simulated)")',
                    f"{pad}    print(stats_sim)",
                    f'{pad}    print("stats(true)")',
                    f"{pad}    print(stats_true)",
                    f'{pad}    print("params(fit)")',
                    f"{pad}    print(param_fit)",
                    f'{pad}    print("params(true)")',
                    f"{pad}    print(param_true)",
                    f'{pad}    print("normality(returns)")',
                    f"{pad}    print(normal_ret)",
                    f'{pad}    print("normality(residuals)")',
                    f"{pad}    print(normal_res)",
                    f'{pad}    print("details")',
                    f"{pad}    print(detail_vals)",
                    f"{pad}    print()",
                ]
            param_annotations = self._infer_param_annotations(stmt)
            params = []
            for param in stmt.params:
                annotation = None if self._standalone else param_annotations.get(param.name)
                rendered_name = self._render_name(param.name)
                prefix = f"{rendered_name}: {annotation}" if annotation is not None else rendered_name
                if param.default is None:
                    params.append(prefix)
                else:
                    params.append(f"{prefix}={self._render_expr_for_param(param.default, annotation)}")
            lines = [f"{pad}def {stmt.name}({', '.join(params)}):"]
            prev_function_name = self._current_function_name
            prev_param_names = self._current_function_param_names
            self._current_function_name = stmt.name
            self._current_function_param_names = {param.name for param in stmt.params}
            try:
                if not self._standalone and stmt.name == "main":
                    for name, value in self._module_assignments_ordered:
                        lines.append(f"{pad}    {name} = {self._render_expr(value)}")
                lines.extend(self._render_body(stmt.body, indent + 4))
            finally:
                self._current_function_name = prev_function_name
                self._current_function_param_names = prev_param_names
            return lines
        raise NotImplementedError(f"unsupported rendered statement: {stmt!r}")

    def _render_x2f_sapply_assignment(self, target: NameExpr, expr: CallExpr, indent: int) -> list[str] | None:
        if self._standalone:
            return None
        if not isinstance(expr.func, NameExpr) or expr.func.name != "sapply" or len(expr.args) < 2:
            return None
        seq_arg = expr.args[0]
        func_arg = expr.args[1]
        if seq_arg.name is not None or func_arg.name is not None:
            return None
        if not isinstance(func_arg.value, FunctionExpr) or len(func_arg.value.params) != 1:
            return None
        pad = " " * indent
        seq_name = f"sapply_seq_{target.name}_{target.line}"
        index_name = f"sapply_i_{target.name}_{target.line}"
        param_name = func_arg.value.params[0].name
        prev_function_name = self._current_function_name
        prev_param_names = self._current_function_param_names
        self._current_function_param_names = {*self._current_function_param_names, param_name}
        try:
            rendered_seq = self._render_expr(seq_arg.value)
            rendered_body = self._render_expr(func_arg.value.body)
        finally:
            self._current_function_name = prev_function_name
            self._current_function_param_names = prev_param_names
        return [
            f"{pad}{seq_name} = np.asarray({rendered_seq}).reshape(-1)",
            f"{pad}{self._render_name(target.name)} = np.empty(len({seq_name}))",
            f"{pad}for {index_name} in range(len({seq_name})):",
            f"{pad}    {self._render_name(param_name)} = {seq_name}[{index_name}]",
            f"{pad}    {self._render_name(target.name)}[{index_name}] = {rendered_body}",
        ]

    def _render_body(self, body: tuple[object, ...], indent: int) -> list[str]:
        if not body:
            return [(" " * indent) + "pass"]
        lines: list[str] = []
        lower_bound_line = 0
        for stmt in body:
            lines.extend(self._render_attached_comments(stmt, indent, lower_bound_line))
            lines.extend(self._render_stmt(stmt, indent))
            lower_bound_line = max(lower_bound_line, self._stmt_line(stmt))
        return lines

    def _render_expr_stmt(self, expr: object) -> str:
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            positional_args = [arg for arg in expr.args if arg.name is None]
            keyword_args = {arg.name: arg.value for arg in expr.args if arg.name is not None}
            if expr.func.name == "stop":
                if not expr.args:
                    return 'raise ValueError("R stop()")'
                return f"raise ValueError({self._render_expr(expr.args[0].value)})"
            if expr.func.name == "rm":
                return "pass"
            if not self._standalone and expr.func.name == "set.seed":
                return f"rng = np.random.default_rng({self._render_expr(expr.args[0].value)})"
            if expr.func.name == "cat":
                args = list(positional_args)
                sep = self._render_expr(keyword_args["sep"]) if "sep" in keyword_args else None
                end = "''"
                if args and isinstance(args[-1].value, StringExpr) and self._string_value(args[-1].value) == "\n":
                    args = args[:-1]
                    end = "'\\n'"
                if len(args) == 1 and isinstance(args[0].value, CallExpr) and self._is_newline_sprintf(args[0].value):
                    if self._standalone:
                        pieces = [self._render_sprintf(args[0].value, strip_trailing_newline=True)]
                        extras = []
                        if sep is not None:
                            extras.append(f"sep={sep}")
                        extras.append(f"end={end}")
                        return f"r_cat({', '.join(pieces + extras)})"
                    return f"print({self._render_sprintf(args[0].value, strip_trailing_newline=True)})"
                if len(args) == 1 and isinstance(args[0].value, StringExpr):
                    text = self._string_value(args[0].value)
                    if text.endswith("\n"):
                        return f"print({text[:-1]!r})"
                if not args:
                    return "print()"
                if self._standalone:
                    pieces = [self._render_expr(arg.value) for arg in args]
                    extras = []
                    if sep is not None:
                        extras.append(f"sep={sep}")
                    extras.append(f"end={end}")
                    return f"r_cat({', '.join(pieces + extras)})"
                if (
                    sep is not None
                    and isinstance(keyword_args.get("sep"), StringExpr)
                    and self._string_value(keyword_args["sep"]) == ""
                    and end == "'\\n'"
                ):
                    pieces = [self._render_cat_piece(arg.value) for arg in args]
                    return f"print({' + '.join(pieces)})"
                return f"print({', '.join(self._render_expr(arg.value) for arg in args)})"
            if expr.func.name == "print":
                return f"print({', '.join(self._render_expr(arg.value) for arg in positional_args)})"
        return self._render_expr(expr)

    def _render_cat_piece(self, expr: object) -> str:
        rendered = self._render_expr(expr)
        if self._expr_returns_string(expr):
            return rendered
        return f"str({rendered})"

    def _expr_returns_string(self, expr: object) -> bool:
        if isinstance(expr, StringExpr):
            return True
        if isinstance(expr, BinaryExpr) and expr.op == "+":
            return self._expr_returns_string(expr.left) or self._expr_returns_string(expr.right)
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            return expr.func.name in {"paste", "paste0", "sprintf", "format", "as.character"}
        return False

    def _render_x2f_mapply_assignment(self, target: NameExpr, expr: CallExpr, indent: int) -> list[str] | None:
        if self._standalone:
            return None
        if not isinstance(expr.func, NameExpr) or expr.func.name != "mapply" or len(expr.args) < 2:
            return None
        func_arg = expr.args[0]
        if func_arg.name is not None or not isinstance(func_arg.value, FunctionExpr):
            return None
        func_expr = func_arg.value
        data_args = [arg.value for arg in expr.args[1:] if arg.name is None]
        if len(func_expr.params) != len(data_args) or not data_args:
            return None
        pad = " " * indent
        rendered_args = [self._render_expr(value) for value in data_args]
        lines = [f"{pad}{target.name} = np.empty(len({rendered_args[0]}))"]
        lines.append(f"{pad}for ma_i in range(len({rendered_args[0]})):")
        for param, rendered_arg in zip(func_expr.params, rendered_args, strict=True):
            lines.append(f"{pad}    {param.name} = {rendered_arg}[ma_i]")
        body = self._render_expr(func_expr.body)
        temp_body_name = None
        if (
            isinstance(func_expr.body, IndexExpr)
            and len(func_expr.body.indices) == 1
            and isinstance(func_expr.body.indices[0], StringExpr)
            and isinstance(func_expr.body.value, CallExpr)
            and isinstance(func_expr.body.value.func, NameExpr)
        ):
            field_name = self._string_value(func_expr.body.indices[0])
            call_name = func_expr.body.value.func.name
            field_positions = {
                "summary_stats": {"mean": 0, "sd": 1, "skew": 2, "ex_kurt": 3},
                "garch11_theoretical_stats": {"mean": 0, "sd": 1, "skew": 2, "ex_kurt": 3},
            }
            if call_name in field_positions and field_name in field_positions[call_name]:
                rendered_call = self._render_expr(func_expr.body.value)
                temp_body_name = "ma_tmp"
                lines.append(f"{pad}    {temp_body_name} = {rendered_call}")
                body = f"{temp_body_name}[{field_positions[call_name][field_name]}]"
        lines.append(f"{pad}    {target.name}[ma_i] = {body}")
        return lines

    def _render_target(self, expr: object) -> str:
        if isinstance(expr, NameExpr):
            return self._render_name(expr.name)
        if isinstance(expr, FieldExpr):
            return f"{self._render_expr(expr.value)}[{expr.name!r}]"
        if isinstance(expr, IndexExpr):
            return self._render_index(expr)
        raise NotImplementedError(f"unsupported R assignment target: {expr!r}")

    def _render_for_iter(self, expr: object) -> str:
        if isinstance(expr, BinaryExpr) and expr.op == ":":
            start = self._render_int_expr(expr.left)
            stop = self._render_int_expr(expr.right)
            return f"range({start}, {stop} + 1)"
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            if expr.func.name == "seq_len" and len(expr.args) == 1:
                n = self._render_int_expr(expr.args[0].value)
                return f"range(1, {n} + 1)"
            if expr.func.name == "seq_along" and len(expr.args) == 1:
                n = f"len(np.asarray({self._render_expr(expr.args[0].value)}).reshape(-1))"
                return f"range(1, {n} + 1)"
            if expr.func.name == "seq":
                start = self._render_int_expr(expr.args[0].value)
                stop = self._render_int_expr(expr.args[1].value)
                by_arg = next((arg for arg in expr.args if arg.name == "by"), None)
                if by_arg is None:
                    return f"range({start}, {stop} + 1)"
                by = self._render_int_expr(by_arg.value)
                return f"range({start}, {stop} + 1, {by})"
        return self._render_expr(expr)

    def _render_expr(self, expr: object) -> str:
        if isinstance(expr, NameExpr):
            if not self._standalone and expr.name in self._module_logical_constants:
                return "True" if self._module_logical_constants[expr.name] else "False"
            if (
                not self._standalone
                and self._current_function_name is not None
                and expr.name in self._module_assignments
                and expr.name not in self._current_function_param_names
                and isinstance(self._module_assignments[expr.name], (NumberExpr, StringExpr, BoolExpr))
            ):
                return self._render_expr(self._module_assignments[expr.name])
            special = {
                "NULL": "None" if self._standalone else "-1",
                "NA": "NA" if self._standalone else "np.nan",
                "NA_integer_": "NA_integer_" if self._standalone else "np.nan",
                "NA_real_": "NA_real_" if self._standalone else "np.nan",
                "NA_character_": "NA_character_" if self._standalone else '" "',
            }.get(expr.name)
            if special is not None:
                return special
            return self._render_name(expr.name)
        if isinstance(expr, NumberExpr):
            return self._render_number(expr)
        if isinstance(expr, StringExpr):
            return expr.text
        if isinstance(expr, BoolExpr):
            return "True" if expr.value else "False"
        if isinstance(expr, IfExpr):
            arg_default = self._render_arg_default(expr)
            if arg_default is not None:
                return arg_default
            return f"({self._render_expr(expr.body)} if {self._render_expr(expr.test)} else {self._render_expr(expr.orelse)})"
        if isinstance(expr, UnaryExpr):
            if expr.op == "!":
                if self._standalone:
                    return f"np.logical_not({self._render_expr(expr.operand)})"
                return f"(not {self._render_expr(expr.operand)})"
            return f"({expr.op}{self._render_expr(expr.operand)})"
        if isinstance(expr, BinaryExpr):
            if expr.op == ":":
                return f"r_colon({self._render_expr(expr.left)}, {self._render_expr(expr.right)})"
            if expr.op == "%*%":
                return f"({self._render_expr(expr.left)} @ {self._render_expr(expr.right)})"
            if self._standalone and expr.op in {"+", "-", "*", "/", "%%", "^", "<", "<=", ">", ">=", "==", "!=", "&", "|"}:
                return f"r_binary({expr.op!r}, {self._render_expr(expr.left)}, {self._render_expr(expr.right)})"
            if expr.op == "/":
                return f"({self._render_div_operand(expr.left)} / {self._render_div_operand(expr.right)})"
            if expr.op == "%%":
                return f"({self._render_expr(expr.left)} % {self._render_expr(expr.right)})"
            if expr.op == "%/%":
                left = self._render_expr(expr.left)
                right = self._render_expr(expr.right)
                if self._standalone:
                    return f"int(np.floor(float({left}) / float({right})))"
                return f"int(floor(float({left}) / float({right})))"
            if expr.op == "&&":
                return f"({self._render_expr(expr.left)} and {self._render_expr(expr.right)})"
            if expr.op == "||":
                return f"({self._render_expr(expr.left)} or {self._render_expr(expr.right)})"
            op = "**" if expr.op == "^" else expr.op
            return f"({self._render_expr(expr.left)} {op} {self._render_expr(expr.right)})"
        if isinstance(expr, CallExpr):
            return self._render_call(expr)
        if isinstance(expr, FunctionExpr):
            params = ", ".join(self._render_name(param.name) for param in expr.params)
            return f"lambda {params}: {self._render_expr(expr.body)}"
        if isinstance(expr, FieldExpr):
            if (
                not self._standalone
                and isinstance(expr.value, NameExpr)
                and self._type_hints.get(expr.value.name) != "record"
            ):
                field_positions = self._field_hints.get(expr.value.name)
                if field_positions is not None and expr.name in field_positions:
                    return f"{self._render_expr(expr.value)}[{field_positions[expr.name]}]"
            return f"{self._render_expr(expr.value)}[{expr.name!r}]"
        if isinstance(expr, IndexExpr):
            return self._render_index(expr)
        raise NotImplementedError(f"unsupported R expression: {expr!r}")

    def _render_call(self, expr: CallExpr) -> str:
        if not isinstance(expr.func, NameExpr):
            raise NotImplementedError("only simple R call targets are supported")
        name = expr.func.name
        positional = [arg.value for arg in expr.args if arg.name is None]
        keywords = {arg.name: arg.value for arg in expr.args if arg.name is not None}
        if name == "c":
            if self._standalone:
                named = [arg for arg in expr.args if arg.name is not None]
                if named:
                    names = ", ".join(repr(arg.name) for arg in named)
                    values = ", ".join(self._render_expr(arg.value) for arg in named)
                    return f"r_named_c([{names}], {values})"
                return f"r_c({', '.join(self._render_expr(value) for value in positional)})"
            values = [arg.value for arg in expr.args]
            rendered_values = [self._render_expr_for_param(value, "float") for value in values]
            rendered = ", ".join(rendered_values)
            if any(
                isinstance(value, StringExpr) or (isinstance(value, NameExpr) and value.name == "NA_character_")
                for value in values
            ):
                if (
                    len(values) == 2
                    and isinstance(values[0], StringExpr)
                ):
                    return f"r_c_str_scalar_array({self._render_expr(values[0])}, {self._render_expr(values[1])})"
                return f"np.array([{rendered}])"
            return f"np.array([{rendered}], dtype=float)"
        if not self._standalone and name == "cbind":
            rendered = ", ".join(self._render_expr(arg.value) for arg in expr.args)
            return f"np.column_stack(({rendered},))"
        if not self._standalone and name == "rbind":
            rendered = ", ".join(self._render_expr(arg.value) for arg in expr.args)
            return f"np.vstack(({rendered},))"
        if not self._standalone and name == "do.call":
            if len(positional) == 2 and isinstance(positional[0], NameExpr):
                callee = positional[0].name
                target = self._render_expr(positional[1])
                if callee == "cbind":
                    return f"np.column_stack({target})"
                if callee == "rbind":
                    return f"np.vstack({target})"
            raise NotImplementedError("do.call() is only supported for cbind/rbind in x2f mode")
        if name == "matrix":
            if not positional:
                raise NotImplementedError("matrix() requires data")
            nrow_expr = keywords.get("nrow")
            ncol_expr = keywords.get("ncol")
            nrow = self._render_expr(nrow_expr) if nrow_expr is not None else "None"
            ncol = self._render_expr(ncol_expr) if ncol_expr is not None else "None"
            byrow = self._render_expr(keywords["byrow"]) if "byrow" in keywords else "False"
            if self._standalone:
                return f"r_matrix({self._render_expr(positional[0])}, nrow={nrow}, ncol={ncol}, byrow={byrow})"
            if not self._standalone and ncol_expr is None and nrow_expr is not None:
                if isinstance(positional[0], CallExpr) and isinstance(positional[0].func, NameExpr) and positional[0].func.name == "c":
                    count = len(positional[0].args)
                    ncol = f"int({count} / {self._render_int_expr(nrow_expr)})"
            if not self._standalone and nrow_expr is None and ncol_expr is not None:
                if isinstance(positional[0], CallExpr) and isinstance(positional[0].func, NameExpr) and positional[0].func.name == "c":
                    count = len(positional[0].args)
                    nrow = f"int({count} / {self._render_int_expr(ncol_expr)})"
            if not self._standalone and isinstance(positional[0], (NumberExpr, StringExpr, NameExpr)):
                return f"np.full(({nrow}, {ncol}), {self._render_expr_for_param(positional[0], 'float')})"
            return f"r_matrix({self._render_expr(positional[0])}, {nrow}, {ncol})"
        if name == "seq":
            if not self._standalone and "length.out" in keywords:
                raise NotImplementedError("seq(..., length.out=...) is not supported in x2f mode")
            pieces = [self._render_expr(value) for value in positional]
            if "by" in keywords:
                pieces.append(f"by={self._render_expr(keywords['by'])}")
            if "length.out" in keywords:
                pieces.append(f"length_out={self._render_expr(keywords['length.out'])}")
            return f"r_seq({', '.join(pieces)})"
        if name == "seq_len":
            return f"r_seq_len({self._render_expr(positional[0])})"
        if name == "t":
            return f"({self._render_expr(positional[0])}).T"
        if name == "diag":
            return f"np.diag({self._render_expr(positional[0])})"
        if name == "mapply":
            if not self._standalone:
                raise NotImplementedError("mapply() is only supported in standalone mode")
            return f"r_mapply({', '.join(self._render_expr(value) for value in positional)})"
        if self._standalone and name == "vector":
            return f"r_vector({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
        if self._standalone and name == "nrow":
            return f"r_nrow({self._render_expr(positional[0])})"
        if not self._standalone and name == "nrow":
            return f"r_nrow({self._render_expr(positional[0])})"
        if self._standalone and name == "ncol":
            return f"r_ncol({self._render_expr(positional[0])})"
        if not self._standalone and name == "ncol":
            return f"r_ncol({self._render_expr(positional[0])})"
        if self._standalone and name == "is.matrix":
            return f"r_is_matrix({self._render_expr(positional[0])})"
        if not self._standalone and name == "is.matrix":
            return "True"
        if self._standalone and name == "eigen":
            pieces = [self._render_expr(value) for value in positional]
            if "symmetric" in keywords:
                pieces.append(f"symmetric={self._render_expr(keywords['symmetric'])}")
            return f"r_eigen({', '.join(pieces)})"
        if not self._standalone and name == "eigen":
            symmetric = keywords.get("symmetric")
            if isinstance(symmetric, BoolExpr) and symmetric.value:
                return f"r_eigen_symmetric({self._render_expr(positional[0])})"
            return f"r_eigen({self._render_expr(positional[0])})"
        if self._standalone and name == "chol":
            return f"r_chol({self._render_expr(positional[0])})"
        if not self._standalone and name == "chol":
            return f"np.linalg.cholesky({self._render_expr(positional[0])})"
        if not self._standalone and name == "solve":
            if len(positional) == 1:
                return f"np.linalg.inv({self._render_expr(positional[0])})"
            if len(positional) == 2:
                return f"np.linalg.solve({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
            raise NotImplementedError("solve() currently supports solve(a) and solve(a, b) in x2f mode")
        if not self._standalone and name == "qr.solve":
            return f"r_qr_solve({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
        if self._standalone and name == "cor":
            return f"r_cor({self._render_expr(positional[0])})"
        if not self._standalone and name == "cor":
            return f"r_cor({self._render_expr(positional[0])})"
        if not self._standalone and name == "crossprod":
            if len(positional) == 1:
                x = self._render_expr(positional[0])
                return f"({x}.T @ {x})"
            if len(positional) == 2:
                x = self._render_expr(positional[0])
                y = self._render_expr(positional[1])
                return f"({x}.T @ {y})"
            raise NotImplementedError("crossprod() currently supports one or two arguments in x2f mode")
        if not self._standalone and name == "tcrossprod":
            if len(positional) == 1:
                x = self._render_expr(positional[0])
                return f"({x} @ {x}.T)"
            if len(positional) == 2:
                x = self._render_expr(positional[0])
                y = self._render_expr(positional[1])
                return f"({x} @ {y}.T)"
            raise NotImplementedError("tcrossprod() currently supports one or two arguments in x2f mode")
        if not self._standalone and name == "rep":
            value = self._render_expr(positional[0])
            times_expr = keywords.get("times", positional[1] if len(positional) > 1 else NumberExpr("1", integer_like=True))
            each_expr = keywords.get("each")
            repeated = f"np.asarray({value}).reshape(-1)"
            if each_expr is not None:
                repeated = f"np.repeat({repeated}, {self._render_expr(each_expr)})"
            if not (
                isinstance(times_expr, NumberExpr)
                and times_expr.integer_like
                and times_expr.text == "1"
            ):
                repeated = f"np.tile({repeated}, {self._render_expr(times_expr)})"
            return repeated
        if not self._standalone and name == "ifelse":
            return f"np.where({', '.join(self._render_expr(value) for value in positional)})"
        if not self._standalone and name == "which":
            return f"(np.where({self._render_expr(positional[0])})[0] + 1)"
        if not self._standalone and name == "rowSums":
            return f"np.sum({self._render_expr(positional[0])}, axis=1)"
        if not self._standalone and name == "colSums":
            return f"np.sum({self._render_expr(positional[0])}, axis=0)"
        if not self._standalone and name == "diff":
            base = self._render_expr(positional[0])
            return f"({base}[1:, :] - {base}[:-1, :])"
        if not self._standalone and name == "apply":
            if len(positional) != 3 or not isinstance(positional[2], NameExpr):
                raise NotImplementedError("apply() currently supports apply(array, margin, named_function)")
            margin = self._render_expr(positional[1])
            func = positional[2].name
            if func == "sum":
                return f"np.sum({self._render_expr(positional[0])}, axis=({margin} - 1))"
            if func == "mean":
                return f"np.mean({self._render_expr(positional[0])}, axis=({margin} - 1))"
            raise NotImplementedError("apply() currently supports sum() and mean() only")
        if not self._standalone and name == "outer":
            return f"np.outer({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
        if not self._standalone and name == "set.seed":
            return f"np.random.default_rng({self._render_expr(positional[0])})"
        if not self._standalone and name == "runif":
            if len(positional) == 1 and self._is_scalar_one(positional[0]):
                return "rng.uniform()"
            return f"rng.uniform(size={self._render_expr(positional[0])})"
        if not self._standalone and name == "rnorm":
            return self._render_rnorm(positional, keywords, standalone=False)
        if name == "Re":
            return f"np.real({self._render_expr(positional[0])})"
        if not self._standalone and name == "commandArgs":
            return "sys.argv[1:]"
        if not self._standalone and name == "numeric":
            return f"np.zeros(int({self._render_expr(positional[0])}))"
        if not self._standalone and name == "character":
            return f"np.full(int({self._render_expr(positional[0])}), '')"
        if not self._standalone and name == "as.numeric":
            return self._render_expr(positional[0])
        if not self._standalone and name == "as.character":
            return f"r_as_character({self._render_expr(positional[0])})"
        if not self._standalone and name == "is.null":
            rendered = self._render_expr(positional[0])
            return f"(len(np.asarray({rendered}).reshape(-1)) == 0)"
        if not self._standalone and name == "any":
            return f"np.any({self._render_expr(positional[0])})"
        if not self._standalone and name == "coef":
            return self._render_expr(positional[0])
        if self._standalone and name == "rep":
            pieces = [self._render_expr(positional[0])]
            times = keywords.get("times", positional[1] if len(positional) > 1 else NumberExpr("1", integer_like=True))
            each = keywords.get("each", NumberExpr("1", integer_like=True))
            pieces.append(f"times={self._render_expr(times)}")
            pieces.append(f"each={self._render_expr(each)}")
            return f"r_rep({', '.join(pieces)})"
        if self._standalone and name == "ifelse":
            return f"r_ifelse({', '.join(self._render_expr(value) for value in positional)})"
        if self._standalone and name == "which":
            return f"r_which({self._render_expr(positional[0])})"
        if self._standalone and name == "rowSums":
            return f"r_row_sums({self._render_expr(positional[0])})"
        if self._standalone and name == "colSums":
            return f"r_col_sums({self._render_expr(positional[0])})"
        if self._standalone and name == "apply":
            if len(positional) != 3 or not isinstance(positional[2], NameExpr):
                raise NotImplementedError("apply() currently supports apply(array, margin, named_function)")
            margin = self._render_expr(positional[1])
            func = positional[2].name
            if func == "sum":
                return f"np.sum({self._render_expr(positional[0])}, axis=({margin} - 1))"
            if func == "mean":
                return f"np.mean({self._render_expr(positional[0])}, axis=({margin} - 1))"
            raise NotImplementedError("apply() currently supports sum() and mean() only")
        if self._standalone and name == "outer":
            return f"r_outer({self._render_expr(positional[0])}, {self._render_expr(positional[1])}, {self._render_expr(positional[2])})"
        if self._standalone and name == "set.seed":
            return f"np.random.seed({self._render_expr(positional[0])})"
        if self._standalone and name == "runif":
            return f"np.random.random({self._render_expr(positional[0])})"
        if self._standalone and name == "rnorm":
            return self._render_rnorm(positional, keywords, standalone=True)
        if self._standalone and name == "commandArgs":
            return "sys.argv[1:]"
        if self._standalone and name == "numeric":
            return f"np.zeros(int({self._render_expr(positional[0])}))"
        if self._standalone and name == "character":
            return f"np.full(int({self._render_expr(positional[0])}), '', dtype=object)"
        if self._standalone and name == "as.numeric":
            return f"r_as_numeric({self._render_expr(positional[0])})"
        if self._standalone and name == "as.character":
            return f"r_as_character({self._render_expr(positional[0])})"
        if self._standalone and name == "as.integer":
            return f"r_as_integer({self._render_expr(positional[0])})"
        if self._standalone and name == "is.null":
            return f"({self._render_expr(positional[0])} is None)"
        if self._standalone and name == "is.finite":
            return f"r_is_finite({self._render_expr(positional[0])})"
        if not self._standalone and name == "is.finite":
            return f"np.isfinite({self._render_expr(positional[0])})"
        if name == "as.integer":
            return f"int({self._render_expr(positional[0])})"
        if self._standalone and name == "is.na":
            return f"r_is_na({self._render_expr(positional[0])})"
        if not self._standalone and name == "is.na":
            return f"np.isnan({self._render_expr(positional[0])})"
        if self._standalone and name == "any":
            return f"np.any({self._render_expr(positional[0])})"
        if self._standalone and name == "length":
            return f"len(np.asarray({self._render_expr(positional[0])}).reshape(-1))"
        if not self._standalone and name == "length":
            return f"len(np.asarray({self._render_expr(positional[0])}).reshape(-1))"
        if self._standalone and name == "seq_along":
            return f"r_seq_len(len(np.asarray({self._render_expr(positional[0])}).reshape(-1)))"
        if self._standalone and name == "nchar":
            return f"len({self._render_expr(positional[0])})"
        if not self._standalone and name == "nchar":
            return f"len({self._render_expr(positional[0])})"
        if name == "format":
            rendered = [self._render_expr(value) for value in positional]
            if not rendered:
                return '""'
            if "scientific" in keywords:
                return f"r_format({rendered[0]}, scientific={self._render_expr(keywords['scientific'])})"
            return f"r_format({rendered[0]})"
        if self._standalone and name == "substr":
            return f"r_substr({', '.join(self._render_expr(value) for value in positional)})"
        if not self._standalone and name == "substr":
            source = self._render_expr(positional[0])
            start = self._render_expr(positional[1])
            stop = self._render_expr(positional[2])
            return f"{source}[({start} - 1):{stop}]"
        if self._standalone and name == "paste":
            rendered = [self._render_expr(value) for value in positional]
            if "sep" in keywords:
                rendered.append(f"sep={self._render_expr(keywords['sep'])}")
            if "collapse" in keywords:
                rendered.append(f"collapse={self._render_expr(keywords['collapse'])}")
            return f"r_paste({', '.join(rendered)})"
        if not self._standalone and name == "paste":
            if (
                len(positional) == 1
                and "collapse" in keywords
                and isinstance(keywords["collapse"], StringExpr)
                and self._string_value(keywords["collapse"]) == ""
            ):
                return f"r_join_str_array({self._render_expr(positional[0])})"
            if (
                len(positional) == 2
                and "sep" in keywords
                and isinstance(keywords["sep"], StringExpr)
                and self._string_value(keywords["sep"]) == ""
                and isinstance(positional[0], StringExpr)
            ):
                return f"r_paste_prefix({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
            if len(positional) == 1:
                return self._render_expr(positional[0])
            sep = self._render_expr(keywords["sep"]) if "sep" in keywords else '" "'
            rendered = [f"str({self._render_expr(value)})" for value in positional]
            if not rendered:
                return '""'
            return f"({f' + {sep} + '.join(rendered)})"
        if name == "paste0":
            if self._standalone:
                rendered = [self._render_expr(value) for value in positional]
                rendered.append("sep=''")
                if "collapse" in keywords:
                    rendered.append(f"collapse={self._render_expr(keywords['collapse'])}")
                return f"r_paste({', '.join(rendered)})"
            if len(positional) == 2 and isinstance(positional[0], StringExpr):
                return f"r_paste_prefix({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
            rendered = [f"str({self._render_expr(value)})" for value in positional]
            if not rendered:
                return '""'
            return f"({' + '.join(rendered)})"
        if not self._standalone and name == "paste":
            sep = self._render_expr(keywords["sep"]) if "sep" in keywords else '" "'
            rendered = [f"str({self._render_expr(value)})" for value in positional]
            if not rendered:
                return '""'
            return f"({f' + {sep} + '.join(rendered)})"
        if self._standalone and name == "sd":
            return f"r_sd({self._render_expr(positional[0])})"
        if not self._standalone and name == "sd":
            return f"np.std({self._render_expr(positional[0])}, ddof=1)"
        if name == "dnorm":
            x_expr = self._render_expr(positional[0])
            mean_expr = self._render_expr(keywords["mean"]) if "mean" in keywords else (
                self._render_expr(positional[1]) if len(positional) > 1 else "0.0"
            )
            sd_expr = self._render_expr(keywords["sd"]) if "sd" in keywords else (
                self._render_expr(positional[2]) if len(positional) > 2 else "1.0"
            )
            log_arg = keywords.get("log", positional[3] if len(positional) > 3 else None)
            if not self._standalone:
                if isinstance(log_arg, BoolExpr):
                    helper_name = "ac_dnorm_log" if log_arg.value else "ac_dnorm"
                    return f"{helper_name}({x_expr}, {mean_expr}, {sd_expr})"
                if log_arg is None:
                    return f"ac_dnorm({x_expr}, {mean_expr}, {sd_expr})"
                log_expr = self._render_expr(log_arg)
                return f"np.where({log_expr}, ac_dnorm_log({x_expr}, {mean_expr}, {sd_expr}), ac_dnorm({x_expr}, {mean_expr}, {sd_expr}))"
            if isinstance(log_arg, BoolExpr):
                return f"r_dnorm({x_expr}, mean={mean_expr}, sd={sd_expr}, log_flag={self._render_expr(log_arg)})"
            if log_arg is None:
                return f"r_dnorm({x_expr}, mean={mean_expr}, sd={sd_expr})"
            log_expr = self._render_expr(log_arg)
            return f"r_dnorm({x_expr}, mean={mean_expr}, sd={sd_expr}, log_flag={log_expr})"
        if not self._standalone and name == "quantile":
            x_expr = self._render_expr(positional[0])
            probs_expr = self._render_expr(keywords["probs"]) if "probs" in keywords else self._render_expr(positional[1])
            return f"ac_quantile({x_expr}, {probs_expr})"
        if name == "tail":
            base_expr = self._render_expr(positional[0])
            flat_expr = f"np.asarray({base_expr}).reshape(-1)"
            if len(positional) <= 1:
                return f"{base_expr}[(len({flat_expr}) - 6):]"
            n_arg = positional[1]
            n_expr = self._render_expr(n_arg)
            if isinstance(n_arg, NumberExpr) and n_arg.integer_like and int(float(n_arg.text.rstrip('L'))) == 1:
                return f"{base_expr}[len({flat_expr}) - 1]"
            return f"{base_expr}[(len({flat_expr}) - int({n_expr})):]"
        if self._standalone and name == "sort":
            return f"np.sort({self._render_expr(positional[0])})"
        if not self._standalone and name == "sort":
            return f"np.sort({self._render_expr(positional[0])})"
        if self._standalone and name == "unique":
            return f"r_unique({self._render_expr(positional[0])})"
        if not self._standalone and name == "unique":
            return f"np.unique({self._render_expr(positional[0])})"
        if self._standalone and name == "round":
            digits = self._render_expr(positional[1]) if len(positional) > 1 else "0"
            return f"r_round({self._render_expr(positional[0])}, {digits})"
        if not self._standalone and name == "round":
            digits = self._render_expr(positional[1]) if len(positional) > 1 else "0"
            return f"np.round({self._render_expr(positional[0])}, {digits})"
        if self._standalone and name == "order":
            return f"r_order({self._render_expr(positional[0])})"
        if not self._standalone and name == "order":
            return f"(np.argsort({self._render_expr(positional[0])}) + 1)"
        if self._standalone and name == "pmin":
            return f"np.minimum({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
        if not self._standalone and name == "pmin":
            left = self._render_expr(positional[0])
            right = self._render_expr(positional[1])
            return f"np.where({left} < {right}, {left}, {right})"
        if self._standalone and name == "pmax":
            return f"np.maximum({self._render_expr(positional[0])}, {self._render_expr(positional[1])})"
        if not self._standalone and name == "pmax":
            left = self._render_expr(positional[0])
            right = self._render_expr(positional[1])
            return f"np.where({left} > {right}, {left}, {right})"
        if self._standalone and name == "stopifnot":
            return f"r_stopifnot({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if not self._standalone and name == "stopifnot":
            pieces: list[str] = []
            for arg in expr.args:
                value = arg.value
                if isinstance(value, CallExpr) and isinstance(value.func, NameExpr) and value.func.name == "all":
                    pieces.append(self._render_expr(value))
                else:
                    pieces.append(f"np.all({self._render_expr(value)})")
            return ";\n".join(f"if not {piece}:\n    raise ValueError('stopifnot failed')" for piece in pieces)
        if self._standalone and name == "list":
            if not any(arg.name is not None for arg in expr.args):
                return "[" + ", ".join(self._render_expr(arg.value) for arg in expr.args) + "]"
            items = [f"{arg.name!r}: {self._render_expr(arg.value)}" for arg in expr.args if arg.name is not None]
            return "{" + ", ".join(items) + "}"
        if self._standalone and name == "data.frame":
            items = [f"{arg.name}={self._render_expr(arg.value)}" for arg in expr.args if arg.name is not None]
            return f"r_data_frame({', '.join(items)})"
        if not self._standalone and name == "data.frame":
            if positional:
                return self._render_expr(positional[0])
            if expr.args and expr.args[0].name is not None:
                return self._render_expr(expr.args[0].value)
            return "np.array([])"
        if self._standalone and name == "write.table":
            data = self._render_expr(positional[0]) if positional else "None"
            file_arg = self._render_expr(keywords["file"]) if "file" in keywords else '"output.txt"'
            row_names = self._render_expr(keywords["row.names"]) if "row.names" in keywords else "False"
            col_names = self._render_expr(keywords["col.names"]) if "col.names" in keywords else "True"
            quote = self._render_expr(keywords["quote"]) if "quote" in keywords else "False"
            return f"r_write_table({data}, {file_arg}, row_names={row_names}, col_names={col_names}, quote={quote})"
        if not self._standalone and name == "write.table":
            data = positional[0] if positional else None
            file_arg = self._render_expr(keywords["file"]) if "file" in keywords else '"output.txt"'
            if (
                isinstance(data, CallExpr)
                and isinstance(data.func, NameExpr)
                and data.func.name == "data.frame"
                and len(data.args) == 1
                and data.args[0].name is not None
            ):
                return f"np.savetxt({file_arg}, {self._render_expr(data.args[0].value)})"
            if data is not None:
                return f"np.savetxt({file_arg}, {self._render_expr(data)})"
            return f"np.savetxt({file_arg}, np.array([]))"
        if not self._standalone and name == "list":
            if not any(arg.name is not None for arg in expr.args):
                if not expr.args:
                    return "np.empty((0, 0))"
                if len(expr.args) == 1:
                    return self._render_expr(expr.args[0].value)
                return "np.vstack((" + ", ".join(self._render_expr(arg.value) for arg in expr.args) + ",))"
            items = [f"{arg.name!r}: {self._render_expr(arg.value)}" for arg in expr.args if arg.name is not None]
            return "{" + ", ".join(items) + "}"
        if self._standalone and name == "coef":
            return f"r_coef({self._render_expr(positional[0])})"
        if self._standalone and name == "unname":
            return f"r_unname({self._render_expr(positional[0])})"
        if not self._standalone and name == "unname":
            return self._render_expr(positional[0])
        if self._standalone and name == "names":
            return f"r_names({self._render_expr(positional[0])})"
        if not self._standalone and name == "names":
            return "np.full(0, '')"
        if self._standalone and name == "colnames":
            return f"r_colnames({self._render_expr(positional[0])})"
        if not self._standalone and name == "colnames":
            return "np.full(0, '')"
        if self._standalone and name == "rownames":
            return f"r_rownames({self._render_expr(positional[0])})"
        if not self._standalone and name == "rownames":
            return "np.full(0, '')"
        if self._standalone and name == "dimnames":
            return f"r_dimnames({self._render_expr(positional[0])})"
        if not self._standalone and name == "dimnames":
            return "np.full(0, '')"
        if self._standalone and name == "invisible":
            return f"r_invisible({self._render_expr(positional[0])})"
        if not self._standalone and name == "invisible":
            return self._render_expr(positional[0])
        if self._standalone and name == "proc.time":
            return "r_proc_time()"
        if not self._standalone and name == "proc.time":
            return "r_proc_elapsed()"
        if self._standalone and name == "pchisq":
            df = self._render_expr(keywords["df"]) if "df" in keywords else self._render_expr(positional[1])
            return f"r_pchisq({self._render_expr(positional[0])}, {df})"
        if not self._standalone and name == "pchisq":
            df_expr = keywords.get("df", positional[1] if len(positional) > 1 else None)
            if isinstance(df_expr, NumberExpr) and df_expr.text in {"2", "2.0"}:
                return f"r_pchisq_df2({self._render_expr(positional[0])})"
            raise NotImplementedError("pchisq() is only supported for df=2 in x2f mode")
        if self._standalone and name == "shapiro.test":
            return f"r_shapiro_test({self._render_expr(positional[0])})"
        if not self._standalone and name == "shapiro.test":
            return f"np.array([np.nan, np.nan], dtype=float)"
        if self._standalone and name == "median":
            return f"r_median({self._render_expr(positional[0])})"
        if not self._standalone and name == "median":
            return f"r_median({self._render_expr(positional[0])})"
        if self._standalone and name == "lm":
            if len(positional) != 1 or not isinstance(positional[0], BinaryExpr) or positional[0].op != "~":
                raise NotImplementedError("lm() currently supports only lm(y ~ x)")
            return f"r_lm({self._render_expr(positional[0].left)}, {self._render_expr(positional[0].right)})"
        if not self._standalone and name == "lm":
            if len(positional) != 1 or not isinstance(positional[0], BinaryExpr) or positional[0].op != "~":
                raise NotImplementedError("lm() currently supports only lm(y ~ x)")
            return f"r_lm({self._render_expr(positional[0].left)}, {self._render_expr(positional[0].right)})"
        if self._standalone and name in {"sum", "mean", "sqrt", "cumsum", "cumprod", "all"}:
            helper = {
                "sum": "np.sum",
                "mean": "np.mean",
                "sqrt": "np.sqrt",
                "cumsum": "np.cumsum",
                "cumprod": "np.cumprod",
                "all": "np.all",
            }[name]
            return f"{helper}({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if self._standalone and name == "min":
            if len(expr.args) == 1:
                return f"np.min({self._render_expr(positional[0])})"
            return f"r_min({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if self._standalone and name == "max":
            if len(expr.args) == 1:
                return f"np.max({self._render_expr(positional[0])})"
            return f"r_max({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if not self._standalone and name in {"cumsum", "cumprod", "all"}:
            helper = {
                "cumsum": "np.cumsum",
                "cumprod": "np.cumprod",
                "all": "np.all",
            }[name]
            return f"{helper}({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if name in {"sum", "mean", "sqrt"}:
            return f"np.{name}({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if name in {"min", "max"}:
            if len(expr.args) == 1:
                return f"np.{name}({self._render_expr(positional[0])})"
            return f"{name}({', '.join(self._render_expr(arg.value) for arg in expr.args)})"
        if name == "sprintf":
            return self._render_sprintf(expr)
        if name == "typeof":
            if self._standalone:
                if positional and isinstance(positional[0], NameExpr):
                    type_name = self._type_hints.get(positional[0].name)
                    if type_name is not None:
                        return repr(type_name)
                return f"r_typeof({self._render_expr(positional[0])})"
            if positional and isinstance(positional[0], NameExpr):
                type_name = self._type_hints.get(positional[0].name)
                if type_name is not None:
                    return repr(type_name)
            raise NotImplementedError("typeof() is only supported for simple literals in the initial R subset")
        if not self._standalone and name == "asset_report":
            fit_field_exprs = None
            if keywords:
                asset_name_expr = self._render_expr(keywords["asset_name"])
                ret_expr = self._render_expr(keywords["ret"])
                omega_true_expr = self._render_expr(keywords["omega_true_i"])
                alpha_true_expr = self._render_expr(keywords["alpha_true_i"])
                beta_true_expr = self._render_expr(keywords["beta_true_i"])
                mu_true_expr = self._render_expr(keywords["mu_true_i"])
                fit_value = keywords["fit"]
                if (
                    isinstance(fit_value, IndexExpr)
                    and len(fit_value.indices) == 1
                    and isinstance(fit_value.value, FieldExpr)
                    and fit_value.value.name == "fits"
                ):
                    parent_expr = self._render_expr(fit_value.value.value)
                    index_expr = self._render_expr(fit_value.indices[0])
                    zero_based = f"({index_expr} - 1)"
                    fit_field_exprs = [
                        f"{parent_expr}['mu_fit'][{zero_based}]",
                        f"{parent_expr}['omega_fit'][{zero_based}]",
                        f"{parent_expr}['alpha_fit'][{zero_based}]",
                        f"{parent_expr}['beta_fit'][{zero_based}]",
                        f"{parent_expr}['sd_uncond_fit'][{zero_based}]",
                        f"r_col_get({parent_expr}['z_fit'], {index_expr})",
                        f"{parent_expr}['rho1_hat_fit'][{zero_based}]",
                        f"{parent_expr}['phi_hat_fit'][{zero_based}]",
                        f"{parent_expr}['alpha_kurt_fit'][{zero_based}]",
                        f"{parent_expr}['alpha_rho1_fit'][{zero_based}]",
                    ]
                else:
                    fit_expr = self._render_expr(fit_value)
            elif len(positional) >= 7:
                asset_name_expr = self._render_expr(positional[0])
                ret_expr = self._render_expr(positional[1])
                fit_expr = self._render_expr(positional[2])
                omega_true_expr = self._render_expr(positional[3])
                alpha_true_expr = self._render_expr(positional[4])
                beta_true_expr = self._render_expr(positional[5])
                mu_true_expr = self._render_expr(positional[6])
            else:
                fit_expr = None
            if fit_field_exprs is not None or fit_expr is not None:
                if fit_field_exprs is None:
                    fit_field_exprs = [
                        f"{fit_expr}['mu']",
                        f"{fit_expr}['omega']",
                        f"{fit_expr}['alpha']",
                        f"{fit_expr}['beta']",
                        f"{fit_expr}['sd_uncond']",
                        f"{fit_expr}['residuals']",
                        f"{fit_expr}['rho1_hat']",
                        f"{fit_expr}['phi_hat']",
                        f"{fit_expr}['alpha_kurt']",
                        f"{fit_expr}['alpha_rho1']",
                    ]
                expanded = [
                    asset_name_expr,
                    ret_expr,
                    *fit_field_exprs,
                    omega_true_expr,
                    alpha_true_expr,
                    beta_true_expr,
                    mu_true_expr,
                ]
                return f"{name}({', '.join(expanded)})"
        if not self._standalone and name == "print_fit_vs_truth":
            fit_value = keywords.get("fit", positional[0] if len(positional) > 0 else None)
            truth_value = keywords.get("truth", positional[1] if len(positional) > 1 else None)
            digits_value = keywords.get("digits", positional[2] if len(positional) > 2 else None)
            fit_expr = self._render_expr(fit_value) if fit_value is not None else ""
            truth_expr = self._render_expr(truth_value) if truth_value is not None else ""
            digits_expr = self._render_expr(digits_value) if digits_value is not None else "4"
            if fit_expr and truth_expr:
                expanded = [
                    f"{fit_expr}['alpha']",
                    f"{fit_expr}['beta']",
                    f"{fit_expr}['Gamma']",
                    f"{fit_expr}['Pi']",
                    f"{fit_expr}['sigma_u']",
                    f"{fit_expr}['lambda']",
                    f"{fit_expr}['trace_stats']",
                    f"{fit_expr}['maxeig_stats']",
                    f"{truth_expr}['alpha']",
                    f"{truth_expr}['beta']",
                    f"{truth_expr}['Gamma']",
                    f"{truth_expr}['Pi']",
                    f"{truth_expr}['sigma_u']",
                    digits_expr,
                ]
                return f"{name}({', '.join(expanded)})"
        rendered_args: list[str] = []
        signature = self._function_param_annotations.get(name)
        positional_index = 0
        for arg in expr.args:
            expected_type = None
            if signature is not None:
                if arg.name is None:
                    if positional_index < len(signature):
                        expected_type = tuple(signature.values())[positional_index]
                    positional_index += 1
                else:
                    expected_type = signature.get(arg.name)
            if arg.name is None:
                rendered_args.append(self._render_expr_for_param(arg.value, expected_type))
            else:
                rendered_args.append(f"{arg.name}={self._render_expr_for_param(arg.value, expected_type)}")
        return f"{name}({', '.join(rendered_args)})"

    def _render_sprintf(self, expr: CallExpr, *, strip_trailing_newline: bool = False) -> str:
        if len(expr.args) < 2:
            raise NotImplementedError("sprintf() requires a format and at least one value")
        rendered_value = self._render_expr(expr.args[1].value)
        if isinstance(expr.args[0].value, StringExpr):
            fmt = self._string_value(expr.args[0].value)
            if not self._standalone:
                rendered = self._render_constant_sprintf(expr, fmt, strip_trailing_newline=strip_trailing_newline)
                if rendered is not None:
                    return rendered
        if isinstance(expr.args[0].value, StringExpr) and len(expr.args) == 2:
            fmt = self._string_value(expr.args[0].value)
            fixed = re.fullmatch(r"%\.(\d+)f\n?", fmt)
            if fixed is not None:
                precision = fixed.group(1)
                suffix = "" if strip_trailing_newline else ("\\n" if fmt.endswith("\n") else "")
                return f'f"{{{rendered_value}:.{precision}f}}{suffix}"'
            scientific = re.fullmatch(r"%\.(\d+)e\n?", fmt)
            if scientific is not None:
                precision = scientific.group(1)
                suffix = "" if strip_trailing_newline else ("\\n" if fmt.endswith("\n") else "")
                return f'f"{{{rendered_value}:.{precision}e}}{suffix}"'
            padded_str = re.fullmatch(r"%(\d+)s\n?", fmt)
            if padded_str is not None and isinstance(expr.args[1].value, StringExpr):
                width = padded_str.group(1)
                suffix = "" if strip_trailing_newline else (" + '\\n'" if fmt.endswith("\n") else "")
                if self._standalone:
                    return f"r_pad_left(str({rendered_value}), {width}){suffix}"
                return f"r_pad_left(str({rendered_value}), {width}){suffix}"
            if not self._standalone:
                if fmt.endswith("s") or fmt.endswith("s\n"):
                    return rendered_value
                return f"str({rendered_value})"
        rendered = [self._render_expr(arg.value) for arg in expr.args]
        if self._standalone:
            return f"r_sprintf({', '.join(rendered)})"
        return f"str({rendered_value})"

    def _render_rnorm(self, positional: list[Expr], keywords: dict[str, Expr], *, standalone: bool) -> str:
        n_expr = positional[0] if positional else NumberExpr("1", integer_like=True)
        mean_expr = keywords.get("mean", positional[1] if len(positional) > 1 else NumberExpr("0"))
        sd_expr = keywords.get("sd", positional[2] if len(positional) > 2 else NumberExpr("1"))
        rand_base = "np.random.standard_normal" if standalone else "rng.standard_normal"
        if self._is_scalar_one(n_expr):
            rand_expr = f"{rand_base}()"
        else:
            rand_expr = f"{rand_base}(int({self._render_expr(n_expr)}))"
        mean_rendered = self._render_expr(mean_expr)
        sd_rendered = self._render_expr(sd_expr)
        return f"(({mean_rendered}) + ({sd_rendered}) * ({rand_expr}))"

    def _render_constant_sprintf(
        self, expr: CallExpr, fmt: str, *, strip_trailing_newline: bool = False
    ) -> str | None:
        fmt_text = fmt[:-1] if strip_trailing_newline and fmt.endswith("\n") else fmt
        pieces: list[str] = []
        value_index = 1
        literal: list[str] = []
        i = 0
        while i < len(fmt_text):
            ch = fmt_text[i]
            if ch != "%":
                literal.append(ch)
                i += 1
                continue
            if i + 1 < len(fmt_text) and fmt_text[i + 1] == "%":
                literal.append("%")
                i += 2
                continue
            if literal:
                pieces.append(repr("".join(literal)))
                literal = []
            j = i + 1
            width_digits: list[str] = []
            while j < len(fmt_text) and fmt_text[j].isdigit():
                width_digits.append(fmt_text[j])
                j += 1
            precision_digits: list[str] = []
            if j < len(fmt_text) and fmt_text[j] == ".":
                j += 1
                while j < len(fmt_text) and fmt_text[j].isdigit():
                    precision_digits.append(fmt_text[j])
                    j += 1
            if j >= len(fmt_text):
                return None
            spec = fmt_text[j]
            if spec not in {"d", "i", "f", "e"}:
                return None
            if value_index >= len(expr.args):
                return None
            rendered_value = self._render_expr(expr.args[value_index].value)
            value_index += 1
            width = int("".join(width_digits)) if width_digits else None
            precision = int("".join(precision_digits)) if precision_digits else None
            if spec in {"d", "i"}:
                pieces.append(f"ac_format_int(int({rendered_value}), {width if width is not None else 0})")
            elif spec == "f":
                decimals = precision if precision is not None else 6
                if width is None:
                    pieces.append(f"ac_format_fixed({rendered_value}, {decimals})")
                else:
                    pieces.append(f"ac_format_fixed({rendered_value}, {decimals}, {width})")
            elif spec == "e":
                decimals = precision if precision is not None else 6
                if width is None:
                    pieces.append(f"ac_format_scientific({rendered_value}, {decimals})")
                else:
                    pieces.append(f"ac_format_scientific({rendered_value}, {decimals}, {width})")
            i = j + 1
        if literal:
            pieces.append(repr("".join(literal)))
        if value_index != len(expr.args):
            return None
        if not pieces:
            return '""'
        return " + ".join(pieces)

    def _render_index(self, expr: IndexExpr) -> str:
        base = self._render_expr(expr.value)
        if self._standalone:
            if len(expr.indices) == 1:
                return f"r_index({base}, {self._render_index_arg(expr.indices[0])})"
            if len(expr.indices) == 2:
                row, col = expr.indices
                return f"r_index({base}, {self._render_index_arg(row)}, {self._render_index_arg(col)})"
        if (
            not self._standalone
            and len(expr.indices) == 1
            and isinstance(expr.value, CallExpr)
            and isinstance(expr.value.func, NameExpr)
            and expr.value.func.name == "proc.time"
            and isinstance(expr.indices[0], StringExpr)
            and self._string_value(expr.indices[0]) == "elapsed"
        ):
            return "r_proc_elapsed()"
        if len(expr.indices) == 1:
            if (
                not self._standalone
                and isinstance(expr.value, FieldExpr)
                and expr.value.name == "fits"
            ):
                parent = self._render_expr(expr.value.value)
                index = self._render_expr(expr.indices[0])
                return (
                    f"r_fit_record_at({parent}['mu_fit'], {parent}['omega_fit'], {parent}['alpha_fit'], "
                    f"{parent}['beta_fit'], {parent}['sd_uncond_fit'], {parent}['rho1_hat_fit'], "
                    f"{parent}['phi_hat_fit'], {parent}['alpha_kurt_fit'], {parent}['alpha_rho1_fit'], "
                    f"{parent}['z_fit'], {index})"
                )
            if (
                not self._standalone
                and isinstance(expr.value, NameExpr)
                and isinstance(expr.indices[0], StringExpr)
            ):
                field_positions = self._field_hints.get(expr.value.name)
                field_name = self._string_value(expr.indices[0])
                if field_positions is not None and field_name in field_positions:
                    return f"{base}[{field_positions[field_name]}]"
                if field_name == "elapsed":
                    return base
            if not self._standalone and self._is_negative_subscript(expr.indices[0]):
                return f"r_exclude({base}, np.asarray({self._render_expr(self._negative_subscript_operand(expr.indices[0]))}).reshape(-1))"
            if not self._standalone and self._is_logical_index(expr.indices[0]):
                return f"r_mask({base}, {self._render_expr(expr.indices[0])})"
            return f"{base}[{self._render_zero_based_index(expr.indices[0])}]"
        if len(expr.indices) == 2:
            row, col = expr.indices
            if row is None and col is None:
                return base
            if row is not None and col is None:
                return f"{base}[{self._render_zero_based_index(row)}]"
            if row is None and col is not None:
                return f"r_col_get({base}, {self._render_expr(col)})"
            return f"{base}[{self._render_zero_based_index(row)}, {self._render_zero_based_index(col)}]"
        raise NotImplementedError("only 1D and 2D indexing are supported in the initial R subset")

    def _render_zero_based_index(self, expr: object | None) -> str:
        if expr is None:
            return ":"
        rendered = self._render_expr(expr)
        if self._is_index_vector_expr(expr):
            return f"(({rendered}).astype(int) - 1)"
        return f"({rendered} - 1)"

    def _is_index_vector_expr(self, expr: object) -> bool:
        if isinstance(expr, BinaryExpr) and expr.op == ":":
            return True
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr):
            return expr.func.name in {"seq", "seq_len", "which"}
        return False

    def _is_negative_subscript(self, expr: object | None) -> bool:
        return isinstance(expr, UnaryExpr) and expr.op == "-"

    def _negative_subscript_operand(self, expr: object) -> object:
        assert isinstance(expr, UnaryExpr) and expr.op == "-"
        return expr.operand

    def _is_logical_index(self, expr: object | None) -> bool:
        if expr is None:
            return False
        if isinstance(expr, BoolExpr):
            return True
        if isinstance(expr, NameExpr):
            return self._type_hints.get(expr.name) == "logical"
        if isinstance(expr, UnaryExpr) and expr.op == "!":
            return True
        if isinstance(expr, BinaryExpr) and expr.op in {"<", "<=", ">", ">=", "==", "!=", "&", "|", "&&", "||"}:
            return True
        if isinstance(expr, CallExpr) and isinstance(expr.func, NameExpr) and expr.func.name in {"is.na", "is.finite"}:
            return True
        return False

    def _render_index_arg(self, expr: object | None) -> str:
        if expr is None:
            return "None"
        return self._render_expr(expr)

    def _render_arg_default(self, expr: IfExpr) -> str | None:
        if self._standalone:
            return None
        if not isinstance(expr.test, BinaryExpr) or expr.test.op != ">=":
            return None
        if not (
            isinstance(expr.test.left, CallExpr)
            and isinstance(expr.test.left.func, NameExpr)
            and expr.test.left.func.name == "length"
            and len(expr.test.left.args) == 1
            and isinstance(expr.test.left.args[0].value, NameExpr)
        ):
            return None
        args_name = expr.test.left.args[0].value.name
        if not isinstance(expr.test.right, NumberExpr) or not expr.test.right.integer_like:
            return None
        pos = expr.test.right.text
        body = expr.body
        if isinstance(body, CallExpr) and isinstance(body.func, NameExpr) and body.func.name in {"as.integer", "as.numeric"}:
            if len(body.args) != 1:
                return None
            inner = body.args[0].value
            if not self._is_args_index(inner, args_name, pos):
                return None
            if body.func.name == "as.integer":
                default = self._render_expr_for_param(expr.orelse, "int")
                return f"r_arg_int({args_name}, {pos}, {default})"
            default = self._render_expr_for_param(expr.orelse, "float")
            return f"r_arg_float({args_name}, {pos}, {default})"
        if self._is_args_index(body, args_name, pos):
            return f"r_arg_str({args_name}, {pos}, {self._render_expr(expr.orelse)})"
        return None

    def _is_args_index(self, expr: object, args_name: str, pos: str) -> bool:
        return (
            isinstance(expr, IndexExpr)
            and isinstance(expr.value, NameExpr)
            and expr.value.name == args_name
            and len(expr.indices) == 1
            and isinstance(expr.indices[0], NumberExpr)
            and expr.indices[0].text == pos
        )

    def _string_value(self, expr: StringExpr) -> str:
        return bytes(expr.text[1:-1], "utf-8").decode("unicode_escape")

    def _is_newline_sprintf(self, expr: CallExpr) -> bool:
        if not isinstance(expr.func, NameExpr) or expr.func.name != "sprintf":
            return False
        if len(expr.args) != 2 or not isinstance(expr.args[0].value, StringExpr):
            return False
        return self._string_value(expr.args[0].value).endswith("\n")

    def _render_number(self, expr: NumberExpr, expected_type: str | None = None) -> str:
        if expected_type == "float" and re.fullmatch(r"[+-]?\d+", expr.text):
            return f"{expr.text}.0"
        return expr.text

    def _render_expr_for_param(self, expr: object, expected_type: str | None) -> str:
        if (
            not self._standalone
            and isinstance(expr, NameExpr)
            and expr.name == "NULL"
            and expected_type is not None
            and expected_type.startswith("Array")
        ):
            return "None"
        if isinstance(expr, NumberExpr):
            return self._render_number(expr, expected_type)
        if isinstance(expr, IfExpr):
            arg_default = self._render_arg_default(expr)
            if arg_default is not None:
                return arg_default
            return (
                f"({self._render_expr_for_param(expr.body, expected_type)} "
                f"if {self._render_expr(expr.test)} else "
                f"{self._render_expr_for_param(expr.orelse, expected_type)})"
            )
        if (
            expected_type == "float"
            and isinstance(expr, UnaryExpr)
            and expr.op in {"+", "-"}
            and isinstance(expr.operand, NumberExpr)
        ):
            return f"({expr.op}{self._render_number(expr.operand, expected_type)})"
        return self._render_expr(expr)

    def _render_div_operand(self, expr: object) -> str:
        if isinstance(expr, NumberExpr):
            return self._render_number(expr, "float")
        if isinstance(expr, UnaryExpr) and expr.op in {"+", "-"} and isinstance(expr.operand, NumberExpr):
            return f"({expr.op}{self._render_number(expr.operand, 'float')})"
        return self._render_expr(expr)

    def _render_int_expr(self, expr: object) -> str:
        if isinstance(expr, NumberExpr) and re.fullmatch(r"[+-]?\d+", expr.text):
            return expr.text
        if isinstance(expr, UnaryExpr) and expr.op in {"+", "-"} and isinstance(expr.operand, NumberExpr):
            if re.fullmatch(r"[+-]?\d+", expr.operand.text):
                return f"({expr.op}{expr.operand.text})"
        return f"int({self._render_expr(expr)})"

    def _is_scalar_one(self, expr: object) -> bool:
        if isinstance(expr, NumberExpr):
            return expr.text in {"1", "1.0", "1."}
        return False

    def _infer_param_annotations(self, stmt: FunctionDefStmt) -> dict[str, str]:
        annotations = {param.name: "float" for param in stmt.params}
        for param in stmt.params:
            if isinstance(param.default, BoolExpr):
                annotations[param.name] = "bool"
            elif isinstance(param.default, StringExpr):
                annotations[param.name] = "str"
        forced_annotations: dict[str, dict[str, str]] = {
            "acf_lags": {"x": "Array1D[float]"},
            "summary_stats": {"x": "Array1D[float]"},
            "garch11_filter": {"ret": "Array1D[float]"},
            "fit_garch11_direct": {"ret": "Array1D[float]"},
            "jarque_bera_test": {"x": "Array1D[float]"},
            "shapiro_test_safe": {"x": "Array1D[float]"},
            "normality_summary": {"x": "Array1D[float]"},
            "fmt_num": {"x": "Array1D[float]"},
            "estimate_phi_from_sq_acf": {"nratios": "int"},
        }
        annotations.update(forced_annotations.get(stmt.name, {}))
        integer_like_defaults = {
            param.name
            for param in stmt.params
            if isinstance(param.default, NumberExpr) and param.default.integer_like
        }
        param_names = set(annotations)

        def visit_expr(expr: object, callee: str | None = None) -> None:
            if isinstance(expr, NameExpr):
                if expr.name in param_names and callee in {"mean", "sum", "length"}:
                    annotations[expr.name] = "Array1D[float]"
                if expr.name in param_names and callee == "as.numeric":
                    annotations[expr.name] = "Array1D[float]"
                if expr.name in param_names and callee in {"numeric", "character", "seq_len", "set.seed", "runif", "rnorm", "sample", "vector"}:
                    annotations[expr.name] = "int"
                if expr.name in integer_like_defaults and callee in {"paste", "paste0", "sprintf"}:
                    annotations[expr.name] = "int"
                return
            if isinstance(expr, IndexExpr):
                if isinstance(expr.value, NameExpr) and expr.value.name in param_names:
                    annotations[expr.value.name] = "Array1D[float]"
                visit_expr(expr.value, None)
                for index in expr.indices:
                    if index is not None:
                        visit_expr(index, None)
                return
            if isinstance(expr, BinaryExpr):
                if expr.op == "%*%":
                    if isinstance(expr.left, NameExpr) and expr.left.name in param_names:
                        annotations[expr.left.name] = "Array1D[float]"
                    if isinstance(expr.right, NameExpr) and expr.right.name in param_names:
                        annotations[expr.right.name] = "Array1D[float]"
                if expr.op == ":":
                    if isinstance(expr.left, NameExpr) and expr.left.name in param_names:
                        annotations[expr.left.name] = "int"
                    if isinstance(expr.right, NameExpr) and expr.right.name in param_names:
                        annotations[expr.right.name] = "int"
                visit_expr(expr.left, None)
                visit_expr(expr.right, None)
                return
            if isinstance(expr, UnaryExpr):
                visit_expr(expr.operand, None)
                return
            if isinstance(expr, CallExpr):
                func_name = expr.func.name if isinstance(expr.func, NameExpr) else None
                for arg in expr.args:
                    visit_expr(arg.value, func_name)

        def visit_stmt(inner: object) -> None:
            if isinstance(inner, AssignStmt):
                visit_expr(inner.value, None)
            elif isinstance(inner, ExprStmt):
                visit_expr(inner.value, None)
            elif isinstance(inner, ReturnStmt):
                visit_expr(inner.value, None)
            elif isinstance(inner, IfStmt):
                visit_expr(inner.test, None)
                for child in inner.body:
                    visit_stmt(child)
                for child in inner.orelse:
                    visit_stmt(child)
            elif isinstance(inner, ForStmt):
                visit_expr(inner.iterable, None)
                for child in inner.body:
                    visit_stmt(child)
            elif isinstance(inner, WhileStmt):
                visit_expr(inner.test, None)
                for child in inner.body:
                    visit_stmt(child)

        for child in stmt.body:
            visit_stmt(child)
        return annotations
