"""Normalized semantic IR for array-oriented code.

This is intentionally small at the start. The first milestone is to stabilize
the abstractions needed for a reusable Fortran backend before migrating any
existing frontend wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScalarType(str, Enum):
    INTEGER = "integer"
    REAL64 = "real64"
    COMPLEX128 = "complex128"
    LOGICAL = "logical"
    STRING = "string"


@dataclass(frozen=True)
class RecordTypeRef:
    name: str


@dataclass(frozen=True)
class ArrayTypeRef:
    element_type: object
    rank: int = 1


class BinaryOperator(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "**"
    MOD = "%"
    AND = ".and."
    OR = ".or."


class UnaryOperator(str, Enum):
    PLUS = "+"
    MINUS = "-"
    NOT = "not"


class CompareOperator(str, Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


@dataclass(frozen=True)
class ValueRef:
    name: str


@dataclass(frozen=True)
class Constant:
    value: object


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple[object, ...]


@dataclass(frozen=True)
class FieldAccess:
    value: object
    field: str


@dataclass(frozen=True)
class IndexAccess:
    value: object
    index: object


@dataclass(frozen=True)
class ConditionalExpr:
    test: object
    body: object
    orelse: object


@dataclass(frozen=True)
class ListComprehension:
    target: str
    iterable: object
    body: object


@dataclass(frozen=True)
class RecordLiteral:
    type_name: str
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class BinaryOp:
    left: object
    op: BinaryOperator
    right: object


@dataclass(frozen=True)
class UnaryOp:
    op: UnaryOperator
    operand: object


@dataclass(frozen=True)
class Compare:
    left: object
    op: CompareOperator
    right: object


@dataclass(frozen=True)
class BooleanOp:
    op: str
    values: tuple[object, ...]


@dataclass(frozen=True)
class Assignment:
    target: ValueRef
    value: object


@dataclass(frozen=True)
class FieldAssignment:
    target: object
    field: str
    value: object


@dataclass(frozen=True)
class IndexAssignment:
    target: object
    index: object
    value: object


@dataclass(frozen=True)
class UnpackAssignment:
    targets: tuple[ValueRef, ...]
    value: object


@dataclass(frozen=True)
class AugmentedAssignment:
    target: ValueRef
    op: BinaryOperator
    value: object


@dataclass(frozen=True)
class Append:
    target: ValueRef
    value: object


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class Break:
    pass


@dataclass(frozen=True)
class Comment:
    text: str


@dataclass(frozen=True)
class Pass:
    pass


@dataclass(frozen=True)
class Return:
    value: object | None = None


@dataclass(frozen=True)
class Raise:
    message: str


@dataclass(frozen=True)
class Print:
    values: tuple[object, ...]


@dataclass(frozen=True)
class ExprStatement:
    expr: object


@dataclass(frozen=True)
class If:
    test: object
    body: tuple[object, ...]
    orelse: tuple[object, ...] = ()


@dataclass(frozen=True)
class ForRange:
    target: str
    stop: object
    body: tuple[object, ...]
    start: object | None = None
    step: object | None = None


@dataclass(frozen=True)
class While:
    test: object
    body: tuple[object, ...]


@dataclass
class Function:
    name: str
    args: list[tuple[str, object]] = field(default_factory=list)
    locals: list[tuple[str, object]] = field(default_factory=list)
    body: list[object] = field(default_factory=list)
    result_type: object | None = None
    leading_comments: list[str] = field(default_factory=list)


@dataclass
class RecordDef:
    name: str
    fields: list[tuple[str, object]] = field(default_factory=list)


@dataclass
class Program:
    name: str
    locals: list[tuple[str, object]] = field(default_factory=list)
    body: list[object] = field(default_factory=list)
    leading_comments: list[str] = field(default_factory=list)
