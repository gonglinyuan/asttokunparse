# coding: utf-8
import io
import tokenize

from .printer import Printer
from .unparser import Unparser, SpanUnparser

__version__ = '2.0.0'

SPAN_OPEN = SpanUnparser.SPAN_OPEN
SPAN_CLOSE = SpanUnparser.SPAN_CLOSE


def unparse(tree):
    v = []
    Unparser(tree, file=v)
    return v


def span_unparse(tree, node_to_span):
    v = []
    SpanUnparser(tree, v, node_to_span)
    return v


def dump(tree):
    v = io.StringIO()
    Printer(file=v).visit(tree)
    return v.getvalue()


def is_docstring(tokens, i):
    return (
        tokens[i][0] == tokenize.STRING
        and (i == 0 or tokens[i - 1][0] in [tokenize.INDENT, tokenize.NEWLINE])
        and (i + 1 < len(tokens) and tokens[i + 1][0] == tokenize.NEWLINE)
    )
