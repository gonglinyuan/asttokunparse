"Usage: unparse.py <path to source file>"
import ast
import io
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Callable

import astunparse

# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)

T_out = List[Tuple[int, str]]


def interleave(inter, f, seq):
    """Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)


class Unparser:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded. """

    def __init__(self, tree, file: T_out):
        """Unparser(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.f = file
        self.future_imports = []
        self._indent = 0
        self._just_changed_indent = True
        self.dispatch(tree)
        self.fill([(tokenize.ENDMARKER, "")])

    def fill(self, tokens: Optional[T_out] = None):
        "Indent a piece of text, according to the current indentation level"
        if tokens is None:
            tokens = []
        if self._just_changed_indent:
            self.write(tokens)
            self._just_changed_indent = False
        else:
            self.write([(tokenize.NEWLINE, "\n")] + tokens)

    def write(self, tokens: T_out):
        "Append a piece of text to the current line."
        self.f.extend(tokens)

    def enter(self):
        "Print ':', and increase the indentation."
        self.write([(tokenize.OP, ":")])
        if not self._just_changed_indent:
            self.write([(tokenize.NEWLINE, "\n")])
        self._indent += 1
        self._just_changed_indent = True
        self.write([(tokenize.INDENT, "    " * self._indent)])

    def leave(self):
        "Decrease the indentation level."
        if not self._just_changed_indent:
            self.write([(tokenize.NEWLINE, "\n")])
        self._indent -= 1
        self._just_changed_indent = True
        self.write([(tokenize.DEDENT, "")])

    def dispatch(self, tree):
        "Dispatcher function, dispatching tree type T to method _T."
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        meth = getattr(self, "_" + tree.__class__.__name__)
        meth(tree)

    ############### Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    ########################################################

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Interactive(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Expression(self, tree):
        self.dispatch(tree.body)
        self._just_changed_indent = False

    # stmt
    def _Expr(self, tree):
        self.fill()
        self.dispatch(tree.value)

    def _NamedExpr(self, tree):
        self.write([(tokenize.OP, "(")])
        self.dispatch(tree.target)
        self.write([(tokenize.OP, ":=")])
        self.dispatch(tree.value)
        self.write([(tokenize.OP, ")")])

    def _Import(self, t):
        self.fill([(tokenize.NAME, "import")])
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.names)

    def _ImportFrom(self, t):
        # A from __future__ import may affect unparsing, so record it.
        if t.module and t.module == '__future__':
            self.future_imports.extend(n.name for n in t.names)

        self.fill([(tokenize.NAME, "from")])
        self.write([(tokenize.OP, ".")] * t.level)
        if t.module:
            self._import_split_helper(t.module)
        self.write([(tokenize.NAME, "import")])
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.names)

    def _Assign(self, t):
        self.fill()
        for target in t.targets:
            self.dispatch(target)
            self.write([(tokenize.OP, "=")])
        self.dispatch(t.value)

    def _AugAssign(self, t):
        self.fill()
        self.dispatch(t.target)
        self.write([(tokenize.OP, self.binop[t.op.__class__.__name__] + "=")])
        self.dispatch(t.value)

    def _AnnAssign(self, t):
        self.fill()
        if not t.simple and isinstance(t.target, ast.Name):
            self.write([(tokenize.OP, "(")])
        self.dispatch(t.target)
        if not t.simple and isinstance(t.target, ast.Name):
            self.write([(tokenize.OP, ")")])
        self.write([(tokenize.OP, ":")])
        self.dispatch(t.annotation)
        if t.value:
            self.write([(tokenize.OP, "=")])
            self.dispatch(t.value)

    def _Return(self, t):
        self.fill([(tokenize.NAME, "return")])
        if t.value:
            self.dispatch(t.value)

    def _Pass(self, t):
        self.fill([(tokenize.NAME, "pass")])

    def _Break(self, t):
        self.fill([(tokenize.NAME, "break")])

    def _Continue(self, t):
        self.fill([(tokenize.NAME, "continue")])

    def _Delete(self, t):
        self.fill([(tokenize.NAME, "del")])
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.targets)

    def _Assert(self, t):
        self.fill([(tokenize.NAME, "assert")])
        self.dispatch(t.test)
        if t.msg:
            self.write([(tokenize.OP, ",")])
            self.dispatch(t.msg)

    def _Global(self, t):
        self.fill([(tokenize.NAME, "global")])
        interleave(
            lambda: self.write([(tokenize.OP, ",")]),
            lambda nm: self.write([(tokenize.NAME, nm)]),
            t.names
        )

    def _Nonlocal(self, t):
        self.fill([(tokenize.NAME, "nonlocal")])
        interleave(
            lambda: self.write([(tokenize.OP, ",")]),
            lambda nm: self.write([(tokenize.NAME, nm)]),
            t.names
        )

    def _Await(self, t):
        self.write([(tokenize.OP, "(")])
        self.write([(tokenize.NAME, "await")])
        if t.value:
            self.dispatch(t.value)
        self.write([(tokenize.OP, ")")])

    def _Yield(self, t):
        self.write([(tokenize.OP, "(")])
        self.write([(tokenize.NAME, "yield")])
        if t.value:
            self.dispatch(t.value)
        self.write([(tokenize.OP, ")")])

    def _YieldFrom(self, t):
        self.write([(tokenize.OP, "(")])
        self.write([(tokenize.NAME, "yield"), (tokenize.NAME, "from")])
        if t.value:
            self.dispatch(t.value)
        self.write([(tokenize.OP, ")")])

    def _Raise(self, t):
        self.fill([(tokenize.NAME, "raise")])
        if not t.exc:
            assert not t.cause
            return
        self.dispatch(t.exc)
        if t.cause:
            self.write([(tokenize.NAME, "from")])
            self.dispatch(t.cause)

    def _Try(self, t):
        self.fill([(tokenize.NAME, "try")])
        self.enter()
        self.dispatch(t.body)
        self.leave()
        for ex in t.handlers:
            self.dispatch(ex)
        if t.orelse:
            self.fill([(tokenize.NAME, "else")])
            self.enter()
            self.dispatch(t.orelse)
            self.leave()
        if t.finalbody:
            self.fill([(tokenize.NAME, "finally")])
            self.enter()
            self.dispatch(t.finalbody)
            self.leave()

    def _ExceptHandler(self, t):
        self.fill([(tokenize.NAME, "except")])
        if t.type:
            self.dispatch(t.type)
        if t.name:
            self.write([(tokenize.NAME, "as"), (tokenize.NAME, t.name)])
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _ClassDef(self, t):
        for deco in t.decorator_list:
            self.fill([(tokenize.OP, "@")])
            self.dispatch(deco)
        self.fill([(tokenize.NAME, "class"), (tokenize.NAME, t.name)])
        self.write([(tokenize.OP, "(")])
        comma = False
        for e in t.bases:
            if comma:
                self.write([(tokenize.OP, ",")])
            else:
                comma = True
            self.dispatch(e)
        for e in t.keywords:
            if comma:
                self.write([(tokenize.OP, ",")])
            else:
                comma = True
            self.dispatch(e)
        self.write([(tokenize.OP, ")")])
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _FunctionDef(self, t):
        self.__FunctionDef_helper(t, [(tokenize.NAME, "def")])

    def _AsyncFunctionDef(self, t):
        self.__FunctionDef_helper(t, [(tokenize.NAME, "async"), (tokenize.NAME, "def")])

    def __FunctionDef_helper(self, t, fill_suffix: T_out):
        for deco in t.decorator_list:
            self.fill([(tokenize.OP, "@")])
            self.dispatch(deco)
        def_str = fill_suffix + [(tokenize.NAME, t.name), (tokenize.OP, "(")]
        self.fill(def_str)
        self.dispatch(t.args)
        self.write([(tokenize.OP, ")")])
        if getattr(t, "returns", False):
            self.write([(tokenize.OP, "->")])
            self.dispatch(t.returns)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _For(self, t):
        self.__For_helper([(tokenize.NAME, "for")], t)

    def _AsyncFor(self, t):
        self.__For_helper([(tokenize.NAME, "async"), (tokenize.NAME, "for")], t)

    def __For_helper(self, fill: T_out, t):
        self.fill(fill)
        self.dispatch(t.target)
        self.write([(tokenize.NAME, "in")])
        self.dispatch(t.iter)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill([(tokenize.NAME, "else")])
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _If(self, t):
        self.fill([(tokenize.NAME, "if")])
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while (t.orelse and len(t.orelse) == 1 and
               isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            self.fill([(tokenize.NAME, "elif")])
            self.dispatch(t.test)
            self.enter()
            self.dispatch(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill([(tokenize.NAME, "else")])
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _While(self, t):
        self.fill([(tokenize.NAME, "while")])
        self.dispatch(t.test)
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill([(tokenize.NAME, "else")])
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _generic_With(self, t, async_=False):
        self.fill([(tokenize.NAME, "async"), (tokenize.NAME, "with")] if async_ else [(tokenize.NAME, "with")])
        if hasattr(t, 'items'):
            interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.items)
        else:
            self.dispatch(t.context_expr)
            if t.optional_vars:
                self.write([(tokenize.NAME, "as")])
                self.dispatch(t.optional_vars)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _With(self, t):
        self._generic_With(t)

    def _AsyncWith(self, t):
        self._generic_With(t, async_=True)

    def _JoinedStr(self, t):
        # JoinedStr(expr* values)
        string = io.StringIO()
        self._fstring_JoinedStr(t, string.write)
        # Deviation from `unparse.py`: Try to find an unused quote.
        # This change is made to handle _very_ complex f-strings.
        v = string.getvalue()
        if '\n' in v or '\r' in v:
            quote_types = ["'''", '"""']
        else:
            quote_types = ["'", '"', '"""', "'''"]
        for quote_type in quote_types:
            if quote_type not in v:
                v = "{quote_type}{v}{quote_type}".format(quote_type=quote_type, v=v)
                break
        else:
            v = repr(v)
        self.write([(tokenize.STRING, "f" + v)])

    def _FormattedValue(self, t):
        # FormattedValue(expr value, int? conversion, expr? format_spec)
        string = io.StringIO()
        self._fstring_JoinedStr(t, string.write)
        self.write([(tokenize.STRING, "f" + repr(string.getvalue()))])

    def _fstring_JoinedStr(self, t, write):
        for value in t.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, write)

    def _fstring_Str(self, t, write):
        value = t.s.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_Constant(self, t, write):
        assert isinstance(t.value, str)
        value = t.value.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_FormattedValue(self, t, write):
        write("{")
        expr = io.StringIO()
        astunparse.Unparser(t.value, expr)
        expr = expr.getvalue().rstrip("\n")
        if expr.startswith("{"):
            write(" ")  # Separate pair of opening brackets as "{ {"
        write(expr)
        if t.conversion != -1:
            conversion = chr(t.conversion)
            assert conversion in "sra"
            write("!{conversion}".format(conversion=conversion))
        if t.format_spec:
            write(":")
            meth = getattr(self, "_fstring_" + type(t.format_spec).__name__)
            meth(t.format_spec, write)
        write("}")

    def _Name(self, t):
        self.write([(tokenize.NAME, t.id)])

    def _Constant(self, t):
        if t.value is Ellipsis:
            self.write([(tokenize.OP, "...")])
        elif isinstance(t.value, (bool, type(None))):
            self.write([(tokenize.NAME, repr(t.value))])
        elif isinstance(t.value, (float, complex)):
            # Substitute overflowing decimal literal for AST infinities.
            self.write([(tokenize.NUMBER, repr(t.value).replace("inf", INFSTR))])
        elif isinstance(t.value, int):
            self.write([(tokenize.NUMBER, repr(t.value))])
        elif isinstance(t.value, (str, bytes)):
            if t.kind == "u":
                self.write([(tokenize.STRING, "u" + repr(t.value))])
            else:
                self.write([(tokenize.STRING, repr(t.value))])
        else:
            raise ValueError()

    def _List(self, t):
        self.write([(tokenize.OP, "[")])
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.elts)
        self.write([(tokenize.OP, "]")])

    def _ListComp(self, t):
        self.write([(tokenize.OP, "[")])
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write([(tokenize.OP, "]")])

    def _GeneratorExp(self, t):
        self.write([(tokenize.OP, "(")])
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write([(tokenize.OP, ")")])

    def _SetComp(self, t):
        self.write([(tokenize.OP, "{")])
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write([(tokenize.OP, "}")])

    def _DictComp(self, t):
        self.write([(tokenize.OP, "{")])
        self.dispatch(t.key)
        self.write([(tokenize.OP, ":")])
        self.dispatch(t.value)
        for gen in t.generators:
            self.dispatch(gen)
        self.write([(tokenize.OP, "}")])

    def _comprehension(self, t):
        if getattr(t, 'is_async', False):
            self.write([(tokenize.NAME, "async"), (tokenize.NAME, "for")])
        else:
            self.write([(tokenize.NAME, "for")])
        self.dispatch(t.target)
        self.write([(tokenize.NAME, "in")])
        self.dispatch(t.iter)
        for if_clause in t.ifs:
            self.write([(tokenize.NAME, "if")])
            self.dispatch(if_clause)

    def _IfExp(self, t):
        self.write([(tokenize.OP, "(")])
        self.dispatch(t.body)
        self.write([(tokenize.NAME, "if")])
        self.dispatch(t.test)
        self.write([(tokenize.NAME, "else")])
        self.dispatch(t.orelse)
        self.write([(tokenize.OP, ")")])

    def _Set(self, t):
        assert (t.elts)  # should be at least one element
        self.write([(tokenize.OP, "{")])
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.elts)
        self.write([(tokenize.OP, "}")])

    def _Dict(self, t):
        self.write([(tokenize.OP, "{")])

        def write_key_value_pair(k, v):
            self.dispatch(k)
            self.write([(tokenize.OP, ":")])
            self.dispatch(v)

        def write_item(item):
            k, v = item
            if k is None:
                # for dictionary unpacking operator in dicts {**{'y': 2}}
                # see PEP 448 for details
                self.write([(tokenize.OP, "**")])
                self.dispatch(v)
            else:
                write_key_value_pair(k, v)

        interleave(lambda: self.write([(tokenize.OP, ",")]), write_item, zip(t.keys, t.values))
        self.write([(tokenize.OP, "}")])

    def _Tuple(self, t):
        self.write([(tokenize.OP, "(")])
        if len(t.elts) == 1:
            elt = t.elts[0]
            self.dispatch(elt)
            self.write([(tokenize.OP, ",")])
        else:
            interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.elts)
        self.write([(tokenize.OP, ")")])

    unop_1 = {"Invert": "~", "UAdd": "+", "USub": "-"}
    unop_2 = {"Not": "not"}

    def _UnaryOp(self, t):
        self.write([(tokenize.OP, "(")])
        if t.op.__class__.__name__ in self.unop_1:
            self.write([(tokenize.OP, self.unop_1[t.op.__class__.__name__])])
        elif t.op.__class__.__name__ in self.unop_2:
            self.write([(tokenize.NAME, self.unop_2[t.op.__class__.__name__])])
        else:
            raise ValueError()
        self.dispatch(t.operand)
        self.write([(tokenize.OP, ")")])

    binop = {"Add": "+", "Sub": "-", "Mult": "*", "MatMult": "@", "Div": "/", "Mod": "%",
             "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^", "BitAnd": "&",
             "FloorDiv": "//", "Pow": "**"}

    def _BinOp(self, t):
        self.write([(tokenize.OP, "(")])
        self.dispatch(t.left)
        self.write([(tokenize.OP, self.binop[t.op.__class__.__name__])])
        self.dispatch(t.right)
        self.write([(tokenize.OP, ")")])

    cmpops_1 = {"Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">="}
    cmpops_2 = {"Is": ["is"], "IsNot": ["is", "not"], "In": ["in"], "NotIn": ["not", "in"]}

    def _Compare(self, t):
        self.write([(tokenize.OP, "(")])
        self.dispatch(t.left)
        for o, e in zip(t.ops, t.comparators):
            if o.__class__.__name__ in self.cmpops_1:
                self.write([(tokenize.OP, self.cmpops_1[o.__class__.__name__])])
            elif o.__class__.__name__ in self.cmpops_2:
                for nm in self.cmpops_2[o.__class__.__name__]:
                    self.write([(tokenize.NAME, nm)])
            else:
                raise ValueError()
            self.dispatch(e)
        self.write([(tokenize.OP, ")")])

    boolops = {ast.And: 'and', ast.Or: 'or'}

    def _BoolOp(self, t):
        self.write([(tokenize.OP, "(")])
        interleave(lambda: self.write([(tokenize.NAME, self.boolops[t.op.__class__])]), self.dispatch, t.values)
        self.write([(tokenize.OP, ")")])

    def _Attribute(self, t):
        self.dispatch(t.value)
        self.write([(tokenize.OP, ".")])
        self.write([(tokenize.NAME, t.attr)])

    def _Call(self, t):
        self.dispatch(t.func)
        self.write([(tokenize.OP, "(")])
        comma = False
        for e in t.args:
            if comma:
                self.write([(tokenize.OP, ",")])
            else:
                comma = True
            self.dispatch(e)
        for e in t.keywords:
            if comma:
                self.write([(tokenize.OP, ",")])
            else:
                comma = True
            self.dispatch(e)
        self.write([(tokenize.OP, ")")])

    def _Subscript(self, t):
        self.dispatch(t.value)
        self.write([(tokenize.OP, "[")])
        self.dispatch(t.slice)
        self.write([(tokenize.OP, "]")])

    def _Starred(self, t):
        self.write([(tokenize.OP, "*")])
        self.dispatch(t.value)

    def _Index(self, t):
        self.dispatch(t.value)

    def _Slice(self, t):
        if t.lower:
            self.dispatch(t.lower)
        self.write([(tokenize.OP, ":")])
        if t.upper:
            self.dispatch(t.upper)
        if t.step:
            self.write([(tokenize.OP, ":")])
            self.dispatch(t.step)

    def _ExtSlice(self, t):
        interleave(lambda: self.write([(tokenize.OP, ",")]), self.dispatch, t.dims)

    # argument
    def _arg(self, t):
        self.write([(tokenize.NAME, t.arg)])
        if t.annotation:
            self.write([(tokenize.OP, ":")])
            self.dispatch(t.annotation)

    # others
    def _arguments(self, t):
        first = True
        # normal arguments
        all_args = getattr(t, 'posonlyargs', []) + t.args
        defaults = [None] * (len(all_args) - len(t.defaults)) + t.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write([(tokenize.OP, ",")])
            self.dispatch(a)
            if d:
                self.write([(tokenize.OP, "=")])
                self.dispatch(d)
            if index == len(getattr(t, 'posonlyargs', ())):
                self.write([(tokenize.OP, ","), (tokenize.OP, "/")])

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        if t.vararg or getattr(t, "kwonlyargs", False):
            if first:
                first = False
            else:
                self.write([(tokenize.OP, ",")])
            self.write([(tokenize.OP, "*")])
            if t.vararg:
                self.write([(tokenize.NAME, t.vararg.arg)])
                if t.vararg.annotation:
                    self.write([(tokenize.OP, ":")])
                    self.dispatch(t.vararg.annotation)

        # keyword-only arguments
        if getattr(t, "kwonlyargs", False):
            for a, d in zip(t.kwonlyargs, t.kw_defaults):
                if first:
                    first = False
                else:
                    self.write([(tokenize.OP, ",")])
                self.dispatch(a),
                if d:
                    self.write([(tokenize.OP, "=")])
                    self.dispatch(d)

        # kwargs
        if t.kwarg:
            if first:
                first = False
            else:
                self.write([(tokenize.OP, ",")])
            self.write([(tokenize.OP, "**"), (tokenize.NAME, t.kwarg.arg)])
            if t.kwarg.annotation:
                self.write([(tokenize.OP, ":")])
                self.dispatch(t.kwarg.annotation)

    def _keyword_arg_helper(self, t):
        self.write([(tokenize.NAME, t.arg)])

    def _keyword(self, t):
        if t.arg is None:
            # starting from Python 3.5 this denotes a kwargs part of the invocation
            self.write([(tokenize.OP, "**")])
        else:
            self._keyword_arg_helper(t)
            self.write([(tokenize.OP, "=")])
        self.dispatch(t.value)

    def _Lambda(self, t):
        self.write([(tokenize.OP, "(")])
        self.write([(tokenize.NAME, "lambda")])
        self.dispatch(t.args)
        self.write([(tokenize.OP, ":")])
        self.dispatch(t.body)
        self.write([(tokenize.OP, ")")])

    def _import_split_helper(self, import_name: str):
        if import_name == "*":
            self.write([(tokenize.OP, "*")])
            return
        for tok in re.split(r"(\.)", import_name):
            if not tok:
                continue
            if tok.isidentifier():
                self.write([(tokenize.NAME, tok)])
            elif tok == '.':
                self.write([(tokenize.OP, tok)])
            else:
                raise ValueError()

    def _alias(self, t):
        self._import_split_helper(t.name)
        if t.asname:
            self.write([(tokenize.NAME, "as"), (tokenize.NAME, t.asname)])

    def _withitem(self, t):
        self.dispatch(t.context_expr)
        if t.optional_vars:
            self.write([(tokenize.NAME, "as")])
            self.dispatch(t.optional_vars)


class SpanUnparser(Unparser):
    SPAN_OPEN = -1
    SPAN_CLOSE = -2

    def __init__(self, tree, file, node_to_span: Callable[[ast.AST], List[str]]):
        self.node_to_span = node_to_span
        super().__init__(tree, file)

    def _keyword_arg_helper(self, t):
        span_types = self.node_to_span(t)
        for st in span_types:
            if st.startswith("keyword:"):
                self.write([(self.SPAN_OPEN, st[len("keyword:"):])])
        super()._keyword_arg_helper(t)
        for st in reversed(span_types):
            if st.startswith("keyword:"):
                self.write([(self.SPAN_CLOSE, st[len("keyword:"):])])

    def dispatch(self, tree):
        if isinstance(tree, list):
            super().dispatch(tree)
        elif isinstance(tree, ast.AST):
            span_types = self.node_to_span(tree)
            for st in span_types:
                if ":" not in st:
                    self.write([(self.SPAN_OPEN, st)])
            super().dispatch(tree)
            for st in reversed(span_types):
                if ":" not in st:
                    self.write([(self.SPAN_CLOSE, st)])
        else:
            raise ValueError()
