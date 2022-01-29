# AST Tokenizing Unparser

[![Build Status](https://app.travis-ci.com/gonglinyuan/asttokunparse.svg?branch=main)](https://app.travis-ci.com/gonglinyuan/asttokunparse)

AST tokenizing unparser unparses a Python abstract syntax tree (AST) into tokens defined in the `tokenize` module.

It also has a utility dubbed `SpanUnparser`.  Given a user-defined function that maps an AST node to tags, the span unparser can wrap all AST nodes with the user-defined tags: one at the beginning and one at the end.

This library is compatible with Python 3.8 only.

Example usages:

```python
import ast
import asttokunparse


# Tokenizing unparser
asttokunparse.unparse(ast.parse("x = x + 1"))

# Expected output:
# [(1, 'x'), (54, '='), (54, '('), (1, 'x'), (54, '+'),
#  (2, '1'), (54, ')'), (4, '\n'), (0, '')]


# Span unparser
def func(node):
    if isinstance(node, ast.Name):
        return ["name"]
    else:
        return []

asttokunparse.span_unparse(ast.parse("x = x + 1"), func)

# Expected output: -1 is for the start tag, -2 for the end tag
# [(-1, 'name'), (1, 'x'), (-2, 'name'), (54, '='), (54, '('),
#  (-1, 'name'), (1, 'x'), (-2, 'name'), (54, '+'), (2, '1'),
#  (54, ')'), (4, '\n'), (0, '')]
```

