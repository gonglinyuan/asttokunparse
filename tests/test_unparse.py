import ast
import tokenize
import unittest
from io import BytesIO

import astunparse

import asttokunparse
from tests.common import AsttokunparseCommonTestCase


class UnparseTestCase(AsttokunparseCommonTestCase, unittest.TestCase):

    def assertASTEqual(self, ast1, ast2):
        self.assertEqual(ast.dump(ast1), ast.dump(ast2))

    def check_roundtrip(self, code1, filename="internal", mode="exec"):
        ast1 = compile(str(code1), filename, mode, ast.PyCF_ONLY_AST)
        code2 = astunparse.unparse(ast1)
        tok1 = asttokunparse.unparse(ast1)
        tok1 = [
            (toktype, tok, asttokunparse.is_docstring(tok1, i))
            for i, (toktype, tok) in enumerate(tok1)
        ]
        tok2 = [
            (tok.type, tok.string, tok.type == tokenize.STRING and tok.string == tok.line.strip())
            for tok in tokenize.tokenize(BytesIO(code2.encode('utf-8')).readline)
            if tok.type not in [tokenize.ENCODING, tokenize.NL]
        ]
        self.assertEqual(tok1, tok2)
