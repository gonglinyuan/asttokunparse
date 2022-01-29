import ast
import re
import unittest

import asttokunparse
from tests.common import AsttokunparseCommonTestCase


class DumpTestCase(AsttokunparseCommonTestCase, unittest.TestCase):

    def assertASTEqual(self, dump1, dump2):
        # undo the pretty-printing
        dump1 = re.sub(r"(?<=[\(\[])\n\s+", "", dump1)
        dump1 = re.sub(r"\n\s+", " ", dump1)
        self.assertEqual(dump1, dump2)

    def check_roundtrip(self, code1, filename="internal", mode="exec"):
        ast_ = compile(str(code1), filename, mode, ast.PyCF_ONLY_AST)
        dump1 = asttokunparse.dump(ast_)
        dump2 = ast.dump(ast_)
        self.assertASTEqual(dump1, dump2)
