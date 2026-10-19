# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.protocol_compatibility.generation.release_fixtures import (
    excerpt,
    sha256,
    writer_evidence,
)


class ReleaseFixtureExtractionTests(unittest.TestCase):
    def test_exact_function_lines_not_ast_reformatted(self):
        source = "# preamble\ndef build(x):\n    # keep comment\n    return  x\n"
        actual = excerpt(source, "build")
        self.assertEqual(actual["source"], source.split("\n", 1)[1])
        self.assertEqual(actual["line"], 2)
        self.assertEqual(actual["sha256"], sha256(actual["source"]))

    def test_missing_or_duplicate_declaration_fails(self):
        for source in ("", "def build(): pass\ndef build(): pass\n"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                excerpt(source, "build")

    def test_decorators_require_review(self):
        with self.assertRaisesRegex(ValueError, "Decorated"):
            excerpt("@decorate\ndef build(): pass\n", "build")

    def test_annotated_literal_constant(self):
        actual = excerpt('KEY: Final = "key"\n', "KEY")
        self.assertEqual(actual["source"], 'KEY: Final = "key"\n')

    def test_writer_literal_keys(self):
        source = (
            "    fn sampling_passthrough_args<R>() {\n"
            '        for key in ["first", "second",] { use_key(key); }\n'
            "    }\n    fn backend_extra_args<R>() {}\n"
        )
        self.assertEqual(writer_evidence(source)["keys"], ["first", "second"])
        with self.assertRaises(ValueError):
            writer_evidence(source.replace('["first", "second",]', "KEYS"))


if __name__ == "__main__":
    unittest.main()
