# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package/CLI regressions replacing legacy flat-script wiring checks."""

import ast
import hashlib
import subprocess
import sys
import unittest

from scripts.protocol_compatibility.common.paths import ROOT
from scripts.protocol_compatibility.common.provenance import tool_provenance


class PackageTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "scripts.protocol_compatibility", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_optional_investigation_layer_is_installed(self):
        package = ROOT / "scripts/protocol_compatibility"
        self.assertTrue((package / "extraction/dynamo_handling.py").exists())
        self.assertTrue((package / "assessment/upstream_changes.py").exists())
        self.assertIn("--no-investigation", self.invoke("assess", "--help").stdout)

    def test_all_public_commands_have_help(self):
        for command in (
            (),
            ("assess",),
            ("check-pins",),
            ("generate-inventory",),
            ("generate-release-fixtures",),
        ):
            with self.subTest(command=command):
                result = self.invoke(*command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_invalid_command_and_incompatible_revision_modes_are_usage_errors(self):
        for args in (
            (),
            ("unknown",),
            (
                "assess",
                "--dynamo-commit",
                "a" * 40,
                "--upstream-repo",
                ".",
                "--output-dir",
                "unused",
                "--platform",
                "cpu",
                "--upstream-commit",
                "b" * 40,
            ),
            (
                "assess",
                "--dynamo-commit",
                "a" * 40,
                "--upstream-repo",
                ".",
                "--output-dir",
                "unused",
                "--upstream-commit",
                "b" * 40,
                "--upstream-candidate",
                "c" * 40,
            ),
        ):
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_provenance_covers_nested_implementation_but_not_tests(self):
        provenance = tool_provenance()
        package = ROOT / "scripts/protocol_compatibility"
        expected = {
            path.relative_to(ROOT)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in package.rglob("*.py")
            if "tests" not in path.relative_to(package).parts
        }
        self.assertEqual(provenance["tools_sha256"], expected)
        self.assertEqual(provenance["python_version"], sys.version.split()[0])
        for part in (
            "inputs",
            "extraction",
            "assessment",
            "reporting",
            "generation",
            "common",
        ):
            self.assertTrue(any(f"/{part}/" in path for path in expected))
        self.assertIn("scripts/protocol_compatibility/__main__.py", expected)

    def test_lower_layers_do_not_import_workflow_or_rendering(self):
        package = ROOT / "scripts/protocol_compatibility"
        for directory in ("common", "extraction"):
            for path in (package / directory).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        parts = (node.module or "").split(".")
                        self.assertFalse(
                            {"assessment", "reporting", "generation"} & set(parts),
                            str(path),
                        )


if __name__ == "__main__":
    unittest.main()
