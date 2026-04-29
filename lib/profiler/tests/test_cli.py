# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests."""

import os
import subprocess
import sys
import tempfile
import unittest


DEMO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sysprofile-demo-output"
)

PROFILER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class TestCLIHelp(unittest.TestCase):
    def test_main_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "dynamo_profiler", "--help"],
            capture_output=True, text=True, cwd=PROFILER_DIR,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("report", result.stdout)
        self.assertIn("nsys-convert", result.stdout)
        self.assertIn("analyze", result.stdout)

    def test_report_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "dynamo_profiler", "report", "--help"],
            capture_output=True, text=True, cwd=PROFILER_DIR,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--merge-result", result.stdout)


@unittest.skipUnless(
    os.path.isdir(DEMO_DIR), "sysprofile-demo-output not found"
)
class TestCLIReport(unittest.TestCase):
    def test_report_from_demo_output(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "dynamo_profiler", "report",
                 DEMO_DIR, "--output", out_path],
                capture_output=True, text=True, cwd=PROFILER_DIR,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path) as f:
                html = f.read()
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("View A", html)
        finally:
            os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()
