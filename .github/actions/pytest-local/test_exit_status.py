# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the action's optional empty-partition handling."""

import os
import subprocess
import textwrap
import unittest
from pathlib import Path


class ExitStatusTest(unittest.TestCase):
    def test_only_opted_in_no_tests_is_accepted(self):
        action = Path(__file__).with_name("action.yml").read_text()
        block = action.split("        TEST_EXIT_CODE=$?\n", 1)[1].split(
            "        # Back to fail-fast:", 1
        )[0]
        script = textwrap.dedent(block) + '\nprintf "%s" "$TEST_EXIT_CODE"\n'
        for allow in ("true", "false", ""):
            for code in range(7):
                with self.subTest(allow=allow, code=code):
                    result = subprocess.run(
                        ["bash", "-c", script],
                        env={
                            **os.environ,
                            "TEST_EXIT_CODE": str(code),
                            "ALLOW_NO_TESTS": allow,
                        },
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    )
                    expected = 0 if code == 5 and allow == "true" else code
                    self.assertEqual(result.stdout.splitlines()[-1], str(expected))

    def test_only_sequential_remainder_opts_in(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / "workflows/shared-test.yml").read_text()
        self.assertEqual(workflow.count("allow_no_tests:"), 1)
        sequential = workflow.split("      - name: Run GPU tests (sequential)", 1)[1]
        self.assertIn("allow_no_tests:", sequential)
        self.assertIn("inputs.run_gpu_parallel_tests && 'true' || 'false'", sequential)


if __name__ == "__main__":
    unittest.main()
