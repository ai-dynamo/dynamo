# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "routing_runner", Path(__file__).resolve().parents[1] / "run_image_test.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RoutingHarnessTests(unittest.TestCase):
    def test_changed_image_requires_reversed_colors_and_authoritative_usage(self):
        response = {"prompt_tokens": 82, "completion_tokens": 20}
        runner.validate_image_response(
            "Blue, then red.", response, range(19), ("blue", "red")
        )
        with self.assertRaisesRegex(AssertionError, "Expected"):
            runner.validate_image_response(
                "Red, then blue.", response, range(19), ("blue", "red")
            )
        with self.assertRaisesRegex(AssertionError, "usage"):
            runner.validate_image_response(
                "Blue, then red.",
                {"prompt_tokens": 19, "completion_tokens": 0},
                range(19),
                ("blue", "red"),
            )

    def test_an_unexecuted_or_ignored_routing_check_cannot_pass(self):
        for output in (
            "test result: ok. 0 passed; 0 failed; 0 ignored;",
            "test result: ok. 0 passed; 0 failed; 1 ignored;",
        ):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as tmp:
                with (
                    patch.object(
                        runner.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0, output),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(
                        RuntimeError, "did not execute exactly once"
                    ),
                ):
                    runner.run_test(Path("binary"), "event_conversion", {}, Path(tmp))

    def test_missing_routing_support_reports_a_failure_instead_of_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_image_test.py",
                "--scenario",
                "image-routing",
                "--binary",
                "unused",
                "--config",
                "unused",
                "--results",
                tmp,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(
                    runner,
                    "run",
                    side_effect=RuntimeError("missing routing capability"),
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 1)
            suite = ET.parse(Path(tmp) / "junit.xml").getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertEqual(suite.attrib["skipped"], "0")
            self.assertIn("image-routing", suite.attrib["name"])
