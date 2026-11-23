# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks for the temporary Rust console report adapter; no Dynamo imports."""

import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from convert import convert_log, parse_log

HEADER = (
    "     Running unittests src/lib.rs (target/debug/deps/example-0123456789abcdef)\n"
)
PASS = (
    HEADER
    + "\nrunning 1 test\ntest module::works ... ok\n\n"
    + "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
    + "0 filtered out; finished in 0.01s\n"
)
SCRIPT = Path(__file__).with_name("convert.py")


class ConsoleReportTests(unittest.TestCase):
    def parse(self, text, group="root/unit"):
        return parse_log(text.splitlines(keepends=True), group)

    def test_pass_omits_unknown_test_duration(self):
        report, warnings = self.parse(PASS)
        self.assertEqual(warnings, [])
        suite = report.find("testsuite")
        self.assertEqual(suite.get("time"), "0.01")
        self.assertEqual(suite.get("name"), "root/unit/example/unittests src/lib.rs")
        self.assertIsNone(suite.find("testcase").get("time"))

    def test_failure_skip_and_expected_panic(self):
        text = (
            HEADER
            + "running 3 tests\n"
            + "test a - should panic ... ok\n"
            + "test b ... FAILED\n"
            + "test c ... ignored, needs a GPU\n"
            + "failures:\nthread 'b' panicked: expected <value> & got \\x00\n"
            + "test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; "
            + "0 filtered out; finished in 1.23s\n"
        )
        report, warnings = self.parse(text.replace("\\x00", "\x00"))
        self.assertEqual(warnings, [])
        cases = report.findall("testsuite/testcase")
        self.assertEqual(cases[0].get("name"), "a")
        self.assertIsNotNone(cases[1].find("failure"))
        self.assertIn("expected <value>", cases[1].find("failure").text)
        self.assertIsNotNone(cases[2].find("skipped"))
        # Special characters and invalid XML bytes cannot corrupt an upload.
        ET.fromstring(ET.tostring(report))

    def test_doctests_and_ansi(self):
        text = PASS.replace(HEADER, "   Doc-tests example\n").replace(
            "module::works", "src/lib.rs - docs (line 42)"
        )
        report, warnings = self.parse(text.replace("ok\n", "\x1b[32mok\x1b[0m\n"))
        self.assertEqual(warnings, [])
        self.assertEqual(report[0].get("name"), "root/unit/doctest/example")
        self.assertEqual(report[0][0].get("name"), "src/lib.rs - docs (line 42)")

    def test_ambiguous_or_mismatched_results_are_omitted(self):
        for text in (
            PASS.replace(
                "test module::works ... ok", "test module::works ... a log\nok"
            ),
            PASS.replace("1 passed; 0 failed", "0 passed; 1 failed"),
            PASS.replace("running 1 test", "running 2 tests"),
            PASS.replace(
                "test module::works ... ok", "test module::works ... ok\n" * 2
            ),
        ):
            with self.subTest(text=text):
                report, warnings = self.parse(text)
                self.assertEqual(len(report), 0)
                self.assertIn("Ambiguous", warnings[0])

    def test_crash_retains_only_preceding_complete_suite(self):
        report, warnings = self.parse(
            PASS + HEADER + "running 2 tests\ntest before_crash ... ok\nSIGSEGV\n"
        )
        self.assertEqual(len(report), 1)
        self.assertIn("Incomplete", warnings[0])

    def test_new_suite_exposes_truncation(self):
        report, warnings = self.parse(HEADER + "running 2 tests\n" + PASS)
        self.assertEqual(len(report), 1)
        self.assertIn("Incomplete", warnings[0])

    def test_unidentified_application_output_is_not_a_test_suite(self):
        report, _ = self.parse(PASS.replace(HEADER, ""))
        self.assertEqual(len(report), 0)

    def test_custom_benchmark_output_is_not_fabricated(self):
        report, _ = self.parse(HEADER + "Testing bench_a\nSuccess\n")
        self.assertEqual(len(report), 0)

    def test_single_thread_nocapture_output(self):
        text = (
            HEADER
            + "running 4 tests\n"
            + "test a ... test log\nok\n"
            + "test b ... stderr log\nok\n"
            + "test c ... \nthread 'c' panicked: failure\nFAILED\n"
            + "test d ... ignored\n"
            + "test result: FAILED. 2 passed; 1 failed; 1 ignored; 0 measured; "
            + "0 filtered out; finished in 0.00s\n"
        )
        report, warnings = parse_log(text.splitlines(keepends=True), "gpu", serial=True)
        self.assertEqual(warnings, [])
        self.assertEqual(len(report[0].findall("testcase")), 4)
        self.assertEqual(report[0].get("failures"), "1")
        parallel, warnings = self.parse(text)
        self.assertEqual(len(parallel), 0)
        self.assertIn("Ambiguous", warnings[0])

    def test_extra_outcome_lines_are_ambiguous(self):
        text = PASS.replace(
            "test module::works ... ok", "test module::works ... log\nok\nFAILED"
        )
        report, warnings = parse_log(text.splitlines(keepends=True), "gpu", serial=True)
        self.assertEqual(len(report), 0)
        self.assertIn("Ambiguous", warnings[0])

    def test_names_distinguish_features_and_binaries(self):
        first, _ = self.parse(PASS, "root/feature-a")
        second, _ = self.parse(PASS, "root/feature-b")
        self.assertNotEqual(first[0].get("name"), second[0].get("name"))
        first, _ = self.parse(PASS + PASS.replace("example-", "another-"))
        self.assertNotEqual(first[0].get("name"), first[1].get("name"))

    def test_zero_tests_produce_no_testcases(self):
        text = (
            PASS.replace("running 1 test", "running 0 tests")
            .replace("test module::works ... ok\n", "")
            .replace("1 passed", "0 passed")
        )
        report, warnings = self.parse(text)
        self.assertEqual(len(report), 0)
        self.assertEqual(warnings, [])

    def test_existing_log_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "unit.log"
            log_path.write_text(PASS)
            warnings = convert_log(log_path, "lib/example", serial=False)
            self.assertEqual(warnings, [])
            report = ET.parse(log_path.with_suffix(".xml")).getroot()
            self.assertTrue(report[0].get("name").startswith("lib/example/unit/"))
            self.assertEqual(log_path.read_text(), PASS)

    def test_incomplete_log_removes_stale_report(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "unit.log"
            log_path.write_text(HEADER + "running 1 test\n")
            stale = log_path.with_suffix(".xml")
            stale.write_text("stale")
            warnings = convert_log(log_path, "root", serial=False)
            self.assertFalse(stale.exists())
            self.assertTrue(any("Incomplete" in warning for warning in warnings))

    def test_test_step_failure_and_later_conversion_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "unit.log"
            # This test step finishes before the converter starts. Bash pipefail
            # retains the original test failure even though tee succeeds.
            environment = {
                **os.environ,
                "TEST_LOG": str(log_path),
                "PYTHON": sys.executable,
            }
            script = root / "failed_test.py"
            script.write_text(f"print({PASS!r})\nraise SystemExit(101)\n")
            environment["TEST_SCRIPT"] = str(script)
            completed = subprocess.run(
                [
                    "bash",
                    "-eo",
                    "pipefail",
                    "-c",
                    '"$PYTHON" "$TEST_SCRIPT" 2>&1 | tee "$TEST_LOG"',
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 101)
            self.assertFalse(log_path.with_suffix(".xml").exists())
            converted = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--log-dir",
                    directory,
                    "--workspace",
                    "root",
                ],
                env={**os.environ, "GITHUB_STEP_SUMMARY": str(root / "summary.md")},
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(converted.returncode, 0, converted.stderr)
            ET.parse(log_path.with_suffix(".xml"))
            self.assertIn("Per-test durations", (root / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
