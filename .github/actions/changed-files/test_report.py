# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the composite action's filename and coverage boundary."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ACTION_DIR = Path(__file__).resolve().parent
ACTION = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
REPORT_STEP = next(
    step
    for step in ACTION["runs"]["steps"]
    if step["name"] == "Report changes and check filter coverage"
)
FILTER_STEP = next(
    step for step in ACTION["runs"]["steps"] if step.get("id") == "filter"
)


class ChangedFilesTests(unittest.TestCase):
    def run_report(self, groups, *, extra_outputs=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "action output"
            output_dir.mkdir()
            for name, filenames in groups.items():
                # The pinned v42 setOutput first escapes JSON quotes, then its
                # file writer removes that layer. Include quotes/backslashes
                # in fixtures to exercise this exact round trip.
                raw = json.dumps(filenames, ensure_ascii=False)
                escaped = raw.replace('"', '\\"')
                file_data = escaped.replace('\\"', '"')
                (output_dir / f"{name}_all_modified_files.json").write_text(file_data)
            for name, value in (extra_outputs or {}).items():
                (output_dir / name).write_text(value)
            env = {
                **os.environ,
                "ACTION_PATH": str(ACTION_DIR),
                "CHANGED_FILES_DIR": str(output_dir),
                "BASE_SHA": "$(touch base-sha-injected)",
            }
            completed = subprocess.run(
                ["bash", "-e", "-c", REPORT_STEP["run"]],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(list(root.iterdir()), [output_dir], completed.stdout)
            return completed

    def test_action_uses_data_files_and_no_context_in_shell_source(self):
        settings = FILTER_STEP["with"]
        for option in ("json", "escape_json", "write_output_files"):
            self.assertEqual(settings[option], "true")
        self.assertEqual(settings["safe_output"], "false")
        self.assertEqual(settings["output_dir"], "${{ steps.output-dir.outputs.path }}")
        self.assertEqual(
            REPORT_STEP["env"]["CHANGED_FILES_DIR"],
            "${{ steps.output-dir.outputs.path }}",
        )
        for step in ACTION["runs"]["steps"]:
            self.assertNotIn("${{", step.get("run", ""), step["name"])

    def test_hostile_filenames_are_preserved_without_shell_execution(self):
        filenames = [
            "gyms/planner-gym/space in name.py",
            'gyms/planner-gym/double"quote.py',
            "gyms/planner-gym/single'quote.py",
            "gyms/planner-gym/back\\slash.py",
            "$(touch injected)",
            "`touch injected-backtick`",
            '"; touch injected-quote; #',
            "line\n::warning::injected workflow command",
            "tab\tname.py",
            "unicode-λ.py",
        ]
        result = self.run_report({"all": filenames, "planner_gym": filenames})
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in filenames:
            self.assertIn(json.dumps(filename), result.stdout)
        self.assertNotIn("\n::warning::", result.stdout)

    def test_new_filters_and_ignored_files_join_the_coverage_union(self):
        result = self.run_report(
            {
                "all": ["gym.py", "README.md", "future.py"],
                "planner_gym": ["gym.py"],
                "ignore": ["README.md"],
                "future_filter": ["future.py"],
            },
            extra_outputs={
                "changed_keys.json": '["planner_gym", "future_filter"]',
                "planner_gym_any_modified.txt": "true",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_whitespace_does_not_split_a_filename_for_coverage(self):
        result = self.run_report({"all": ["one two"], "planner_gym": ["one", "two"]})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('"one two"', result.stdout)

    def test_catch_all_cannot_hide_an_uncovered_filename(self):
        result = self.run_report({"all": ["unclaimed.py"], "planner_gym": []})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('"unclaimed.py"', result.stdout)

    def test_empty_change_set_passes(self):
        result = self.run_report({"all": [], "planner_gym": []})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_catch_all_output_fails(self):
        result = self.run_report({"planner_gym": ["gym.py"]})
        self.assertNotEqual(result.returncode, 0)

    def test_malformed_filename_array_fails(self):
        result = self.run_report({"all": ["gym.py"], "planner_gym": "gym.py"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected a JSON array", result.stderr)


if __name__ == "__main__":
    unittest.main()
