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
from report import OPERATOR_ONLY_FILES

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
            github_output = root / "github-output"
            env = {
                **os.environ,
                "GITHUB_OUTPUT": str(github_output),
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
            completed.outputs = (
                dict(
                    line.split("=", 1)
                    for line in github_output.read_text().splitlines()
                )
                if github_output.exists()
                else {}
            )
            if github_output.exists():
                github_output.unlink()
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

    def test_runtime_admission_requires_verified_isolated_modifications(self):
        # Regression: uncertain status or a shared input could silently omit an
        # affected backend's tests; exercise the real action output boundary.
        path = "components/src/dynamo/frontend/tests/test_vllm_processor_unit.py"
        status = {
            f"all_{name}_files.json": "[]"
            for name in (
                "added",
                "copied",
                "deleted",
                "renamed",
                "type_changed",
                "unmerged",
                "unknown",
            )
        }
        status["all_modified_files.json"] = json.dumps([path])
        status["all_all_changed_and_modified_files.json"] = json.dumps([path])
        cases = [([path], status, "false"), ([], status, "true")]
        for extra in (
            "unknown.py",
            "components/src/dynamo/common/utils.py",
            ".github/workflows/pr.yaml",
            "container/context.yaml",
        ):
            cases.append(([path, extra], status, "true"))
        for name in status:
            for value in (None, "{", "[17]", json.dumps([path])):
                changed = dict(status)
                if value is None:
                    del changed[name]
                else:
                    changed[name] = value
                if changed == status:
                    continue
                cases.append(([path], changed, "true"))
        for files, outputs, expected in cases:
            with self.subTest(files=files, outputs=outputs):
                result = self.run_report(
                    {"all": files, "core": files}, extra_outputs=outputs
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.outputs["sglang_runtime"], expected)
                self.assertEqual(result.outputs["trtllm_runtime"], "true")
                self.assertEqual(result.outputs["vllm_runtime"], "true")

    def test_operator_admission_retains_unlisted_and_mixed_inputs(self):
        # Regression: additions in the audited operator class were rejected;
        # incomplete/conflicting status must still never omit runtime coverage.
        files = sorted(OPERATOR_ONLY_FILES)
        outputs = {
            f"all_{name}_files.json": "[]"
            for name in (
                "added",
                "modified",
                "copied",
                "deleted",
                "renamed",
                "type_changed",
                "unmerged",
                "unknown",
            )
        }
        outputs["all_added_files.json"] = json.dumps(files[:8])
        outputs["all_modified_files.json"] = json.dumps(files[8:])
        outputs["all_all_changed_and_modified_files.json"] = json.dumps(files)
        cases = [(files, outputs, "false")]
        for added in ([], files):
            changed = dict(outputs)
            changed["all_added_files.json"] = json.dumps(added)
            changed["all_modified_files.json"] = json.dumps(
                sorted(set(files) - set(added))
            )
            cases.append((files, changed, "false"))
        for extra in (
            "deploy/operator/api/v1beta2/unreviewed.go",
            "components/src/dynamo/frontend/tests/test_vllm_processor_unit.py",
            "components/src/dynamo/common/utils.py",
        ):
            changed = dict(outputs)
            changed["all_modified_files.json"] = json.dumps(files[8:] + [extra])
            changed["all_all_changed_and_modified_files.json"] = json.dumps(
                files + [extra]
            )
            cases.append((files + [extra], changed, "true"))
        for name in outputs:
            for value in (None, "{", "[17]", json.dumps([files[0]])):
                changed = dict(outputs)
                if value is None:
                    del changed[name]
                else:
                    changed[name] = value
                cases.append((files, changed, "true"))
        # Complete accounting also rejects a path listed as both A and M.
        changed = dict(outputs)
        changed["all_modified_files.json"] = json.dumps(files)
        cases.append((files, changed, "true"))
        cases.append((files[8:], outputs, "true"))
        for changed_files, status, expected in cases:
            with self.subTest(files=changed_files, status=status):
                result = self.run_report(
                    {"all": changed_files, "deploy": changed_files},
                    extra_outputs=status,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.outputs["sglang_runtime"], expected)
                self.assertEqual(result.outputs["trtllm_runtime"], expected)
                self.assertEqual(result.outputs["vllm_runtime"], expected)

    def test_runtime_gate_defaults_full_and_supports_force_full(self):
        workflow = yaml.safe_load(
            (ACTION_DIR.parents[1] / "workflows/pr.yaml").read_text()
        )
        jobs = workflow["jobs"]
        for backend, value, force, expected in (
            ("sglang", "false", "", False),
            ("sglang", "false", "true", True),
            ("sglang", "", "", True),
            ("sglang", "true", "", True),
            ("trtllm", "false", "", False),
            ("trtllm", "false", "true", True),
            ("trtllm", "", "", True),
            ("trtllm", "true", "", True),
            ("vllm", "false", "", False),
            ("vllm", "false", "true", True),
            ("vllm", "", "", True),
            ("vllm", "true", "", True),
        ):
            output = jobs["changed-files"]["outputs"][f"{backend}_runtime"]
            expression = output.removeprefix("${{").removesuffix("}}")
            expression = expression.replace("vars.FORCE_FULL_CI", repr(force))
            expression = expression.replace(
                f"steps.changes.outputs.{backend}_runtime", repr(value)
            )
            required = eval(expression.replace("||", "or"), {"__builtins__": {}})
            for job in (f"{backend}-test", f"{backend}-multi-gpu-test"):
                condition = jobs[job]["if"].replace(
                    f"needs.changed-files.outputs.{backend}_runtime",
                    repr(str(required).lower()),
                )
                for name in (
                    "core",
                    "sglang",
                    "trtllm",
                    "vllm",
                    "sample",
                    "deploy",
                    "run_multigpu_tests",
                ):
                    condition = condition.replace(
                        f"needs.changed-files.outputs.{name}", repr("true")
                    )
                expression = (
                    " ".join(condition.split()).replace("&&", "and").replace("||", "or")
                )
                self.assertEqual(eval(expression, {"__builtins__": {}}), expected)

    def test_trt_router_selection_is_limited_to_the_audited_change_class(self):
        # Regression: a loose path match could add unrelated nightly cases or
        # lose the launcher case; check the actual action's selection output.
        paths = [
            "examples/backends/trtllm/launch/disagg.sh",
            "examples/backends/trtllm/launch/disagg_router.sh",
            "tests/serve/test_trtllm.py",
        ]
        cases = [([path], "true") for path in paths] + [(paths, "true")]
        cases += [
            ([], "false"),
            ([paths[0], "tests/serve/common.py"], "false"),
            ([paths[0], "examples/common/gpu_utils.sh"], "false"),
            (["examples/backends/trtllm/launch/disagg_same_gpu.sh"], "false"),
        ]
        for files, expected in cases:
            status = {
                f"all_{name}_files.json": "[]"
                for name in (
                    "added",
                    "copied",
                    "deleted",
                    "renamed",
                    "type_changed",
                    "unmerged",
                    "unknown",
                )
            }
            status["all_modified_files.json"] = json.dumps(files)
            status["all_all_changed_and_modified_files.json"] = json.dumps(files)
            for outputs, result_expected in ((status, expected), ({}, "false")):
                with self.subTest(files=files, outputs=outputs):
                    result = self.run_report(
                        {"all": files, "trtllm": files}, extra_outputs=outputs
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.outputs["trtllm_disagg_router"], result_expected
                    )

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
