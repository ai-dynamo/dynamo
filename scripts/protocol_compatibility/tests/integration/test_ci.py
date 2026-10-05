# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline workflow wiring checks; not evidence that hosted scheduling is active."""

import ast
import json
import re
import subprocess
import unittest

import yaml

from scripts.protocol_compatibility.common.paths import ROOT


class AssessmentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(
            (ROOT / ".github/workflows/protocol-assessment.yml").read_text()
        )
        self.job = self.workflow["jobs"]["assess"]

    def test_changed_runtime_inputs_and_weekly_candidate_are_wired(self):
        # PyYAML's YAML 1.1 parser treats the Actions 'on' key as a boolean.
        triggers = self.workflow.get("on", self.workflow.get(True))
        self.assertIn("container/context.yaml", triggers["pull_request"]["paths"])
        self.assertIn("Cargo.lock", triggers["pull_request"]["paths"])
        self.assertIn("lib/llm/src/types.rs", triggers["pull_request"]["paths"])
        self.assertIn("lib/llm/src/protocols.rs", triggers["pull_request"]["paths"])
        self.assertTrue(triggers["schedule"])
        self.assertIn(
            "scripts/protocol_compatibility/**", triggers["pull_request"]["paths"]
        )
        tests = next(
            step["run"]
            for step in self.job["steps"]
            if step.get("name")
            == "Install tooling dependency and run source-only regression tests"
        )
        self.assertIn("discover -s scripts/protocol_compatibility/tests -t .", tests)
        candidate = json.loads(
            (
                ROOT
                / "lib/llm/src/protocols/openai/compatibility/assessment/candidate.json"
            ).read_text()
        )
        self.assertEqual(
            candidate["repository"], "https://github.com/vllm-project/vllm"
        )
        self.assertRegex(candidate["ref"], r"^refs/(heads|tags)/[A-Za-z0-9._/-]+$")

    def test_matrix_covers_configured_platforms_and_preserves_failed_reports(self):
        context = yaml.safe_load((ROOT / "container/context.yaml").read_text())
        platforms = {
            key
            for key, value in context["vllm"].items()
            if isinstance(value, dict) and "runtime_image_tag" in value
        }
        self.assertEqual(set(self.job["strategy"]["matrix"]["platform"]), platforms)
        uploads = [
            step
            for step in self.job["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["if"], "always()")
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})

    def test_compact_vocabulary_freshness_is_checked_without_inventory_json(self):
        script = next(
            step["run"]
            for step in self.job["steps"]
            if step.get("name")
            == "Check compact runtime vocabulary against configured pins"
        )
        self.assertIn("protocol_compatibility check-pins", script)
        self.assertIn("protocol_compatibility generate-inventory", script)
        self.assertIn("--check", script)
        self.assertNotIn("--inventory-output", script)

    def test_embedded_shell_and_python_parse_without_execution(self):
        for step in self.job["steps"]:
            if "run" not in step:
                continue
            with self.subTest(step=step["name"]):
                result = subprocess.run(
                    ["bash", "-n"],
                    input=step["run"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for body in re.findall(
                    r"<<'PY'[^\n]*\n(.*?)\nPY", step["run"], re.DOTALL
                ):
                    ast.parse(body)


if __name__ == "__main__":
    unittest.main()
