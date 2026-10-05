# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CLI/Git integration on synthetic source, not native runtime conformance.

No model, engine import, network or compilation is used. Set
DYNAMO_ASSESSMENT_TEST_ARTIFACTS to an unused output directory to retain fixture
repositories and actual CLI outputs as explicitly synthetic acceptance evidence.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from protocol_test_utils import fixture_git as git
from test_protocol_comparison import registry

SCRIPT = Path(__file__).with_name("assess_protocol_compatibility.py")
CHAT = "/v1/chat/completions"
NATIVE_PATH = "vllm/entrypoints/openai/protocol.py"
DYNAMO_PATH = "lib/llm/src/protocols/openai/fixture.rs"


def native_source(fields="    model: str\n    missing: int = 3", method=""):
    return (
        "from pydantic import BaseModel\n"
        f"class ChatCompletionRequest(BaseModel):\n{fields or '    pass'}\n{method}\n"
        "class CompletionRequest(BaseModel):\n    pass\n"
    )


def dynamo_source(fields="pub model: String", attributes=""):
    return (
        "#[derive(Deserialize)] "
        + attributes
        + f" struct NvCreateChatCompletionRequest {{ {fields} }}\n"
        + "#[derive(Deserialize)] struct NvCreateCompletionRequest {}\n"
    )


class AssessmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = self.root / "native"
        self.dynamo = self.root / "dynamo"
        for repo in (self.native, self.dynamo):
            repo.mkdir()
            git(repo, "init")
            git(repo, "config", "user.email", "fixture@example.invalid")
            git(repo, "config", "user.name", "Synthetic protocol fixture")
        self.write(self.dynamo, "Cargo.lock", "version = 3\npackage = []\n")
        self.write(self.native, NATIVE_PATH, native_source())
        self.write(self.dynamo, DYNAMO_PATH, dynamo_source())
        self.upstream_sha = self.commit(self.native)
        self.dynamo_sha = self.commit(self.dynamo)
        self.number = 0

    def tearDown(self):
        destination = os.environ.get("DYNAMO_ASSESSMENT_TEST_ARTIFACTS")
        if destination:
            shutil.copytree(self.root, Path(destination) / self._testMethodName)

    def write(self, repo, path, content):
        output = repo / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        return output

    def commit(self, repo):
        git(repo, "add", ".")
        git(repo, "commit", "-m", "synthetic assessment input")
        return git(repo, "rev-parse", "HEAD")

    def assess(self, *, expected=1, previous=None, decisions=None, extra=()):
        self.number += 1
        output = self.root / f"assessment-{self.number}"
        command = [
            sys.executable,
            str(SCRIPT),
            "--dynamo-repo",
            str(self.dynamo),
            "--dynamo-commit",
            self.dynamo_sha,
            "--upstream-repo",
            str(self.native),
            "--upstream-commit",
            self.upstream_sha,
            "--output-dir",
            str(output),
        ]
        if previous:
            command += ["--previous", str(previous / "report.json")]
        if decisions is not None:
            path = self.write(
                self.root, f"decisions-{self.number}.json", json.dumps(decisions)
            )
            command += ["--decisions", str(path)]
        command += list(extra)
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.write(
            self.root,
            f"command-{self.number}.json",
            json.dumps(
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "synthetic": True,
                },
                indent=2,
            ),
        )
        self.assertEqual(process.returncode, expected, process.stderr + process.stdout)
        self.assertNotIn("Traceback", process.stderr)
        if expected == 2:
            self.assertIn("Assessment tool error", process.stderr)
            self.assertFalse((output / "report.json").exists())
            return output, None
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["exit_code"], expected)
        self.assertEqual(
            report["revisions"], {"dynamo": self.dynamo_sha, "vllm": self.upstream_sha}
        )
        self.assertEqual(
            report["gates"]["runtime_conformance"],
            "not_established_by_static_assessment",
        )
        for artifact in (
            "report.md",
            "native-contract.json",
            "dynamo-contract.json",
            "native-source.json",
        ):
            self.assertTrue((output / artifact).is_file())
        return output, report

    def get_finding(self, report, path, aspect):
        return next(
            item
            for item in report["findings"]
            if item["identity"] == f"{CHAT}#{path}:{aspect}"
        )

    def test_baseline_existing_gap_review_and_dynamo_only_invalidation(self):
        original_pin = self.write(
            self.dynamo, "pins.json", '{"candidate_adoption": false}\n'
        ).read_bytes()
        folder, first = self.assess()
        self.assertEqual(
            self.get_finding(first, "missing", "handling")["lifecycle"], "new"
        )
        decisions = registry(first, carry_static_disposition=True)
        _, reviewed = self.assess(previous=folder, decisions=decisions)
        self.assertEqual(reviewed["gates"]["review"], "complete")
        self.assertEqual(reviewed["gates"]["extraction"], "incomplete")
        self.assertTrue(
            all(item["lifecycle"] == "unchanged" for item in reviewed["findings"])
        )
        self.write(self.dynamo, DYNAMO_PATH, dynamo_source("pub model: Option<String>"))
        self.dynamo_sha = self.commit(self.dynamo)
        _, changed = self.assess(previous=folder, decisions=decisions)
        self.assertEqual(changed["previous_revisions"], first["revisions"])
        self.assertEqual(changed["revisions"]["vllm"], first["revisions"]["vllm"])
        self.assertEqual(
            self.get_finding(changed, "model", "required")["lifecycle"], "new"
        )
        self.assertEqual(
            self.get_finding(changed, "model", "handling")["decision_status"],
            "stale_facts",
        )
        self.assertEqual(
            self.get_finding(changed, "missing", "handling")["decision_status"],
            "applicable",
        )
        self.assertEqual((self.dynamo / "pins.json").read_bytes(), original_pin)
        self.assertEqual(git(self.dynamo, "status", "--porcelain"), "")
        markdown = (folder / "report.md").read_text()
        for text in (
            first["revisions"]["dynamo"],
            self.upstream_sha,
            "missing",
            "Next action",
            "incomplete",
        ):
            self.assertIn(text, markdown)

    def test_nested_alias_change_and_coverage_loss_not_resolved(self):
        self.write(
            self.native,
            "vllm/payload.py",
            "from pydantic import BaseModel\nclass Payload(BaseModel):\n    n: int = 3\nAlias = list[Payload]\n",
        )
        self.write(
            self.native,
            NATIVE_PATH,
            "from vllm.payload import Alias\n" + native_source("    payload: Alias"),
        )
        self.write(self.dynamo, DYNAMO_PATH, dynamo_source("pub payload: String"))
        self.upstream_sha = self.commit(self.native)
        self.dynamo_sha = self.commit(self.dynamo)
        folder, first = self.assess()
        initial = self.get_finding(first, "payload", "wire_type")
        self.assertEqual(initial["category"], "compatibility")
        self.write(
            self.native,
            "vllm/payload.py",
            "from pydantic import BaseModel\nclass Payload(BaseModel):\n    n: str = 'three'\nAlias = list[Payload]\n",
        )
        self.upstream_sha = self.commit(self.native)
        _, changed = self.assess(
            previous=folder, decisions=registry(first, carry_static_disposition=True)
        )
        self.assertEqual(
            self.get_finding(changed, "payload", "wire_type")["lifecycle"], "changed"
        )
        self.assertEqual(
            self.get_finding(changed, "payload", "wire_type")["decision_status"],
            "stale_facts",
        )
        self.write(self.native, NATIVE_PATH, native_source("    payload: ExternalType"))
        self.upstream_sha = self.commit(self.native)
        _, unknown = self.assess(previous=folder)
        self.assertEqual(
            self.get_finding(unknown, "payload", "wire_type")["category"], "coverage"
        )
        self.write(
            self.dynamo,
            DYNAMO_PATH,
            dynamo_source(attributes='#[serde(from = "Dynamic")]'),
        )
        self.dynamo_sha = self.commit(self.dynamo)
        _, lost = self.assess(previous=folder)
        retired = next(
            item
            for item in lost["retired_findings"]
            if item["identity"] == initial["identity"]
        )
        self.assertEqual(retired["lifecycle"], "no_longer_assessable")

    def test_added_removed_fields_and_behavior_only_change(self):
        folder, first = self.assess()
        self.write(
            self.native,
            NATIVE_PATH,
            native_source("    model: str\n    added: bool = False"),
        )
        self.upstream_sha = self.commit(self.native)
        _, changed = self.assess(previous=folder)
        self.assertEqual(
            self.get_finding(changed, "added", "handling")["lifecycle"], "new"
        )
        removed = next(
            item for item in changed["retired_findings"] if item["path"] == "missing"
        )
        self.assertEqual(removed["lifecycle"], "resolved")
        self.write(
            self.native,
            NATIVE_PATH,
            native_source(
                method="    def validate_model(self):\n        return self.model.strip()"
            ),
        )
        self.upstream_sha = self.commit(self.native)
        _, behavior = self.assess(previous=folder)
        self.assertTrue(
            any(item["category"] == "behavior" for item in behavior["findings"])
        )
        self.assertEqual(
            first["contracts"]["native"]["endpoints"],
            behavior["contracts"]["native"]["endpoints"],
        )

    def test_empty_scope_zero_diff_does_not_claim_runtime_or_release(self):
        self.write(self.native, NATIVE_PATH, native_source(""))
        self.write(self.dynamo, DYNAMO_PATH, dynamo_source(""))
        self.upstream_sha = self.commit(self.native)
        self.dynamo_sha = self.commit(self.dynamo)
        _, report = self.assess(expected=0)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["gates"]["release"]["status"], "not_evaluated")

    def test_missing_request_root_is_coverage_not_success_or_tool_failure(self):
        self.write(
            self.native, NATIVE_PATH, "class CompletionRequest(BaseModel):\n    pass\n"
        )
        self.upstream_sha = self.commit(self.native)
        _, report = self.assess()
        self.assertEqual(report["gates"]["extraction"], "incomplete")
        self.assertTrue(
            any(
                item["category"] == "coverage" and item["endpoint"] == CHAT
                for item in report["findings"]
            )
        )

    def test_workflow_connects_version_pins_and_checks_candidate_without_adoption(self):
        pins_path = "lib/llm/src/protocols/openai/compatibility/vllm_pins.json"

        def configure(version, sha):
            self.write(
                self.dynamo,
                "container/context.yaml",
                f"vllm:\n  cpu:\n    runtime_image_tag: v{version}\n",
            )
            self.write(
                self.dynamo,
                pins_path,
                json.dumps(
                    {
                        "format_version": 1,
                        "target": "vllm",
                        "versions": [
                            {"version": version, "commit": sha, "platforms": ["cpu"]}
                        ],
                    }
                ),
            )
            return self.commit(self.dynamo)

        old_native = self.upstream_sha
        before = configure("0.29.0", old_native)
        self.write(
            self.native,
            NATIVE_PATH,
            native_source("    model: str\n    missing: int = 4"),
        )
        candidate = self.commit(self.native)
        after = configure("0.30.0", candidate)
        command = [
            sys.executable,
            str(SCRIPT.with_name("run_protocol_assessment.py")),
            "--dynamo-repo",
            str(self.dynamo),
            "--dynamo-commit",
            after,
            "--upstream-repo",
            str(self.native),
            "--platform",
            "cpu",
        ]
        output = self.root / "workflow-bump"
        result = subprocess.run(
            command + ["--baseline-dynamo", before, "--output-dir", str(output)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads((output / "current/report.json").read_text())
        self.assertEqual(
            report["previous_revisions"], {"dynamo": before, "vllm": old_native}
        )
        self.assertEqual(report["revisions"], {"dynamo": after, "vllm": candidate})
        self.assertTrue((output / "report.md").is_file())
        pins_before = (self.dynamo / pins_path).read_bytes()
        periodic = self.root / "workflow-candidate"
        result = subprocess.run(
            command
            + ["--upstream-candidate", old_native, "--output-dir", str(periodic)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        metadata = json.loads((periodic / "workflow.json").read_text())
        self.assertFalse(metadata["candidate_adopted"])
        self.assertEqual(metadata["configured_pin"]["commit"], candidate)
        self.assertEqual(metadata["candidate_commit"], old_native)
        self.assertEqual((self.dynamo / pins_path).read_bytes(), pins_before)
        self.assertEqual(git(self.dynamo, "status", "--porcelain"), "")

    def test_invalid_inputs_and_scope_changes_are_tool_errors(self):
        folder, report = self.assess()
        for bad in (
            None,
            [],
            True,
            {},
            {"schema": "dynamo-native-decisions/v1", "decisions": [None]},
        ):
            with self.subTest(bad=bad):
                path = self.write(self.root, "malformed.json", json.dumps(bad))
                self.assess(expected=2, extra=("--decisions", str(path)))
        for field, bad in (
            ("carry_static_disposition", "false"),
            ("evidence", ["approved"]),
            ("reviewed_revisions", {}),
        ):
            decision = registry(report)
            decision["decisions"][0][field] = bad
            with self.subTest(field=field):
                self.assess(expected=2, decisions=decision)
        self.assess(expected=2, previous=folder, extra=("--pipeline", "text"))
        altered = copy.deepcopy(report)
        altered["findings"].append(altered["findings"][0])
        path = self.write(self.root, "invalid-previous.json", json.dumps(altered))
        self.assess(expected=2, extra=("--previous", str(path)))

    def test_invalid_upstream_syntax_is_tool_error_not_review_required(self):
        self.write(self.native, NATIVE_PATH, "class ChatCompletionRequest(:\n")
        self.upstream_sha = self.commit(self.native)
        self.assess(expected=2)

    def test_malformed_configured_pins_are_clean_workflow_errors(self):
        pin_path = "lib/llm/src/protocols/openai/compatibility/vllm_pins.json"
        self.write(self.dynamo, "container/context.yaml", "vllm: {}\n")
        for value in ([], {"format_version": 1, "target": "vllm", "versions": [None]}):
            with self.subTest(value=value):
                self.write(self.dynamo, pin_path, json.dumps(value))
                sha = self.commit(self.dynamo)
                process = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT.with_name("run_protocol_assessment.py")),
                        "--dynamo-repo",
                        str(self.dynamo),
                        "--dynamo-commit",
                        sha,
                        "--upstream-repo",
                        str(self.native),
                        "--platform",
                        "cpu",
                        "--output-dir",
                        str(self.root / "invalid-workflow"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertIn("Assessment workflow error", process.stderr)
                self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
