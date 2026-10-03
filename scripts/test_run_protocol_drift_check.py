# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_protocol_drift_check import OUTPUT, ROOT, apply_decisions, validate_decisions


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "previous_upstream_commit": "a" * 40,
            "candidate_upstream_commit": "b" * 40,
            "endpoints": ["/v1/chat/completions", "/v1/completions"],
            "changes": [
                {"id": "change-1", "status": "Unverified", "runtime_evidence": []}
            ],
        }
        self.decisions = {
            **self.report,
            "changes": {
                "change-1": {
                    "status": "Unverified",
                    "owner": "frontend",
                    "rationale": "New field needs runtime investigation",
                    "next_step": "Run a native-server probe",
                    "tracking_issue": "https://github.com/example/project/issues/1",
                }
            },
        }

    def test_exact_reviewed_inventory_accepted(self):
        validate_decisions(self.report, self.decisions)

    def test_malformed_shapes_are_reportable_validation_errors(self):
        for document in [
            None,
            [],
            True,
            "reviewed",
            {},
            {**self.decisions, "changes": []},
        ]:
            with self.subTest(document=document), self.assertRaises(ValueError):
                validate_decisions(self.report, document)
        for field, value in [
            ("status", []),
            ("owner", True),
            ("rationale", "  "),
            ("next_step", {}),
            ("tracking_issue", 7),
            ("decision", None),
            ("evidence", True),
            ("evidence", []),
            ("runtime_evidence", [""]),
            ("runtime_evidence", [{"claim": "passed"}]),
        ]:
            decisions = copy.deepcopy(self.decisions)
            decisions["changes"]["change-1"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_decisions(self.report, decisions)

    def test_stale_commit_missing_or_extra_change_rejected(self):
        for mutation in ["commit", "missing", "extra"]:
            decisions = copy.deepcopy(self.decisions)
            if mutation == "commit":
                decisions["candidate_upstream_commit"] = "c" * 40
            elif mutation == "missing":
                decisions["changes"] = {}
            else:
                decisions["changes"]["extra"] = decisions["changes"]["change-1"]
            with self.assertRaises(ValueError):
                validate_decisions(self.report, decisions)

    def test_compatibility_claim_requires_runtime_evidence(self):
        item = self.decisions["changes"]["change-1"]
        item["status"] = "Compatible"
        with self.assertRaisesRegex(ValueError, "runtime"):
            validate_decisions(self.report, self.decisions)
        item["runtime_evidence"] = ["tests/native-parity.json"]
        validate_decisions(self.report, self.decisions)

    def test_unresolved_change_requires_tracking(self):
        item = self.decisions["changes"]["change-1"]
        del item["tracking_issue"]
        item["decision"] = "No action"
        with self.assertRaisesRegex(ValueError, "tracking issue"):
            validate_decisions(self.report, self.decisions)

    def test_explicit_rejection_requires_evidence(self):
        item = self.decisions["changes"]["change-1"]
        item["status"] = "Unsupported"
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_decisions(self.report, self.decisions)

    def test_decision_cannot_overwrite_source_evidence(self):
        self.decisions["changes"]["change-1"]["path"] = ["unrelated"]
        with self.assertRaisesRegex(ValueError, "identity or payload"):
            validate_decisions(self.report, self.decisions)

    def non_material_decisions(self):
        return {
            **self.decisions,
            "changes": {
                "change-1": {
                    "material": False,
                    "scope_endpoints": self.report["endpoints"].copy(),
                    "owner": "frontend",
                    "rationale": "Added helper is used only by the excluded Responses API",
                    "next_step": "Review again if the endpoint scope expands",
                    "decision": "No action for chat/completions or completions",
                    "evidence": ["pinned-source-call-sites"],
                }
            },
        }

    def test_non_material_review_has_no_compatibility_claim(self):
        decisions = self.non_material_decisions()
        apply_decisions(self.report, decisions)
        item = self.report["changes"][0]
        self.assertIs(item["material"], False)
        self.assertNotIn("status", item)
        self.assertNotIn("runtime_evidence", item)
        self.assertEqual(item["id"], "change-1")

    def test_non_material_review_requires_scope_and_evidence(self):
        for field in (
            "scope_endpoints",
            "decision",
            "evidence",
            "rationale",
            "owner",
            "next_step",
        ):
            decisions = self.non_material_decisions()
            del decisions["changes"]["change-1"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_decisions(self.report, decisions)

    def test_non_material_cannot_smuggle_status_or_relax_scope(self):
        for field, value in [
            ("material", "false"),
            ("material", 0),
            ("material", None),
            ("status", "Compatible"),
            ("status", "Unverified"),
            ("runtime_evidence", ["native-parity"]),
            ("scope_endpoints", []),
            ("scope_endpoints", ["/v1/chat/completions"]),
            (
                "scope_endpoints",
                ["/v1/chat/completions", "/v1/completions", "/v1/responses"],
            ),
        ]:
            decisions = self.non_material_decisions()
            decisions["changes"]["change-1"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_decisions(self.report, decisions)
        expanded = copy.deepcopy(self.report)
        expanded["endpoints"].append("/v1/responses")
        with self.assertRaises(ValueError):
            validate_decisions(expanded, self.non_material_decisions())

    def test_invalid_decisions_do_not_partially_mutate_report(self):
        decisions = self.non_material_decisions()
        decisions["changes"]["extra"] = {"material": False}
        original = copy.deepcopy(self.report)
        with self.assertRaises(ValueError):
            apply_decisions(self.report, decisions)
        self.assertEqual(self.report, original)

    def test_material_decisions_keep_existing_requirements(self):
        self.decisions["changes"]["change-1"]["material"] = True
        apply_decisions(self.report, self.decisions)
        self.assertEqual(self.report["changes"][0]["status"], "Unverified")
        self.decisions["changes"]["change-1"]["scope_endpoints"] = self.report[
            "endpoints"
        ]
        with self.assertRaises(ValueError):
            validate_decisions(self.report, self.decisions)


class DriftDriverIntegrationTests(unittest.TestCase):
    def test_real_git_version_bump_report_and_triage_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            dynamo = root / "dynamo"

            def git(repo, *args):
                return subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "-c",
                        "commit.gpgsign=false",
                        "-c",
                        "tag.gpgsign=false",
                        "-c",
                        "core.hooksPath=/dev/null",
                        *args,
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip()

            for repo in (upstream, dynamo):
                repo.mkdir()
                git(repo, "init")
                git(repo, "config", "user.email", "fixture@example.invalid")
                git(repo, "config", "user.name", "Fixture")
            protocol = upstream / "vllm/entrypoints/openai/protocol.py"
            protocol.parent.mkdir(parents=True)
            source = "\n".join(
                f"class {name}(BaseModel):\n    value: int = 0\n"
                for name in (
                    "ChatCompletionRequest",
                    "CompletionRequest",
                    "ChatCompletionResponse",
                    "CompletionResponse",
                    "SamplingParams",
                )
            )
            protocol.write_text(source)
            git(upstream, "add", ".")
            git(upstream, "commit", "-m", "baseline")
            previous = git(upstream, "rev-parse", "HEAD")
            git(upstream, "tag", "v0.29.0")
            protocol.write_text(source.replace("value: int = 0", "value: int = 1", 1))
            git(upstream, "commit", "-am", "new default")
            candidate = git(upstream, "rev-parse", "HEAD")
            git(upstream, "tag", "v0.30.0")
            container = dynamo / "container/context.yaml"
            container.parent.mkdir()
            container.write_text("vllm:\n  cpu:\n    runtime_image_tag: v0.29.0\n")
            git(dynamo, "add", ".")
            git(dynamo, "commit", "-m", "baseline without inventory (initial rollout)")
            baseline = git(dynamo, "rev-parse", "HEAD")
            container.write_text("vllm:\n  cpu:\n    runtime_image_tag: v0.30.0\n")
            pins = dynamo / OUTPUT.relative_to(ROOT) / "vllm_pins.json"
            pins.parent.mkdir(parents=True)
            pins.write_text(
                json.dumps(
                    {
                        "versions": [
                            {
                                "version": "0.30.0",
                                "commit": candidate,
                                "platforms": ["cpu"],
                            }
                        ]
                    }
                )
            )
            command = [
                sys.executable,
                str(ROOT / "scripts/run_protocol_drift_check.py"),
                "--upstream-repo",
                str(upstream),
                "--dynamo-repo",
                str(dynamo),
                "--baseline-dynamo",
                baseline,
                "--require-triage",
            ]
            report_dir = root / "report"
            result = subprocess.run(
                command + ["--output-dir", str(report_dir)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(
                (report_dir / f"{previous}-{candidate}.json").read_text()
            )
            self.assertEqual(len(report["changes"]), 1)
            self.assertEqual(report["dynamo_commit"], baseline)
            self.assertEqual(report["platforms"], ["cpu"])
            self.assertTrue((report_dir / "summary.md").exists())
            provenance = report["provenance"]
            self.assertEqual(provenance["baseline_dynamo_commit"], baseline)
            self.assertEqual(
                provenance["current_inputs_sha256"],
                {
                    "container/context.yaml": hashlib.sha256(
                        container.read_bytes()
                    ).hexdigest(),
                    str(pins.relative_to(dynamo)): hashlib.sha256(
                        pins.read_bytes()
                    ).hexdigest(),
                },
            )
            self.assertIsNone(
                provenance["baseline_inputs_sha256"][str(pins.relative_to(dynamo))]
            )
            self.assertEqual(
                provenance["baseline_inputs_sha256"]["container/context.yaml"],
                hashlib.sha256(
                    b"vllm:\n  cpu:\n    runtime_image_tag: v0.29.0\n"
                ).hexdigest(),
            )
            tool_paths = {
                "scripts/protocol_drift.py",
                "scripts/run_protocol_drift_check.py",
                "scripts/check_protocol_pins.py",
                "scripts/generate_protocol_inventory.py",
            }
            self.assertEqual(set(provenance["tools_sha256"]), tool_paths)
            for path in tool_paths:
                self.assertEqual(
                    provenance["tools_sha256"][path],
                    hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                )
            self.assertEqual(provenance["python_version"], sys.version.split()[0])
            self.assertIsNone(provenance["decision_input"]["sha256"])
            decisions = {
                "previous_upstream_commit": previous,
                "candidate_upstream_commit": candidate,
                "changes": {
                    report["changes"][0]["id"]: {
                        "status": "Unverified",
                        "owner": "frontend",
                        "rationale": "Default changed",
                        "next_step": "Run probe",
                        "tracking_issue": "https://github.com/example/project/issues/1",
                    }
                },
            }
            decision_file = pins.parent / "decisions" / f"{previous}-{candidate}.json"
            decision_file.parent.mkdir()
            decision_file.write_text(json.dumps(decisions))
            result = subprocess.run(
                command + ["--output-dir", str(root / "reviewed")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reviewed = json.loads(
                (root / "reviewed" / f"{previous}-{candidate}.json").read_text()
            )
            self.assertEqual(
                reviewed["provenance"]["decision_input"],
                {
                    "path": str(decision_file.relative_to(dynamo)),
                    "sha256": hashlib.sha256(decision_file.read_bytes()).hexdigest(),
                },
            )

            # Invalid JSON/shape must retain the drift report for triage instead
            # of escaping before report/summary generation with a traceback.
            for index, malformed in enumerate(["{", "null", '{"changes": []}']):
                decision_file.write_text(malformed)
                output = root / f"malformed-{index}"
                result = subprocess.run(
                    command + ["--output-dir", str(output)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                invalid_report = json.loads(
                    (output / f"{previous}-{candidate}.json").read_text()
                )
                self.assertIn("invalid decision file", invalid_report["triage"])
                self.assertEqual(
                    invalid_report["provenance"]["decision_input"]["sha256"],
                    hashlib.sha256(malformed.encode()).hexdigest(),
                )
                self.assertTrue((output / "summary.md").exists())
                self.assertNotIn("Traceback", result.stderr)

            # Periodic checks and zero-diff reports need the same input identity.
            periodic = command.copy()
            mode_index = periodic.index("--baseline-dynamo")
            periodic[mode_index : mode_index + 2] = ["--upstream-candidate", candidate]
            result = subprocess.run(
                periodic + ["--output-dir", str(root / "periodic")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            unchanged = json.loads(
                (root / "periodic" / f"{candidate}-{candidate}.json").read_text()
            )
            self.assertEqual(unchanged["changes"], [])
            self.assertIsNone(unchanged["provenance"]["baseline_dynamo_commit"])
            self.assertEqual(
                unchanged["provenance"]["current_inputs_sha256"],
                provenance["current_inputs_sha256"],
            )

            # A later rollout has committed pins, unlike the initial baseline.
            git(
                dynamo,
                "add",
                str(container.relative_to(dynamo)),
                str(pins.relative_to(dynamo)),
            )
            git(dynamo, "commit", "-m", "version bump with source pins")
            pinned_baseline = git(dynamo, "rev-parse", "HEAD")
            pinned = command.copy()
            pinned[mode_index + 1] = pinned_baseline
            result = subprocess.run(
                pinned + ["--output-dir", str(root / "pinned-baseline")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pinned_report = json.loads(
                (root / "pinned-baseline" / f"{candidate}-{candidate}.json").read_text()
            )
            self.assertEqual(
                pinned_report["provenance"]["baseline_inputs_sha256"],
                provenance["current_inputs_sha256"],
            )

            # Standalone extraction must also identify the actual tool/interpreter.
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/protocol_drift.py"),
                    "--upstream-repo",
                    str(upstream),
                    "--dynamo-repo",
                    str(dynamo),
                    "--previous",
                    previous,
                    "--candidate",
                    candidate,
                    "--output-dir",
                    str(root / "standalone"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            standalone = json.loads((root / "standalone/report.json").read_text())
            self.assertEqual(
                standalone["provenance"],
                {
                    "python_version": sys.version.split()[0],
                    "tools_sha256": {
                        "scripts/protocol_drift.py": provenance["tools_sha256"][
                            "scripts/protocol_drift.py"
                        ]
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
