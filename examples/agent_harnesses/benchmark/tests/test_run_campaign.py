# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_campaign.py"
MODULE_SPEC = importlib.util.spec_from_file_location("agent_load_campaign", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"could not load campaign wrapper from {MODULE_PATH}")
campaign = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(campaign)


FAKE_LOADGEN = """#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
invocation_log = pathlib.Path(os.environ["FAKE_LOADGEN_INVOCATIONS"])
with invocation_log.open("a", encoding="utf-8") as output:
    output.write(json.dumps(arguments) + "\\n")

if arguments == ["--version"]:
    print("agent-loadgen 0.1.0")
    raise SystemExit(0)

command = arguments[0]
output_dir = pathlib.Path(arguments[arguments.index("--output") + 1])
secret = os.environ.get("FAKE_LOADGEN_SECRET", "")
print(f"diagnostic-secret={secret}", file=sys.stderr)

if command == "plan":
    if os.environ.get("FAKE_LOADGEN_FAIL_PLAN") == "1":
        print(f"plan failed with {secret}", file=sys.stderr)
        raise SystemExit(23)
    output_dir.mkdir(parents=True)
    profile = pathlib.Path(arguments[arguments.index("--config") + 1])
    file_digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    profile_digest = hashlib.sha256((file_digest + "-semantic").encode()).hexdigest()
    scenario_digest = "1" * 64
    scenario = {
        "profile_digest_sha256": "0" * 64 if os.environ.get("FAKE_LOADGEN_MISMATCH_SCENARIO") == "1" else profile_digest,
        "scenario_digest_sha256": scenario_digest,
        "trace_manifest": {"source_digest_sha256": profile_digest},
    }
    (output_dir / "scenario.json").write_text(json.dumps(scenario) + "\\n", encoding="utf-8")
    (output_dir / "plan.dot").write_text("digraph smoke {}\\n", encoding="utf-8")
    print(json.dumps({"profile_digest_sha256": profile_digest, "requests": 4, "scenario_digest_sha256": scenario_digest, "echo": secret}))
    raise SystemExit(0)

if command == "generate":
    output_dir.mkdir(parents=True)
    performance_allowed = "--token-path-verified" in arguments and "--engine-cache-mode" in arguments
    summary = {
        "capacity_performance_conclusions_allowed": performance_allowed,
        "conclusion_blockers": [] if performance_allowed else ["token_path_unverified", "engine_cache_mode_undeclared"],
        "passed": True,
        "protocol_surface": "chat_completions",
        "request_count": 4,
        "run_id": "fake-run",
    }
    (output_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_dir / "requests.jsonl").write_text("{}\\n", encoding="utf-8")
    print(json.dumps(summary))
    raise SystemExit(0)

raise SystemExit(2)
"""


class CampaignTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.loadgen = self.root / "agent-loadgen"
        self.loadgen.write_text(FAKE_LOADGEN, encoding="utf-8")
        self.loadgen.chmod(0o755)
        self.loadgen_source = self.root / "agent-loadgen-source"
        self.loadgen_source.mkdir()
        self.profile = self.root / "profile.toml"
        self.profile.write_text(
            'schema_version = 4\nagent = "codex"\n', encoding="utf-8"
        )
        self.output_root = self.root / "output"
        self.invocations = self.root / "invocations.jsonl"
        self.secret = "Bearer test-secret-value"
        self.environment = mock.patch.dict(
            os.environ,
            {
                "FAKE_LOADGEN_INVOCATIONS": str(self.invocations),
                "FAKE_LOADGEN_SECRET": self.secret,
            },
        )
        self.source_state = mock.patch.object(
            campaign,
            "_loadgen_source_state",
            return_value=(campaign.PINNED_LOADGEN_COMMIT, False),
        )
        self.environment.start()
        self.source_state_mock = self.source_state.start()

    def tearDown(self) -> None:
        self.source_state.stop()
        self.environment.stop()
        self.temporary_directory.cleanup()

    def arguments(self, campaign_id: str = "test-campaign") -> list[str]:
        return [
            "--loadgen",
            str(self.loadgen),
            "--loadgen-source",
            str(self.loadgen_source),
            "--base-url",
            "https://dynamo.example.test",
            "--model",
            "served-model",
            "--tokenizer",
            "tokenizer-id",
            "--profile",
            str(self.profile),
            "--output-root",
            str(self.output_root),
            "--campaign-id",
            campaign_id,
            "--header",
            f"Authorization={self.secret}",
        ]

    def read_invocations(self) -> list[list[str]]:
        return [
            json.loads(line)
            for line in self.invocations.read_text(encoding="utf-8").splitlines()
        ]

    def test_transport_smoke_plans_before_generate_and_redacts_evidence(self) -> None:
        campaign_dir = campaign.run_campaign(self.arguments())

        invocations = self.read_invocations()
        self.assertEqual(invocations[0], ["--version"])
        self.assertEqual(invocations[1][0], "plan")
        self.assertEqual(invocations[2][0], "generate")
        metadata = json.loads(
            (campaign_dir / "campaign.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["classification"]["transport_passed"])
        self.assertFalse(
            metadata["classification"]["agent_loadgen_performance_eligible"]
        )
        self.assertFalse(metadata["classification"]["performance_qualified"])
        self.assertEqual(metadata["target"]["static_header_names"], ["Authorization"])
        self.assertEqual(metadata["plan"]["requests"], 4)
        self.assertNotEqual(
            metadata["profile"]["file_sha256"],
            metadata["profile"]["semantic_digest_sha256"],
        )
        for evidence_path in campaign_dir.rglob("*"):
            if evidence_path.is_file():
                self.assertNotIn(self.secret, evidence_path.read_text(encoding="utf-8"))
        self.assertIn("Authorization=<redacted>", json.dumps(metadata["commands"]))
        self.assertIn(
            "diagnostic-secret=<redacted>",
            (campaign_dir / "plan.stderr.log").read_text(encoding="utf-8"),
        )

    def test_existing_campaign_directory_is_not_overwritten(self) -> None:
        existing = self.output_root / "existing"
        existing.mkdir(parents=True)
        sentinel = existing / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(campaign.CampaignError, "will not be overwritten"):
            campaign.run_campaign(self.arguments("existing"))

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.invocations.exists())

    def test_failed_plan_blocks_generate_and_preserves_redacted_failure(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_LOADGEN_FAIL_PLAN": "1"}):
            with self.assertRaisesRegex(campaign.CampaignError, "plan failed"):
                campaign.run_campaign(self.arguments("failed-plan"))

        invocations = self.read_invocations()
        self.assertEqual(
            [invocation[0] for invocation in invocations], ["--version", "plan"]
        )
        campaign_dir = self.output_root / "failed-plan"
        metadata_text = (campaign_dir / "campaign.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(metadata_text)["status"], "failed")
        self.assertNotIn(self.secret, metadata_text)
        self.assertNotIn(
            self.secret, (campaign_dir / "plan.stderr.log").read_text(encoding="utf-8")
        )
        self.assertFalse((campaign_dir / "run").exists())

    def test_oversized_plan_blocks_generate(self) -> None:
        with self.assertRaisesRegex(
            campaign.CampaignError, "planned request count 4 exceeds"
        ):
            campaign.run_campaign(
                [*self.arguments("oversized-plan"), "--max-planned-requests", "3"]
            )

        invocations = self.read_invocations()
        self.assertEqual(
            [invocation[0] for invocation in invocations], ["--version", "plan"]
        )
        metadata = json.loads(
            (self.output_root / "oversized-plan" / "campaign.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["status"], "failed")

    def test_inconsistent_semantic_digest_blocks_generate(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_LOADGEN_MISMATCH_SCENARIO": "1"}):
            with self.assertRaisesRegex(
                campaign.CampaignError, "profile digest does not match plan output"
            ):
                campaign.run_campaign(self.arguments("mismatched-digest"))

        invocations = self.read_invocations()
        self.assertEqual(
            [invocation[0] for invocation in invocations], ["--version", "plan"]
        )

    def test_unpinned_source_is_rejected_before_binary_execution(self) -> None:
        self.source_state_mock.return_value = ("0" * 40, False)
        try:
            with self.assertRaisesRegex(
                campaign.CampaignError, "expected pinned commit"
            ):
                campaign.run_campaign(self.arguments("unpinned"))
        finally:
            self.source_state_mock.return_value = (
                campaign.PINNED_LOADGEN_COMMIT,
                False,
            )

        self.assertFalse(self.invocations.exists())

    def test_performance_measurement_requires_fidelity_declarations(self) -> None:
        with self.assertRaisesRegex(
            campaign.CampaignError, "requires --token-path-verified"
        ):
            campaign.run_campaign(
                [
                    *self.arguments("missing-token-gate"),
                    "--intent",
                    "performance-measurement",
                ]
            )
        with self.assertRaisesRegex(
            campaign.CampaignError, "requires at least one --engine-cache-mode"
        ):
            campaign.run_campaign(
                [
                    *self.arguments("missing-cache-gate"),
                    "--intent",
                    "performance-measurement",
                    "--token-path-verified",
                ]
            )
        self.assertFalse(self.invocations.exists())

    def test_performance_measurement_is_eligible_but_not_fully_qualified(self) -> None:
        campaign_dir = campaign.run_campaign(
            [
                *self.arguments("performance-candidate"),
                "--intent",
                "performance-measurement",
                "--token-path-verified",
                "--engine-cache-mode",
                "ownership=session",
            ]
        )

        metadata = json.loads(
            (campaign_dir / "campaign.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            metadata["classification"]["agent_loadgen_performance_eligible"]
        )
        self.assertFalse(metadata["classification"]["performance_qualified"])

    def test_base_url_must_be_the_service_root(self) -> None:
        arguments = self.arguments("bad-url")
        base_url_index = arguments.index("--base-url") + 1
        arguments[base_url_index] = "https://dynamo.example.test/v1"

        with self.assertRaisesRegex(campaign.CampaignError, "root service URL"):
            campaign.run_campaign(arguments)

        self.assertFalse(self.invocations.exists())


if __name__ == "__main__":
    unittest.main()
