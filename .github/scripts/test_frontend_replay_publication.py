# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication must select this run's image or report incomplete evaluation."""

import hashlib
import http.client
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "frontend_replay_publication",
    Path(__file__).with_name("frontend_replay_publication.py"),
)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.descriptor_path = self.directory / "publication.json"
        self.target = self.directory / "target.json"
        self.metadata_path = self.directory / "run.json"
        self.platform_digest = "sha256:" + "a" * 64
        self.manifest = {
            "manifests": [
                {
                    "platform": {"os": "linux", "architecture": "amd64"},
                    "digest": self.platform_digest,
                }
            ]
        }
        self.body = json.dumps(self.manifest)
        self.index_digest = "sha256:" + hashlib.sha256(self.body.encode()).hexdigest()
        self.environment = {
            "SOURCE_SHA": "1234567" + "0" * 33,
            "NGC_VERSION_TAG": "20261109-1234567",
            "GITHUB_RUN_ID": "42",
            "GITHUB_RUN_ATTEMPT": "1",
            "FRONTEND_STAGED": "true",
            "STAGING_REFERENCE": "private.invalid/secret/image:tag",
        }
        self.metadata = {
            "id": 42,
            "run_attempt": 1,
            "head_sha": self.environment["SOURCE_SHA"],
            "repository": {"full_name": "ai-dynamo/dynamo"},
            "head_repository": {"full_name": "ai-dynamo/dynamo"},
            "path": ".github/workflows/nightly-ci.yml",
            "name": "Nightly CI Pipeline",
            "head_branch": "main",
            "event": "schedule",
            "status": "completed",
            "pull_requests": [],
            "conclusion": "failure",
        }
        publication.write_json(self.metadata_path, self.metadata)

    def record(self):
        commands = [
            subprocess.CompletedProcess([], 0, self.index_digest + "\n", ""),
            subprocess.CompletedProcess([], 0, self.body, ""),
        ]
        with patch.dict(os.environ, self.environment, clear=True), patch.object(
            publication.subprocess, "run", side_effect=commands
        ) as crane:
            publication.record(self.descriptor_path)
        self.assertEqual(
            crane.call_args_list[1].args[0],
            ["crane", "manifest", "private.invalid/secret/image@" + self.index_digest],
        )

    def test_success_resolves_only_matching_public_platform_digest(self):
        self.record()
        with patch.object(
            publication,
            "public_manifest",
            return_value=(self.index_digest, self.manifest),
        ) as fetch:
            self.assertEqual(
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/amd64",
                    0,
                    self.metadata_path,
                ),
                0,
            )
        fetch.assert_called_once_with("20261109-1234567")
        target = json.loads(self.target.read_text())
        self.assertEqual(
            target["image"], publication.PUBLIC_REPOSITORY + "@" + self.platform_digest
        )
        self.assertNotIn("private.invalid", self.descriptor_path.read_text())

    def test_not_staged_never_queries_an_older_floating_image(self):
        with patch.dict(
            os.environ, {**self.environment, "FRONTEND_STAGED": "false"}, clear=True
        ), patch.object(publication.subprocess, "run") as crane:
            publication.record(self.descriptor_path)
        crane.assert_not_called()
        with patch.object(publication, "public_manifest") as fetch:
            self.assertEqual(
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/amd64",
                    0,
                    self.metadata_path,
                ),
                2,
            )
        fetch.assert_not_called()
        self.assertEqual(
            json.loads(self.target.read_text())["status"], "target_unavailable"
        )
        self.assertEqual(
            json.loads(self.target.read_text())["reason"], "frontend_not_staged"
        )

    def test_staging_errors_do_not_persist_the_secret_reference(self):
        failure = subprocess.CalledProcessError(
            1,
            ["crane", "digest", self.environment["STAGING_REFERENCE"]],
            stderr="credentials and private staging reference",
        )
        with patch.dict(os.environ, self.environment, clear=True), patch.object(
            publication.subprocess, "run", side_effect=failure
        ):
            publication.record(self.descriptor_path)
        result = self.descriptor_path.read_text()
        self.assertNotIn("private", result)
        self.assertNotIn("credentials", result)
        self.assertEqual(json.loads(result)["reason"], "staged_digest_unavailable")

    def test_mismatched_publication_is_incomplete_not_a_fallback(self):
        self.record()
        with patch.object(
            publication,
            "public_manifest",
            return_value=("sha256:" + "b" * 64, self.manifest),
        ):
            self.assertEqual(
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/amd64",
                    0,
                    self.metadata_path,
                ),
                2,
            )
        self.assertEqual(
            json.loads(self.target.read_text())["reason"],
            "public_digest_differs_from_staging",
        )

    def test_missing_platform_cannot_be_substituted(self):
        self.record()
        with patch.object(
            publication,
            "public_manifest",
            return_value=(self.index_digest, self.manifest),
        ), patch.object(
            publication.time,
            "sleep",
            side_effect=AssertionError("immutable platform mismatch must not poll"),
        ):
            self.assertEqual(
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/arm64",
                    7200,
                    self.metadata_path,
                ),
                2,
            )

    def test_staging_digest_failure_reason_is_preserved(self):
        with patch.dict(os.environ, self.environment, clear=True), patch.object(
            publication.subprocess, "run", side_effect=OSError("unavailable")
        ):
            publication.record(self.descriptor_path)
        self.assertEqual(
            publication.resolve(
                self.descriptor_path, self.target, "linux/amd64", 0, self.metadata_path
            ),
            2,
        )
        self.assertEqual(
            json.loads(self.target.read_text())["reason"], "staged_digest_unavailable"
        )

    def test_invalid_source_identity_is_rejected_before_registry_access(self):
        self.record()
        descriptor = json.loads(self.descriptor_path.read_text())
        descriptor["source_sha"] = "f" * 40
        publication.write_json(self.descriptor_path, descriptor)
        with patch.object(publication, "public_manifest") as fetch:
            with self.assertRaisesRegex(ValueError, "invalid publication descriptor"):
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/amd64",
                    0,
                    self.metadata_path,
                )
        fetch.assert_not_called()

    def test_startup_failures_have_junit_infrastructure_evidence(self):
        publication.ensure_report(self.directory)
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["status"], "infra_error")
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["executions"], 0)
        self.assertIn('errors="1"', (self.directory / "junit.xml").read_text())
        publication.write_json(self.directory / "summary.json", {"executions": 20})
        (self.directory / "junit.xml").unlink()
        publication.ensure_report(self.directory)
        self.assertTrue((self.directory / "junit.xml").is_file())
        self.assertEqual(
            json.loads((self.directory / "summary.json").read_text()),
            {"executions": 20},
        )

    def test_only_completed_main_nightly_can_supply_evidence(self):
        for event in ("schedule", "workflow_dispatch"):
            metadata = {**self.metadata, "event": event}
            self.assertEqual(
                publication.verify_run(metadata, "ai-dynamo/dynamo", "42")["id"], 42
            )
        changed_fields = {
            "id": 43,
            "run_attempt": 0,
            "head_sha": "short",
            "repository": {"full_name": "another/repository"},
            "head_repository": {"full_name": "fork/dynamo"},
            "path": ".github/workflows/pr.yaml",
            "name": "PR",
            "head_branch": "feature",
            "event": "pull_request",
            "status": "in_progress",
            "pull_requests": [{"number": 1}],
        }
        for field, value in changed_fields.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    publication.verify_run(
                        {**self.metadata, field: value}, "ai-dynamo/dynamo", "42"
                    )
        for run_id in ("0", "-1", "42\nimage=bad", "main"):
            with self.assertRaises(ValueError):
                publication.verify_run(self.metadata, "ai-dynamo/dynamo", run_id)

    def test_descriptor_must_match_trusted_run_before_registry_access(self):
        self.record()
        for field, value in {"id": 43, "run_attempt": 2, "head_sha": "f" * 40}.items():
            publication.write_json(self.metadata_path, {**self.metadata, field: value})
            with self.subTest(field=field), patch.object(
                publication, "public_manifest"
            ) as fetch:
                with self.assertRaisesRegex(ValueError, "trusted nightly run"):
                    publication.resolve(
                        self.descriptor_path,
                        self.target,
                        "linux/amd64",
                        0,
                        self.metadata_path,
                    )
                fetch.assert_not_called()

    def test_target_unavailable_reason_is_not_replaced_by_generic_infra(self):
        publication.write_json(
            self.target,
            {"status": "target_unavailable", "reason": "frontend_not_staged"},
        )
        publication.ensure_report(self.directory, self.target)
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["status"], "target_unavailable")
        self.assertEqual(summary["reason"], "frontend_not_staged")
        self.assertIn('errors="1"', (self.directory / "junit.xml").read_text())

    def test_missing_descriptor_path_has_incomplete_evaluation_report(self):
        publication.ensure_report(self.directory, self.target)
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["status"], "target_unavailable")
        self.assertEqual(summary["reason"], "publication_evidence_unavailable")
        self.assertEqual(summary["exit_code"], 2)

    def test_truncated_manifest_response_is_retried_within_budget(self):
        self.record()
        with patch.object(
            publication,
            "public_manifest",
            side_effect=[
                http.client.IncompleteRead(b""),
                (self.index_digest, self.manifest),
            ],
        ) as fetch, patch.object(
            publication.time, "monotonic", return_value=0
        ), patch.object(
            publication.time, "sleep"
        ):
            self.assertEqual(
                publication.resolve(
                    self.descriptor_path,
                    self.target,
                    "linux/amd64",
                    1,
                    self.metadata_path,
                ),
                0,
            )
        self.assertEqual(fetch.call_count, 2)

    def test_missing_attempt_descriptor_is_distinct_from_delayed_promotion(self):
        for attempt, reason in (
            (1, "publication_descriptor_missing"),
            (2, "publication_not_rerun_in_attempt"),
        ):
            metadata = {**self.metadata, "run_attempt": attempt}
            responses = [
                io.BytesIO(json.dumps(metadata).encode()),
                io.BytesIO(b'{"artifacts":[]}'),
            ]
            with patch.dict(os.environ, {"GITHUB_TOKEN": "test-only"}), patch.object(
                publication.urllib.request, "urlopen", side_effect=responses
            ):
                self.assertEqual(
                    publication.validate_run(
                        "ai-dynamo/dynamo",
                        "42",
                        self.metadata_path,
                        self.target,
                        str(attempt),
                    ),
                    2,
                )
            self.assertEqual(json.loads(self.target.read_text())["reason"], reason)

    def test_trigger_attempt_is_verified_before_descriptor_download(self):
        metadata = {**self.metadata, "run_attempt": 2}
        responses = [
            io.BytesIO(json.dumps(metadata).encode()),
            io.BytesIO(
                b'{"artifacts":[{"name":"frontend-replay-publication-2","expired":false}]}'
            ),
        ]
        output = self.directory / "github-output"
        with patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-only", "GITHUB_OUTPUT": str(output)}
        ), patch.object(
            publication.urllib.request, "urlopen", side_effect=responses
        ) as fetch:
            self.assertEqual(
                publication.validate_run(
                    "ai-dynamo/dynamo", "42", self.metadata_path, self.target, "2"
                ),
                0,
            )
        self.assertTrue(
            fetch.call_args_list[0].args[0].full_url.endswith("/attempts/2")
        )
        self.assertIn("run_attempt=2\n", output.read_text())
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-only"}), patch.object(
            publication.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(self.metadata).encode()),
        ):
            with self.assertRaises(publication.ProvenanceError):
                publication.validate_run(
                    "ai-dynamo/dynamo", "42", self.metadata_path, self.target, "2"
                )

    def test_provenance_rejections_keep_their_own_report_status(self):
        self.record()
        publication.write_json(self.metadata_path, {**self.metadata, "run_attempt": 2})
        commands = [
            [
                "validate-run",
                "--repository",
                "ai-dynamo/dynamo",
                "--run-id",
                "0",
                "--output",
                str(self.metadata_path),
                "--target",
                str(self.target),
            ],
            [
                "resolve",
                "--descriptor",
                str(self.descriptor_path),
                "--run-metadata",
                str(self.metadata_path),
                "--output",
                str(self.target),
                "--wait-seconds",
                "0",
            ],
        ]
        for args in commands:
            with patch("sys.argv", ["publication.py", *args]), patch.object(
                publication, "public_manifest"
            ) as fetch:
                self.assertEqual(publication.main(), 2)
                fetch.assert_not_called()
            self.assertEqual(
                json.loads(self.target.read_text())["status"], "provenance_rejected"
            )
        publication.ensure_report(self.directory, self.target)
        self.assertEqual(
            json.loads((self.directory / "summary.json").read_text())["status"],
            "provenance_rejected",
        )


if __name__ == "__main__":
    unittest.main()
