#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import subprocess


MODULE_PATH = Path(__file__).with_name("capture_metadata.py")
FIXTURE_PATH = MODULE_PATH.parents[1] / "fixtures" / "runtime-binding.json"
SPEC = importlib.util.spec_from_file_location("terminal_capture_metadata", MODULE_PATH)
assert SPEC and SPEC.loader
capture_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_metadata)

from source_provenance import build_source_provenance  # noqa: E402


class FakeResponse:
    status = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class CaptureMetadataTests(unittest.TestCase):
    @staticmethod
    def source_tree(root: Path, commit: str) -> tuple[Path, Path]:
        source_root = root / "workspace"
        (source_root / "eval").mkdir(parents=True)
        (source_root / "campaign.env").write_text("MAX_MODEL_LEN=409600\n")
        (source_root / "eval/run.sh").write_text("#!/bin/sh\n")
        provenance = source_root / "source-provenance.json"
        provenance.write_text(
            json.dumps(
                build_source_provenance(
                    source_root,
                    source_commit=commit,
                    source_branch="rmccormick/glm52",
                    bundle_sha256="b" * 64,
                )
            )
            + "\n"
        )
        return source_root, provenance

    def test_runtime_binding_is_canonical_and_phase_bound(self) -> None:
        deployment = json.loads(FIXTURE_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text(json.dumps(deployment, indent=2) + "\n")
            source_root, provenance = self.source_tree(
                Path(directory), deployment["recipe"]["source_commit"]
            )
            wrapper, campaign_source = capture_metadata.runtime_binding(
                path,
                "dynamo-vllm",
                "ab",
                "http://glm52-dynamo-vllm-frontend:8000/v1",
                provenance,
                source_root,
            )

        deployment_canonical = json.dumps(
            deployment, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        content = {
            "deployment": deployment,
            "evaluator": {"campaign_source": campaign_source},
        }
        content_canonical = json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        self.assertEqual(wrapper["file"], "runtime-binding.json")
        self.assertEqual(
            wrapper["deployment_sha256"],
            hashlib.sha256(deployment_canonical).hexdigest(),
        )
        self.assertEqual(
            wrapper["content_sha256"], hashlib.sha256(content_canonical).hexdigest()
        )
        self.assertEqual(wrapper["content"]["deployment"], deployment)
        self.assertEqual(
            wrapper["content"]["evaluator"]["campaign_source"], campaign_source
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text(json.dumps(deployment))
            source_root, provenance = self.source_tree(
                Path(directory), deployment["recipe"]["source_commit"]
            )
            with self.assertRaisesRegex(ValueError, "campaign_phase"):
                capture_metadata.runtime_binding(
                    path,
                    "dynamo-vllm",
                    "ba",
                    "http://glm52-dynamo-vllm-frontend:8000/v1",
                    provenance,
                    source_root,
                )

    def test_harbor_environment_is_lock_checked_and_hashed(self) -> None:
        checked = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        inventoried = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {"python": "3.12.11", "packages": [["harbor", "0.17.1"]]}
            ),
            stderr="",
        )
        with mock.patch.object(
            capture_metadata.subprocess, "run", side_effect=[checked, inventoried]
        ) as run:
            identity = capture_metadata.harbor_environment(Path("/harbor"))
        self.assertEqual(identity["uv_sync_check"], "passed")
        self.assertEqual(identity["package_count"], 1)
        self.assertEqual(identity["packages"], [["harbor", "0.17.1"]])
        self.assertEqual(len(identity["packages_sha256"]), 64)
        self.assertIn("--check", run.call_args_list[0].args[0])

    def test_harbor_environment_rejects_duplicate_normalized_names(self) -> None:
        checked = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        inventoried = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "python": "3.12.11",
                    "packages": [["demo_pkg", "1"], ["demo-pkg", "2"]],
                }
            ),
            stderr="",
        )
        with mock.patch.object(
            capture_metadata.subprocess, "run", side_effect=[checked, inventoried]
        ):
            with self.assertRaisesRegex(RuntimeError, "duplicate normalized"):
                capture_metadata.harbor_environment(Path("/harbor"))

    def test_harbor_clean_tree_checks_include_untracked_files(self) -> None:
        directory = Path(__file__).resolve().parent
        for path in (directory / "common.sh", directory / "bootstrap.sh", MODULE_PATH):
            source = path.read_text()
            self.assertIn("--untracked-files=all", source)
            self.assertNotIn("--untracked-files=no", source)

    def test_resume_rechecks_exact_harbor_environment(self) -> None:
        run_spec = {"mode": "full"}
        binding = {"deployment_sha256": "a" * 64}
        campaign_source = {"source_commit": "c" * 40}
        environment = {"packages_sha256": "d" * 64}
        metadata = {
            "schema_version": 2,
            "run_spec": run_spec,
            "runtime_binding": binding,
            "campaign_source": campaign_source,
            "harbor_environment": environment,
        }
        capture_metadata.validate_resume_metadata(
            metadata, run_spec, binding, campaign_source, environment
        )
        with self.assertRaisesRegex(RuntimeError, "Harbor environment differs"):
            capture_metadata.validate_resume_metadata(
                metadata,
                run_spec,
                binding,
                campaign_source,
                {"packages_sha256": "e" * 64},
            )

    def test_resume_rejects_old_metadata_schema(self) -> None:
        metadata = {
            "schema_version": 1,
            "run_spec": {},
            "runtime_binding": {},
            "campaign_source": {},
            "harbor_environment": {},
        }
        with self.assertRaisesRegex(RuntimeError, "schema version 2"):
            capture_metadata.validate_resume_metadata(metadata, {}, {}, {}, {})

    def test_endpoint_requires_exact_serving_context(self) -> None:
        for field in ("context_window", "max_model_len"):
            with self.subTest(field=field):
                payload = json.dumps(
                    {"data": [{"id": "zai-org/GLM-5.2", field: 409600}]}
                ).encode()
                with mock.patch.object(
                    capture_metadata.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(payload),
                ):
                    identity = capture_metadata.endpoint_models(
                        "http://example.test/v1", "zai-org/GLM-5.2", 409600
                    )
                self.assertEqual(identity["advertised_model"]["context_window"], 409600)
                self.assertNotIn("max_model_len", identity["advertised_model"])

        invalid_models = (
            {"id": "zai-org/GLM-5.2"},
            {"id": "zai-org/GLM-5.2", "max_model_len": 262144},
            {
                "id": "zai-org/GLM-5.2",
                "context_window": 409600,
                "max_model_len": 262144,
            },
        )
        for advertised in invalid_models:
            with self.subTest(advertised=advertised):
                payload = json.dumps({"data": [advertised]}).encode()
                with (
                    mock.patch.object(
                        capture_metadata.urllib.request,
                        "urlopen",
                        return_value=FakeResponse(payload),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "context_window or max_model_len"
                    ),
                ):
                    capture_metadata.endpoint_models(
                        "http://example.test/v1", "zai-org/GLM-5.2", 409600
                    )


if __name__ == "__main__":
    unittest.main()
