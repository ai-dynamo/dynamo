#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("prefill_swebench.py")
SPEC = importlib.util.spec_from_file_location("prefill_swebench", MODULE_PATH)
assert SPEC and SPEC.loader
prefill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill)


class PrefillSwebenchTest(unittest.TestCase):
    binding = "4" * 64

    @staticmethod
    def identity(index: int) -> dict[str, object]:
        repository = f"swebench/sweb.eval.x86_64.test_1776_test-{index}"
        return {
            "requested_ref": f"docker.io/{repository}:latest",
            "image_id": "sha256:" + f"{index:064x}",
            "repo_digests": [f"{repository}@sha256:" + f"{index + 1:064x}"],
            "content_identity_sha256": f"{index + 2:064x}",
        }

    def args(self, manifest: Path, state_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            manifest=manifest,
            state_dir=state_dir,
            cache_binding_sha256=self.binding,
            registry="http://registry.invalid",
            quota_reserve=5,
            poll_seconds=1,
            retry_seconds=1,
            max_pull_failures=1,
            pull_timeout_seconds=1,
            revalidate_completed=False,
        )

    def test_parse_rate_limit(self) -> None:
        self.assertEqual(prefill.parse_rate_limit("99;w=3600"), (99, 3600))
        with self.assertRaisesRegex(ValueError, "RateLimit-Remaining"):
            prefill.parse_rate_limit("unknown")

    def test_repository_from_reference(self) -> None:
        self.assertEqual(
            prefill.repository_from_reference(
                "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
            ),
            "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",
        )
        with self.assertRaisesRegex(ValueError, "not a SWE-bench"):
            prefill.repository_from_reference("docker.io/library/hello-world:latest")

    def test_load_completed_rejects_identity_drift(self) -> None:
        identity = {
            "requested_ref": "docker.io/swebench/sweb.eval.x86_64.test_1776_test-1:latest",
            "image_id": "sha256:" + "1" * 64,
            "repo_digests": ["swebench/sweb.eval.x86_64.test_1776_test-1@sha256:" + "2" * 64],
            "content_identity_sha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            entries = Path(temporary)
            (entries / "test__test-1.json").write_text(
                json.dumps({"instance_id": "test__test-1", "identity": {}})
            )
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                prefill.load_completed(
                    entries,
                    {"test__test-1": identity},
                    self.binding,
                )

    def test_load_completed_rejects_cache_binding_drift(self) -> None:
        identity = self.identity(1)
        with tempfile.TemporaryDirectory() as temporary:
            entries = Path(temporary)
            (entries / "test__test-1.json").write_text(
                json.dumps({"instance_id": "test__test-1", "identity": identity})
            )
            with self.assertRaisesRegex(ValueError, "cache binding drifted"):
                prefill.load_completed(
                    entries,
                    {"test__test-1": identity},
                    self.binding,
                )
            loaded = prefill.load_completed(
                entries,
                {"test__test-1": identity},
                self.binding,
                allow_missing_binding=True,
            )
            self.assertEqual(set(loaded), {"test__test-1"})

    def test_same_binding_zero_work_resume_completes_without_quota_probe(self) -> None:
        images = {f"test__test-{i}": self.identity(i) for i in range(500)}
        repositories = {
            prefill.repository_from_reference(identity["requested_ref"])
            for identity in images.values()
        }
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            entries = state_dir / "entries"
            entries.mkdir(parents=True)
            manifest = Path(temporary) / "task-images.json"
            manifest.write_text("synthetic manifest\n")
            for instance_id, identity in images.items():
                prefill.atomic_write_json(
                    entries / f"{instance_id}.json",
                    {
                        "instance_id": instance_id,
                        "identity": identity,
                        "cache_binding_sha256": self.binding,
                    },
                )
            prefill.atomic_write_json(
                state_dir / "initial-catalog.json",
                {"repositories": sorted(repositories)},
            )
            with (
                mock.patch.object(prefill, "load_manifest", return_value=("verified", images)),
                mock.patch.object(prefill, "registry_catalog", return_value=repositories),
                mock.patch.object(
                    prefill,
                    "docker_hub_quota",
                    side_effect=AssertionError("zero-work resume probed quota"),
                ),
            ):
                prefill.run(self.args(manifest, state_dir))
            status = json.loads((state_dir / "status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed"], 500)
            self.assertNotIn("quota_last_observed_before_pull", status)

    def test_interrupted_legacy_revalidation_keeps_metadata_unbound(self) -> None:
        images = {f"test__test-{i}": self.identity(i) for i in range(500)}
        repositories = {
            prefill.repository_from_reference(identity["requested_ref"])
            for identity in images.values()
        }
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            entries = state_dir / "entries"
            entries.mkdir(parents=True)
            manifest = Path(temporary) / "task-images.json"
            manifest.write_text("legacy synthetic manifest\n")
            first_id = "test__test-0"
            prefill.atomic_write_json(
                entries / f"{first_id}.json",
                {"instance_id": first_id, "identity": images[first_id]},
            )
            prefill.atomic_write_json(
                state_dir / "initial-catalog.json",
                {"repositories": sorted(repositories)},
            )
            prefill.atomic_write_json(
                state_dir / "metadata.json",
                {
                    "schema_version": 1,
                    "suite": "verified",
                    "source_manifest_sha256": prefill.file_sha256(manifest),
                    "total": 500,
                    "started_at": "2026-07-07T00:00:00Z",
                    "script_sha256s": ["5" * 64],
                },
            )
            with (
                mock.patch.object(prefill, "load_manifest", return_value=("verified", images)),
                mock.patch.object(prefill, "docker_hub_quota", return_value=(100, 3600, "100;w=3600")),
                mock.patch.object(prefill, "pull_and_validate", side_effect=RuntimeError("stop")),
                mock.patch.object(prefill, "remove_local_image"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    prefill.run(self.args(manifest, state_dir))
            metadata = json.loads((state_dir / "metadata.json").read_text())
            self.assertNotIn("cache_binding_sha256", metadata)


if __name__ == "__main__":
    unittest.main()
