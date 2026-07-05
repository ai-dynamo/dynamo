# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("terminalbench_summarize", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summarize = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarize
SPEC.loader.exec_module(summarize)


def valid_run_metadata() -> dict[str, object]:
    campaign_source = {
        "schema_version": 1,
        "source_commit": "c" * 40,
        "source_clean": True,
        "source_changed_path_count": 0,
        "bundle_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "eval_tree_sha256": "d" * 64,
        "campaign_env_sha256": "e" * 64,
        "source_file_count": 3,
        "eval_file_count": 2,
    }
    packages = [["Harbor", "0.17.1"], ["typing_extensions", "4.15.0"]]
    packages_sha256 = hashlib.sha256(
        json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    deployment = {"recipe": {"source_commit": campaign_source["source_commit"]}}
    content = {
        "deployment": deployment,
        "evaluator": {"campaign_source": campaign_source},
    }
    deployment_sha256 = summarize.canonical_sha256(deployment)
    return {
        "schema_version": 2,
        "run_spec": {"runtime_deployment_sha256": deployment_sha256},
        "campaign_source": campaign_source,
        "runtime_binding": {
            "file": "runtime-binding.json",
            "deployment_sha256": deployment_sha256,
            "content_sha256": summarize.canonical_sha256(content),
            "content": content,
        },
        "harbor_environment": {
            "uv_sync_check": "passed",
            "python": "3.12.11",
            "package_count": len(packages),
            "packages_sha256": packages_sha256,
            "packages": packages,
        },
    }


class ExpectedTaskNamesTest(unittest.TestCase):
    def test_uses_pinned_dataset_order_for_smoke(self) -> None:
        metadata = {
            "task_count": 4,
            "task_refs": [
                {"org": "terminal-bench", "name": "a"},
                {"org": "terminal-bench", "name": "b"},
                {"org": "terminal-bench", "name": "c"},
                {"org": "terminal-bench", "name": "d"},
            ],
        }
        self.assertEqual(
            summarize.expected_task_names(metadata, 3),
            ["terminal-bench/a", "terminal-bench/b", "terminal-bench/c"],
        )

    def test_rejects_duplicate_or_inconsistent_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize.expected_task_names(
                {"task_count": 2, "task_refs": [{"name": "a"}, {"name": "a"}]},
                2,
            )
        with self.assertRaisesRegex(ValueError, "task_count"):
            summarize.expected_task_names(
                {"task_count": 3, "task_refs": [{"name": "a"}, {"name": "b"}]},
                2,
            )


class RunMetadataValidationTest(unittest.TestCase):
    def test_accepts_schema_v2_with_bound_source_and_environment(self) -> None:
        metadata = valid_run_metadata()
        self.assertIs(summarize.validate_run_metadata(metadata), metadata)

    def test_rejects_schema_v1(self) -> None:
        metadata = valid_run_metadata()
        metadata["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            summarize.validate_run_metadata(metadata)

    def test_rejects_package_inventory_digest_drift(self) -> None:
        metadata = copy.deepcopy(valid_run_metadata())
        metadata["harbor_environment"]["packages_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inventory digest mismatch"):
            summarize.validate_run_metadata(metadata)

    def test_rejects_campaign_source_envelope_drift(self) -> None:
        metadata = copy.deepcopy(valid_run_metadata())
        metadata["runtime_binding"]["content"]["evaluator"]["campaign_source"][
            "bundle_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            summarize.validate_run_metadata(metadata)


class TaskImageSummaryTest(unittest.TestCase):
    def test_summary_embeds_and_hashes_exact_task_image_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            job_dir.mkdir()
            task_ref = {
                "org": "terminal-bench",
                "name": "synthetic-task",
                "ref": "sha256:" + "a" * 64,
            }
            (job_dir / "dataset-metadata.json").write_text(
                json.dumps({"task_count": 1, "task_refs": [task_ref]}) + "\n"
            )
            (job_dir / "result.json").write_text(
                json.dumps(
                    {
                        "n_total_trials": 1,
                        "stats": {"n_completed_trials": 1},
                    }
                )
                + "\n"
            )
            trial_name = "synthetic-task__0"
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            result_path = trial_dir / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "task_name": "terminal-bench/synthetic-task",
                        "trial_name": trial_name,
                        "task_id": task_ref,
                        "task_checksum": "b" * 64,
                        "verifier_result": {"rewards": {"reward": 1}},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            evidence = {
                "schema_version": 1,
                "task_count": 1,
                "trial_count": 1,
                "tasks": [
                    {
                        "task_name": "terminal-bench/synthetic-task",
                        "task_ref": task_ref,
                        "task_checksum": "b" * 64,
                        "task_toml_sha256": "c" * 64,
                        "requested_ref": "registry.example/task:v1",
                        "image_id": "sha256:" + "d" * 64,
                        "repo_digests": ["registry.example/task@sha256:" + "e" * 64],
                    }
                ],
            }
            task_images_path = job_dir / "task-images.json"
            task_images_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n"
            )
            args = SimpleNamespace(
                job_dir=job_dir,
                output_dir=job_dir / "summary",
                metadata=None,
                task_images=task_images_path,
                expected_tasks=1,
                expected_attempts=1,
                strict=True,
            )
            summary, errors, _trials = summarize.summarize(args)
            self.assertEqual(errors, [])
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["task_images"], evidence)
            self.assertEqual(
                summary["input_hashes"]["task_images_sha256"],
                summarize.sha256_file(task_images_path),
            )


if __name__ == "__main__":
    unittest.main()
