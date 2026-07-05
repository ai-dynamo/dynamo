#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import capture_terminal_task_images


class TaskImageEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.job_dir = self.root / "job"
        self.job_dir.mkdir()
        self.cache_dir = self.root / "cache"
        self.ref = {
            "org": "terminal-bench",
            "name": "synthetic-task",
            "ref": "sha256:" + "a" * 64,
        }
        self.dataset = {"task_count": 1, "task_refs": [self.ref]}
        task_dir = (
            self.cache_dir
            / self.ref["org"]
            / self.ref["name"]
            / self.ref["ref"].removeprefix("sha256:")
        )
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            '[environment]\ndocker_image = "registry.example/task:v1"\n'
        )
        for attempt in range(2):
            trial_name = f"synthetic-task__{attempt}"
            trial_dir = self.job_dir / trial_name
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "terminal-bench/synthetic-task",
                        "trial_name": trial_name,
                        "task_id": self.ref,
                        "task_checksum": "b" * 64,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def inspector(_requested_ref: str) -> dict[str, object]:
        return {
            "Id": "sha256:" + "c" * 64,
            "RepoDigests": ["registry.example/task@sha256:" + "d" * 64],
        }

    def build(self) -> dict[str, object]:
        return capture_terminal_task_images.build_evidence(
            self.job_dir,
            self.dataset,
            self.cache_dir,
            1,
            2,
            inspector=self.inspector,
            toml_loader=lambda _text: {
                "environment": {"docker_image": "registry.example/task:v1"}
            },
        )

    def test_builds_and_validates_canonical_cross_attempt_evidence(self) -> None:
        evidence = self.build()
        self.assertEqual(evidence["task_count"], 1)
        self.assertEqual(evidence["trial_count"], 2)
        task = evidence["tasks"][0]
        self.assertEqual(task["task_ref"], self.ref)
        self.assertEqual(task["task_checksum"], "b" * 64)
        self.assertEqual(task["requested_ref"], "registry.example/task:v1")
        self.assertEqual(task["image_id"], "sha256:" + "c" * 64)
        self.assertNotIn("trials", task)
        self.assertIs(
            capture_terminal_task_images.validate_evidence(
                evidence, self.dataset, self.job_dir, 1, 2
            ),
            evidence,
        )

    def test_rejects_cross_attempt_task_checksum_drift(self) -> None:
        path = self.job_dir / "synthetic-task__1/result.json"
        result = json.loads(path.read_text())
        result["task_checksum"] = "e" * 64
        path.write_text(json.dumps(result) + "\n")
        with self.assertRaisesRegex(
            capture_terminal_task_images.TaskImageError, "differs across attempts"
        ):
            self.build()

    def test_rejects_missing_repo_digest(self) -> None:
        with self.assertRaisesRegex(
            capture_terminal_task_images.TaskImageError, "no valid RepoDigests"
        ):
            capture_terminal_task_images.build_evidence(
                self.job_dir,
                self.dataset,
                self.cache_dir,
                1,
                2,
                inspector=lambda _ref: {
                    "Id": "sha256:" + "c" * 64,
                    "RepoDigests": [],
                },
                toml_loader=lambda _text: {
                    "environment": {"docker_image": "registry.example/task:v1"}
                },
            )

    def test_validation_rejects_result_identity_drift(self) -> None:
        evidence = self.build()
        for attempt in range(2):
            path = self.job_dir / f"synthetic-task__{attempt}/result.json"
            result = json.loads(path.read_text())
            result["task_checksum"] = "f" * 64
            path.write_text(json.dumps(result) + "\n")
        with self.assertRaisesRegex(
            capture_terminal_task_images.TaskImageError,
            "task_checksum differs from results",
        ):
            capture_terminal_task_images.validate_evidence(
                evidence, self.dataset, self.job_dir, 1, 2
            )

    def test_existing_evidence_is_immutable(self) -> None:
        evidence = self.build()
        path = self.root / "task-images.json"
        capture_terminal_task_images.write_exclusive_or_verify(path, evidence)
        capture_terminal_task_images.write_exclusive_or_verify(path, evidence)
        changed = copy.deepcopy(evidence)
        changed["tasks"][0]["image_id"] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(
            capture_terminal_task_images.TaskImageError, "refusing to overwrite"
        ):
            capture_terminal_task_images.write_exclusive_or_verify(path, changed)


if __name__ == "__main__":
    unittest.main()
