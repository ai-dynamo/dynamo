#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run-full.sh")


class FullRunContractTests(unittest.TestCase):
    def run_full(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(SCRIPT),
                "--api-base",
                "http://glm52-dynamo-vllm-frontend:8000/v1",
                "--label",
                "dynamo-vllm",
                "--phase",
                "ab",
                "--job-name",
                "dynamo-vllm-ab-test",
                "--dry-run",
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_official_recipe_overrides_are_rejected_before_execution(self) -> None:
        cases = {
            "temperature": ("--temperature", "0"),
            "top-p": ("--top-p", "0.5"),
            "max-turns": ("--max-turns", "499"),
            "max-context": ("--max-context", "131072"),
            "max-output": ("--max-output", "32000"),
            "timeout-multiplier": ("--timeout-multiplier", "8"),
            "served-model": ("--served-model", "other-model"),
            "model": ("--model", "openai/other-model"),
        }
        for name, arguments in cases.items():
            with self.subTest(name=name):
                result = self.run_full(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"Full runs require {name}=", result.stderr)

    def test_job_name_cannot_escape_jobs_directory(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--api-base",
                "http://glm52-dynamo-vllm-frontend:8000/v1",
                "--label",
                "dynamo-vllm",
                "--phase",
                "ab",
                "--job-name",
                "../dynamo-vllm-ab-test",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--job-name must contain only", result.stderr)

    def test_success_path_requires_task_image_attestation(self) -> None:
        source = SCRIPT.with_name("run.sh").read_text()
        self.assertIn("if (( harbor_rc == 0 )); then", source)
        self.assertIn('"${SCRIPT_DIR}/capture_terminal_task_images.py"', source)
        self.assertIn('--task-images "${task_images_path}"', source)
        self.assertIn('"${HARBOR_SOURCE_DIR}/.venv/bin/python"', source)


if __name__ == "__main__":
    unittest.main()
