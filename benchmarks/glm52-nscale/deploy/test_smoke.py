#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("smoke.sh").read_text()
MODELS_BLOCK = SCRIPT.split('/v1/models" | jq -e', maxsplit=1)[1]
MODELS_FILTER = MODELS_BLOCK.split(" ' >/dev/null", maxsplit=1)[0].split(
    " '\n", maxsplit=1
)[1]


def accepts(payload: dict[str, object]) -> bool:
    result = subprocess.run(
        [
            "jq",
            "-e",
            "--arg",
            "model",
            "zai-org/GLM-5.2",
            "--argjson",
            "context",
            "409600",
            MODELS_FILTER,
        ],
        input=json.dumps(payload),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise AssertionError(result.stderr)
    return result.returncode == 0


class SmokeModelsContractTests(unittest.TestCase):
    def test_accepts_canonical_and_engine_context_aliases(self) -> None:
        models = (
            {"id": "zai-org/GLM-5.2", "context_window": 409600},
            {"id": "zai-org/GLM-5.2", "max_model_len": 409600},
            {
                "id": "zai-org/GLM-5.2",
                "context_window": 409600,
                "max_model_len": 409600,
            },
            {
                "id": "zai-org/GLM-5.2",
                "context_window": None,
                "max_model_len": 409600,
            },
        )
        for model in models:
            with self.subTest(model=model):
                self.assertTrue(accepts({"data": [model]}))

    def test_rejects_missing_conflicting_wrong_or_duplicate_context(self) -> None:
        payloads = (
            {"data": [{"id": "zai-org/GLM-5.2"}]},
            {
                "data": [
                    {
                        "id": "zai-org/GLM-5.2",
                        "context_window": 409600,
                        "max_model_len": 262144,
                    }
                ]
            },
            {"data": [{"id": "zai-org/GLM-5.2", "max_model_len": 262144}]},
            {
                "data": [
                    {"id": "zai-org/GLM-5.2", "context_window": 409600},
                    {"id": "zai-org/GLM-5.2", "context_window": 409600},
                ]
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertFalse(accepts(payload))


if __name__ == "__main__":
    unittest.main()
