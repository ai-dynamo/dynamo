#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


TEMPLATE_DIR = Path(__file__).with_name("templates")
TEMPLATES = (
    TEMPLATE_DIR / "dynamo-sglang.yaml",
    TEMPLATE_DIR / "sglang-serve.yaml",
)
PYTHON_MARKER = (
    'python3 - "${MODEL_PATH}/config.json" "${model_view}/config.json" <<\'PY\'\n'
)


def embedded_transforms(path: Path) -> list[str]:
    source = path.read_text()
    return [
        textwrap.dedent(re.split(r"\n[ \t]+PY\n", fragment, maxsplit=1)[0])
        for fragment in source.split(PYTHON_MARKER)[1:]
    ]


def embedded_transform(path: Path) -> str:
    return embedded_transforms(path)[0]


class SglangModelViewTests(unittest.TestCase):
    def test_manifests_use_identical_model_view_transforms(self) -> None:
        dynamo_programs = embedded_transforms(TEMPLATES[0])
        native_programs = embedded_transforms(TEMPLATES[1])
        self.assertEqual(len(dynamo_programs), 2)
        self.assertEqual(len(native_programs), 1)
        self.assertEqual(dynamo_programs, native_programs * 2)
        for path in TEMPLATES:
            source = path.read_text()
            self.assertIn('--model-path="${model_view}"', source)
            self.assertNotIn('--model-path="${MODEL_PATH}"', source)
        dynamo_source = TEMPLATES[0].read_text()
        frontend, worker = dynamo_source.split("    - name: SglangWorker", maxsplit=1)
        self.assertEqual(frontend.count(PYTHON_MARKER), 1)
        self.assertEqual(worker.count(PYTHON_MARKER), 1)
        self.assertLess(
            frontend.index(PYTHON_MARKER),
            frontend.index("exec python3 -m dynamo.frontend"),
        )

    def test_transform_removes_only_the_validated_layer_types(self) -> None:
        document = {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "layer_types": ["deepseek_sparse_attention"] * 80,
            "model_type": "glm_moe_dsa",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            target = root / "target.json"
            source.write_text(json.dumps(document))
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    embedded_transform(TEMPLATES[0]),
                    source,
                    target,
                ],
                check=True,
            )
            transformed = json.loads(target.read_text())
            self.assertNotIn("layer_types", transformed)
            self.assertEqual(
                transformed,
                {key: value for key, value in document.items() if key != "layer_types"},
            )
            self.assertEqual(target.stat().st_mode & 0o777, 0o444)

    def test_transform_rejects_unexpected_layer_types(self) -> None:
        invalid_values = (None, [], ["full_attention"])
        for value in invalid_values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source.json"
                target = root / "target.json"
                source.write_text(json.dumps({"layer_types": value}))
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        embedded_transform(TEMPLATES[0]),
                        source,
                        target,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unexpected layer_types", result.stderr)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
