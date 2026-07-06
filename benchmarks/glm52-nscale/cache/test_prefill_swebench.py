#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prefill_swebench.py")
SPEC = importlib.util.spec_from_file_location("prefill_swebench", MODULE_PATH)
assert SPEC and SPEC.loader
prefill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill)


class PrefillSwebenchTest(unittest.TestCase):
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
                prefill.load_completed(entries, {"test__test-1": identity})


if __name__ == "__main__":
    unittest.main()
