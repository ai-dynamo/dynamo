#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("capture-identity.sh").read_text()
CRI_CONTAINER_ID = re.compile(
    r"^(?:containerd|docker|cri-o)://(?:sha256:)?[0-9a-f]{64}$"
)


class CaptureIdentityContractTests(unittest.TestCase):
    def test_accepts_standard_cri_container_id_forms(self) -> None:
        digest = "a" * 64
        for runtime in ("containerd", "docker", "cri-o"):
            with self.subTest(runtime=runtime):
                self.assertIsNotNone(
                    CRI_CONTAINER_ID.fullmatch(f"{runtime}://{digest}")
                )
                self.assertIsNotNone(
                    CRI_CONTAINER_ID.fullmatch(f"{runtime}://sha256:{digest}")
                )

    def test_rejects_unscoped_or_malformed_container_ids(self) -> None:
        digest = "a" * 64
        for value in (digest, f"unknown://{digest}", "containerd://short"):
            with self.subTest(value=value):
                self.assertIsNone(CRI_CONTAINER_ID.fullmatch(value))

    def test_capture_guard_uses_the_cri_uri_predicate(self) -> None:
        self.assertIn(
            'test("^(containerd|docker|cri-o)://(sha256:)?[0-9a-f]{64}$")',
            SCRIPT,
        )
        self.assertIn("and (.containerID | valid_container_id)", SCRIPT)


if __name__ == "__main__":
    unittest.main()
