# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pro_adapter import adapt_row, docker_image, format_problem_statement  # noqa: E402


class ProAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "instance_id": "instance_owner__repo-deadbeef-vnan",
            "problem_statement": "Fix the behavior.",
            "requirements": "- Preserve compatibility.\n- Add coverage.",
            "interface": "Type: Method\nName: Widget.run",
            "dockerhub_tag": "owner.repo-owner__repo-deadbeef-vnan",
        }

    def test_problem_statement_matches_public_scaffold(self) -> None:
        self.assertEqual(
            format_problem_statement(self.row),
            "Fix the behavior.\n\n"
            "Requirements:\n- Preserve compatibility.\n- Add coverage.\n\n"
            "New interfaces introduced:\nType: Method\nName: Widget.run",
        )

    def test_image_uses_authoritative_dataset_tag(self) -> None:
        self.assertEqual(
            docker_image(self.row),
            "docker.io/jefzda/sweap-images:owner.repo-owner__repo-deadbeef-vnan",
        )

    def test_adapt_row_preserves_evaluator_fields(self) -> None:
        adapted = adapt_row({**self.row, "base_commit": "abc"})
        self.assertEqual(adapted["base_commit"], "abc")
        self.assertEqual(adapted["image_name"], docker_image(self.row))
        self.assertEqual(
            adapted["problem_statement"], format_problem_statement(self.row)
        )

    def test_invalid_tag_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            docker_image({**self.row, "dockerhub_tag": "owner/image:tag"})


if __name__ == "__main__":
    unittest.main()
