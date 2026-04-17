# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PIL", reason="Pillow required for image generation benchmarks")

# The generators use relative imports, so add the source dir to sys.path
JSONL_DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "multimodal" / "jsonl"
sys.path.insert(0, str(JSONL_DIR))

from main import main  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _run_main(tmp_path: Path, argv: list[str]) -> list[dict]:
    """Run main() with given argv, return parsed JSONL lines."""
    with patch("sys.argv", ["main.py"] + argv):
        main()
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL file, got {jsonl_files}"
    with open(jsonl_files[0]) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestSingleTurnDefault:
    """single-turn is the default when no subcommand is given."""

    def test_default_produces_independent_requests(self, tmp_path: Path) -> None:
        lines = _run_main(
            tmp_path,
            [
                "-n",
                "4",
                "--images-per-request",
                "2",
                "--image-size",
                "32",
                "32",
                "--image-dir",
                str(tmp_path / "imgs"),
                "--seed",
                "1",
                "-o",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert len(lines) == 4
        for line in lines:
            assert "text" in line
            assert len(line["images"]) == 2
            assert "session_id" not in line


class TestSlidingWindow:
    """sliding-window produces causal sessions with image overlap."""

    def test_output_structure(self, tmp_path: Path) -> None:
        lines = _run_main(
            tmp_path,
            [
                "sliding-window",
                "--num-users",
                "2",
                "--turns-per-user",
                "3",
                "--window-size",
                "5",
                "--image-size",
                "32",
                "32",
                "--image-dir",
                str(tmp_path / "imgs"),
                "--seed",
                "42",
                "-o",
                str(tmp_path / "sw.jsonl"),
            ],
        )

        assert len(lines) == 6  # 2 users x 3 turns

        # Round-robin interleaving: user_0, user_1, user_0, user_1, ...
        session_ids = [row["session_id"] for row in lines]
        assert session_ids == [
            "user_0",
            "user_1",
            "user_0",
            "user_1",
            "user_0",
            "user_1",
        ]

        for line in lines:
            assert len(line["images"]) == 5

    def test_image_overlap(self, tmp_path: Path) -> None:
        lines = _run_main(
            tmp_path,
            [
                "sliding-window",
                "--num-users",
                "1",
                "--turns-per-user",
                "3",
                "--window-size",
                "4",
                "--image-size",
                "32",
                "32",
                "--image-dir",
                str(tmp_path / "imgs"),
                "--seed",
                "7",
                "-o",
                str(tmp_path / "overlap.jsonl"),
            ],
        )

        assert len(lines) == 3
        # Consecutive turns should share window_size-1 images
        for i in range(len(lines) - 1):
            prev = lines[i]["images"]
            curr = lines[i + 1]["images"]
            # Sliding by 1: prev[1:] == curr[:-1]
            assert (
                prev[1:] == curr[:-1]
            ), f"Turn {i} and {i + 1} should share 3/4 images"
