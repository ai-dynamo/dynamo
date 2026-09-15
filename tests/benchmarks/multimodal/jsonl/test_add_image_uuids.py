# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.multimodal.jsonl.add_image_uuids import add_image_uuids
from benchmarks.multimodal.jsonl.generate_images import compute_image_uuid

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_add_image_uuids_preserves_rows_and_reuses_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    rows = [
        {"session_id": "one", "text": "first", "images": ["a.png", "b.png"]},
        {"session_id": "one", "text": "second", "images": ["b.png", "c.png"]},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    row_count, image_count = add_image_uuids(input_path, output_path)

    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert (row_count, image_count) == (2, 4)
    assert output_rows[0]["image_uuids"] == [
        compute_image_uuid("a.png"),
        compute_image_uuid("b.png"),
    ]
    assert output_rows[1]["image_uuids"][0] == output_rows[0]["image_uuids"][1]
    for original, output in zip(rows, output_rows):
        assert {key: output[key] for key in original} == original


def test_add_image_uuids_rejects_in_place_update(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"images":[]}\n')

    with pytest.raises(ValueError, match="must differ"):
        add_image_uuids(input_path, input_path)
