# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.multimodal.sweep.dataset_shape import count_uuid_expectations

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _write_jsonl(tmp_path, rows: list[dict]) -> str:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(dataset)


def test_uuid_expectations_are_deduplicated_per_session(tmp_path) -> None:
    dataset = _write_jsonl(
        tmp_path,
        [
            {"session_id": "a", "image_uuids": ["x", "y"]},
            {"session_id": "b", "image_uuids": ["x"]},
            {"session_id": "a", "image_uuids": ["x", "z"]},
            {"session_id": "b", "image_uuids": ["x", "y"]},
        ],
    )

    assert count_uuid_expectations(dataset) == (5, 2)


def test_uuid_expectations_respect_conversation_limit(tmp_path) -> None:
    dataset = _write_jsonl(
        tmp_path,
        [
            {"session_id": "a", "image_uuids": ["a0"]},
            {"session_id": "b", "image_uuids": ["b0"]},
            {"session_id": "a", "image_uuids": ["a0"]},
            {"session_id": "c", "image_uuids": ["c0"]},
        ],
    )

    assert count_uuid_expectations(dataset, conversation_num=2) == (2, 1)


def test_uuid_expectations_treat_null_and_missing_sessions_as_rows(tmp_path) -> None:
    dataset = _write_jsonl(
        tmp_path,
        [
            {"session_id": None, "image_uuids": ["x", "x"]},
            {"image_uuids": ["x"]},
            {"session_id": 7, "image_uuids": ["z"]},
            {"session_id": "7", "image_uuids": ["z"]},
        ],
    )

    assert count_uuid_expectations(dataset) == (3, 2)
