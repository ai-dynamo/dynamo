# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.multimodal.jsonl.prepare_ec_h2d_overlap import (
    build_chat_template,
    make_exact_text,
    write_sliding_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return "".join(chr(token_id) for token_id in ids)


def test_exact_text_hits_requested_token_count() -> None:
    tokenizer = CharacterTokenizer()

    value = make_exact_text(tokenizer, 257)

    assert len(tokenizer.encode(value)) == 257


def test_chat_template_prepends_system_message_only_when_missing() -> None:
    base = "{% for message in messages %}[{{ message['role'] }}]{{ message['content'] }}{% endfor %}"
    template = build_chat_template(base, "shared prefix")

    assert "messages[0]['role'] != 'system'" in template
    assert "'role': 'system', 'content': \"shared prefix\"" in template
    assert template.endswith(base)


def test_sliding_datasets_share_fixed_text_and_stable_overlap(tmp_path: Path) -> None:
    pool = [str(tmp_path / f"image-{index}.png") for index in range(8)]
    output = tmp_path / "dataset.jsonl"

    counts = write_sliding_dataset(
        output,
        pool,
        num_users=2,
        turns_per_user=2,
        window_size=3,
        images_per_user=4,
        user_text="same text",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert counts == {
        "rows": 4,
        "unique_images": 8,
        "content_images": 8,
        "stripped_images": 4,
    }
    assert {row["text"] for row in rows} == {"same text"}
    user_zero = [row for row in rows if row["session_id"] == "user_0"]
    assert user_zero[0]["images"][1:] == user_zero[1]["images"][:-1]
    assert all(len(row["image_uuids"]) == 3 for row in rows)
