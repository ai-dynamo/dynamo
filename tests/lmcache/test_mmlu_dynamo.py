# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tests.lmcache.mmlu_utils import extract_choice, prompt_string

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", "A"),
        (" c. ", "C"),
        ("Answer: D", "D"),
        ("Answer: C because it is correct", None),
        ("not a choice", None),
        ("", None),
    ],
)
def test_extract_choice_rejects_ambiguous_output(
    response: str, expected: str | None
) -> None:
    assert extract_choice(response) == expected


class StubFrame:
    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]))

    @property
    def iloc(self) -> StubFrame:
        return self

    def __getitem__(self, key: tuple[int, int]) -> str:
        row, column = key
        return self._rows[row][column]


def test_prompt_string_uses_the_answer_column() -> None:
    frame = StubFrame(
        [["Question?", "option a", "option b", "option c", "option d", "D"]]
    )

    prompt = prompt_string(frame, 0)

    assert prompt.endswith("Answer: D\n\n")
    assert "D. option d" in prompt
