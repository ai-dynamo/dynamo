# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light helpers for MMLU prompt construction and answer parsing."""

from __future__ import annotations

import re
from typing import Any, Protocol

CHOICES = ("A", "B", "C", "D")
_CHOICE_RESPONSE = re.compile(r"^(?:answer\s*:\s*)?([A-D])(?:[.)])?$", re.IGNORECASE)


class _FrameIndexer(Protocol):
    def __getitem__(self, key: tuple[int, int]) -> Any:
        ...


class TabularFrame(Protocol):
    @property
    def shape(self) -> tuple[int, int]:
        ...

    @property
    def iloc(self) -> _FrameIndexer:
        ...


def prompt_string(frame: TabularFrame, index: int, include_answer: bool = True) -> str:
    prompt = frame.iloc[index, 0]
    option_count = frame.shape[1] - 2
    for option_index in range(option_count):
        prompt += f"\n{CHOICES[option_index]}. {frame.iloc[index, option_index + 1]}"
    prompt += (
        "\nRespond with **only the letter** (A, B, C, D).  Do **not** output "
        "any explanation, analysis, or extra words. Answer:"
    )
    if include_answer:
        prompt += f" {frame.iloc[index, option_count + 1]}\n\n"
    return prompt


def extract_choice(response: str) -> str | None:
    match = _CHOICE_RESPONSE.fullmatch(response.strip())
    return match.group(1).upper() if match else None
