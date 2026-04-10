# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for hf_processor_bypass placeholder expansion logic.

The ``_expand_placeholders`` function below is a standalone reimplementation
of ``QwenHFProcessorBypass._expand_placeholders`` so that tests can run
without a vLLM installation.  Keep this in sync with the real implementation
in ``hf_processor_bypass.py``.
"""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]

# Token IDs (made up for testing)
VS = 100  # vision_start
IP = 101  # image_pad
VE = 102  # vision_end


class _FakePlaceholderRange:
    """Mimics PlaceholderRange for testing without vllm dependency."""

    def __init__(self, offset: int, length: int) -> None:
        self.offset = offset
        self.length = length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FakePlaceholderRange):
            return NotImplemented
        return self.offset == other.offset and self.length == other.length

    def __repr__(self) -> str:
        return f"PR(offset={self.offset}, length={self.length})"


def _expand_placeholders(
    token_ids: list[int],
    per_image_tokens: list[int],
    vs: int = VS,
    ip: int = IP,
    ve: int = VE,
) -> tuple[list[int], list[_FakePlaceholderRange]]:
    """Standalone copy of the expansion logic for isolated testing."""
    expanded: list[int] = []
    placeholders: list[_FakePlaceholderRange] = []
    img_idx = 0
    i = 0
    n = len(token_ids)

    while i < n:
        if (
            i + 2 < n
            and token_ids[i] == vs
            and token_ids[i + 1] == ip
            and token_ids[i + 2] == ve
            and img_idx < len(per_image_tokens)
        ):
            n_tokens = per_image_tokens[img_idx]
            # Keep vision_start; Qwen2-VL needs boundary tokens for mRoPE
            expanded.append(vs)
            offset = len(expanded)
            expanded.extend([ip] * n_tokens)
            placeholders.append(_FakePlaceholderRange(offset=offset, length=n_tokens))
            expanded.append(ve)
            img_idx += 1
            i += 3
        else:
            expanded.append(token_ids[i])
            i += 1

    return expanded, placeholders


class TestExpandPlaceholders:
    def test_single_image(self) -> None:
        # System text + 1 image placeholder + user text
        token_ids = [1, 2, 3, VS, IP, VE, 4, 5]
        expanded, phs = _expand_placeholders(token_ids, [10])
        # Boundary tokens VS/VE are preserved around the expanded pad tokens
        assert expanded == [1, 2, 3, VS] + [IP] * 10 + [VE, 4, 5]
        assert len(phs) == 1
        assert phs[0].offset == 4  # after [1, 2, 3, VS]
        assert phs[0].length == 10

    def test_two_images(self) -> None:
        token_ids = [1, VS, IP, VE, 2, VS, IP, VE, 3]
        expanded, phs = _expand_placeholders(token_ids, [5, 8])
        # [1, VS, IP*5, VE, 2, VS, IP*8, VE, 3]
        assert expanded == [1, VS] + [IP] * 5 + [VE, 2, VS] + [IP] * 8 + [VE, 3]
        assert len(phs) == 2
        assert phs[0] == _FakePlaceholderRange(offset=2, length=5)  # after [1, VS]
        assert phs[1] == _FakePlaceholderRange(offset=2 + 5 + 3, length=8)  # after [VE, 2, VS]

    def test_no_images(self) -> None:
        token_ids = [1, 2, 3, 4]
        expanded, phs = _expand_placeholders(token_ids, [])
        assert expanded == [1, 2, 3, 4]
        assert phs == []

    def test_adjacent_images(self) -> None:
        token_ids = [VS, IP, VE, VS, IP, VE]
        expanded, phs = _expand_placeholders(token_ids, [3, 4])
        # [VS, IP*3, VE, VS, IP*4, VE]
        assert expanded == [VS] + [IP] * 3 + [VE, VS] + [IP] * 4 + [VE]
        assert len(phs) == 2
        assert phs[0] == _FakePlaceholderRange(offset=1, length=3)  # after [VS]
        assert phs[1] == _FakePlaceholderRange(offset=1 + 3 + 2, length=4)  # after [VE, VS]

    def test_partial_pattern_not_matched(self) -> None:
        # vision_start followed by wrong token should not match
        token_ids = [VS, 99, VE, VS, IP, VE]
        expanded, phs = _expand_placeholders(token_ids, [6])
        # First VS,99,VE is kept literally; second is expanded with boundary tokens
        assert expanded == [VS, 99, VE, VS] + [IP] * 6 + [VE]
        assert len(phs) == 1
        assert phs[0] == _FakePlaceholderRange(offset=4, length=6)  # after [VS, 99, VE, VS]


@pytest.mark.vllm
class TestExpandPlaceholdersIntegration:
    """Integration tests that exercise the real QwenHFProcessorBypass class.

    Requires vLLM to be installed. Skipped in environments without it.
    """

    @pytest.mark.skip(reason="Requires vLLM + model weights; run manually")
    def test_expand_matches_standalone(self) -> None:
        """Verify the real _expand_placeholders produces the same output."""
        # TODO: instantiate QwenHFProcessorBypass with a real model and
        # compare its _expand_placeholders output against the standalone
        # implementation above.
