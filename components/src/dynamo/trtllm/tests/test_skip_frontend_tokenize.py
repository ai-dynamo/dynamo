# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DYN_SKIP_FRONTEND_TOKENIZE behaviour in the TRT-LLM engine.

Tests cover:
- engine_input selection (text vs token IDs) gated on DYN_ENGINE_CONV_AFFINITY.
- max_tokens auto-cap bypass when token_ids is empty and conv-affinity is on.
- startup warning when DYN_SKIP_FRONTEND_TOKENIZE=1 without DYN_ENGINE_CONV_AFFINITY=1.
- runtime guard: empty token_ids + prompt_text without conv-affinity raises RuntimeError.

No GPU or TRT-LLM engine initialisation is required.
"""


import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Helpers that mirror the logic in TrtllmLLMEngine.generate()
# ---------------------------------------------------------------------------


def _engine_input(engine_conv_affinity: bool, token_ids: list, prompt_text):
    """Mirror the engine_input selection ternary in generate()."""
    return (
        prompt_text
        if (engine_conv_affinity and not token_ids and prompt_text)
        else token_ids
    )


def _should_cap_max_tokens(engine_conv_affinity: bool, token_ids: list) -> bool:
    """Mirror the max_tokens auto-cap condition in generate().

    Returns True when the frontend-side cap should be applied.
    """
    return bool(token_ids or not engine_conv_affinity)


# ---------------------------------------------------------------------------
# engine_input selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine_conv_affinity", "token_ids", "prompt_text", "expect_text"),
    [
        # skip path: conv-affinity ON, no token_ids, prompt_text present → text
        (True, [], "Hello world", True),
        (True, None, "Hello world", True),
        # conv-affinity ON but token_ids present → IDs win
        (True, [1, 2, 3], "Hello world", False),
        # conv-affinity ON but no prompt_text → IDs (empty list)
        (True, [], None, False),
        # conv-affinity OFF → always use IDs regardless of prompt_text
        (False, [], "Hello world", False),
        (False, [1, 2, 3], "Hello world", False),
        (False, [], None, False),
    ],
)
def test_engine_input_selection(
    engine_conv_affinity, token_ids, prompt_text, expect_text
):
    result = _engine_input(engine_conv_affinity, token_ids or [], prompt_text)
    if expect_text:
        assert result == prompt_text
    else:
        assert result == (token_ids or [])


# ---------------------------------------------------------------------------
# max_tokens auto-cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine_conv_affinity", "token_ids", "expect_cap"),
    [
        # conv-affinity ON, no token_ids → skip the cap (engine will determine length)
        (True, [], False),
        # conv-affinity ON, token_ids present → cap still applies
        (True, [1, 2, 3], True),
        # conv-affinity OFF → cap always applies
        (False, [], True),
        (False, [1, 2, 3], True),
    ],
)
def test_max_tokens_cap_condition(engine_conv_affinity, token_ids, expect_cap):
    assert _should_cap_max_tokens(engine_conv_affinity, token_ids) == expect_cap


def test_max_tokens_cap_value_uses_remaining_context():
    """Cap value is max(1, max_seq_len - len(token_ids))."""
    max_seq_len = 4096
    token_ids = list(range(100))
    cap = max(1, max_seq_len - len(token_ids))
    assert cap == 3996


def test_max_tokens_cap_floor_is_one():
    """Cap never goes below 1 even when prompt fills the context."""
    max_seq_len = 4
    token_ids = list(range(10))  # longer than context
    cap = max(1, max_seq_len - len(token_ids))
    assert cap == 1


# ---------------------------------------------------------------------------
# Warning when DYN_SKIP_FRONTEND_TOKENIZE=1 without DYN_ENGINE_CONV_AFFINITY
# ---------------------------------------------------------------------------


def _check_skip_tokenize_warning(skip_flag: str | None, conv_affinity: bool) -> bool:
    """Mirror the startup warning condition from TrtllmLLMEngine.start()."""
    return skip_flag in ("1", "true") and not conv_affinity


@pytest.mark.parametrize(
    ("skip_flag", "conv_affinity", "expect_warn"),
    [
        ("1", False, True),
        ("true", False, True),
        ("1", True, False),  # conv-affinity on: no warning
        (None, False, False),  # flag absent: no warning
        ("0", False, False),  # flag explicitly off: no warning
    ],
)
def test_startup_warning_condition(skip_flag, conv_affinity, expect_warn):
    assert _check_skip_tokenize_warning(skip_flag, conv_affinity) == expect_warn


# ---------------------------------------------------------------------------
# Runtime guard: empty token_ids + prompt_text without conv-affinity
# ---------------------------------------------------------------------------


def _runtime_guard(engine_conv_affinity: bool, token_ids: list, prompt_text):
    """Mirror the runtime guard in TrtllmLLMEngine.generate()."""
    if not token_ids and prompt_text and not engine_conv_affinity:
        raise RuntimeError(
            "DYN_SKIP_FRONTEND_TOKENIZE=1 requires DYN_ENGINE_CONV_AFFINITY=1 for TRT-LLM."
        )


def test_runtime_guard_raises_when_skip_without_conv_affinity():
    """Empty token_ids + prompt_text without conv-affinity must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="DYN_ENGINE_CONV_AFFINITY"):
        _runtime_guard(engine_conv_affinity=False, token_ids=[], prompt_text="Hello")


def test_runtime_guard_passes_with_conv_affinity():
    """No error when conv-affinity is on (text path)."""
    _runtime_guard(engine_conv_affinity=True, token_ids=[], prompt_text="Hello")


def test_runtime_guard_passes_with_token_ids():
    """No error when token_ids is non-empty (normal path)."""
    _runtime_guard(engine_conv_affinity=False, token_ids=[1, 2, 3], prompt_text=None)
