# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared logprob helpers.

The legacy per-backend extraction logic moved into this module; these
tests cover the contract directly so both the unified engines and the
legacy handlers (which now delegate here) get verified at the same time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from dynamo.common.backend.logprobs import (
    DYN_SGL_ALLOW_TOP_LOGPROBS_ENV,
    build_sglang_logprob_kwargs,
    extract_from_completion_output,
    extract_from_sglang_meta,
    extract_prompt_logprobs_from_completion_output,
    extract_prompt_logprobs_from_sglang_meta,
    parse_logprob_options,
    sglang_top_logprobs_allowed,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# parse_logprob_options
# ---------------------------------------------------------------------------


def test_parse_options_returns_both_when_present():
    assert parse_logprob_options({"logprobs": 3, "prompt_logprobs": 5}) == (3, 5)


def test_parse_options_handles_empty_string_as_absent():
    # vLLM legacy treated empty string as "field absent" — preserved.
    assert parse_logprob_options({"logprobs": "", "prompt_logprobs": ""}) == (
        None,
        None,
    )


def test_parse_options_warns_and_drops_on_negative(caplog):
    with caplog.at_level("WARNING"):
        result = parse_logprob_options({"logprobs": -2})
    assert result == (None, None)
    assert any("logprobs" in r.getMessage() for r in caplog.records)


def test_parse_options_drops_non_integer():
    assert parse_logprob_options({"logprobs": "abc"}) == (None, None)


# ---------------------------------------------------------------------------
# extract_from_completion_output (vLLM/TRT-LLM shape)
# ---------------------------------------------------------------------------


def _logprob(lp: float, rank: int = 1, decoded: str | None = None):
    return SimpleNamespace(logprob=lp, rank=rank, decoded_token=decoded)


def test_extract_completion_returns_none_when_logprobs_absent():
    output = SimpleNamespace(token_ids=[1, 2], logprobs=None)
    assert extract_from_completion_output(output, 0) == (None, None)


def test_extract_completion_returns_none_past_end_of_tokens():
    output = SimpleNamespace(
        token_ids=[7, 8], logprobs=[{7: _logprob(-0.7)}, {8: _logprob(-0.8)}]
    )
    assert extract_from_completion_output(output, 5) == (None, None)


def test_extract_completion_slices_from_offset():
    output = SimpleNamespace(
        token_ids=[7, 8, 9],
        logprobs=[
            {7: _logprob(-0.7, decoded="a")},
            {8: _logprob(-0.8, decoded="b")},
            {9: _logprob(-0.9, decoded="c")},
        ],
    )
    log_probs, top_logprobs = extract_from_completion_output(output, 1)
    assert log_probs == [-0.8, -0.9]
    assert [pos[0]["token_id"] for pos in top_logprobs] == [8, 9]


def test_extract_completion_includes_bytes_when_requested():
    output = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: _logprob(-0.5, decoded="ab")}],
    )
    _, top_logprobs = extract_from_completion_output(output, 0, include_bytes=True)
    assert top_logprobs[0][0]["bytes"] == list("ab".encode("utf-8"))


def test_extract_completion_omits_bytes_by_default():
    output = SimpleNamespace(
        token_ids=[7], logprobs=[{7: _logprob(-0.5, decoded="ab")}]
    )
    _, top_logprobs = extract_from_completion_output(output, 0)
    assert "bytes" not in top_logprobs[0][0]


def test_extract_completion_skips_missing_selected_by_default():
    # vLLM legacy behaviour: when the selected token isn't in the dict,
    # skip that position rather than substitute.
    output = SimpleNamespace(
        token_ids=[7],
        logprobs=[{99: _logprob(-1.0, decoded="x")}],
    )
    log_probs, top_logprobs = extract_from_completion_output(output, 0)
    assert log_probs is None
    assert top_logprobs is None


def test_extract_completion_falls_back_to_first_when_flag_set():
    # TRT-LLM legacy behaviour: fall back to first dict entry.
    output = SimpleNamespace(
        token_ids=[7],
        logprobs=[{99: _logprob(-1.0, decoded="x")}],
    )
    log_probs, _ = extract_from_completion_output(
        output, 0, fallback_to_first_on_missing=True
    )
    assert log_probs == [-1.0]


def test_extract_completion_handles_trtllm_float_edge_case():
    # TRT-LLM sometimes returns a list[float] instead of list[dict].
    output = SimpleNamespace(token_ids=[7, 8], logprobs=[-0.7, -0.8])
    log_probs, top_logprobs = extract_from_completion_output(output, 0)
    assert log_probs == [-0.7, -0.8]
    assert top_logprobs is None


def test_extract_completion_bails_on_none_positions_in_logprobs_array():
    # Engines occasionally emit `None` for a position in the logprobs
    # list (no logprobs computed at that slot). The Rust response
    # builder zips log_probs against token_ids positionally, so emitting
    # a shorter array would misalign every later token — drop the
    # chunk's logprobs entirely instead.
    output = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[None, {8: _logprob(-0.8, decoded="b")}],
    )
    log_probs, top_logprobs = extract_from_completion_output(output, 0)
    assert log_probs is None
    assert top_logprobs is None


def test_extract_completion_bails_when_any_selected_token_missing():
    # Same invariant: a single missing selected token would shrink
    # log_probs out of alignment with token_ids — bail on the chunk.
    output = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: _logprob(-0.7, decoded="a")},
            {99: _logprob(-0.8, decoded="x")},  # 8 not present
        ],
    )
    log_probs, top_logprobs = extract_from_completion_output(output, 0)
    assert log_probs is None
    assert top_logprobs is None


@pytest.mark.parametrize(
    "missing_row",
    [
        None,
        {99: _logprob(-0.8, decoded="x")},
    ],
)
def test_extract_completion_can_preserve_missing_rows(missing_row):
    output = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: _logprob(-0.7, decoded="a")},
            missing_row,
        ],
    )

    log_probs, top_logprobs = extract_from_completion_output(
        output,
        0,
        preserve_missing_rows=True,
    )

    assert log_probs == [-0.7, -9999.0]
    assert [entry["token_id"] for entry in top_logprobs[0]] == [7]
    assert top_logprobs[1] == []


def test_extract_completion_uses_tokenizer_when_decoded_missing():
    class FakeTok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    output = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: _logprob(-0.5, decoded=None)}],
    )
    _, top_logprobs = extract_from_completion_output(output, 0, tokenizer=FakeTok())
    assert top_logprobs[0][0]["token"] == "<7>"


# ---------------------------------------------------------------------------
# extract_prompt_logprobs_from_completion_output (vLLM/TRT-LLM shape)
# ---------------------------------------------------------------------------


def test_prompt_logprobs_completion_returns_none_when_absent():
    output = SimpleNamespace(prompt_logprobs=None)
    assert extract_prompt_logprobs_from_completion_output(output) is None


def test_prompt_logprobs_completion_preserves_none_bos_position():
    # vLLM emits `None` at index 0 (no logprob for the very first token).
    output = SimpleNamespace(
        prompt_logprobs=[
            None,
            {7: _logprob(-0.5, decoded="a")},
        ]
    )
    payload = extract_prompt_logprobs_from_completion_output(output)
    assert payload == [
        None,
        {"7": {"logprob": -0.5, "rank": 1, "decoded_token": "a"}},
    ]


def test_prompt_logprobs_completion_includes_top_k_alternatives():
    output = SimpleNamespace(
        prompt_logprobs=[
            None,
            {
                7: _logprob(-0.5, rank=1, decoded="a"),
                70: _logprob(-1.5, rank=2, decoded="A"),
            },
        ]
    )
    payload = extract_prompt_logprobs_from_completion_output(output)
    assert payload[1] == {
        "7": {"logprob": -0.5, "rank": 1, "decoded_token": "a"},
        "70": {"logprob": -1.5, "rank": 2, "decoded_token": "A"},
    }


def test_prompt_logprobs_completion_falls_back_to_tokenizer():
    class FakeTok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    output = SimpleNamespace(prompt_logprobs=[None, {9: _logprob(-0.7, decoded=None)}])
    payload = extract_prompt_logprobs_from_completion_output(
        output, tokenizer=FakeTok()
    )
    assert payload[1]["9"]["decoded_token"] == "<9>"


# ---------------------------------------------------------------------------
# extract_prompt_logprobs_from_sglang_meta
# ---------------------------------------------------------------------------


def test_prompt_logprobs_sglang_returns_none_when_absent():
    assert extract_prompt_logprobs_from_sglang_meta({}) is None
    assert (
        extract_prompt_logprobs_from_sglang_meta({"input_token_logprobs": []}) is None
    )


def test_prompt_logprobs_sglang_prepends_none_for_bos():
    # SGLang's `input_token_logprobs` starts at prompt position 1; we
    # add `None` at index 0 to align with the Rust PromptLogprobs
    # invariant (BOS has no logprob).
    meta = {
        "input_token_logprobs": [
            (-0.5, 7, "a"),
            (-0.6, 8, "b"),
        ]
    }
    payload = extract_prompt_logprobs_from_sglang_meta(meta)
    assert payload == [
        None,
        {"7": {"logprob": -0.5, "decoded_token": "a"}},
        {"8": {"logprob": -0.6, "decoded_token": "b"}},
    ]


def test_prompt_logprobs_sglang_merges_input_top_logprobs():
    meta = {
        "input_token_logprobs": [(-0.5, 7, "a")],
        "input_top_logprobs": [[(-0.5, 7, "a"), (-1.5, 70, "A")]],
    }
    payload = extract_prompt_logprobs_from_sglang_meta(meta)
    assert payload[1] == {
        "7": {"logprob": -0.5, "decoded_token": "a"},
        "70": {"logprob": -1.5, "decoded_token": "A"},
    }


def test_prompt_logprobs_sglang_handles_missing_decoded_token():
    meta = {"input_token_logprobs": [(-0.7, 9, None)]}
    payload = extract_prompt_logprobs_from_sglang_meta(meta)
    assert payload[1] == {"9": {"logprob": -0.7}}


# ---------------------------------------------------------------------------
# build_sglang_logprob_kwargs
# ---------------------------------------------------------------------------


def test_sglang_kwargs_empty_when_no_options():
    assert build_sglang_logprob_kwargs({}, allow_top_logprobs=True) == {}


def test_sglang_kwargs_logprobs_zero_allowed_without_gate():
    # The default gate forbids logprobs >= 1; logprobs=0 always works.
    kwargs = build_sglang_logprob_kwargs({"logprobs": 0}, allow_top_logprobs=False)
    assert kwargs == {"return_logprob": True, "top_logprobs_num": 0}


def test_sglang_kwargs_top_logprobs_rejected_without_gate():
    with pytest.raises(ValueError, match="does not currently support logprobs >= 1"):
        build_sglang_logprob_kwargs({"logprobs": 2}, allow_top_logprobs=False)


def test_sglang_kwargs_top_logprobs_allowed_with_gate():
    kwargs = build_sglang_logprob_kwargs({"logprobs": 2}, allow_top_logprobs=True)
    assert kwargs == {"return_logprob": True, "top_logprobs_num": 2}


def test_sglang_kwargs_prompt_logprobs_sets_start_len_zero():
    kwargs = build_sglang_logprob_kwargs(
        {"prompt_logprobs": 0}, allow_top_logprobs=False
    )
    assert kwargs == {
        "return_logprob": True,
        "top_logprobs_num": 0,
        "logprob_start_len": 0,
    }


def test_sglang_kwargs_both_set_picks_max():
    kwargs = build_sglang_logprob_kwargs(
        {"logprobs": 3, "prompt_logprobs": 5}, allow_top_logprobs=True
    )
    assert kwargs["top_logprobs_num"] == 5
    assert kwargs["logprob_start_len"] == 0


def test_sglang_gate_reads_env(monkeypatch):
    monkeypatch.setenv(DYN_SGL_ALLOW_TOP_LOGPROBS_ENV, "1")
    assert sglang_top_logprobs_allowed() is True
    monkeypatch.setenv(DYN_SGL_ALLOW_TOP_LOGPROBS_ENV, "0")
    assert sglang_top_logprobs_allowed() is False
    monkeypatch.delenv(DYN_SGL_ALLOW_TOP_LOGPROBS_ENV, raising=False)
    assert sglang_top_logprobs_allowed() is False


# ---------------------------------------------------------------------------
# extract_from_sglang_meta
# ---------------------------------------------------------------------------


def test_sglang_extract_returns_none_when_meta_empty():
    assert extract_from_sglang_meta({}, 0) == (None, None, 0)


def test_sglang_extract_slices_cumulative_array():
    meta = {
        "output_token_logprobs": [(-0.1, 1, "a"), (-0.2, 2, "b"), (-0.3, 3, "c")],
    }
    log_probs, top_logprobs, new_total = extract_from_sglang_meta(meta, 1)
    assert log_probs == [-0.2, -0.3]
    assert top_logprobs is None
    assert new_total == 3


def test_sglang_extract_with_top():
    meta = {
        "output_token_logprobs": [(-0.1, 101, "a")],
        "output_top_logprobs": [[(-0.1, 101, "a"), (-0.2, 102, "b")]],
    }
    log_probs, top_logprobs, _ = extract_from_sglang_meta(meta, 0)
    assert log_probs == [-0.1]
    assert top_logprobs == [
        [
            {"rank": 1, "token_id": 101, "token": "a", "logprob": -0.1},
            {"rank": 2, "token_id": 102, "token": "b", "logprob": -0.2},
        ]
    ]


def test_sglang_extract_return_tokens_as_token_ids():
    meta = {
        "output_token_logprobs": [(-0.1, 101, "a")],
        "output_top_logprobs": [[(-0.1, 101, "a")]],
    }
    _, top_logprobs, _ = extract_from_sglang_meta(
        meta, 0, return_tokens_as_token_ids=True
    )
    assert top_logprobs[0][0]["token"] == "token_id:101"


def test_sglang_extract_none_top_position_becomes_empty_list():
    # SGLang's output_top_logprobs entry can be None for positions
    # without top-k data — preserve as an empty list per-position so the
    # caller can still index by position without breaking alignment.
    meta = {
        "output_token_logprobs": [(-0.1, 101, "a"), (-0.2, 102, "b")],
        "output_top_logprobs": [None, [(-0.2, 102, "b")]],
    }
    _, top_logprobs, _ = extract_from_sglang_meta(meta, 0)
    assert top_logprobs == [
        [],
        [{"rank": 1, "token_id": 102, "token": "b", "logprob": -0.2}],
    ]


def test_sglang_extract_returns_offset_unchanged_when_no_new_entries():
    meta = {"output_token_logprobs": [(-0.1, 1, "a")]}
    _, _, new_total = extract_from_sglang_meta(meta, 1)
    assert new_total == 1


# ---------------------------------------------------------------------------
# Legacy ↔ unified behavioural parity corner cases.
#
# The legacy handlers and unified engines call the same shared helpers, so
# the call sites should produce byte-identical output for any given input.
# The tests below simulate each engine's wire format and confirm parity by
# driving both call patterns side-by-side.
# ---------------------------------------------------------------------------


def _vllm_legacy_call(output, num_so_far, tokenizer=None):
    """Mirror of ``BaseWorkerHandler._extract_logprobs`` — vLLM legacy.
    `fallback_to_first_on_missing=True` matches the pre-shared-helper
    extractor that always emitted a logprob when vLLM returned a dict."""
    return extract_from_completion_output(
        output,
        num_so_far,
        tokenizer=tokenizer,
        fallback_to_first_on_missing=True,
        include_bytes=True,
    )


def _vllm_unified_call(output, tokenizer=None):
    """Mirror of ``VllmLLMEngine.generate``'s per-chunk extraction.
    DELTA outputs always slice from offset 0."""
    return extract_from_completion_output(
        output,
        0,
        tokenizer=tokenizer,
        fallback_to_first_on_missing=True,
        include_bytes=True,
    )


def _trtllm_call(output, num_so_far):
    """Mirror of both ``HandlerBase._extract_logprobs`` and
    ``TrtllmLLMEngine.generate``'s per-chunk extraction — TRT-LLM uses
    cumulative arrays on both call sites with the same offset."""
    return extract_from_completion_output(
        output,
        num_so_far,
        fallback_to_first_on_missing=True,
        include_bytes=False,
    )


def test_parity_vllm_delta_vs_cumulative_yields_same_logprobs():
    """vLLM's legacy path slices a *cumulative* CompletionOutput with a
    growing offset; the unified path receives DELTA chunks and slices
    from offset 0. For an equivalent token stream both paths must emit
    the same flat list of logprobs."""
    cumulative_tokens = [7, 8, 9]
    cumulative_logprobs = [
        {7: _logprob(-0.7, decoded="a")},
        {8: _logprob(-0.8, decoded="b")},
        {9: _logprob(-0.9, decoded="c")},
    ]

    legacy_flat: list[float] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=cumulative_tokens[: i + 1],
            logprobs=cumulative_logprobs[: i + 1],
        )
        lp, _ = _vllm_legacy_call(output, i)
        if lp:
            legacy_flat.extend(lp)

    unified_flat: list[float] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=[cumulative_tokens[i]],
            logprobs=[cumulative_logprobs[i]],
        )
        lp, _ = _vllm_unified_call(output)
        if lp:
            unified_flat.extend(lp)

    assert legacy_flat == unified_flat == [-0.7, -0.8, -0.9]


def test_parity_vllm_delta_vs_cumulative_yields_same_top_logprobs():
    """Top-logprobs are also identical across the two slicing schemes."""
    cumulative_tokens = [7, 8]
    cumulative_logprobs = [
        {7: _logprob(-0.7, decoded="a"), 70: _logprob(-1.7, decoded="A")},
        {8: _logprob(-0.8, decoded="b"), 80: _logprob(-1.8, decoded="B")},
    ]

    legacy_top: list[list[dict]] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=cumulative_tokens[: i + 1],
            logprobs=cumulative_logprobs[: i + 1],
        )
        _, top = _vllm_legacy_call(output, i)
        if top:
            legacy_top.extend(top)

    unified_top: list[list[dict]] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=[cumulative_tokens[i]],
            logprobs=[cumulative_logprobs[i]],
        )
        _, top = _vllm_unified_call(output)
        if top:
            unified_top.extend(top)

    assert legacy_top == unified_top
    # Confirm `bytes` field is included on both paths (vLLM-specific).
    assert legacy_top[0][0]["bytes"] == list("a".encode("utf-8"))


def test_parity_trtllm_extraction_is_idempotent_across_call_sites():
    """TRT-LLM's legacy `_extract_logprobs` and the unified call use the
    same flags and same offset — confirm a representative payload
    produces byte-identical output through both call patterns."""
    output = SimpleNamespace(
        token_ids=[7, 8, 9],
        logprobs=[
            {7: _logprob(-0.7), 70: _logprob(-1.7)},
            {8: _logprob(-0.8)},
            # Selected-token missing — fallback_to_first must kick in.
            {99: _logprob(-1.9)},
        ],
    )

    legacy_log, legacy_top = _trtllm_call(output, num_so_far=0)
    unified_log, unified_top = _trtllm_call(output, num_so_far=0)

    assert legacy_log == unified_log == [-0.7, -0.8, -1.9]
    assert legacy_top == unified_top
    # Confirm bytes NOT included on TRT-LLM (legacy & unified).
    assert "bytes" not in legacy_top[0][0]


def test_parity_sglang_per_choice_offset_tracking_multi_choice():
    """The unified SGLang generate() tracks `num_logprobs_per_choice` —
    a dict keyed by output_idx. Without per-choice tracking, interleaved
    n>1 chunks would mis-slice each other's cumulative arrays. Simulate
    interleaved chunks for choice 0 and choice 1 and confirm each
    choice's logprobs are extracted correctly."""
    # SGLang emits ONE cumulative array per choice. The unified engine
    # receives chunks tagged with `index` and tracks an offset per index.
    # Sim: choice 0 emits 3 tokens, choice 1 emits 2, interleaved.
    choice_0_arrays = [
        {  # after step 1
            "output_token_logprobs": [(-0.1, 100, "a")],
        },
        {  # after step 2 (cumulative)
            "output_token_logprobs": [(-0.1, 100, "a"), (-0.2, 101, "b")],
        },
        {  # after step 3 (cumulative)
            "output_token_logprobs": [
                (-0.1, 100, "a"),
                (-0.2, 101, "b"),
                (-0.3, 102, "c"),
            ],
        },
    ]
    choice_1_arrays = [
        {"output_token_logprobs": [(-0.5, 200, "x")]},
        {
            "output_token_logprobs": [(-0.5, 200, "x"), (-0.6, 201, "y")],
        },
    ]

    # Interleaved arrival order: c0[0], c1[0], c0[1], c1[1], c0[2].
    arrival = [
        (0, choice_0_arrays[0]),
        (1, choice_1_arrays[0]),
        (0, choice_0_arrays[1]),
        (1, choice_1_arrays[1]),
        (0, choice_0_arrays[2]),
    ]

    num_logprobs_per_choice: dict[int, int] = {}
    extracted_per_choice: dict[int, list[float]] = {}
    for idx, meta in arrival:
        lp, _, next_total = extract_from_sglang_meta(
            meta, num_logprobs_per_choice.get(idx, 0)
        )
        num_logprobs_per_choice[idx] = next_total
        if lp:
            extracted_per_choice.setdefault(idx, []).extend(lp)

    assert extracted_per_choice[0] == [-0.1, -0.2, -0.3]
    assert extracted_per_choice[1] == [-0.5, -0.6]


def test_parity_sglang_single_offset_misslices_under_n_gt_1():
    """Negative control for the bug we fixed: a *scalar* offset (the
    pre-fix unified behaviour) would mis-slice when chunks for two
    different choices are interleaved on the same offset cursor. This
    test exists so the regression is obvious if the per-choice dict
    ever gets simplified back into a scalar."""
    choice_0 = {"output_token_logprobs": [(-0.1, 100, "a")]}
    choice_1 = {"output_token_logprobs": [(-0.5, 200, "x")]}

    # Single scalar offset across both choices (buggy mode).
    num_logprobs_so_far = 0
    extracted_logprobs: list[float] = []
    for meta in (choice_0, choice_1):
        lp, _, num_logprobs_so_far = extract_from_sglang_meta(meta, num_logprobs_so_far)
        if lp:
            extracted_logprobs.extend(lp)

    # The second extraction returns nothing because the scalar offset
    # (=1 after the first chunk) skips choice 1's first entry. That's the
    # bug — we'd lose choice 1's logprobs entirely.
    assert extracted_logprobs == [-0.1]


def test_parity_parse_logprob_options_consistent_across_engines():
    """Every engine path goes through `parse_logprob_options` for the
    options→ints normalization. Confirm the function is the single source
    of truth — a battery of edge-case inputs returns the same parsed
    output regardless of which engine consumes it next."""
    cases: list[tuple[dict, tuple[Optional[int], Optional[int]]]] = [
        ({}, (None, None)),
        ({"logprobs": 0}, (0, None)),
        ({"logprobs": 5}, (5, None)),
        ({"prompt_logprobs": 3}, (None, 3)),
        ({"logprobs": 2, "prompt_logprobs": 4}, (2, 4)),
        ({"logprobs": ""}, (None, None)),
        ({"logprobs": "7"}, (7, None)),  # str → int coercion
        ({"logprobs": -1}, (None, None)),
        ({"logprobs": "garbage"}, (None, None)),
    ]
    for opts, expected in cases:
        assert parse_logprob_options(opts) == expected, opts


def test_parity_sglang_kwargs_rejects_top_logprobs_consistently():
    """Both legacy decode handler and unified engine call into
    `build_sglang_logprob_kwargs` with the env-derived gate. With the
    gate off, any `logprobs >= 1` (or `prompt_logprobs >= 1`) request
    must raise — regardless of which call site invoked it."""
    for opts in (
        {"logprobs": 1},
        {"logprobs": 5},
        {"prompt_logprobs": 2},
        {"logprobs": 0, "prompt_logprobs": 3},
    ):
        with pytest.raises(ValueError, match="does not currently support"):
            build_sglang_logprob_kwargs(opts, allow_top_logprobs=False)


# ---------------------------------------------------------------------------
# Direct wrapper-vs-shared parity. These tests import the legacy static
# methods on each engine's handler class and assert byte-identical output
# vs a direct shared call with the same flags. They catch wrapper rot —
# e.g. someone editing the static method to drop a flag.
#
# `pytest.importorskip` skips when the engine package isn't installed
# (developer machines without vllm/sglang/trtllm). CI envs with the
# engines will execute them.
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_parity_vllm_legacy_wrapper_matches_shared():
    # `exc_type=ImportError` also skips on missing native deps (libcuda)
    # on CPU-only lanes where the wheel is installed but no GPU runtime.
    pytest.importorskip("vllm", reason="vLLM not installed", exc_type=ImportError)
    from dynamo.vllm.handlers import BaseWorkerHandler

    output = SimpleNamespace(
        token_ids=[11, 12],
        logprobs=[
            {11: _logprob(-0.1, decoded="a"), 110: _logprob(-1.1, decoded="A")},
            {12: _logprob(-0.2, decoded="b")},
        ],
    )

    wrapper_lp, wrapper_top = BaseWorkerHandler._extract_logprobs(output, 0)
    direct_lp, direct_top = extract_from_completion_output(
        output, 0, fallback_to_first_on_missing=True, include_bytes=True
    )
    assert wrapper_lp == direct_lp
    assert wrapper_top == direct_top


@pytest.mark.trtllm
def test_parity_trtllm_legacy_wrapper_matches_shared():
    pytest.importorskip(
        "tensorrt_llm", reason="TRT-LLM not installed", exc_type=ImportError
    )
    from dynamo.trtllm.request_handlers.handler_base import HandlerBase

    output = SimpleNamespace(
        token_ids=[11, 12],
        logprobs=[
            {11: _logprob(-0.1), 110: _logprob(-1.1)},
            # Selected token missing — exercises the fallback flag.
            {99: _logprob(-9.9)},
        ],
    )

    wrapper_lp, wrapper_top = HandlerBase._extract_logprobs(output, 0)
    direct_lp, direct_top = extract_from_completion_output(
        output, 0, fallback_to_first_on_missing=True, include_bytes=False
    )
    assert wrapper_lp == direct_lp
    assert wrapper_top == direct_top


@pytest.mark.sglang
def test_parity_sglang_legacy_extract_wrapper_matches_shared():
    pytest.importorskip("sglang", reason="SGLang not installed", exc_type=ImportError)
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

    meta = {
        "output_token_logprobs": [(-0.1, 101, "a"), (-0.2, 102, "b")],
        "output_top_logprobs": [
            [(-0.1, 101, "a"), (-0.5, 105, "x")],
            [(-0.2, 102, "b")],
        ],
    }

    wrapper = DecodeWorkerHandler._extract_logprobs(meta, 0)
    direct = extract_from_sglang_meta(meta, 0)
    assert wrapper == direct


@pytest.mark.sglang
def test_parity_sglang_legacy_kwargs_wrapper_matches_shared(monkeypatch):
    pytest.importorskip("sglang", reason="SGLang not installed", exc_type=ImportError)
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

    # Gate off — both must reject logprobs >= 1 identically.
    monkeypatch.delenv(DYN_SGL_ALLOW_TOP_LOGPROBS_ENV, raising=False)
    request = {"output_options": {"logprobs": 0, "prompt_logprobs": 0}}

    wrapper_kwargs = DecodeWorkerHandler._build_logprob_kwargs(request)
    direct_kwargs = build_sglang_logprob_kwargs(
        request["output_options"], allow_top_logprobs=False
    )
    assert wrapper_kwargs == direct_kwargs


# ---------------------------------------------------------------------------
# Whole-stream parity. Drive a simulated multi-chunk generation through both
# paths and confirm not just the per-chunk logprobs, but the byte-identical
# top-logprob dict shape (rank/token_id/token/logprob/bytes) lines up.
# ---------------------------------------------------------------------------


def test_parity_vllm_full_stream_byte_identical_top_logprobs():
    """Whole-stream parity: a 4-token generation with non-trivial top-k
    alternatives must emit byte-identical top_logprobs through both
    legacy cumulative+offset and unified DELTA+0 paths."""
    cumulative_tokens = [10, 20, 30, 40]
    # Each position has 3 alternatives — selected first, then two off-pol.
    cumulative_logprobs = [
        {
            10: _logprob(-0.1, rank=1, decoded="ten"),
            11: _logprob(-1.1, rank=2, decoded="11"),
            12: _logprob(-2.1, rank=3, decoded="12"),
        },
        {
            20: _logprob(-0.2, rank=1, decoded="twenty"),
            21: _logprob(-1.2, rank=2, decoded="21"),
        },
        {
            30: _logprob(-0.3, rank=1, decoded="thirty"),
            31: _logprob(-1.3, rank=2, decoded="31"),
            32: _logprob(-2.3, rank=3, decoded="32"),
            33: _logprob(-3.3, rank=4, decoded="33"),
        },
        {
            40: _logprob(-0.4, rank=1, decoded="forty"),
        },
    ]

    def _run_legacy() -> list[list[dict]]:
        # Legacy: receives the cumulative array each step, slices by offset.
        flat: list[list[dict]] = []
        for i in range(len(cumulative_tokens)):
            output = SimpleNamespace(
                token_ids=cumulative_tokens[: i + 1],
                logprobs=cumulative_logprobs[: i + 1],
            )
            _, top = _vllm_legacy_call(output, i)
            if top:
                flat.extend(top)
        return flat

    def _run_unified() -> list[list[dict]]:
        # Unified: receives the DELTA chunk each step, offset always 0.
        flat: list[list[dict]] = []
        for i in range(len(cumulative_tokens)):
            output = SimpleNamespace(
                token_ids=[cumulative_tokens[i]],
                logprobs=[cumulative_logprobs[i]],
            )
            _, top = _vllm_unified_call(output)
            if top:
                flat.extend(top)
        return flat

    legacy = _run_legacy()
    unified = _run_unified()

    # Byte-identical comparison on the full structure: rank, token_id,
    # token, logprob, bytes are all included.
    assert legacy == unified
    assert len(legacy) == 4
    # Spot-check a representative entry.
    assert legacy[2] == [
        {
            "rank": 1,
            "token_id": 30,
            "token": "thirty",
            "logprob": -0.3,
            "bytes": list("thirty".encode("utf-8")),
        },
        {
            "rank": 2,
            "token_id": 31,
            "token": "31",
            "logprob": -1.3,
            "bytes": list("31".encode("utf-8")),
        },
        {
            "rank": 3,
            "token_id": 32,
            "token": "32",
            "logprob": -2.3,
            "bytes": list("32".encode("utf-8")),
        },
        {
            "rank": 4,
            "token_id": 33,
            "token": "33",
            "logprob": -3.3,
            "bytes": list("33".encode("utf-8")),
        },
    ]


def test_parity_vllm_with_none_position_mid_stream():
    """If the engine emits None for a mid-stream position (no logprobs
    sampled at that slot), both paths must skip the position consistently
    without misaligning subsequent entries."""
    cumulative_tokens = [10, 20, 30]
    cumulative_logprobs = [
        {10: _logprob(-0.1, decoded="a")},
        None,  # engine emitted no logprobs for position 1
        {30: _logprob(-0.3, decoded="c")},
    ]

    legacy: list[float] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=cumulative_tokens[: i + 1],
            logprobs=cumulative_logprobs[: i + 1],
        )
        lp, _ = _vllm_legacy_call(output, i)
        if lp:
            legacy.extend(lp)

    unified: list[float] = []
    for i in range(len(cumulative_tokens)):
        output = SimpleNamespace(
            token_ids=[cumulative_tokens[i]],
            logprobs=[cumulative_logprobs[i]],
        )
        lp, _ = _vllm_unified_call(output)
        if lp:
            unified.extend(lp)

    # Both skip the None position (one fewer entry than tokens emitted).
    assert legacy == unified == [-0.1, -0.3]


def test_parity_trtllm_selected_token_missing_fallback_consistent():
    """TRT-LLM fallback: when the engine doesn't include the selected
    token in the logprob dict, both legacy and unified fall back to the
    first dict entry. Confirm the fallback is consistent across a
    multi-position stream where SOME positions hit the fallback."""
    output = SimpleNamespace(
        token_ids=[100, 200, 300],
        logprobs=[
            # Position 0: selected present, normal path.
            {100: _logprob(-0.5), 101: _logprob(-1.5)},
            # Position 1: selected missing, fallback to first.
            {201: _logprob(-2.5), 202: _logprob(-3.5)},
            # Position 2: selected present again.
            {300: _logprob(-0.7), 301: _logprob(-1.7)},
        ],
    )

    log_probs, top_logprobs = _trtllm_call(output, num_so_far=0)
    # Position 1 falls back to first dict entry = 201 (-2.5).
    assert log_probs == [-0.5, -2.5, -0.7]
    assert top_logprobs is not None
    # Top remains the full dict at each position — fallback doesn't
    # change the top list itself.
    assert [len(p) for p in top_logprobs] == [2, 2, 2]


def test_parity_sglang_cumulative_two_chunks_yields_full_stream():
    """SGLang receives cumulative arrays. After two chunks (3 tokens
    total), the running extracted-logprobs list matches the engine's
    final state — proving the offset advances correctly."""
    chunk_1_meta = {
        "output_token_logprobs": [(-0.1, 100, "a")],
        "output_top_logprobs": [[(-0.1, 100, "a"), (-0.5, 105, "x")]],
    }
    chunk_2_meta = {
        "output_token_logprobs": [
            (-0.1, 100, "a"),
            (-0.2, 200, "b"),
            (-0.3, 300, "c"),
        ],
        "output_top_logprobs": [
            [(-0.1, 100, "a"), (-0.5, 105, "x")],
            [(-0.2, 200, "b"), (-0.6, 205, "y")],
            [(-0.3, 300, "c")],
        ],
    }

    extracted_lp: list[float] = []
    extracted_top: list[list[dict]] = []
    offset = 0
    for meta in (chunk_1_meta, chunk_2_meta):
        lp, top, offset = extract_from_sglang_meta(meta, offset)
        if lp:
            extracted_lp.extend(lp)
        if top:
            extracted_top.extend(top)

    assert extracted_lp == [-0.1, -0.2, -0.3]
    assert [pos[0]["token_id"] for pos in extracted_top] == [100, 200, 300]
    # First chunk: 1 entry; second chunk: 2 new entries. Total 3.
    assert len(extracted_top) == 3
    assert offset == 3
