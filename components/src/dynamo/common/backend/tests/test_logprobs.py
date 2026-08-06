# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared logprob helpers and active handler adapters."""

from __future__ import annotations

from types import SimpleNamespace

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
# Active handler adapters. These tests import the static methods on each
# backend's handler class and compare them with the shared helper. They catch
# adapter drift, such as dropping a backend-specific flag.
#
# `pytest.importorskip` skips when the engine package isn't installed
# (developer machines without vllm/sglang/trtllm). CI envs with the
# engines will execute them.
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_vllm_handler_matches_shared():
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
def test_trtllm_handler_matches_shared():
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
def test_sglang_extract_handler_matches_shared():
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
def test_sglang_kwargs_handler_matches_shared(monkeypatch):
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
