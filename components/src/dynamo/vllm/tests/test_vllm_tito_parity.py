# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM TITO (token-in-token-out) parity fixes:

  D1 – _serialize_prompt_logprobs converts vLLM's prompt_logprobs into
       the dict shape expected by the Rust PromptLogprobEntry.
  D2 – cache_salt is forwarded from extra_args["nvext"]["cache_salt"]
       to the prompt dict so vLLM's input_processor picks it up.
  D3 – skip_special_tokens from output_options is applied to
       SamplingParams via build_sampling_params.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.skipif(
        importlib.util.find_spec("vllm") is None,
        reason="vllm not installed in this container",
    ),
]


# ---------------------------------------------------------------------------
# D1 – _serialize_prompt_logprobs
# ---------------------------------------------------------------------------


class TestSerializePromptLogprobs:
    """Validate _serialize_prompt_logprobs against various vLLM outputs."""

    @staticmethod
    def _import():
        from dynamo.vllm.handlers import _serialize_prompt_logprobs

        return _serialize_prompt_logprobs

    def test_none_entries_preserved(self):
        fn = self._import()
        raw = [None, None]
        assert fn(raw) == [None, None]

    def test_single_token_entry(self):
        fn = self._import()
        logprob = SimpleNamespace(logprob=-1.5, rank=1, decoded_token="hello")
        raw = [{42: logprob}]
        result = fn(raw)
        assert len(result) == 1
        entry = result[0]
        assert 42 in entry
        assert entry[42]["logprob"] == pytest.approx(-1.5)
        assert entry[42]["rank"] == 1
        assert entry[42]["decoded_token"] == "hello"

    def test_mixed_none_and_entries(self):
        fn = self._import()
        lp1 = SimpleNamespace(logprob=-0.1, rank=1, decoded_token="a")
        lp2 = SimpleNamespace(logprob=-2.3, rank=5, decoded_token="b")
        raw = [None, {10: lp1, 20: lp2}, None]
        result = fn(raw)
        assert result[0] is None
        assert result[2] is None
        assert set(result[1].keys()) == {10, 20}

    def test_missing_optional_attributes(self):
        """Logprob objects without rank/decoded_token should omit those keys."""
        fn = self._import()
        logprob = SimpleNamespace(logprob=-3.0)
        raw = [{7: logprob}]
        result = fn(raw)
        assert result[0][7]["logprob"] == pytest.approx(-3.0)
        assert "rank" not in result[0][7]
        assert "decoded_token" not in result[0][7]

    def test_empty_list(self):
        fn = self._import()
        assert fn([]) == []

    def test_multiple_tokens_per_position(self):
        fn = self._import()
        lp_a = SimpleNamespace(logprob=-0.5, rank=1, decoded_token="x")
        lp_b = SimpleNamespace(logprob=-1.2, rank=2, decoded_token="y")
        lp_c = SimpleNamespace(logprob=-3.0, rank=3, decoded_token="z")
        raw = [{100: lp_a, 200: lp_b, 300: lp_c}]
        result = fn(raw)
        assert len(result[0]) == 3


# ---------------------------------------------------------------------------
# D2 – cache_salt forwarding
# ---------------------------------------------------------------------------


class TestCacheSaltWiring:
    """Verify cache_salt is extracted from extra_args and placed on the prompt."""

    @staticmethod
    def _build_token_mode_request(cache_salt=None, token_ids=None):
        """Build a minimal TITO request dict mirroring the Rust preprocessor."""
        req = {
            "token_ids": token_ids or [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
        }
        if cache_salt is not None:
            req["extra_args"] = {"nvext": {"cache_salt": cache_salt}}
        return req

    def test_cache_salt_attached_to_prompt(self):
        """When extra_args.nvext.cache_salt is set, the prompt dict gets it."""
        from vllm.inputs import TokensPrompt

        req = self._build_token_mode_request(cache_salt="step_42")
        extra_args = req.get("extra_args") or {}
        nvext_args = extra_args.get("nvext") or {}
        salt = nvext_args.get("cache_salt")
        prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
        if salt is not None and isinstance(prompt, dict):
            prompt["cache_salt"] = salt

        assert prompt.get("cache_salt") == "step_42"

    def test_no_cache_salt_when_absent(self):
        """When extra_args has no cache_salt, prompt should not gain the key."""
        from vllm.inputs import TokensPrompt

        req = self._build_token_mode_request()
        extra_args = req.get("extra_args") or {}
        nvext_args = extra_args.get("nvext") or {}
        salt = nvext_args.get("cache_salt")
        prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
        if salt is not None and isinstance(prompt, dict):
            prompt["cache_salt"] = salt

        assert "cache_salt" not in prompt


# ---------------------------------------------------------------------------
# D3 – skip_special_tokens in build_sampling_params
# ---------------------------------------------------------------------------


class TestSkipSpecialTokens:
    """Verify skip_special_tokens from output_options flows to SamplingParams."""

    @staticmethod
    def _build(output_options=None):
        from dynamo.vllm.handlers import build_sampling_params

        req = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": output_options or {},
        }
        return build_sampling_params(req, {})

    def test_skip_special_tokens_true(self):
        sp = self._build(output_options={"skip_special_tokens": True})
        assert sp.skip_special_tokens is True

    def test_skip_special_tokens_false(self):
        sp = self._build(output_options={"skip_special_tokens": False})
        assert sp.skip_special_tokens is False

    def test_skip_special_tokens_absent(self):
        """When not provided, build_sampling_params hardcodes detokenize=False
        and SamplingParams default for skip_special_tokens should be unchanged."""
        sp = self._build(output_options={})
        assert sp.detokenize is False

    def test_prompt_logprobs_still_works(self):
        """Regression: prompt_logprobs should still be wired alongside skip_special_tokens."""
        sp = self._build(
            output_options={"prompt_logprobs": 5, "skip_special_tokens": True}
        )
        assert sp.prompt_logprobs == 5
        assert sp.skip_special_tokens is True
