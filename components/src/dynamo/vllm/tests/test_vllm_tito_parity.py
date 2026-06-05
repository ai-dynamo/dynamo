# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM token-in/token-out request handling."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.skipif(
        importlib.util.find_spec("vllm") is None,
        reason="vllm not installed in this container",
    ),
]


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
        assert "42" in entry
        assert entry["42"]["logprob"] == pytest.approx(-1.5)
        assert entry["42"]["rank"] == 1
        assert entry["42"]["decoded_token"] == "hello"

    def test_mixed_none_and_entries(self):
        fn = self._import()
        lp1 = SimpleNamespace(logprob=-0.1, rank=1, decoded_token="a")
        lp2 = SimpleNamespace(logprob=-2.3, rank=5, decoded_token="b")
        raw = [None, {10: lp1, 20: lp2}, None]
        result = fn(raw)
        assert result[0] is None
        assert result[2] is None
        assert set(result[1].keys()) == {"10", "20"}

    def test_missing_optional_attributes(self):
        """Logprob objects without rank/decoded_token should omit those keys."""
        fn = self._import()
        logprob = SimpleNamespace(logprob=-3.0)
        raw = [{7: logprob}]
        result = fn(raw)
        assert result[0]["7"]["logprob"] == pytest.approx(-3.0)
        assert "rank" not in result[0]["7"]
        assert "decoded_token" not in result[0]["7"]

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

        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request(cache_salt="step_42")
        prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
        _apply_nvext_cache_salt(req, prompt)

        assert prompt.get("cache_salt") == "step_42"

    def test_no_cache_salt_when_absent(self):
        """When extra_args has no cache_salt, prompt should not gain the key."""
        from vllm.inputs import TokensPrompt

        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request()
        prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
        _apply_nvext_cache_salt(req, prompt)

        assert "cache_salt" not in prompt

    def test_prefill_and_decode_share_cache_salt_helper(self):
        """Regression: disaggregated prefill and decode both salt prompt cache."""
        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request(cache_salt="step_43")
        prefill_prompt = {"prompt_token_ids": req["token_ids"]}
        decode_prompt = {"prompt_token_ids": req["token_ids"]}

        _apply_nvext_cache_salt(req, prefill_prompt)
        _apply_nvext_cache_salt(req, decode_prompt)

        assert prefill_prompt["cache_salt"] == "step_43"
        assert decode_prompt["cache_salt"] == "step_43"


class TestTokenInSamplingDefaults:
    @staticmethod
    def _build(extra_args=None, enable_rl=False, nvext=None):
        from dynamo.vllm.handlers import build_sampling_params

        req = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "output_options": {},
        }
        if extra_args is not None:
            req["extra_args"] = extra_args
        if nvext is not None:
            req["nvext"] = nvext
        return build_sampling_params(req, {"top_p": 0.5}, enable_rl=enable_rl)

    def test_metadata_extra_fields_keep_generation_defaults(self):
        sp = self._build({"nvext": {"extra_fields": ["timing", "worker_id"]}})
        assert sp.top_p == pytest.approx(0.5)

    def test_engine_data_extra_field_keeps_generation_defaults(self):
        sp = self._build({"nvext": {"extra_fields": ["engine_data"]}})
        assert sp.top_p == pytest.approx(0.5)

    def test_token_in_marker_skips_generation_defaults(self):
        sp = self._build({"nvext": {"token_in": True}}, enable_rl=True)
        assert sp.top_p == pytest.approx(1.0)

    def test_token_data_keeps_generation_defaults_when_rl_disabled(self):
        sp = self._build(nvext={"token_data": [1, 2, 3]}, enable_rl=False)
        assert sp.top_p == pytest.approx(0.5)

    def test_token_data_skips_generation_defaults_when_rl_enabled(self):
        sp = self._build(nvext={"token_data": [1, 2, 3]}, enable_rl=True)
        assert sp.top_p == pytest.approx(1.0)


class TestFlattenLogprobs:
    def test_nested_lists_are_fully_flattened(self):
        from dynamo.vllm.handlers import _flatten_logprobs

        assert _flatten_logprobs(
            [[{"logprob": -0.1}, {"logprob": -0.2}], [-0.3]]
        ) == pytest.approx([-0.1, -0.2, -0.3])


class TestEngineDataAccumulation:
    def test_prompt_token_ids_come_from_built_prompt_when_available(self):
        from dynamo.vllm.handlers import _prompt_token_ids_for_engine_data

        request = {"token_ids": [1, 2, 3]}
        prompt = {"prompt_token_ids": [1, 2, 99, 100]}

        assert _prompt_token_ids_for_engine_data(request, prompt) == [1, 2, 99, 100]

    def test_prompt_token_ids_fall_back_to_request_tokens(self):
        from dynamo.vllm.handlers import _prompt_token_ids_for_engine_data

        request = {"token_ids": [1, 2, 3]}

        assert _prompt_token_ids_for_engine_data(request, "plain prompt") == [1, 2, 3]

    def test_accumulates_per_output_index(self):
        from dynamo.vllm.handlers import _accumulate_engine_data

        token_ids: dict[int, list[int]] = {}
        logprobs: dict[int, list[float]] = {}

        _accumulate_engine_data(
            {"index": 0, "token_ids": [10], "log_probs": [-0.1]},
            [1, 2],
            token_ids,
            logprobs,
        )
        _accumulate_engine_data(
            {"index": 1, "token_ids": [20], "log_probs": [-0.2]},
            [1, 2],
            token_ids,
            logprobs,
        )

        final = {
            "index": 0,
            "token_ids": [11],
            "log_probs": [-0.3],
            "finish_reason": "stop",
        }
        _accumulate_engine_data(final, [1, 2], token_ids, logprobs)

        assert final["engine_data"]["completion_token_ids"] == [10, 11]
        assert final["engine_data"]["completion_logprobs"] == pytest.approx(
            [-0.1, -0.3]
        )
        assert token_ids[1] == [20]

    def test_engine_data_preserves_prompt_logprobs(self):
        from dynamo.vllm.handlers import (
            _accumulate_engine_data,
            _attach_prompt_logprobs_engine_data,
        )

        final = {
            "index": 0,
            "token_ids": [11],
            "finish_reason": "stop",
        }
        payload = [None, {"42": {"logprob": -0.5, "rank": 1}}]

        _attach_prompt_logprobs_engine_data(final, payload)
        _accumulate_engine_data(final, [1, 2], {}, {})

        assert final["engine_data"]["prompt_logprobs"] == payload
        assert final["engine_data"]["completion_token_ids"] == [11]


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

    def test_token_id_constraints(self):
        from dynamo.vllm.handlers import build_sampling_params

        req = {
            "token_ids": [1, 2, 3],
            "sampling_options": {
                "allowed_token_ids": [10, 11],
                "bad_words_token_ids": [[12, 13]],
                "detokenize": True,
            },
            "stop_conditions": {},
            "output_options": {},
        }

        sp = build_sampling_params(req, {})
        assert sp.allowed_token_ids == [10, 11]
        assert sp.bad_words_token_ids == [[12, 13]]
        assert sp.detokenize is True
