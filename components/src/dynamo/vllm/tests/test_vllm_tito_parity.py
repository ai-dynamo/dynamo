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
        from dynamo.vllm.response_adapters import serialize_prompt_logprobs

        return serialize_prompt_logprobs

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
    """Verify cache_salt is extracted and tagged for vLLM's prompt."""

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
        """The prompt receives an internal tag around the public cache salt."""
        from vllm.inputs import TokensPrompt

        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request(cache_salt="step_42")
        prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
        _apply_nvext_cache_salt(req, prompt)

        assert prompt.get("cache_salt") == "dynamo-cache-salt:step_42"

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

        assert prefill_prompt["cache_salt"] == "dynamo-cache-salt:step_43"
        assert decode_prompt["cache_salt"] == "dynamo-cache-salt:step_43"

    def test_cache_salt_from_top_level_nvext(self):
        """cache_salt under the raw request["nvext"] shape is also honored,
        matching _nvext_extra_field_requested/_is_token_in_request (tzulingk)."""
        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = {"token_ids": [1, 2, 3], "nvext": {"cache_salt": "top_level"}}
        prompt = {"prompt_token_ids": req["token_ids"]}
        _apply_nvext_cache_salt(req, prompt)

        assert prompt["cache_salt"] == "dynamo-cache-salt:top_level"

    def test_empty_cache_salt_is_absent(self):
        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request(cache_salt="")
        prompt = {"prompt_token_ids": req["token_ids"]}
        _apply_nvext_cache_salt(req, prompt)

        assert "cache_salt" not in prompt

    def test_cache_salt_prefix_is_escaped_by_reprefixing(self):
        from dynamo.vllm.handlers import _apply_nvext_cache_salt

        req = self._build_token_mode_request(cache_salt="dynamo-cache-salt:tenant-a")
        prompt = {"prompt_token_ids": req["token_ids"]}
        _apply_nvext_cache_salt(req, prompt)

        assert prompt["cache_salt"] == ("dynamo-cache-salt:dynamo-cache-salt:tenant-a")


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
        # enable_rl=True so the gate is ACTIVE: a metadata-only extra_fields
        # request is not a token-in request, so generation defaults are kept.
        sp = self._build(
            {"nvext": {"extra_fields": ["timing", "worker_id"]}}, enable_rl=True
        )
        assert sp.top_p == pytest.approx(0.5)

    def test_engine_data_extra_field_keeps_generation_defaults(self):
        # engine_data is a response opt-in, not a token-in marker; with RL on it
        # must still keep generation defaults.
        sp = self._build({"nvext": {"extra_fields": ["engine_data"]}}, enable_rl=True)
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


class TestNonFiniteLogprobs:
    """Regression: -inf/nan logprobs (from bad_words/allowed_token_ids masking
    or full-vocab prompt_logprobs) must be clamped to a finite sentinel. JSON
    has no inf/nan, so pythonize -> serde_json would rewrite them to null and
    the Rust typed deserialization would then silently drop the whole logprobs
    payload."""

    def test_finite_logprob_clamps_non_finite(self):
        import math

        from dynamo.vllm.response_adapters import MIN_FINITE_LOGPROB, finite_logprob

        assert finite_logprob(-0.5) == pytest.approx(-0.5)
        assert finite_logprob(0.0) == 0.0  # legitimate finite value preserved
        assert finite_logprob(float("-inf")) == MIN_FINITE_LOGPROB
        assert finite_logprob(float("inf")) == MIN_FINITE_LOGPROB
        assert finite_logprob(float("nan")) == MIN_FINITE_LOGPROB
        assert math.isfinite(finite_logprob(float("-inf")))

    def test_flatten_logprobs_clamps_inf_and_drops_bool(self):
        from dynamo.vllm.handlers import _flatten_logprobs
        from dynamo.vllm.response_adapters import MIN_FINITE_LOGPROB

        assert _flatten_logprobs(
            [float("-inf"), -0.5, [{"logprob": float("nan")}]]
        ) == [
            MIN_FINITE_LOGPROB,
            pytest.approx(-0.5),
            MIN_FINITE_LOGPROB,
        ]
        # bool is an int subclass; drop it rather than coerce to 1.0/0.0.
        assert _flatten_logprobs([True, -0.5, False]) == [pytest.approx(-0.5)]

    def test_serialize_prompt_logprobs_clamps_inf(self):
        from types import SimpleNamespace

        from dynamo.vllm.response_adapters import (
            MIN_FINITE_LOGPROB,
            serialize_prompt_logprobs,
        )

        out = serialize_prompt_logprobs([{7: SimpleNamespace(logprob=float("-inf"))}])
        assert out[0]["7"]["logprob"] == MIN_FINITE_LOGPROB

    def test_sentinel_is_json_safe(self):
        import json
        import math

        from dynamo.vllm.response_adapters import MIN_FINITE_LOGPROB

        assert math.isfinite(MIN_FINITE_LOGPROB)
        # Must serialize without becoming null (the whole point of the clamp).
        assert json.loads(json.dumps({"logprob": MIN_FINITE_LOGPROB})) == {
            "logprob": MIN_FINITE_LOGPROB
        }


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
        from dynamo.vllm.handlers import _accumulate_engine_data
        from dynamo.vllm.response_adapters import attach_prompt_logprobs_engine_data

        final = {
            "index": 0,
            "token_ids": [11],
            "finish_reason": "stop",
        }
        payload = [None, {"42": {"logprob": -0.5, "rank": 1}}]

        attach_prompt_logprobs_engine_data(final, payload)
        _accumulate_engine_data(final, [1, 2], {}, {})

        assert final["engine_data"]["prompt_logprobs"] == payload
        assert final["engine_data"]["completion_token_ids"] == [11]


class TestSkipSpecialTokens:
    """skip_special_tokens is handled by Dynamo's Rust backend (which reads it
    from the request output_options), NOT forwarded to vLLM SamplingParams: the
    internal token path forces detokenize=False, so vLLM never reads it."""

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

    def test_skip_special_tokens_not_forwarded(self):
        # detokenize is forced off so vLLM ignores skip_special_tokens; it must
        # be left at the vLLM default rather than overridden from output_options.
        sp = self._build(output_options={"skip_special_tokens": False})
        assert sp.detokenize is False
        assert sp.skip_special_tokens is True  # vLLM default, not overridden

    def test_detokenize_forced_off(self):
        sp = self._build(output_options={})
        assert sp.detokenize is False

    def test_prompt_logprobs_still_works(self):
        """Regression: prompt_logprobs is still wired through output_options."""
        sp = self._build(output_options={"prompt_logprobs": 5})
        assert sp.prompt_logprobs == 5

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
        # The internal token path detokenizes downstream in Dynamo, so
        # build_sampling_params forces detokenize=False even when the request
        # asks for True.
        assert sp.detokenize is False
