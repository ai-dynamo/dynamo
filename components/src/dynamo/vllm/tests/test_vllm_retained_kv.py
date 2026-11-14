# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private retained-KV hints at the ordinary prefill/decode handler boundary."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_vllm_worker_handler import (
    _make_decode_handler,
    _make_engine_response,
    _make_raw_frontend_request,
)
from test_vllm_worker_handler import (
    prefill_probe_handler as _shared_prefill_probe_handler,
)
from vllm.outputs import CompletionOutput, RequestOutput

import dynamo.vllm.handlers as handlers
from dynamo.vllm.cache_info import DYNAMO_KV_CACHE_GROUP_COUNT_KEY
from dynamo.vllm.kv_connector_protocols import make_kv_connector_protocol
from dynamo.vllm.multimodal_utils.request_processor import PreparedMultimodalInput
from dynamo.vllm.retained_kv import (
    RETAINED_KV_KEY,
    RetainedKvCapture,
    retained_kv_capture,
    retained_prefill_params,
    retained_terminal_metadata,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

prefill_probe_handler = _shared_prefill_probe_handler


def _config(
    *, policy="recompute", bidirectional=True, connector="NixlConnector", group_count=1
):
    extra = {"bidirectional_kv_xfer": bidirectional, "decoder_kv_blocks_ttl": 30}
    return SimpleNamespace(
        additional_config={DYNAMO_KV_CACHE_GROUP_COUNT_KEY: group_count}
        if group_count is not None
        else {},
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_load_failure_policy=policy,
            get_from_extra_config=lambda name, default: extra.get(name, default),
        ),
    )


def _params():
    return {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_block_ids": [[0, 7]],
        "remote_engine_id": "donor-engine",
        "remote_request_id": "donor-request",
        "remote_host": "donor.internal",
        "remote_port": 5555,
        "remote_num_tokens": 4,
        "remote_blocks_expiry_time": 123456.25,
        "tp_size": 2,
        "opaque_layout": {"group_offsets": [0, 64]},
    }


def _request(*, params=None):
    request = _make_raw_frontend_request()
    request["extra_args"] = {RETAINED_KV_KEY: {"version": 1, "capture": True}}
    if params is not None:
        request["extra_args"][RETAINED_KV_KEY]["kv_transfer_params"] = params
    return request


@pytest.mark.parametrize(
    "config",
    [
        _config(policy="fail"),
        _config(policy=None),
        _config(bidirectional=False),
        _config(bidirectional="true"),
        _config(connector="LMCacheMPConnector"),
        SimpleNamespace(kv_transfer_config=None),
        _config(group_count=None),
        _config(group_count=2),
        _config(group_count=True),
        _config(group_count="1"),
    ],
)
def test_capture_requires_supported_recompute_profile(config):
    assert (
        retained_kv_capture(
            _request(),
            config,
            SimpleNamespace(n=1),
            is_decode_only=True,
            lora_request=None,
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_embeds": "embeddings"},
        {"multi_modal_data": {"image": []}},
        {"encoder_result": {}},
        {
            "extra_args": {
                RETAINED_KV_KEY: {"version": 1, "capture": True},
                "mm_kwargs_shm": {},
            }
        },
        {"routing": {"lora_name": "adapter"}},
        {"extra_args": {RETAINED_KV_KEY: {"version": True, "capture": True}}},
        {
            "extra_args": {
                "vllm_tito": {},
                RETAINED_KV_KEY: {"version": 1, "capture": True},
            }
        },
    ],
)
def test_ambiguous_or_native_requests_do_not_capture(change):
    request = {**_request(), **change}
    assert (
        retained_kv_capture(
            request,
            _config(),
            SimpleNamespace(n=1),
            is_decode_only=True,
            lora_request=None,
        )
        is None
    )


@pytest.mark.parametrize("n,lora", [(2, None), (1, object())])
def test_branching_or_lora_do_not_capture(n, lora):
    assert (
        retained_kv_capture(
            _request(),
            _config(),
            SimpleNamespace(n=n),
            is_decode_only=True,
            lora_request=lora,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["recompute", "fail"])
async def test_prefill_hint_preserves_sampling_and_opaque_descriptor(
    prefill_probe_handler, policy
):
    handler = prefill_probe_handler
    handler.engine_client.vllm_config = _config(policy=policy)
    params = _params()
    original = copy.deepcopy(params)
    request = _request(params=params)
    request["extra_args"][RETAINED_KV_KEY]["cache_salt"] = "trusted-scope:user-salt"
    request["extra_args"]["nvext"] = {"cache_salt": "user-salt"}
    request["token_ids"] = [1, 2, 3, 4, 5]
    request["sampling_options"] = {"temperature": 0.37, "top_p": 0.81}
    chunks = [chunk async for chunk in handler.generate(request, MagicMock())]

    sampling = handler.engine_client.generate.call_args.args[1]
    assert (
        handler.engine_client.generate.call_args.args[0]["cache_salt"]
        == "dynamo-cache-salt:trusted-scope:user-salt"
    )
    assert request["extra_args"]["nvext"]["cache_salt"] == "user-salt"
    assert sampling.temperature == 0.37
    assert sampling.top_p == 0.81
    assert sampling.max_tokens == sampling.min_tokens == 1
    actual = sampling.extra_args["kv_transfer_params"]
    if policy == "recompute":
        assert actual == {
            **original,
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        actual["remote_block_ids"][0].append(99)
        actual["opaque_layout"]["group_offsets"].append(128)
    else:
        assert actual["remote_block_ids"] is None
    assert params == original
    assert chunks[0]["disaggregated_params"] is not None
    assert "vllm_tito" not in request["extra_args"]


@pytest.mark.asyncio
async def test_existing_prefill_configuration_errors_remain_visible(
    prefill_probe_handler,
):
    handler = prefill_probe_handler
    error = ValueError("existing connector configuration error")
    with patch.object(handlers, "make_kv_connector_protocol", side_effect=error):
        with pytest.raises(ValueError, match="existing connector configuration error"):
            _ = [chunk async for chunk in handler.generate(_request(), MagicMock())]
    handler.engine_client.generate.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"remote_block_ids": []},
        {"remote_block_ids": [[0, 7], [4]]},
        {"remote_block_ids": [[True]]},
        {"remote_num_tokens": None},
        {"remote_port": 0},
        {"remote_engine_id": ""},
        {"tp_size": 0},
        {"remote_blocks_expiry_time": float("nan")},
        {"_remote_blocks_processed": True},
    ],
)
def test_malformed_optional_hints_fall_back(change):
    config = _config()
    assert (
        retained_prefill_params(
            _request(params={**_params(), **change}),
            config,
            make_kv_connector_protocol(config),
            SimpleNamespace(n=1),
            lora_request=None,
        )
        is None
    )


@pytest.mark.parametrize("group_count", [None, 0, 2, True, "1"])
def test_prefill_without_authoritative_single_group_uses_cold_path(group_count):
    config = _config(group_count=group_count)
    assert (
        retained_prefill_params(
            _request(params=_params()),
            config,
            make_kv_connector_protocol(config),
            SimpleNamespace(n=1),
            lora_request=None,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,engine_ids,capture,expected",
    [
        ("stop", [1, 99, 3], True, True),
        ("length", [1, 2, 3], True, True),
        ("stop", None, True, False),
        ("error", [1, 2, 3], True, False),
        ("stop", [1, 2, 3], False, False),
    ],
)
async def test_ordinary_decode_exports_private_authoritative_terminal_only(
    finish, engine_ids, capture, expected
):
    handler = _make_decode_handler()
    request = _request() if capture else _make_raw_frontend_request()
    request["extra_args"] = {
        **request.get("extra_args", {}),
        "nvext": {"extra_fields": ["engine_data"], "cache_salt": "user-salt"},
    }
    if capture:
        request["extra_args"][RETAINED_KV_KEY]["cache_salt"] = "trusted-scope:user-salt"
    request["sampling_options"] = {"temperature": 0.37}
    request["stop_conditions"] = {"max_tokens": 2}
    request["prefill_result"] = {
        "disaggregated_params": {"kv_transfer_params": {"do_remote_prefill": True}}
    }
    handler._multimodal_request_processor.prepare_input = AsyncMock(
        return_value=PreparedMultimodalInput(
            request=request, multi_modal_data=None, mm_processor_kwargs=None
        )
    )
    handler._build_prompt_from_request = MagicMock(
        return_value={"prompt_token_ids": [1, 2, 3]}
    )
    handler._resolve_lora_request = MagicMock(return_value=None)
    handler._abort_monitor = MagicMock(return_value=nullcontext())
    handler._extract_logprobs = MagicMock(return_value=(None, None))

    async def generate(*args, **kwargs):
        for index, token in enumerate((11, 12)):
            response = RequestOutput(
                request_id="ordinary-openai",
                prompt=None,
                prompt_token_ids=engine_ids,
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=[token],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason=finish if index else None,
                        stop_reason=None,
                    )
                ],
                finished=bool(index),
            )
            response.kv_transfer_params = _params() if index else None
            yield response

    handler.engine_client = SimpleNamespace(
        generate=MagicMock(side_effect=generate),
        abort=AsyncMock(),
        tokenizer=None,
        vllm_config=_config(),
    )
    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            request, MagicMock(), "ordinary-openai"
        )
    ]
    assert [chunk["token_ids"] for chunk in chunks] == [[11], [12]]
    assert all(
        "kv_transfer_params" not in chunk.get("engine_data", {}) for chunk in chunks
    )
    assert RETAINED_KV_KEY not in chunks[0].get("extra_args", {})
    metadata = chunks[-1].get("extra_args", {}).get(RETAINED_KV_KEY)
    assert (metadata is not None) is expected
    if expected:
        assert metadata["prompt_token_ids"] == engine_ids
        assert metadata["kv_transfer_params"] == _params()
        assert metadata["ttl_seconds"] == 30
    sampling = handler.engine_client.generate.call_args.args[1]
    expected_salt = "trusted-scope:user-salt" if capture else "user-salt"
    assert (
        handler.engine_client.generate.call_args.args[0]["cache_salt"]
        == "dynamo-cache-salt:" + expected_salt
    )
    assert request["extra_args"]["nvext"]["cache_salt"] == "user-salt"
    assert sampling.temperature == 0.37
    assert "vllm_tito" not in request["extra_args"]


@pytest.mark.asyncio
async def test_existing_native_coordinate_export_is_preserved():
    handler = _make_decode_handler(disaggregation_mode="AGGREGATED")
    handler._extract_logprobs = MagicMock(return_value=(None, None))
    response = _make_engine_response()
    response.outputs = [
        SimpleNamespace(index=0, token_ids=[11], finish_reason="stop", stop_reason=None)
    ]
    response.num_cached_tokens = 0
    response.kv_transfer_params = _params()

    async def generate(*args, **kwargs):
        yield response

    handler.engine_client = SimpleNamespace(generate=generate, tokenizer=None)
    chunks = [
        chunk
        async for chunk in handler.generate_tokens(
            {"prompt_token_ids": [1, 2, 3]},
            SimpleNamespace(n=1),
            "native",
            expose_kv_transfer_params=True,
        )
    ]
    assert chunks[-1]["engine_data"]["kv_transfer_params"] == _params()
    assert RETAINED_KV_KEY not in chunks[-1].get("extra_args", {})


def test_inconsistent_multi_group_native_export_is_not_retained():
    params = {**_params(), "remote_block_ids": [[0, 7], [4]]}
    assert (
        retained_terminal_metadata(RetainedKvCapture(30), [1, 2, 3, 4], params) is None
    )


def test_metadata_without_prompt_provenance_is_not_exported():
    assert retained_terminal_metadata(RetainedKvCapture(None), None, _params()) is None


def test_native_envelope_does_not_use_automatic_private_cache_salt():
    request = _request()
    request["extra_args"][RETAINED_KV_KEY]["cache_salt"] = "server-salt"
    request["extra_args"]["vllm_tito"] = {"cache_salt": "native-salt"}
    prompt = {"prompt_token_ids": [1, 2, 3]}
    handlers._apply_nvext_cache_salt(request, prompt)
    assert prompt["cache_salt"] == "dynamo-cache-salt:native-salt"


@pytest.mark.parametrize("salt", [None, "", 7, "x" * 4097])
def test_invalid_private_cache_salt_keeps_existing_public_salt(salt):
    request = _request()
    request["extra_args"][RETAINED_KV_KEY]["cache_salt"] = salt
    request["extra_args"]["nvext"] = {"cache_salt": "user-salt"}
    prompt = {"prompt_token_ids": [1, 2, 3]}
    handlers._apply_nvext_cache_salt(request, prompt)
    assert prompt["cache_salt"] == "dynamo-cache-salt:user-salt"
