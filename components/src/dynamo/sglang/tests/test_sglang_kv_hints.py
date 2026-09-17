# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.sglang._compat import kv_hints_kwargs, request_kv_hint
from dynamo.sglang.kv_hints import (
    dp_port_stride,
    kvcr_mode,
    parse_kvcr_extra_config,
    publish_kv_hint_capabilities,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _linker_args(**overrides):
    args = {
        "enable_unified_cache_external_linker": True,
        "unified_cache_external_linker_backend": "kvcr",
        "hicache_storage_backend": None,
        "disaggregation_mode": "null",
        "dp_size": 1,
        "tp_size": 1,
        "enable_dp_attention": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _hicache_args(**overrides):
    args = {
        "enable_unified_cache_external_linker": False,
        "unified_cache_external_linker_backend": "mooncake",
        "hicache_storage_backend": "kvcr",
        "disaggregation_mode": "null",
        "dp_size": 1,
        "tp_size": 1,
        "enable_dp_attention": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _kvcr_config(**overrides):
    config = {
        "enable_remote_hint": True,
        "control_host": "0.0.0.0",
        "control_advertise_host": "127.0.0.1",
        "control_port": 25000,
    }
    config.update(overrides)
    return config


def _published(runtime_config):
    return dict(call.args for call in runtime_config.set_engine_specific.call_args_list)


def test_detects_linker_and_hicache_modes():
    assert kvcr_mode(_linker_args()) == "linker"
    assert kvcr_mode(_hicache_args()) == "hicache"
    assert (
        kvcr_mode(_linker_args(unified_cache_external_linker_backend="mooncake"))
        is None
    )
    assert kvcr_mode(_hicache_args(hicache_storage_backend="mooncake")) is None


@pytest.mark.parametrize("args", [_linker_args(), _hicache_args()])
def test_publishes_single_dp_rank_endpoint_in_both_modes(args):
    runtime_config = MagicMock()

    assert publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=args,
        extra_config=_kvcr_config(),
        dp_bounds=(0, 1),
    )

    assert _published(runtime_config) == {
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: json.dumps(
            {"0": "tcp://127.0.0.1:25000"}
        ),
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: json.dumps("aggregated"),
        KV_HINT_TRANSFER_CAPABILITY_KEY: json.dumps(True),
    }
    # The capability flag is the last key published, so a partially observed
    # registration can never be capability-only.
    assert list(_published(runtime_config))[-1] == KV_HINT_TRANSFER_CAPABILITY_KEY


@pytest.mark.parametrize(
    "disaggregation_mode,expected_worker_type",
    [
        (None, "aggregated"),
        ("null", "aggregated"),
        ("prefill", "prefill"),
        ("decode", "decode"),
    ],
)
def test_publishes_worker_type_from_disaggregation_mode(
    disaggregation_mode, expected_worker_type
):
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=_linker_args(disaggregation_mode=disaggregation_mode),
        extra_config=_kvcr_config(),
        dp_bounds=(0, 1),
    )
    assert _published(runtime_config)[
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY
    ] == json.dumps(expected_worker_type)


def test_dp_ranks_get_one_port_block_each_at_the_attention_stride():
    # tp 8 with dp attention 2: each DP rank owns 4 attention ranks, so DP rank
    # d's first scheduler binds base + 4d. Node 1 of 2 owns global ranks 2, 3.
    args = _linker_args(dp_size=4, tp_size=8, enable_dp_attention=True)
    assert dp_port_stride(args) == 2
    args = _linker_args(dp_size=2, tp_size=8, enable_dp_attention=True)
    assert dp_port_stride(args) == 4
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=_linker_args(dp_size=4, tp_size=8, enable_dp_attention=True),
        extra_config=_kvcr_config(),
        dp_bounds=(2, 4),
    )
    assert json.loads(
        _published(runtime_config)[
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
        ]
    ) == {"2": "tcp://127.0.0.1:25000", "3": "tcp://127.0.0.1:25002"}


def test_no_publication_without_kvcr_or_remote_hint():
    runtime_config = MagicMock()
    assert not publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=_linker_args(unified_cache_external_linker_backend="mooncake"),
        extra_config=_kvcr_config(),
        dp_bounds=(0, 1),
    )
    assert not publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=_linker_args(),
        extra_config=_kvcr_config(enable_remote_hint=False),
        dp_bounds=(0, 1),
    )
    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_port": 0},
        {"control_advertise_host": "0.0.0.0"},
        {"control_advertise_host": None},
        {"control_port": 65535},  # the second DP rank would leave the port range
    ],
)
def test_undialable_endpoints_fail_registration(overrides):
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match="advertisable source control endpoints"):
        publish_kv_hint_capabilities(
            runtime_config=runtime_config,
            server_args=_linker_args(dp_size=2, tp_size=2, enable_dp_attention=True),
            extra_config=_kvcr_config(**overrides),
            dp_bounds=(0, 2),
        )
    runtime_config.set_engine_specific.assert_not_called()


def test_ipv6_advertise_host_is_bracketed():
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config=runtime_config,
        server_args=_linker_args(),
        extra_config=_kvcr_config(control_advertise_host="fd00::1"),
        dp_bounds=(0, 1),
    )
    assert json.loads(
        _published(runtime_config)[
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
        ]
    ) == {"0": "tcp://[fd00::1]:25000"}


def test_parse_kvcr_extra_config_accepts_dict_json_and_empty():
    assert parse_kvcr_extra_config(None) == {}
    assert parse_kvcr_extra_config("") == {}
    assert parse_kvcr_extra_config({"a": 1}) == {"a": 1}
    assert parse_kvcr_extra_config('{"control_port": 25000}') == {"control_port": 25000}
    with pytest.raises(ValueError):
        parse_kvcr_extra_config("[1]")


_ENVELOPE = {
    "protocol_version": "0.1",
    "message_id": "m",
    "actions": [
        {
            "action_id": "a",
            "action_type": "kv.fetch",
            "action_version": "1.0",
            "payload": {
                "source_control_endpoint": "tcp://10.0.0.1:25000",
                "block_hashes": [1, 2],
            },
        }
    ],
}


def _engine(accepts_kv_hints: bool):
    if accepts_kv_hints:

        async def async_generate(
            prompt=None, *, sampling_params=None, kv_hints=None, **_
        ):
            return None

    else:

        async def async_generate(
            prompt=None, *, sampling_params=None, require_reasoning=False
        ):
            return None

    return SimpleNamespace(async_generate=async_generate)


def test_request_kv_hint_reads_top_level_then_nested_location():
    assert request_kv_hint({"kv_hint": _ENVELOPE}) == _ENVELOPE
    nested = {"extra_args": {"kv_transfer_params": {"kv_hint": _ENVELOPE}}}
    assert request_kv_hint(nested) == _ENVELOPE
    assert request_kv_hint({"kv_hint": "not-a-mapping"}) is None
    assert request_kv_hint({}) is None


def test_kv_hints_kwargs_forwards_envelope_only_when_engine_accepts_it():
    assert kv_hints_kwargs(_engine(True), {"kv_hint": _ENVELOPE}) == {
        "kv_hints": _ENVELOPE
    }
    assert kv_hints_kwargs(_engine(True), {}) == {}
    # An engine without the kwarg recomputes the prefix instead of failing.
    assert kv_hints_kwargs(_engine(False), {"kv_hint": _ENVELOPE}) == {}
