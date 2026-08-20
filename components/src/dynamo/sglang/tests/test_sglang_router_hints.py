# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.sglang._compat import router_hint_kwargs
from dynamo.sglang.router_hints import enable_router_hint_support

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _server_args(**overrides):
    args = {"hicache_storage_backend": "kvcr"}
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


def test_publishes_single_dp_rank_endpoint():
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(),
        extra_config=_kvcr_config(),
        dp_bounds=(0, 1),
    )

    assert _published(runtime_config) == {
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY: json.dumps(True),
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY: json.dumps("aggregated"),
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: json.dumps(
            {"0": "tcp://127.0.0.1:25000"}
        ),
    }


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
    """The router only hints between workers reporting the same role.

    ``router_hint_metadata_for_dp_rank`` (lib/llm/src/local_model/runtime_config.rs)
    returns None without this key, and ``router_hint_for_selection``
    (lib/llm/src/kv_router.rs) then matches source to target on it. A worker
    that omits it is invisible to the router as both hint target and hint
    source, so hints silently never fire. SGLang spells a non-disaggregated
    engine "null"; the router's vocabulary calls it "aggregated".
    """
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(disaggregation_mode=disaggregation_mode),
        extra_config=_kvcr_config(),
        dp_bounds=(0, 1),
    )

    assert _published(runtime_config)[
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY
    ] == json.dumps(expected_worker_type)


def test_publishes_one_endpoint_per_dp_rank():
    """One DP rank per port when each rank owns a single scheduler (attn_tp=1)."""
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(
            enable_dp_attention=True, dp_size=4, tp_size=4, nnodes=1, node_rank=0
        ),
        extra_config=_kvcr_config(control_advertise_host="worker-a"),
        dp_bounds=(0, 4),
    )

    assert json.loads(
        _published(runtime_config)[ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]
    ) == {
        "0": "tcp://worker-a:25000",
        "1": "tcp://worker-a:25001",
        "2": "tcp://worker-a:25002",
        "3": "tcp://worker-a:25003",
    }


def test_dp_ranks_are_strided_by_the_schedulers_each_one_owns():
    """A DP rank owns a *block* of ports, one per attention rank under it.

    SGLang builds a KVCR store in every attention rank's scheduler process, all
    offsetting the same configured base port by their own rank coordinate. At
    DP=2/TP=4 that is four schedulers laid out as ports P..P+3, so DP rank 1
    starts at P+2, not P+1. Advertising a bare DP rank here (which is what the
    vLLM module correctly does, since it has one scheduler per DP rank) would
    name a port belonging to DP rank 0's second attention rank -- and because
    KVCR block keys are token hashes carrying no rank identity, the peer would
    accept the wrong attention shard instead of erroring.
    """
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(
            enable_dp_attention=True, dp_size=2, tp_size=4, nnodes=1, node_rank=0
        ),
        extra_config=_kvcr_config(control_advertise_host="worker-a"),
        dp_bounds=(0, 2),
    )

    assert json.loads(
        _published(runtime_config)[ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]
    ) == {
        "0": "tcp://worker-a:25000",
        "1": "tcp://worker-a:25002",
    }


def test_multinode_keys_global_dp_ranks_but_strides_from_the_local_base():
    """Each node numbers its own ports from the same configured base port.

    The map key is the router-visible global rank, but the offset is the local
    one -- node 1 of a DP=4/TP=8 engine binds the same ports as node 0, on its
    own host. Striding by the global rank would advertise ports nothing binds.
    """
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(
            enable_dp_attention=True, dp_size=4, tp_size=8, nnodes=2, node_rank=1
        ),
        extra_config=_kvcr_config(control_advertise_host="worker-b"),
        dp_bounds=(2, 4),
    )

    assert json.loads(
        _published(runtime_config)[ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]
    ) == {
        "2": "tcp://worker-b:25000",
        "3": "tcp://worker-b:25002",
    }


@pytest.mark.parametrize(
    "server_args_overrides",
    [
        # Plain --tp-size: one DP group, so every scheduler is under DP rank 0
        # and the backend's own tp_rank offset already spans the whole engine.
        {"enable_dp_attention": False, "dp_size": 1, "tp_size": 8},
        # --dp-size without --enable-dp-attention is replicated, not attention
        # DP: the schedulers still form a single rank space per engine, and
        # `local_dp_rank_bounds` reports a single rank for it.
        {"enable_dp_attention": False, "dp_size": 4, "tp_size": 8},
    ],
)
def test_no_attention_dp_advertises_the_base_port_unstrided(server_args_overrides):
    """Guards the TP-only path that was validated on hardware."""
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(**server_args_overrides),
        extra_config=_kvcr_config(control_advertise_host="worker-a"),
        dp_bounds=(0, 1),
    )

    assert json.loads(
        _published(runtime_config)[ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]
    ) == {"0": "tcp://worker-a:25000"}


def test_advertise_host_overrides_wildcard_bind_host():
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=_server_args(),
        extra_config=_kvcr_config(control_host="::"),
        dp_bounds=(0, 1),
    )

    assert (
        json.loads(
            _published(runtime_config)[ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]
        )["0"]
        == "tcp://127.0.0.1:25000"
    )


@pytest.mark.parametrize(
    "server_args,extra_config",
    [
        # A different HiCache backend cannot serve a hint at all.
        (_server_args(hicache_storage_backend="mooncake"), _kvcr_config()),
        (_server_args(hicache_storage_backend=None), _kvcr_config()),
        # KVCR without remote hints is local-only; advertising it would route
        # hints to a worker that will not act on them.
        (_server_args(), _kvcr_config(enable_remote_hint=False)),
        (_server_args(), {}),
        # A disaggregation mode the router has no worker_type for. Publishing
        # the other two keys without it registers a worker the router filters
        # out anyway, so say nothing rather than half-advertise.
        (_server_args(disaggregation_mode="encode"), _kvcr_config()),
    ],
)
def test_publishes_nothing_when_not_a_hint_source(server_args, extra_config):
    runtime_config = MagicMock()

    enable_router_hint_support(
        runtime_config=runtime_config,
        server_args=server_args,
        extra_config=extra_config,
        dp_bounds=(0, 1),
    )

    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        # A wildcard bind with no advertise host is not dialable by a peer.
        {"control_advertise_host": None, "control_host": "0.0.0.0"},
        {"control_advertise_host": None, "control_host": "::"},
        {"control_advertise_host": None, "control_host": ""},
        # An ephemeral port is only known inside the scheduler process.
        {"control_port": 0},
        {"control_port": None},
        {"control_port": "not-a-port"},
    ],
)
def test_raises_when_endpoint_is_not_advertisable(overrides):
    """Configured for hints but unreachable is a misconfiguration, not a no-op.

    Silently skipping would leave the worker running while every peer fetch
    fails, which is far harder to diagnose than failing registration.
    """
    with pytest.raises(ValueError, match="advertisable source control endpoints"):
        enable_router_hint_support(
            runtime_config=MagicMock(),
            server_args=_server_args(),
            extra_config=_kvcr_config(**overrides),
            dp_bounds=(0, 1),
        )


class _Engine:
    """Stand-in whose async_generate accepts kv_router_hint."""

    async def async_generate(self, *, kv_router_hint=None, **kwargs):
        return None


class _LegacyEngine:
    """An older SGLang install with no kv_router_hint kwarg."""

    async def async_generate(self, *, sampling_params=None):
        return None


_HINT = {"source_control_endpoint": "tcp://peer:25000", "block_hashes": [1, 2]}


def test_router_hint_kwargs_extracts_hint_from_kv_transfer_params():
    request = {"extra_args": {"kv_transfer_params": {"router_hint": _HINT}}}

    assert router_hint_kwargs(_Engine(), request) == {"kv_router_hint": _HINT}


@pytest.mark.parametrize(
    "request_payload",
    [
        {},
        {"extra_args": None},
        {"extra_args": {}},
        {"extra_args": {"kv_transfer_params": None}},
        {"extra_args": {"kv_transfer_params": {}}},
        # A non-dict hint is not something the backend can parse.
        {"extra_args": {"kv_transfer_params": {"router_hint": "nope"}}},
    ],
)
def test_router_hint_kwargs_is_empty_without_a_hint(request_payload):
    assert router_hint_kwargs(_Engine(), request_payload) == {}


def test_router_hint_kwargs_dropped_when_engine_cannot_accept_it():
    """A stale SGLang recomputes the prefix rather than failing the request."""
    request = {"extra_args": {"kv_transfer_params": {"router_hint": _HINT}}}

    assert router_hint_kwargs(_LegacyEngine(), request) == {}
