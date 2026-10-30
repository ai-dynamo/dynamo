# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router hint forwarding, HiCache endpoint layout, and registration gating."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import ModelType, WorkerType
from dynamo.sglang.kv_hints import (
    kv_hint_kwargs,
    kvcr_hint_endpoints,
    publish_kvcr_hint_capabilities,
)

# Marker-only collection may provide only partial SGLang stubs.
_register_import_error = None
try:
    from dynamo.sglang import register
except ImportError as exc:
    register = None
    _register_import_error = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _HintEngine:
    async def async_generate(self, *, kv_hints=None):
        return kv_hints


class _OldEngine:
    async def async_generate(self, *, input_ids):
        return input_ids


def test_router_hint_forwarding_is_private_and_lossless():
    hint = {
        "protocol_version": "0.1",
        "message_id": "selected-source",
        "actions": [
            {"action_type": "kv.fetch", "payload": {"block_hashes": [(1 << 63) + 1]}}
        ],
    }
    assert kv_hint_kwargs(_HintEngine(), {"kv_hint": hint}) == {"kv_hints": hint}
    assert kv_hint_kwargs(_OldEngine(), {"kv_hint": hint}) == {}
    assert kv_hint_kwargs(_HintEngine(), {"extra_args": {"kv_hint": hint}}) == {}
    assert kv_hint_kwargs(_HintEngine(), {"nvext": {"kv_hint": hint}}) == {}


@pytest.mark.parametrize("tp_size", [1, 2])
def test_linker_hint_advertisement_requires_real_inventory_and_supported_engine(
    tp_size,
):
    args = SimpleNamespace(
        enable_hierarchical_cache=False,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        nnodes=1,
        tp_size=tp_size,
        dp_size=1,
        pp_size=1,
    )
    source = {
        "dp_rank": 0,
        "capabilities": ["router_hint", "kvcr_inventory"],
        "source_control_endpoint": "tcp://worker:24100",
        "inventory_lifetime": "process_local",
        "reset_clears_owner_inventory": True,
        "owner_incarnation_scoped": True,
    }
    engine = _HintEngine()
    engine._scheduler_init_result = SimpleNamespace(
        scheduler_infos=[{"kv_hint_sources": [source]}]
    )
    values = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda k, v: values.update({k: json.loads(v)})
    )
    publish_kvcr_hint_capabilities(runtime, engine, args, WorkerType.Aggregated)
    assert values == {
        KV_HINT_TRANSFER_CAPABILITY_KEY: True,
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: "aggregated",
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: {
            "0": "tcp://worker:24100"
        },
    }
    values.clear()
    source["capabilities"] = ["router_hint"]
    publish_kvcr_hint_capabilities(runtime, engine, args, WorkerType.Aggregated)
    assert values == {}
    source["capabilities"].append("kvcr_inventory")
    source["source_control_endpoint"] = "tcp://0.0.0.0:24100"
    with pytest.raises(ValueError, match="wildcard"):
        publish_kvcr_hint_capabilities(runtime, engine, args, WorkerType.Aggregated)
    assert values == {}
    source["source_control_endpoint"] = "tcp://worker:24100"
    publish_kvcr_hint_capabilities(runtime, engine, args, WorkerType.Prefill)
    assert values == {}


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"source_control_endpoint": "tcp://0.0.0.0:28071"},
        {"source_control_endpoint": "tcp://worker-a:65536"},
        {"source_control_endpoint": "http://worker-a:28071"},
        {"dp_rank": -1},
        {"dp_rank": 1},
    ],
)
def test_hicache_endpoint_layout_comes_from_scheduler_metadata(change):
    sources = [
        {
            "dp_rank": 0,
            "capabilities": ["router_hint"],
            "source_control_endpoint": "tcp://worker-a:28071",
        },
        {
            "dp_rank": 1,
            "capabilities": ["router_hint"],
            "source_control_endpoint": "tcp://[2001:db8::1]:31217",
        },
    ]
    infos = [{"kv_hint_sources": [source]} for source in sources]
    if change is not None:
        sources[0].update(change)
        with pytest.raises(ValueError):
            kvcr_hint_endpoints(infos, (0, 2))
    else:
        assert kvcr_hint_endpoints(infos, (0, 2)) == {
            "0": "tcp://worker-a:28071",
            "1": "tcp://[2001:db8::1]:31217",
        }
        assert kvcr_hint_endpoints([{}], (0, 2)) == {}
        with pytest.raises(ValueError, match="every managed DP rank"):
            kvcr_hint_endpoints(infos[:1], (0, 2))


@pytest.mark.parametrize(
    "change, forwards_kv_hints, engine_type",
    [
        ({}, False, _HintEngine),
        ({}, True, _HintEngine),
        ({}, True, _OldEngine),
        ({"nnodes": 2}, True, _HintEngine),
        ({"pp_size": 2}, True, _HintEngine),
        ({"enable_hierarchical_cache": False}, True, _HintEngine),
        (
            {
                "enable_hierarchical_cache": False,
                "enable_unified_cache_external_linker": True,
                "unified_cache_external_linker_backend": "kvcr",
            },
            True,
            _HintEngine,
        ),
    ],
)
def test_registration_requires_a_hint_forwarding_hicache_worker(
    monkeypatch, change, forwards_kv_hints, engine_type
):
    if _register_import_error is not None:
        raise _register_import_error
    args = SimpleNamespace(
        **{
            "enable_hierarchical_cache": True,
            "hicache_storage_backend": "kvcr",
            "nnodes": 1,
            "attn_dp_size": 2,
            "attn_cp_size": 1,
            "tp_size": 4,
            "pp_size": 1,
            "page_size": 64,
            "model_path": "test-model",
            "served_model_name": "test-model",
            # Publication must not parse config or derive ports from it.
            "hicache_storage_backend_extra_config": "not JSON",
            **change,
        }
    )
    metadata = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: metadata.__setitem__(
            key, json.loads(value)
        )
    )
    monkeypatch.setattr(register, "get_runtime_config", AsyncMock(return_value=runtime))
    register_model = AsyncMock()
    monkeypatch.setattr(register, "register_model", register_model)
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False, model_source_uri=None, custom_jinja_template=None
    )
    kwargs = {"forwards_kv_hints": True} if forwards_kv_hints else {}
    engine = engine_type()
    engine._scheduler_init_result = SimpleNamespace(
        scheduler_infos=[
            {
                "kv_hint_sources": [
                    {
                        "dp_rank": rank,
                        "capabilities": ["router_hint"],
                        "source_control_endpoint": endpoint,
                    }
                    for rank, endpoint in enumerate(
                        ("tcp://worker-a:28071", "tcp://worker-a:31217")
                    )
                ]
            }
        ]
    )
    asyncio.run(
        register.register_model_with_readiness_gate(
            engine,
            None,
            args,
            dynamo_args,
            output_type=ModelType.Chat | ModelType.Completions,
            worker_type=WorkerType.Aggregated,
            **kwargs,
        )
    )
    register_model.assert_awaited_once()
    metadata = {
        key: value
        for key, value in metadata.items()
        if key
        in (
            KV_HINT_TRANSFER_CAPABILITY_KEY,
            KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        )
    }
    if not change and forwards_kv_hints and engine_type is _HintEngine:
        assert metadata == {
            KV_HINT_TRANSFER_CAPABILITY_KEY: True,
            KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: "aggregated",
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: {
                "0": "tcp://worker-a:28071",
                "1": "tcp://worker-a:31217",
            },
        }
    else:
        assert not metadata
