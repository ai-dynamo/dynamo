# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurrent checkpoint events for hybrid models: layout, config and publisher wiring."""

import json
from types import SimpleNamespace

import pytest

import dynamo.sglang._compat as sglang_compat
import dynamo.sglang.publisher as publisher_mod
import dynamo.sglang.recurrent_kv as recurrent_kv
from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.recurrent_kv import (
    MAMBA_GROUP_IDX,
    RECURRENT_KV_EVENTS_ENV,
    RECURRENT_LAYOUT_CONTRACT,
    recurrent_kv_events_enabled,
    sglang_recurrent_kv_layout,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

HYBRID_ARCH = "Qwen3_5ForConditionalGeneration"
ZMQ_CONFIG = '{"publisher": "zmq", "endpoint": "tcp://*:5557", "format": "dynamo"}'


def _server_args(**overrides):
    args = dict(
        kv_events_config=ZMQ_CONFIG,
        page_size=64,
        dcp_size=1,
        disable_radix_cache=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def hybrid_model(monkeypatch):
    """A Qwen3.5-like model as SGLang's model config would describe it."""
    hf_config = SimpleNamespace(architectures=[HYBRID_ARCH], mamba_chunk_size=64)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        linear_attn_registry_result=None,
    )
    monkeypatch.setattr(recurrent_kv, "_model_config", lambda server_args: model_config)
    monkeypatch.setattr(
        recurrent_kv, "_mamba_radix_cache_archs", lambda: frozenset({HYBRID_ARCH})
    )
    monkeypatch.setattr(
        recurrent_kv, "sglang_supports_recurrent_checkpoint_events", lambda: True
    )
    # SGLang's chunk is max(model chunk, tree page).
    monkeypatch.setattr(
        recurrent_kv,
        "_mamba_cache_chunk_size",
        lambda server_args, tree_page: max(64, tree_page),
    )
    return model_config


def test_opt_in_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv(RECURRENT_KV_EVENTS_ENV, raising=False)
    assert not recurrent_kv_events_enabled()
    monkeypatch.setenv(RECURRENT_KV_EVENTS_ENV, "1")
    assert recurrent_kv_events_enabled()


def test_layout_is_none_unless_enabled(hybrid_model):
    assert sglang_recurrent_kv_layout(_server_args()) is None
    assert sglang_recurrent_kv_layout(_server_args(), enabled=False) is None


def test_layout_uses_one_recurrent_group_on_the_page_grid(hybrid_model):
    layout = sglang_recurrent_kv_layout(_server_args(page_size=64), enabled=True)

    assert layout is not None
    assert layout.recurrent_groups == [MAMBA_GROUP_IDX]
    assert layout.indexer_block_size == 64
    assert layout.checkpoint_grid_tokens == 64
    assert layout.alignment_blocks == 1
    metadata = layout.runtime_metadata()
    assert metadata["contract"] == RECURRENT_LAYOUT_CONTRACT
    assert metadata["recurrent_groups"] == [1]
    assert metadata["alignment_blocks"] == 1
    assert metadata["indexer_block_size"] == 64


def test_layout_widens_the_page_by_dcp(hybrid_model):
    layout = sglang_recurrent_kv_layout(
        _server_args(page_size=32, dcp_size=2), enabled=True
    )

    assert layout is not None
    assert layout.indexer_block_size == 64
    assert layout.alignment_blocks == 1


def test_layout_rejects_pages_below_the_mamba_chunk(hybrid_model):
    with pytest.raises(ValueError, match="multiple of 64"):
        sglang_recurrent_kv_layout(_server_args(page_size=16), enabled=True)


def test_layout_rejects_a_dense_model(hybrid_model):
    hybrid_model.hf_config.architectures = ["Qwen3ForCausalLM"]

    with pytest.raises(ValueError, match="no recurrent layers"):
        sglang_recurrent_kv_layout(_server_args(), enabled=True)


def test_layout_accepts_hf_layer_types_for_unlisted_architectures(hybrid_model):
    hybrid_model.hf_config.architectures = ["SomeNewHybridForCausalLM"]
    hybrid_model.hf_config.layer_types = ["linear_attention", "full_attention"]

    layout = sglang_recurrent_kv_layout(_server_args(), enabled=True)

    assert layout is not None


def test_layout_rejects_disabled_radix_cache(hybrid_model):
    with pytest.raises(ValueError, match="radix cache"):
        sglang_recurrent_kv_layout(_server_args(disable_radix_cache=True), enabled=True)


@pytest.mark.parametrize("config", [None, "", '{"publisher": "null"}'])
def test_layout_requires_a_publishing_kv_events_config(hybrid_model, config):
    with pytest.raises(ValueError, match="kv-events-config"):
        sglang_recurrent_kv_layout(_server_args(kv_events_config=config), enabled=True)


def test_layout_requires_an_sglang_that_publishes_checkpoints(
    hybrid_model, monkeypatch
):
    monkeypatch.setattr(
        recurrent_kv, "sglang_supports_recurrent_checkpoint_events", lambda: False
    )

    with pytest.raises(ValueError, match="sgl-project/sglang#51647"):
        sglang_recurrent_kv_layout(_server_args(), enabled=True)


def test_kv_events_config_gains_the_publish_flag_when_opted_in(monkeypatch):
    monkeypatch.setattr(
        sglang_compat, "_kv_events_config_has_recurrent_checkpoints", lambda: True
    )

    result = sglang_compat.kv_events_config_with_recurrent_checkpoints(
        ZMQ_CONFIG, enabled=True
    )

    assert json.loads(result)["publish_recurrent_checkpoints"] is True
    assert json.loads(result)["format"] == "dynamo"


def test_kv_events_config_is_untouched_without_opt_in_or_publisher(monkeypatch):
    monkeypatch.setattr(
        sglang_compat, "_kv_events_config_has_recurrent_checkpoints", lambda: True
    )

    assert (
        sglang_compat.kv_events_config_with_recurrent_checkpoints(
            ZMQ_CONFIG, enabled=False
        )
        == ZMQ_CONFIG
    )
    null_config = '{"publisher": "null"}'
    assert (
        sglang_compat.kv_events_config_with_recurrent_checkpoints(
            null_config, enabled=True
        )
        == null_config
    )
    assert (
        sglang_compat.kv_events_config_with_recurrent_checkpoints(None, enabled=True)
        is None
    )


def test_kv_events_config_opt_in_fails_on_an_old_sglang(monkeypatch):
    monkeypatch.setattr(
        sglang_compat, "_kv_events_config_has_recurrent_checkpoints", lambda: False
    )

    with pytest.raises(InvalidArgument, match="sgl-project/sglang#51647"):
        sglang_compat.kv_events_config_with_recurrent_checkpoints(
            ZMQ_CONFIG, enabled=True
        )


def _publisher(monkeypatch, calls, *, enable_local_indexer=True, inventory=False):
    class FakeKvEventPublisher:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(publisher_mod, "KvEventPublisher", FakeKvEventPublisher)
    monkeypatch.setattr(publisher_mod, "get_local_ip_auto", lambda: "127.0.0.1")
    monkeypatch.setattr(
        publisher_mod,
        "ZmqEventPublisher",
        SimpleNamespace(
            offset_endpoint_port=staticmethod(
                lambda base_ep, dp_rank: f"tcp://*:{5557 + dp_rank}"
            )
        ),
    )
    monkeypatch.setattr(
        publisher_mod,
        "format_zmq_endpoint",
        lambda endpoint, ip_address: endpoint.replace("*", ip_address),
    )
    monkeypatch.setattr(publisher_mod, "sglang_has_wide_mm_pad_values", lambda: False)
    monkeypatch.setattr(
        publisher_mod, "inventory_sources", lambda engine: ["kvcr"] if inventory else []
    )
    # A non-leader DP-attention node owns one local rank and opens no metrics
    # sockets, so the publisher needs no live engine (as in test_sglang_publisher).
    server_args = SimpleNamespace(
        kv_events_config=ZMQ_CONFIG,
        page_size=64,
        dcp_size=1,
        dp_size=2,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
        disable_radix_cache=False,
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=SimpleNamespace(
            enable_local_indexer=enable_local_indexer,
            kv_state_endpoint=None,
            use_kv_events=True,
        ),
    )
    return DynamoSglangPublisher(
        engine=SimpleNamespace(),
        config=config,
        generate_endpoint=SimpleNamespace(),
        component_gauges=SimpleNamespace(),
        kv_worker_id=1234,
    )


def test_init_kv_event_publish_passes_the_recurrent_layout(hybrid_model, monkeypatch):
    monkeypatch.setenv(RECURRENT_KV_EVENTS_ENV, "1")
    calls = []
    publisher = _publisher(monkeypatch, calls)

    publishers = publisher.init_kv_event_publish()

    assert len(publishers) == 1
    assert calls[0]["recurrent_groups"] == [MAMBA_GROUP_IDX]
    assert calls[0]["recurrent_alignment_blocks"] == 1
    assert calls[0]["kv_block_size"] == 64


def test_init_kv_event_publish_stays_dense_without_opt_in(hybrid_model, monkeypatch):
    monkeypatch.delenv(RECURRENT_KV_EVENTS_ENV, raising=False)
    calls = []
    publisher = _publisher(monkeypatch, calls)

    publisher.init_kv_event_publish()

    assert calls[0]["recurrent_groups"] is None
    assert calls[0]["recurrent_alignment_blocks"] == 1


def test_init_kv_event_publish_needs_the_local_indexer(hybrid_model, monkeypatch):
    monkeypatch.setenv(RECURRENT_KV_EVENTS_ENV, "1")
    publisher = _publisher(monkeypatch, [], enable_local_indexer=False)

    with pytest.raises(ValueError, match="local indexer"):
        publisher.init_kv_event_publish()


def test_init_kv_event_publish_rejects_the_state_agent_source(
    hybrid_model, monkeypatch
):
    monkeypatch.setenv(RECURRENT_KV_EVENTS_ENV, "1")
    publisher = _publisher(monkeypatch, [], inventory=True)

    with pytest.raises(ValueError, match="state-agent"):
        publisher.init_kv_event_publish()
