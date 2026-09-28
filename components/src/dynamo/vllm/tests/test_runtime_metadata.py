# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY
from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType
from dynamo.vllm.capacity import get_metrics_model_name, get_spec_decode_runtime_data
from dynamo.vllm.engine_generate import (
    VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY,
    VLLM_GENERATE_CAPABILITY,
    publish_engine_generate_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_spec_decode_runtime_data_uses_vllm_speculative_config():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config={"num_speculative_tokens": 99, "method": "ignored"}
        )
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3, method="eagle")
    )

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 3,
        "method": "eagle",
        "source": "backend_config",
    }


def test_metrics_model_name_prefers_served_model_name():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name="llama")

    assert get_metrics_model_name(config) == "llama"


def test_metrics_model_name_falls_back_to_model():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name=None)

    assert get_metrics_model_name(config) == "meta-llama/Llama-3.1-8B"


def test_vllm_token_budget_matches_rejection_policy():
    from dynamo.vllm.capacity import publish_vllm_token_budget

    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    publish_vllm_token_budget(runtime_config, 4096)

    runtime_config.set_engine_specific.assert_called_once()
    key, value = runtime_config.set_engine_specific.call_args.args
    assert key == TOKEN_BUDGET_RUNTIME_KEY
    assert json.loads(value) == {
        "combined_limit": 4096,
        "reject_prompt_overflow": True,
        "reject_total_overflow": True,
    }


@pytest.mark.parametrize(
    (
        "model_input",
        "model_type",
        "worker_type",
        "tower_connector_lora_enabled",
        "expected",
    ),
    [
        (ModelInput.Tokens, ModelType.Prefill, WorkerType.Prefill, False, True),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Decode, True, True),
        (
            ModelInput.Tokens,
            ModelType.Completions,
            WorkerType.Aggregated,
            False,
            True,
        ),
        (ModelInput.Tokens, ModelType.Empty, WorkerType.Prefill, False, False),
        (ModelInput.Tokens, ModelType.Empty, WorkerType.Decode, False, False),
        (ModelInput.Text, ModelType.Chat, WorkerType.Aggregated, True, False),
        (
            ModelInput.Tokens,
            ModelType.Embedding,
            WorkerType.Aggregated,
            False,
            False,
        ),
    ],
)
def test_vllm_generate_capability_publication(
    model_input,
    model_type,
    worker_type,
    tower_connector_lora_enabled,
    expected,
):
    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    published = publish_engine_generate_capability(
        runtime_config,
        model_input,
        model_type,
        worker_type,
        tower_connector_lora_enabled,
    )

    assert published is expected
    if expected:
        expected_calls = [
            call(VLLM_GENERATE_CAPABILITY, json.dumps(True)),
            call(
                VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY,
                json.dumps(tower_connector_lora_enabled),
            ),
        ]
        assert runtime_config.set_engine_specific.call_args_list == expected_calls
    else:
        runtime_config.set_engine_specific.assert_not_called()


def test_spec_decode_runtime_data_falls_back_to_engine_args_json():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config='{"num_speculative_tokens": "4", "method": "ngram"}'
        )
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 4,
        "method": "ngram",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_config",
    [None, {}, {"num_speculative_tokens": 0}, {"num_speculative_tokens": "bad"}],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_config):
    config = SimpleNamespace(
        engine_args=SimpleNamespace(speculative_config=speculative_config)
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) is None


@pytest.mark.parametrize("reply_state", ["agreed", "missing", "unsupported"])
def test_mooncake_optional_json_metadata_preserves_serving_capabilities(
    monkeypatch, reply_state
):
    from dynamo.vllm import mooncake_store_runtime as runtime
    from dynamo.vllm.capacity import publish_vllm_token_budget

    metadata = ModelRuntimeConfig()
    publish_vllm_token_budget(metadata, 4096)
    assert publish_engine_generate_capability(
        metadata, ModelInput.Tokens, ModelType.Chat, WorkerType.Aggregated, False
    )
    descriptor = {
        "schema_version": 1,
        "adapter": "vllm-1085b644",
        "vllm_revision": runtime.VLLM_REVISION,
        "hash": {
            "algorithm": "sha256",
            "digest_encoding": "hex",
            "gpu_event_hash": "low64",
            "seed_policy": "pythonhashseed-0",
            "key_separator": "@",
        },
        "input": {"text_only": True, "normalized_namespaces": True, "lora": True},
        "main_event_group": 0,
        "main_event_block_size": 16,
        "gpu_to_store_group": [0],
        "coordinator": {
            "lcm_block_size": 16,
            "speculative": False,
            "drop_blocks": False,
            "partial_hash_hits": False,
        },
        "groups": [
            {
                "group_id": 0,
                "kind": "full_attention",
                "spec": "FullAttentionSpec",
                "manager": "FullAttentionManager",
                "block_size": 16,
                "hash_block_size": 16,
                "key_prefixes": ["deployment@model@group:0"],
                "sliding_window": None,
                "mamba_cache_mode": None,
            }
        ],
    }
    replies = [{"rank": 0, "dp_rank": 0, "descriptor": descriptor}]
    if reply_state == "missing":
        replies = []
    elif reply_state == "unsupported":
        replies = [{"unsupported": "unknown revision"}]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_extension_cls=runtime.WORKER_EXTENSION,
            world_size=1,
        )
    )
    monkeypatch.setattr(runtime, "_verify_pinned_sources", lambda: None)
    monkeypatch.setattr(runtime, "_validate_input_config", lambda _: None)
    engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=replies))
    published = asyncio.run(
        runtime.publish_mooncake_store_runtime(
            metadata,
            engine,
            config,
            dp_range=(0, 1),
            event_span=16,
        )
    )
    assert published is (reply_state == "agreed")
    # `get_engine_specific` only decodes string values; these keys hold JSON
    # objects and booleans, so read the raw JSON through `runtime_data`.
    runtime_data = metadata.runtime_data
    if published:
        assert json.loads(runtime_data[runtime.RUNTIME_KEY]) == descriptor
    else:
        assert runtime.RUNTIME_KEY not in runtime_data
    assert json.loads(runtime_data[VLLM_GENERATE_CAPABILITY]) is True
    assert json.loads(runtime_data[TOKEN_BUDGET_RUNTIME_KEY])["combined_limit"] == 4096
