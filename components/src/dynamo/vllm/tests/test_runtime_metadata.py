# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY
from dynamo.llm import ModelInput, ModelType, WorkerType
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


def test_engine_generate_metadata_imports_without_vllm(monkeypatch):
    module_name = "dynamo.vllm.engine_generate"
    loaded_module = sys.modules.pop(module_name)
    original_import = builtins.__import__

    def reject_vllm_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("vLLM is not installed", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)
    try:
        metadata_module = importlib.import_module(module_name)
        assert metadata_module.VLLM_GENERATE_CAPABILITY
    finally:
        sys.modules[module_name] = loaded_module


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
        (ModelInput.Tokens, ModelType.Prefill, WorkerType.Prefill, False, False),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Decode, True, False),
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
