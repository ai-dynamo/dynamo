# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime LoRA advertisement is limited to the implemented worker path."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.lora.runtime import RuntimeLoRAConfigurationError
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.args import Config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def registration(monkeypatch, tmp_path):
    main = importlib.import_module("dynamo.vllm.main")
    protocol = importlib.import_module("dynamo.vllm.runtime_lora_protocol")
    register = AsyncMock()
    manager = Mock(
        return_value=SimpleNamespace(
            cache_root=tmp_path, runtime_lora_schemes=frozenset({"wandb-artifact"})
        )
    )
    monkeypatch.setattr(main, "register_model", register)
    monkeypatch.setattr(protocol, "get_lora_manager", manager)
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    for name in (
        "publish_vllm_structural_tag_reasoning_policy",
        "publish_vllm_qwen_video_processor_contract",
        "publish_vllm_nemotron_video_processor_contract",
        "apply_data_parallel_runtime_config",
        "publish_kv_hint_capabilities",
        "publish_vllm_token_budget",
        "apply_topology_config",
        "build_router_config",
    ):
        monkeypatch.setattr(main, name, Mock())
    monkeypatch.setattr(main, "get_dp_range_for_worker", lambda _: (0, 1))
    monkeypatch.setattr(main, "state_agent_settings", lambda _: None)
    monkeypatch.setattr(main, "publish_engine_generate_capability", lambda *_: False)
    monkeypatch.setattr(main, "per_rank_kv_blocks", lambda blocks, _: blocks)
    monkeypatch.setattr(main, "get_spec_decode_runtime_data", lambda *_: None)
    monkeypatch.setattr(main, "_worker_topology_config", lambda: None)
    monkeypatch.setattr(main, "_register_model_source_path", lambda *_: "base")
    monkeypatch.setattr(main, "should_register_model_ignore_weights", lambda _: False)
    monkeypatch.setattr(main, "create_frontend_media_config", lambda _: (None, None))
    monkeypatch.setattr(
        main,
        "get_engine_cache_info",
        lambda _: {
            "num_gpu_blocks": 16,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 512,
            "kv_event_block_size": 16,
        },
    )
    config = Config()
    config.engine_args = SimpleNamespace(
        enable_lora=True, max_loras=2, max_cpu_loras=2, max_num_seqs=2
    )
    for name, value in {
        "use_vllm_tokenizer": False,
        "enable_local_indexer": False,
        "use_kv_events": False,
        "kv_state_endpoint": None,
        "dyn_tool_call_parser": None,
        "dyn_reasoning_parser": None,
        "dyn_default_thinking_mode": None,
        "exclude_tools_when_tool_choice_none": False,
        "structural_tag": None,
        "frontend_decoding": None,
        "freeze_gc_after_init": False,
        "served_model_name": "base",
        "custom_jinja_template": None,
        "router_advertisement": None,
        "served_model_aliases": (),
    }.items():
        setattr(config, name, value)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024), lora_config=None
    )
    return main, register, manager, config, vllm_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_input,model_type",
    [
        (ModelInput.Text, ModelType.Realtime),
        (ModelInput.Text, ModelType.Chat),
        (ModelInput.Tokens, ModelType.Embedding),
        (ModelInput.Tokens, ModelType.Classify | ModelType.Pooling),
        (ModelInput.Tokens, ModelType.Chat | ModelType.Embedding),
        (ModelInput.Tokens, ModelType.Images),
        (ModelInput.Tokens, ModelType.Audios),
    ],
)
async def test_runtime_capability_rejects_unimplemented_surface(
    registration, model_input, model_type
):
    main, register, manager, config, vllm_config = registration

    with pytest.raises(RuntimeLoRAConfigurationError):
        await main.register_vllm_model(
            model_input,
            model_type,
            Mock(),
            config,
            Mock(),
            vllm_config,
            worker_type=WorkerType.Aggregated,
        )

    register.assert_not_awaited()
    manager.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type",
    [ModelType.Chat, ModelType.Completions, ModelType.Chat | ModelType.Completions],
)
async def test_runtime_capability_registers_supported_token_worker(
    registration, model_type
):
    main, register, manager, config, vllm_config = registration

    await main.register_vllm_model(
        ModelInput.Tokens,
        model_type,
        Mock(),
        config,
        Mock(),
        vllm_config,
        worker_type=WorkerType.Aggregated,
    )

    register.assert_awaited_once()
    manager.assert_called_once_with(configure_runtime=True)
    runtime = register.await_args.kwargs["runtime_config"]
    assert "dynamo.runtime-lora/v2" in runtime.taints
    assert runtime.runtime_data["supports_runtime_lora_resolution"] == "true"
    assert runtime.runtime_data["runtime_lora_protocol_versions"] == "[2]"
    assert runtime.runtime_data["runtime_lora_schemes"] == '["wandb-artifact"]'


@pytest.mark.asyncio
async def test_runtime_disabled_preserves_other_worker_registration(
    registration, monkeypatch
):
    main, register, manager, config, vllm_config = registration
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "false")

    await main.register_vllm_model(
        ModelInput.Text,
        ModelType.Realtime,
        Mock(),
        config,
        Mock(),
        vllm_config,
        worker_type=WorkerType.Aggregated,
    )

    register.assert_awaited_once()
    manager.assert_not_called()
    runtime = register.await_args.kwargs["runtime_config"]
    assert "dynamo.runtime-lora/v2" not in runtime.taints
    assert "supports_runtime_lora_resolution" not in runtime.runtime_data
