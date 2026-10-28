# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("vllm.lora.request")
pytest.importorskip("safetensors.numpy")
pytest.importorskip("vllm.config.model_arch")

from safetensors.numpy import save_file
from vllm.config.model_arch import ModelArchitectureConfig

from dynamo.common.lora.runtime import ResolvedLoRA, RuntimeLoRAConfigurationError
from dynamo.llm import HttpError
from dynamo.vllm.runtime_lora import RuntimeLoRACoordinator
from dynamo.vllm.tests import test_vllm_runtime_lora as coordinator_tests
from dynamo.vllm.tests.test_vllm_runtime_lora import _handler, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


def _packed_snapshot(tmp_path, *, tensor_rank=16, layer_experts=(2,), layer_ranks=None):
    snapshot = _snapshot(tmp_path, target_modules=["experts"])
    config_path = snapshot / "adapter_config.json"
    config = {**json.loads(config_path.read_text()), "r": 16}
    config_path.write_text(json.dumps(config))
    tensors = {
        f"base_model.model.layers.{layer}.mlp.{module}.lora_{suffix}.weight": np.zeros(
            shape, dtype=np.float32
        )
        for layer, experts in enumerate(layer_experts)
        if experts > 0
        for rank in [tensor_rank if layer_ranks is None else layer_ranks[layer]]
        for module, output in [("experts", 2), ("experts.base_layer", 4)]
        for suffix, shape in [
            ("A", [experts * rank, 2]),
            ("B", [output, experts * rank]),
        ]
    }
    save_file(tensors, str(snapshot / "adapter_model.safetensors"))
    return snapshot


def _model_config(experts, *, nemotron=False):
    base = ModelArchitectureConfig(
        architectures=["fixture"],
        model_type="fixture",
        text_model_type=None,
        hidden_size=2,
        total_num_hidden_layers=len(experts),
        total_num_attention_heads=1,
        head_size=2,
        vocab_size=4,
        total_num_kv_heads=1,
        num_experts=max(experts),
        num_experts_per_token=1,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=False,
        rswa_window=None,
        derived_max_model_len_and_key=(128, "fixture"),
    )
    architecture = (
        base
        if nemotron
        else ModelArchitectureConfig.from_layers(
            [replace(base, num_experts=count) for count in experts]
        )
    )
    text_config = (
        SimpleNamespace(
            block_configs=tuple({} for _ in experts),
            get_nemotron_h_config_for_layer=Mock(
                side_effect=lambda layer: SimpleNamespace(
                    n_routed_experts=experts[layer]
                )
            ),
        )
        if nemotron
        else SimpleNamespace()
    )
    return SimpleNamespace(
        model_arch_config=architecture,
        hf_text_config=text_config,
        get_num_experts=Mock(return_value=max(experts)),
    )


@pytest.fixture
async def coordinator_factory(monkeypatch, tmp_path):
    coordinators = []

    def create(snapshot, model_config=None, *, redundant_experts=None):
        handler = _handler()
        monkeypatch.setattr(handler.engine_args, "max_lora_rank", 16)
        if model_config is not None:
            monkeypatch.setattr(handler, "model_config", model_config, raising=False)
        if redundant_experts is not None:
            monkeypatch.setattr(
                handler.engine_client,
                "vllm_config",
                SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        eplb_config=SimpleNamespace(
                            num_redundant_experts=redundant_experts
                        )
                    )
                ),
                raising=False,
            )
        monkeypatch.setattr(coordinator_tests, "_handler", lambda: handler)
        handler, coordinator = coordinator_tests._coordinator(
            monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
        )
        coordinators.append(coordinator)
        return handler, coordinator

    yield create
    for coordinator in coordinators:
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("experts", [2, 0, None])
async def test_packed_snapshot_admission_requires_trusted_base_expert_count(
    coordinator_factory, tmp_path, experts
):
    snapshot = _packed_snapshot(tmp_path)
    model_config = (
        SimpleNamespace(get_num_experts=Mock(return_value=experts))
        if experts is not None
        else None
    )
    handler, coordinator = coordinator_factory(snapshot, model_config)
    async with coordinator.pending_admission_guard("request"):
        if experts == 2:
            result = await coordinator.ensure_from_request(
                _request("wandb-artifact:///entity/project/packed:v1"), "request"
            )
            assert result is not None
            assert result.lora_path == str(snapshot)
            handler.engine_client.add_lora.assert_awaited_once()
        else:
            with pytest.raises(HttpError, match="invalid_lora_adapter") as error:
                await coordinator.ensure_from_request(
                    _request("wandb-artifact:///entity/project/packed:v1"), "request"
                )
            assert error.value.code == 422
            handler.engine_client.add_lora.assert_not_awaited()
    if model_config is not None:
        model_config.get_num_experts.assert_called_once_with()


@pytest.mark.asyncio
async def test_trusted_expert_count_does_not_allow_oversized_per_expert_rank(
    coordinator_factory, tmp_path
):
    snapshot = _packed_snapshot(tmp_path, tensor_rank=32)
    model_config = SimpleNamespace(get_num_experts=Mock(return_value=2))
    handler, coordinator = coordinator_factory(snapshot, model_config)

    with pytest.raises(HttpError, match="invalid_lora_adapter") as error:
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/oversized:v1"), "request"
        )

    assert error.value.code == 422
    handler.engine_client.add_lora.assert_not_awaited()
    model_config.get_num_experts.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("nemotron", [False, True], ids=["canonical", "nemotron"])
@pytest.mark.parametrize("ranks,accepted", [((16, 16), True), ((32, 16), False)])
async def test_heterogeneous_layer_counts_do_not_use_model_maximum(
    coordinator_factory, tmp_path, nemotron, ranks, accepted
):
    model_config = _model_config((2, 4), nemotron=nemotron)
    snapshot = _packed_snapshot(tmp_path, layer_experts=(2, 4), layer_ranks=ranks)
    handler, coordinator = coordinator_factory(snapshot, model_config)
    async with coordinator.pending_admission_guard("request"):
        if accepted:
            result = await coordinator.ensure_from_request(
                _request("wandb-artifact:///entity/project/heterogeneous:v1"), "request"
            )
            assert result is not None
            handler.engine_client.add_lora.assert_awaited_once()
        else:
            with pytest.raises(HttpError, match="invalid_lora_adapter") as error:
                await coordinator.ensure_from_request(
                    _request("wandb-artifact:///entity/project/heterogeneous:v1"),
                    "request",
                )
            assert error.value.code == 422
            handler.engine_client.add_lora.assert_not_awaited()
    model_config.get_num_experts.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rank,accepted", [(16, True), (32, False)])
async def test_resolved_redundant_experts_use_physical_packed_count(
    coordinator_factory, tmp_path, rank, accepted
):
    snapshot = _packed_snapshot(tmp_path, layer_experts=(4,), tensor_rank=rank)
    handler, coordinator = coordinator_factory(
        snapshot, _model_config((2,)), redundant_experts=2
    )
    async with coordinator.pending_admission_guard("request"):
        if accepted:
            result = await coordinator.ensure_from_request(
                _request("wandb-artifact:///entity/project/redundant:v1"), "request"
            )
            assert result is not None
            handler.engine_client.add_lora.assert_awaited_once()
        else:
            with pytest.raises(HttpError, match="invalid_lora_adapter") as error:
                await coordinator.ensure_from_request(
                    _request("wandb-artifact:///entity/project/redundant:v1"), "request"
                )
            assert error.value.code == 422
            handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_redundant_experts_do_not_turn_a_dense_layer_into_moe(
    coordinator_factory, tmp_path
):
    snapshot = _packed_snapshot(tmp_path, layer_experts=(0, 4))
    handler, coordinator = coordinator_factory(
        snapshot, _model_config((0, 2)), redundant_experts=2
    )
    assert coordinator._num_experts == (0, 4)
    async with coordinator.pending_admission_guard("request"):
        result = await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/mixed:v1"), "request"
        )
    assert result is not None
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", ["blocks", "heterogeneous"])
@pytest.mark.parametrize("has_architecture", [False, True])
async def test_unknown_heterogeneous_metadata_preserves_unadjusted_dense_admission(
    coordinator_factory, tmp_path, metadata, has_architecture
):
    text_config = (
        SimpleNamespace(block_configs=({},))
        if metadata == "blocks"
        else SimpleNamespace(is_heterogeneous=True)
    )
    model_config = SimpleNamespace(
        hf_text_config=text_config, get_num_experts=Mock(return_value=4)
    )
    if has_architecture:
        model_config.model_arch_config = replace(
            _model_config((2, 4)).model_arch_config, per_layer_overrides=None
        )
    handler, coordinator = coordinator_factory(_snapshot(tmp_path), model_config)
    assert coordinator._num_experts is None
    async with coordinator.pending_admission_guard("request"):
        result = await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/attention:v1"), "request"
        )
    assert result is not None
    model_config.get_num_experts.assert_not_called()
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_config_kind", ["missing", "missing_getter", "dense"])
async def test_missing_or_dense_model_config_preserves_ordinary_admission(
    coordinator_factory, tmp_path, model_config_kind
):
    model_config = {
        "missing": None,
        "missing_getter": SimpleNamespace(),
        "dense": SimpleNamespace(get_num_experts=Mock(return_value=0)),
    }[model_config_kind]
    handler, coordinator = coordinator_factory(_snapshot(tmp_path), model_config)
    async with coordinator.pending_admission_guard("request"):
        result = await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/dense:v1"), "request"
        )
    assert result is not None
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("experts", [True, False, -1, 2.0, "2", None])
async def test_invalid_trusted_expert_count_fails_coordinator_startup(
    coordinator_factory, tmp_path, experts
):
    model_config = SimpleNamespace(get_num_experts=Mock(return_value=experts))
    with pytest.raises(RuntimeLoRAConfigurationError, match="expert count"):
        coordinator_factory(_snapshot(tmp_path), model_config)


@pytest.mark.asyncio
async def test_noncallable_expert_count_getter_fails_coordinator_startup(
    coordinator_factory, tmp_path
):
    with pytest.raises(RuntimeLoRAConfigurationError, match="expert count"):
        coordinator_factory(_snapshot(tmp_path), SimpleNamespace(get_num_experts=2))


@pytest.mark.asyncio
@pytest.mark.parametrize("experts", [True, -1, 2.0, None])
async def test_invalid_per_layer_expert_count_fails_startup(
    coordinator_factory, tmp_path, experts
):
    model_config = _model_config((2, 4))
    model_config.model_arch_config = replace(
        model_config.model_arch_config,
        per_layer_overrides=[{}, {"num_experts": experts}],
    )
    with pytest.raises(RuntimeLoRAConfigurationError, match="expert count"):
        coordinator_factory(_snapshot(tmp_path), model_config)


@pytest.mark.asyncio
@pytest.mark.parametrize("experts", [True, -1, 2.0, "2"])
async def test_invalid_resolved_redundant_expert_count_fails_startup(
    coordinator_factory, tmp_path, experts
):
    with pytest.raises(RuntimeLoRAConfigurationError, match="expert count"):
        coordinator_factory(
            _snapshot(tmp_path), _model_config((2,)), redundant_experts=experts
        )


def test_disabled_runtime_does_not_read_base_expert_count(monkeypatch):
    handler = _handler()
    model_config = SimpleNamespace(
        get_num_experts=Mock(side_effect=AssertionError("must not read model config"))
    )
    monkeypatch.setattr(handler, "model_config", model_config, raising=False)
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "false")

    coordinator = RuntimeLoRACoordinator(handler)

    assert not coordinator.enabled
    model_config.get_num_experts.assert_not_called()
