# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.mx_refit import extension


pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_weights = None

    def load_weights(self, *, weights):
        self.loaded_weights = list(weights)


class RecordingDraftModel(RecordingModel):
    def __init__(self, *, org_vocab_size: int | None = None):
        super().__init__()
        self._org_vocab_size = org_vocab_size

    def named_modules(self):
        if self._org_vocab_size is None:
            return []
        return [
            (
                "lm_head",
                SimpleNamespace(org_vocab_size=self._org_vocab_size),
            )
        ]


def _worker_with(model_runner):
    worker = object.__new__(extension.MxRefitWorkerExtension)
    worker.model_runner = model_runner
    return worker


def test_torch_dtype_handles_draft_d2t_int64():
    assert extension._torch_dtype("int64") is torch.int64
    assert extension._torch_dtype("torch.int64") is torch.int64


def test_derives_qwen_llama_qkv_hf_names_without_sidecar_map():
    assert extension._derive_qwen_llama_hf_names(
        "module.decoder.layers.0.self_attention.linear_qkv.weight",
        "qkv_column",
    ) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]


def test_derives_qwen_llama_gated_mlp_hf_names_without_sidecar_map():
    assert extension._derive_qwen_llama_hf_names(
        "module.decoder.layers.1.mlp.linear_fc1.weight",
        "gated_mlp_column",
    ) == [
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
    ]


def test_resolves_identity_megatron_name_map_with_derived_hf_name():
    assert extension._resolve_hf_names(
        "module.decoder.layers.2.input_layernorm.weight",
        "replicated",
        {
            "decoder.layers.2.input_layernorm.weight": [
                "module.decoder.layers.2.input_layernorm.weight"
            ]
        },
    ) == ["model.layers.2.input_layernorm.weight"]


def test_resolves_module_prefixed_hf_name_map_without_wrapper_prefix():
    assert extension._resolve_hf_names(
        "module.decoder.layers.2.input_layernorm.weight",
        "replicated",
        {
            "decoder.layers.2.input_layernorm.weight": [
                "module.model.layers.2.input_layernorm.weight"
            ]
        },
    ) == ["model.layers.2.input_layernorm.weight"]


def test_derives_fallback_name_without_module_wrapper_prefix():
    assert extension._derive_qwen_llama_hf_names(
        "module.decoder.layers.2.unhandled.weight",
        "replicated",
    ) == ["decoder.layers.2.unhandled.weight"]


def test_mx_load_weights_routes_draft_weights_to_drafter():
    policy_model = RecordingModel()
    draft_model = RecordingDraftModel(org_vocab_size=2)
    worker = _worker_with(
        SimpleNamespace(
            model=policy_model,
            drafter=SimpleNamespace(model=draft_model),
        )
    )

    policy_tensor = torch.ones(2, 2)
    draft_tensor = torch.arange(8).reshape(4, 2)

    worker._mx_load_weights(
        [
            ("model.layers.0.weight", policy_tensor),
            ("draft.lm_head.weight", draft_tensor),
        ]
    )

    assert len(policy_model.loaded_weights) == 1
    policy_name, loaded_policy_tensor = policy_model.loaded_weights[0]
    assert policy_name == "model.layers.0.weight"
    assert loaded_policy_tensor is policy_tensor
    assert len(draft_model.loaded_weights) == 1
    draft_name, loaded_draft_tensor = draft_model.loaded_weights[0]
    assert draft_name == "lm_head.weight"
    assert torch.equal(loaded_draft_tensor, draft_tensor[:2])


def test_mx_load_weights_routes_draft_weights_to_speculator_model():
    policy_model = RecordingModel()
    draft_model = RecordingDraftModel(org_vocab_size=2)
    worker = _worker_with(
        SimpleNamespace(
            model=policy_model,
            speculator=SimpleNamespace(model=draft_model),
        )
    )

    draft_tensor = torch.arange(8).reshape(4, 2)

    worker._mx_load_weights([("draft.lm_head.weight", draft_tensor)])

    assert policy_model.loaded_weights == []
    assert len(draft_model.loaded_weights) == 1
    draft_name, loaded_draft_tensor = draft_model.loaded_weights[0]
    assert draft_name == "lm_head.weight"
    assert torch.equal(loaded_draft_tensor, draft_tensor[:2])


def test_mx_load_weights_rejects_draft_weights_without_drafter():
    policy_model = RecordingModel()
    worker = _worker_with(SimpleNamespace(model=policy_model))

    with pytest.raises(RuntimeError, match="vLLM has no drafter"):
        worker._mx_load_weights([("draft.eagle_module.fc.weight", torch.ones(2, 2))])

    assert policy_model.loaded_weights == []
