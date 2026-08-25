# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that every DiffusionParallelConfig field is either exposed in Dynamo or intentionally skipped."""

import dataclasses
from unittest.mock import MagicMock

import pytest

try:
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    from dynamo.vllm.omni.args import OmniDiffusionKwargs, OmniParallelKwargs
    from dynamo.vllm.omni.base_handler import BaseOmniHandler
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# These fields are not exposed in OmniParallelKwargs, because they are derived from other fields.
_SKIP_FIELDS = {
    "sequence_parallel_size",
    "enable_expert_parallel",
    "ulysses_mode",
    "mask_sp_padding",
}

# DiffusionParallelConfig fields deliberately sourced from vLLM's shared
# engine arguments rather than Dynamo's Omni-only parallel argument group.
_ENGINE_ARG_PARALLEL_FIELDS = {
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
}


def _diffusion_parallel_fields() -> set:
    return {f.name for f in dataclasses.fields(DiffusionParallelConfig)}


def _make_config(**parallel_overrides):
    cfg = MagicMock()
    cfg.model = "test-model"
    cfg.stage_configs_path = None
    cfg.output_modalities = None
    cfg.engine_args.trust_remote_code = False
    cfg.engine_args.enable_lora = False
    cfg.engine_args.max_cpu_loras = None
    cfg.engine_args.max_loras = None
    cfg.engine_args.tensor_parallel_size = 1
    cfg.engine_args.pipeline_parallel_size = 1
    cfg.engine_args.data_parallel_size = 1
    cfg.diffusion = OmniDiffusionKwargs()
    cfg.parallel = dataclasses.replace(OmniParallelKwargs(), **parallel_overrides)
    return cfg


def _build_kwargs(config):
    handler = BaseOmniHandler.__new__(BaseOmniHandler)
    return handler._build_omni_kwargs(config)


class TestDiffusionParallelConfigCoverage:
    def test_all_diffusion_parallel_config_fields_covered(self):
        """Every DiffusionParallelConfig field must be in OmniParallelKwargs, engine_args, or _SKIP_FIELDS.

        When vllm-omni adds a new parallelism field to DiffusionParallelConfig, this test fails.
        Fix by adding it to OmniParallelKwargs and OmniArgGroup, or to _SKIP_FIELDS
        """
        parallel_kwarg_fields = {f.name for f in dataclasses.fields(OmniParallelKwargs)}
        uncovered = [
            f
            for f in _diffusion_parallel_fields()
            if f not in _SKIP_FIELDS
            and f not in parallel_kwarg_fields
            and f not in _ENGINE_ARG_PARALLEL_FIELDS
        ]
        assert not uncovered, (
            f"DiffusionParallelConfig fields not covered: {uncovered}. "
            f"Add to OmniParallelKwargs and OmniArgGroup, or add to _SKIP_FIELDS with a reason."
        )

    def test_parallel_fields_forwarded_from_separate_configs(self):
        """Construct the real vLLM-Omni config from both argument groups."""
        config = _make_config(text_encoder_tp_size=2)
        config.engine_args.tensor_parallel_size = 4
        config.engine_args.pipeline_parallel_size = 3
        config.engine_args.data_parallel_size = 5

        parallel_config = _build_kwargs(config)["parallel_config"]

        assert parallel_config.tensor_parallel_size == 4
        assert parallel_config.pipeline_parallel_size == 3
        assert parallel_config.data_parallel_size == 5
        assert parallel_config.text_encoder_tp_size == 2

    def test_output_modalities_forwarded_to_async_omni(self):
        config = _make_config()
        config.output_modalities = ["image"]

        kwargs = _build_kwargs(config)

        assert kwargs["output_modalities"] == ["image"]

    def test_model_defined_diffusion_fields_forwarded_to_async_omni(self):
        config = _make_config()
        config.diffusion = dataclasses.replace(
            OmniDiffusionKwargs(),
            task_type="t2va",
            diffusion_attention_backend="TRTLLM_ATTN",
        )

        kwargs = _build_kwargs(config)

        assert kwargs["task_type"] == "t2va"
        assert kwargs["diffusion_attention_backend"] == "TRTLLM_ATTN"

    def test_lora_disabled_resolves_no_capacity(self):
        config = _make_config()
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) is None

    def test_lora_enabled_with_unset_max_loras_resolves_no_capacity_limit(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) is None

    def test_lora_enabled_uses_configured_max_cpu_loras(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = 3
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) == 3

    def test_lora_enabled_falls_back_to_max_loras_when_max_cpu_loras_unset(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = None
        config.engine_args.max_loras = 2
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) == 2

    def test_advertised_gpu_capacity_uses_max_loras_even_when_max_cpu_loras_set(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = 8
        config.engine_args.max_loras = 2
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_advertised_gpu_lora_capacity(config) == 2

    def test_advertised_gpu_capacity_none_when_lora_disabled(self):
        config = _make_config()
        config.engine_args.enable_lora = False
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_advertised_gpu_lora_capacity(config) is None
