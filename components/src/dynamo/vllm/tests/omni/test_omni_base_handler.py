# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that every DiffusionParallelConfig field is either exposed in Dynamo or intentionally skipped."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.engine.arg_utils import OmniEngineArgs

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

# Fields intentionally not in OmniParallelKwargs — must include a reason.
_SKIP_FIELDS = {
    "sequence_parallel_size",  # derived: ulysses_degree * ring_degree
    "enable_expert_parallel",  # MoE-only, no diffusion model uses this yet
    "ulysses_mode",            # advanced Ulysses variant, not yet exposed
}


def _diffusion_parallel_fields() -> set:
    return {f.name for f in dataclasses.fields(DiffusionParallelConfig)}


def _engine_args_fields() -> set:
    fields: set = set()
    for cls in OmniEngineArgs.__mro__:
        fields |= set(getattr(cls, "__annotations__", {}).keys())
    return fields


def _make_config(**parallel_overrides):
    cfg = MagicMock()
    cfg.model = "test-model"
    cfg.stage_configs_path = None
    cfg.engine_args.trust_remote_code = False
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
        Fix by adding it to OmniParallelKwargs and OmniArgGroup, or to _SKIP_FIELDS with a reason.
        """
        parallel_kwarg_fields = {f.name for f in dataclasses.fields(OmniParallelKwargs)}
        engine_fields = _engine_args_fields()

        uncovered = [
            f for f in _diffusion_parallel_fields()
            if f not in _SKIP_FIELDS
            and f not in parallel_kwarg_fields
            and f not in engine_fields
        ]
        assert not uncovered, (
            f"DiffusionParallelConfig fields not covered: {uncovered}. "
            f"Add to OmniParallelKwargs and OmniArgGroup, or add to _SKIP_FIELDS with a reason."
        )

    def test_tensor_parallel_size_read_from_engine_args(self):
        """tensor_parallel_size must come from engine_args (vLLM's --tensor-parallel-size),
        not from OmniParallelKwargs, so it applies to both LLM encoder and diffusion transformer."""
        config = _make_config()
        config.engine_args.tensor_parallel_size = 4
        with patch("dynamo.vllm.omni.base_handler.DiffusionParallelConfig") as MockCfg:
            MockCfg.return_value = SimpleNamespace()
            _build_kwargs(config)
            _, kwargs = MockCfg.call_args
            assert kwargs.get("tensor_parallel_size") == 4
