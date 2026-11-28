# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: the kernel-selecting settings against the installed vLLM."""

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)
# Not importorskip("vllm"): in an image without vLLM, pytest puts
# components/src/dynamo on sys.path while it collects the component tests,
# and "vllm" then resolves to Dynamo's own vllm component.
pytest.importorskip("vllm.model_executor")

from gpu_memory_service.integrations.vllm.load_diff import descriptor  # noqa: E402
from vllm import envs  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_kernel_env_names_are_vllm_env_vars():
    """A variable vLLM renames or removes must not stay in the list unnoticed."""
    assert set(descriptor._KERNEL_ENV) <= set(envs.environment_variables)
