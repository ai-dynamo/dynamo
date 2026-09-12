# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM-Omni stage config compatibility."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from dynamo.vllm.omni import stage_config_compat
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),
]


def test_resolve_stage_configs_uses_structured_resolver(monkeypatch):
    resolution = SimpleNamespace(
        config_path="/deploy/model.yaml",
        stage_configs=("stage-0", "stage-1"),
        omni_lb_policy="round_robin",
    )
    resolver = MagicMock(return_value=resolution)
    monkeypatch.setattr(stage_config_compat, "resolve_omni_config", resolver)

    result = stage_config_compat.resolve_stage_configs(
        "model",
        kwargs={"tensor_parallel_size": 4},
        trust_remote_code=True,
        deploy_config_path="/deploy/model.yaml",
    )

    assert result == (
        "/deploy/model.yaml",
        ["stage-0", "stage-1"],
        "round_robin",
    )
    resolver.assert_called_once_with(
        "model",
        trust_remote_code=True,
        deploy_config_path="/deploy/model.yaml",
        cli_overrides={"tensor_parallel_size": 4},
        stage_overrides=None,
        strategy_config_path=None,
    )


@pytest.mark.parametrize("has_legacy_path", [False, True])
def test_resolve_stage_configs_uses_legacy_resolver(monkeypatch, has_legacy_path):
    if has_legacy_path:

        def legacy_resolver(
            model,
            stage_configs_path,
            kwargs,
            *,
            trust_remote_code,
            deploy_config_path,
        ):
            assert stage_configs_path is None
            return deploy_config_path, [kwargs], None

    else:

        def legacy_resolver(
            model,
            kwargs,
            *,
            trust_remote_code,
            deploy_config_path,
        ):
            return deploy_config_path, [kwargs], None

    monkeypatch.setattr(stage_config_compat, "resolve_omni_config", None)
    monkeypatch.setattr(
        stage_config_compat, "load_and_resolve_stage_configs", legacy_resolver
    )

    result = stage_config_compat.resolve_stage_configs(
        "model",
        kwargs={"tensor_parallel_size": 4},
        trust_remote_code=False,
        deploy_config_path="/deploy/model.yaml",
    )

    assert result == ("/deploy/model.yaml", [{"tensor_parallel_size": 4}], None)
