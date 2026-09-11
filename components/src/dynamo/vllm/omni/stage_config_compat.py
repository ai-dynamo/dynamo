# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for vLLM-Omni stage configuration resolution."""

import inspect
from typing import Any

try:
    from vllm_omni.config.resolver import resolve_omni_config
except ImportError:
    resolve_omni_config = None  # type: ignore[assignment]
    from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
else:
    load_and_resolve_stage_configs = None  # type: ignore[assignment]


def resolve_stage_configs(
    model: str,
    *,
    kwargs: dict[str, Any],
    trust_remote_code: bool,
    deploy_config_path: str | None,
) -> tuple[str | None, list[Any], str | None]:
    """Resolve stage configs across the vLLM-Omni 0.27-0.29 API transition."""
    if resolve_omni_config is not None:
        resolution = resolve_omni_config(
            model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
            cli_overrides=kwargs,
            stage_overrides=None,
            strategy_config_path=None,
        )
        return (
            resolution.config_path,
            list(resolution.stage_configs),
            resolution.omni_lb_policy,
        )

    assert load_and_resolve_stage_configs is not None
    positional_args: list[Any] = [model]
    if (
        "stage_configs_path"
        in inspect.signature(load_and_resolve_stage_configs).parameters
    ):
        positional_args.append(None)
    positional_args.append(kwargs)
    return load_and_resolve_stage_configs(
        *positional_args,
        trust_remote_code=trust_remote_code,
        deploy_config_path=deploy_config_path,
    )
