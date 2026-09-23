# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore the legacy backend's cache metadata RPC for vLLM 0.30.0."""

import os
from importlib.metadata import version

from packaging.version import Version

_ACTIVATION_ENV = "DYN_VLLM_KV_CACHE_METADATA_COMPAT"
_PLUGIN_NAME = "dynamo_kv_cache_metadata"
_RAY_COPY_ENV = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"


def enable_kv_cache_metadata_compat() -> None:
    """Enable the compatibility plugin in this legacy launcher and its engines."""
    os.environ[_ACTIVATION_ENV] = "1"
    ray_vars = os.environ.get(_RAY_COPY_ENV, "")
    if _ACTIVATION_ENV not in {name.strip() for name in ray_vars.split(",")}:
        os.environ[_RAY_COPY_ENV] = (
            f"{ray_vars},{_ACTIVATION_ENV}" if ray_vars else _ACTIVATION_ENV
        )
    if "VLLM_PLUGINS" in os.environ:
        plugins = os.environ["VLLM_PLUGINS"].split(",")
        if _PLUGIN_NAME not in plugins:
            plugins.append(_PLUGIN_NAME)
            os.environ["VLLM_PLUGINS"] = ",".join(plugins)
    register()


def register() -> None:
    """Install the missing method only for an opted-in vLLM 0.30.0 process."""
    # TODO: Reassess next vLLM bump; remove once native metadata replaces this.
    if os.environ.get(_ACTIVATION_ENV) != "1":
        return
    if Version(version("vllm")).public != "0.30.0":
        return

    # General plugins must not import EngineCore before legacy activation.
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    if hasattr(EngineCore, "get_kv_cache_group_metadata"):
        return

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        managers = self.scheduler.kv_cache_manager.coordinator.single_type_managers
        metadata = []
        for group_idx, (group, manager) in enumerate(
            zip(kv_cache_config.kv_cache_groups, managers, strict=True)
        ):
            spec = group.kv_cache_spec
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": get_kv_cache_spec_kind(spec).value,
                    "block_size": manager.block_size,
                    "sliding_window": getattr(spec, "sliding_window", None),
                }
            )
        return metadata

    setattr(EngineCore, "get_kv_cache_group_metadata", get_kv_cache_group_metadata)
