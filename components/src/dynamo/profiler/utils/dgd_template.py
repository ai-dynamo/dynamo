# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the profiler's private DynamoGraphDeployment blueprints."""

from importlib.resources import files

import yaml

_SUPPORTED_TEMPLATES = frozenset(
    {
        ("mocker", "disagg"),
        ("sglang", "agg"),
        ("sglang", "disagg"),
        ("trtllm", "agg"),
        ("trtllm", "disagg"),
        ("vllm", "agg"),
        ("vllm", "disagg"),
    }
)
_INTERNAL_API_VERSION = "nvidia.com/v1beta1"


def load_dgd_template(backend: str, mode: str) -> dict:
    """Load an internal v1beta1 blueprint for profiler modification.

    Profiler blueprints are private copies so changes to user-facing examples do
    not silently alter generated deployments.
    """
    if (backend, mode) not in _SUPPORTED_TEMPLATES:
        raise ValueError(f"Unsupported profiler DGD template: {backend}/{mode}")

    resource = (
        files("dynamo.profiler")
        .joinpath("templates")
        .joinpath("dgd")
        .joinpath(backend)
        .joinpath(f"{mode}.yaml")
    )
    config = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Profiler DGD template is not an object: {resource}")
    if config.get("apiVersion") != _INTERNAL_API_VERSION:
        raise ValueError(
            f"Profiler DGD template must use {_INTERNAL_API_VERSION}: {resource}"
        )
    if not isinstance(config.get("spec", {}).get("components"), list):
        raise ValueError(
            f"Profiler DGD template must define spec.components: {resource}"
        )
    return config
