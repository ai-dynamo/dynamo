# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata
from typing import Optional


def vllm_omni_skip_reason() -> Optional[str]:
    """Return why Omni-only tests must be skipped, or None when they can run.

    Callers must consult this before importing anything that pulls in the Omni
    package, because an Omni build compiled against a different vLLM minor
    version fails at import time rather than at collection time.
    """
    try:
        vllm_version = metadata.version("vllm")
        omni_version = metadata.version("vllm-omni")
    except metadata.PackageNotFoundError:
        return "vLLM-Omni dependencies not available"

    if vllm_version.split(".")[:2] != omni_version.split(".")[:2]:
        return f"vLLM {vllm_version} is incompatible with vLLM-Omni {omni_version}"

    return None
