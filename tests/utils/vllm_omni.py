# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata
from typing import Optional


def vllm_omni_skip_reason() -> Optional[str]:
    """Return why Omni-only tests must be skipped, or None when they can run.

    Callers must consult this before importing anything that pulls in the Omni
    package.
    """
    try:
        metadata.version("vllm")
        metadata.version("vllm-omni")
    except metadata.PackageNotFoundError:
        return "vLLM-Omni dependencies not available"

    return None
