# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni integration for Dynamo."""

from .version_check import check_vllm_omni_compatibility

# Fail fast on a vLLM / vLLM-Omni version mismatch, which otherwise
# surfaces as an opaque ImportError deep inside vllm_omni.
check_vllm_omni_compatibility()

from .base_handler import BaseOmniHandler  # noqa: E402
from .omni_handler import OmniHandler  # noqa: E402
from .realtime_handler import RealtimeOmniHandler  # noqa: E402

__all__ = ["BaseOmniHandler", "OmniHandler", "RealtimeOmniHandler"]
