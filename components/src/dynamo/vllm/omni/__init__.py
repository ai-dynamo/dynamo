# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni integration for Dynamo."""

__all__ = ["BaseOmniHandler", "OmniHandler"]


def __getattr__(name: str):
    if name == "BaseOmniHandler":
        from .base_handler import BaseOmniHandler

        return BaseOmniHandler
    if name == "OmniHandler":
        from .omni_handler import OmniHandler

        return OmniHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
