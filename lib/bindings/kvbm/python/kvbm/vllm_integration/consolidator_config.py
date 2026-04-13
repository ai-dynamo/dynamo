# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim. Canonical module is kvbm.v1.vllm_integration.consolidator_config."""

from kvbm.v1.vllm_integration.consolidator_config import (  # noqa: F401
    get_consolidator_endpoints,
    is_truthy,
    should_enable_consolidator,
)

__all__ = ["get_consolidator_endpoints", "is_truthy", "should_enable_consolidator"]
