# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for runtime environment helpers."""

from dynamo.planner.environment.runtime import (
    RuntimeNamespaceBinding,
    RuntimeNamespaceResolver,
)

__all__ = ["RuntimeNamespaceBinding", "RuntimeNamespaceResolver"]
