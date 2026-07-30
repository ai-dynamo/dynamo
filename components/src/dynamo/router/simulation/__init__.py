# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spica search adapter for Dynamo Router simulation."""

from .adapter import DynamoRouterSimulationAdapter, create_adapter

__all__ = ["DynamoRouterSimulationAdapter", "create_adapter"]
