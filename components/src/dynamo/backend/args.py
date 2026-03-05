# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common argument handling for Dynamo backend workers.

This module re-exports the canonical DynamoRuntimeConfig and
DynamoRuntimeArgGroup from dynamo.common.configuration.groups.runtime_args
so that backend implementations can import from dynamo.backend.
"""

from dynamo.common.configuration.groups.runtime_args import (  # noqa: F401
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
