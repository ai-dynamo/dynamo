# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible mocker wrapper around the shared internal AIC bridge."""

from dynamo._internal.aic import AicSession, create_session

__all__ = ["AicSession", "create_session"]
