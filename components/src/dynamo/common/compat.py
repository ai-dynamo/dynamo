# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for supported Python versions."""

try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self

__all__ = ["Self"]
