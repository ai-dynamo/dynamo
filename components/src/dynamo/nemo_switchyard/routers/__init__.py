#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Auto-import concrete router modules to trigger registration.

RouteLLM is optional — if ``routellm`` / ``transformers`` are not
installed, the RouteLLM router simply won't be available.
"""

from . import random_router, round_robin_router  # noqa: F401

try:
    from . import routellm_router  # noqa: F401
except ImportError:
    pass
