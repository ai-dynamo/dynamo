# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake in-process plugin module consumed by test_in_process_loader.

The loader imports this module by path and constructs ``FakePlugin``
with the configured ``kwargs``.
"""

from __future__ import annotations

from dynamo.planner.plugins.types import (
    AcceptResult,
    ProposeStageResponse,
)


class FakePlugin:
    def __init__(self, tag: str = "default"):
        self.tag = tag

    async def Propose(self, request):
        return ProposeStageResponse(result_kind="accept", accept=AcceptResult())
