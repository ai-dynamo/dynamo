# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Must match `TOKEN_BUDGET_RUNTIME_KEY` in
# `lib/llm/src/local_model/runtime_config.rs`.
TOKEN_BUDGET_RUNTIME_KEY = "token_budget"


@dataclass(frozen=True)
class TokenBudget:
    """Advertise which token-overflow requests the frontend may reject early.

    A false flag delegates that overflow dimension to the backend. The backend
    remains responsible for any clamping, truncation, or rejection after that.
    """

    combined_limit: int
    reject_prompt_overflow: bool
    reject_total_overflow: bool

    def __post_init__(self) -> None:
        if self.combined_limit < 0:
            raise ValueError("combined_limit must be non-negative")


def publish_token_budget(runtime_config: Any, token_budget: TokenBudget) -> None:
    """Publish an engine's token-overflow contract to the Dynamo frontend."""
    runtime_config.set_engine_specific(
        TOKEN_BUDGET_RUNTIME_KEY,
        json.dumps(
            {
                "combined_limit": token_budget.combined_limit,
                "reject_prompt_overflow": token_budget.reject_prompt_overflow,
                "reject_total_overflow": token_budget.reject_total_overflow,
            }
        ),
    )
