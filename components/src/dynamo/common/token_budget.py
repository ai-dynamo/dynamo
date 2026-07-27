# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Must match `TOKEN_BUDGET_RUNTIME_KEY` in
# `lib/llm/src/local_model/runtime_config.rs`.
TOKEN_BUDGET_RUNTIME_KEY = "token_budget"


class OutputOverflow(str, Enum):
    REJECT = "reject"
    CLAMP = "clamp"
    BACKEND = "backend"


class PromptOverflow(str, Enum):
    REJECT = "reject"
    TRUNCATE = "truncate"
    BACKEND = "backend"


@dataclass(frozen=True)
class TokenBudget:
    combined_limit: int
    output_overflow: OutputOverflow
    prompt_overflow: PromptOverflow

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
                "output_overflow": token_budget.output_overflow.value,
                "prompt_overflow": token_budget.prompt_overflow.value,
            }
        ),
    )
