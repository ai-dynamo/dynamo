#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared behavioral cases for backend thinking-control tests.

Rust `normalize_reasoning_template_args` resolves every thinking control at the
HTTP boundary, before either processor runs, so these cases start from its
output. A processor forwards that decision; it never makes one.
"""

from dataclasses import dataclass
from typing import Any

# Model families read different keys, so a decision is written in all three.
RESOLVED_ENABLED = {
    "thinking": True,
    "enable_thinking": True,
    "thinking_mode": "enabled",
}
RESOLVED_DISABLED = {
    "thinking": False,
    "enable_thinking": False,
    "thinking_mode": "disabled",
}
# `adaptive` defers to the model, so it carries no toggle.
RESOLVED_ADAPTIVE = {"thinking_mode": "adaptive"}


@dataclass(frozen=True)
class ThinkingParityCase:
    name: str
    # Fields added to a minimal chat request, as the frontend hands them over.
    request: dict[str, Any]
    # Template kwargs every backend must produce for that request.
    expected: dict[str, Any]


THINKING_PARITY_CASES = (
    ThinkingParityCase(
        "enabled", {"chat_template_args": RESOLVED_ENABLED}, RESOLVED_ENABLED
    ),
    ThinkingParityCase(
        "disabled", {"chat_template_args": RESOLVED_DISABLED}, RESOLVED_DISABLED
    ),
    ThinkingParityCase(
        "adaptive", {"chat_template_args": RESOLVED_ADAPTIVE}, RESOLVED_ADAPTIVE
    ),
    # `none` is the regression this guards: it used to put a toggle back.
    # `high` pins the documented promise that a grade still reaches graders.
    *(
        ThinkingParityCase(
            f"adaptive-ignores-effort-{effort}",
            {
                "chat_template_args": {**RESOLVED_ADAPTIVE, "reasoning_effort": effort},
                "reasoning_effort": effort,
            },
            {**RESOLVED_ADAPTIVE, "reasoning_effort": effort},
        )
        for effort in ("none", "high")
    ),
    ThinkingParityCase("no-controls", {}, {}),
)
