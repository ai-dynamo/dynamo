# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a deployment can actually do, derived from its configuration.

Two rules from DEP 0017 shape this module:

* **Capability comes from configuration, never from a status code.** A
  deployment with no constrained-decoding backend accepts
  ``tool_choice: "required"`` with HTTP 200 and then ignores it, so probing by
  sending a request and reading the status would report it as working.
* **Evaluation is three-valued.** ``UNKNOWN`` must not collapse into
  ``UNSATISFIED``: "we could not determine this" is a different fact from "this
  is not supported", and silently skipping on the former hides harness defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Capability(Enum):
    TOOL_CALLING = "tool_calling"
    REASONING_PARSER = "reasoning_parser"
    CONSTRAINED_DECODING = "constrained_decoding"


class Verdict(Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Report:
    """A verdict plus why, and which fact source produced it."""

    capability: Capability
    verdict: Verdict
    reason: str
    source: str

    def __str__(self) -> str:
        return f"{self.capability.value}={self.verdict.value} ({self.reason}; via {self.source})"


# Backends with a built-in grammar/constrained-decoding engine. TensorRT-LLM
# needs guided_decoding_backend set explicitly -- without it a deployment
# accepts tool_choice:"required" with HTTP 200 and silently ignores it,
# which is exactly the accept-but-ignore trap this module exists to avoid.
GRAMMAR_BUILTIN = {"vllm", "sglang"}


def from_worker_flags(
    flags: Dict[str, str], source: str, backend: str = ""
) -> Dict[Capability, Report]:
    """Derive capabilities from the flags a deployment was launched with.

    Only what the flags actually prove. Everything else stays UNKNOWN rather
    than being guessed at.
    """
    reports: Dict[Capability, Report] = {}

    parser = flags.get("dyn-tool-call-parser")
    reports[Capability.TOOL_CALLING] = Report(
        Capability.TOOL_CALLING,
        Verdict.SATISFIED if parser else Verdict.UNSATISFIED,
        f"--dyn-tool-call-parser={parser}"
        if parser
        else "no --dyn-tool-call-parser set",
        source,
    )

    reasoning = flags.get("dyn-reasoning-parser")
    reports[Capability.REASONING_PARSER] = Report(
        Capability.REASONING_PARSER,
        Verdict.SATISFIED if reasoning else Verdict.UNSATISFIED,
        f"--dyn-reasoning-parser={reasoning}"
        if reasoning
        else "no --dyn-reasoning-parser set",
        source,
    )

    guided = flags.get("guided-decoding-backend")
    if guided:
        verdict, reason = Verdict.SATISFIED, f"guided_decoding_backend={guided}"
    elif backend in GRAMMAR_BUILTIN:
        verdict, reason = Verdict.SATISFIED, f"{backend} has a built-in grammar backend"
    elif backend:
        verdict, reason = (
            Verdict.UNSATISFIED,
            f"{backend} needs guided_decoding_backend set; without it "
            "tool_choice/response_format are accepted but not enforced",
        )
    else:
        verdict, reason = Verdict.UNKNOWN, "backend unknown, cannot determine"
    reports[Capability.CONSTRAINED_DECODING] = Report(
        Capability.CONSTRAINED_DECODING, verdict, reason, source
    )
    return reports


def all_unknown(source: str) -> Dict[Capability, Report]:
    """Used when we did not create the deployment and cannot read its config."""
    return {
        cap: Report(
            cap,
            Verdict.UNKNOWN,
            "attached to a deployment whose configuration is not readable",
            source,
        )
        for cap in Capability
    }
