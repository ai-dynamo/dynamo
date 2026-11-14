# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime evidence used by the Rust frontend to interpret a guided answer."""

import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

GUIDED_OUTPUT_REASONING_RUNTIME_KEY = "guided_output_reasoning"


@dataclass
class GuidedOutputCapability:
    engine: str
    mode: str
    reasoner: str | None = None
    request_controls: bool = False
    start_token_ids: list[int] = field(default_factory=list)
    end_token_ids: list[int] = field(default_factory=list)
    start_text: str = ""
    end_text: str = ""
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def has_named_parameter(function: Callable[..., Any], name: str) -> bool:
    """A catch-all kwargs parameter is not evidence that a control is honored."""
    try:
        parameter = inspect.signature(function).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def explicit_gate_capability(
    engine: str,
    reasoner: str,
    tokenizer: Any,
    start_text: str,
    end_text: str,
    end_token_ids: list[int],
    request_controls: bool,
) -> GuidedOutputCapability:
    """V1 advertises only a single, losslessly decoded explicit gate token."""
    unsupported = GuidedOutputCapability(
        engine, "unsupported", reasoner, request_controls
    )
    # Supported engine versions differ in their treatment of multi-ID closers.
    # Do not advertise a sequence while an engine might gate on its first ID.
    if not request_controls or len(end_token_ids) != 1 or not end_text:
        return unsupported
    decoded = tokenizer.decode(end_token_ids, skip_special_tokens=False)
    if decoded != end_text or "\ufffd" in decoded:
        return unsupported
    start_ids = (
        list(tokenizer.encode(start_text, add_special_tokens=False))
        if start_text
        else []
    )
    if (
        start_ids
        and tokenizer.decode(start_ids, skip_special_tokens=False) != start_text
    ):
        start_ids = []
        start_text = ""
    return GuidedOutputCapability(
        engine,
        "explicit_gate",
        reasoner,
        request_controls,
        start_ids,
        list(end_token_ids),
        start_text,
        end_text,
    )
