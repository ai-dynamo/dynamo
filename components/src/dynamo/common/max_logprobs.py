# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine's log-probability limit, as published to the Dynamo frontend."""

from __future__ import annotations

import json
from typing import Any

# Must match `MAX_LOGPROBS_RUNTIME_KEY` in
# `lib/llm/src/local_model/runtime_config.rs`.
MAX_LOGPROBS_RUNTIME_KEY = "max_logprobs"


def publish_max_logprobs(runtime_config: Any, max_logprobs: int) -> None:
    """Publish the most per-token log-probability entries the engine returns.

    The frontend accepts chat and responses ``top_logprobs`` and completions
    ``logprobs`` up to this value for the model. A worker that publishes
    nothing gets the OpenAI limit of 20.
    """
    if type(max_logprobs) is not int or max_logprobs < 0:
        raise ValueError(
            f"max_logprobs must be a non-negative integer, got {max_logprobs!r}"
        )
    runtime_config.set_engine_specific(
        MAX_LOGPROBS_RUNTIME_KEY, json.dumps(max_logprobs)
    )


def read_max_logprobs(runtime_config: Any) -> int | None:
    """Return the limit published in a model card's runtime config, if any.

    ``runtime_config`` is the dict form of the card's runtime config. A missing
    or malformed value returns ``None``, which callers treat as "not published".
    """
    if not isinstance(runtime_config, dict):
        return None
    runtime_data = runtime_config.get("runtime_data")
    if not isinstance(runtime_data, dict):
        return None
    max_logprobs = runtime_data.get(MAX_LOGPROBS_RUNTIME_KEY)
    if type(max_logprobs) is not int or max_logprobs < 0:
        return None
    return max_logprobs
