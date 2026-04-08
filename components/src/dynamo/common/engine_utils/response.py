# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.common.utils.engine_response import normalize_finish_reason  # noqa: F401


def build_completion_usage(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_tokens_details: dict | None = None,
) -> dict:
    """Build the completion_usage dict yielded on the final streaming chunk."""
    usage: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if prompt_tokens_details is not None:
        usage["prompt_tokens_details"] = prompt_tokens_details
    return usage
