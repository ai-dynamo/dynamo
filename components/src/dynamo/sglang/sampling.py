# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from dynamo.llm import HttpError


def enforce_max_output_tokens(
    sampling_params: dict[str, Any], max_output_tokens: int | None
) -> dict[str, Any]:
    """Apply Dynamo's configured hard output-token ceiling to SGLang params."""
    if max_output_tokens is None:
        return sampling_params

    requested_max = sampling_params.get("max_new_tokens")
    if requested_max is None or requested_max > max_output_tokens:
        sampling_params["max_new_tokens"] = max_output_tokens

    requested_min = sampling_params.get("min_new_tokens")
    if requested_min is not None and requested_min > max_output_tokens:
        raise HttpError(
            400,
            f"min_tokens cannot exceed the configured SGLang output limit "
            f"of {max_output_tokens}",
        )

    return sampling_params
