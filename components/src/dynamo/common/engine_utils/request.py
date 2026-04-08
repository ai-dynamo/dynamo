# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def normalize_request_format(request: dict) -> None:
    """Ensure stop_conditions and sampling_options dicts exist, moving
    top-level OpenAI-style fields into them.

    The Rust frontend may send requests in either the internal protocol
    format (with stop_conditions/sampling_options) or OpenAI format
    (with top-level max_tokens/temperature). This normalizes both to
    the internal format so engines don't need to handle both.

    Modifies *request* in place.
    """
    if "stop_conditions" not in request:
        request["stop_conditions"] = {}
    if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
        request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

    if "sampling_options" not in request:
        request["sampling_options"] = {}
    if (
        "temperature" in request
        and "temperature" not in request["sampling_options"]
    ):
        request["sampling_options"]["temperature"] = request.pop("temperature")
