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
    stop_cond = request.get("stop_conditions")
    if stop_cond is None:
        request["stop_conditions"] = {}
    elif not isinstance(stop_cond, dict):
        raise TypeError(
            f"request['stop_conditions'] must be a dict, got {type(stop_cond).__name__}"
        )
    if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
        request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

    sampling_opts = request.get("sampling_options")
    if sampling_opts is None:
        request["sampling_options"] = {}
    elif not isinstance(sampling_opts, dict):
        raise TypeError(
            f"request['sampling_options'] must be a dict, got {type(sampling_opts).__name__}"
        )
    if "temperature" in request and "temperature" not in request["sampling_options"]:
        request["sampling_options"]["temperature"] = request.pop("temperature")
