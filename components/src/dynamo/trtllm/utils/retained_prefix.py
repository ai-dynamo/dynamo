# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, fields, replace
from typing import Any

from dynamo.llm.exceptions import InvalidArgument

# Retained-prefix support is optional across supported TRT runtime versions.
try:
    from tensorrt_llm.retained_prefix import parse_retained_prefix
except ImportError:
    parse_retained_prefix = None


def request_options(request: dict) -> dict | None:
    nvext = (request.get("extra_args") or {}).get("nvext") or {}
    options = nvext.get("trtllm_retained_prefix")
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) - {
        "retain_prefix",
        "retained_prefix",
    }:
        raise InvalidArgument("invalid nvext.trtllm_retained_prefix options")
    if type(options.get("retain_prefix", False)) is not bool:
        raise InvalidArgument("retain_prefix must be a boolean")
    return options


def apply_options(params: Any, options: dict, *, enabled: bool, prefill: bool) -> Any:
    if not enabled or parse_retained_prefix is None:
        raise InvalidArgument(
            "retained-prefix reuse requires an enabled compatible TRT engine"
        )
    if params is None or not {"retain_prefix", "retained_prefix"}.issubset(
        field.name for field in fields(params)
    ):
        raise InvalidArgument("TRT engine does not support retained-prefix parameters")
    reference = options.get("retained_prefix")
    try:
        # Native validation copies an immutable capability reference. It grants
        # no address authority; the engine validates identity and trusted peers.
        reference = parse_retained_prefix(reference) if reference is not None else None
    except (TypeError, ValueError) as error:
        raise InvalidArgument(str(error)) from error
    # Decode consumes P's new forward handoff, never the prior completed donor.
    return replace(
        params,
        retain_prefix=options.get("retain_prefix", False),
        retained_prefix=reference if prefill else None,
    )


def export_options(params: Any, *, enabled: bool) -> dict | None:
    if not enabled or params is None:
        return None
    if parse_retained_prefix is None:
        raise InvalidArgument(
            "TRT engine returned unsupported retained-prefix parameters"
        )
    reference = params.retained_prefix
    if reference is None:
        return None
    return {"retained_prefix": asdict(parse_retained_prefix(reference))}
