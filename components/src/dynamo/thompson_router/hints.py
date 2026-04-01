# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Extract Thompson routing hints from native Dynamo PreprocessedRequest fields.

Reads from two Dynamo sources that the frontend populates automatically:
  - request["routing"] (RoutingHints): structured fields from nvext.agent_hints
    * expected_output_tokens <- agent_hints.osl
    * priority_jump          <- agent_hints.latency_sensitivity (seconds)
  - request["annotations"] (list[str]): key:value strings from nvext.annotations
    * "prefix_id:<id>", "total_requests:<n>", "osl:<n>", "iat:<ms>"

Both continuous integer values and categorical string bins (LOW/MEDIUM/HIGH)
are supported for osl and iat.  Categorical bins are mapped to representative
integer values so the downstream feature vector and learner always operate on
continuous scales.  Continuous values from a trie model are preferred when
available.

This allows the standalone Thompson router to receive hints without a custom
processor -- clients just use the standard nvext API.
"""

# Categorical bin → representative integer mappings.
# These are the midpoints of the ranges used by _osl_bin / _iat_factor so that
# a categorical hint round-trips cleanly through binning.
_OSL_CATEGORY_MAP = {
    "low": 64,       # short output (< 128 tokens)
    "medium": 250,   # moderate output (128-512 tokens)
    "high": 768,     # long output (> 512 tokens)
}

# IAT category semantics follow NeMo Agent Toolkit's trie profiler convention:
# the label describes inter-arrival *time* (gap between requests), not rate.
#   LOW  = short gaps  → rapid-fire arrivals → max stickiness
#   HIGH = long gaps   → infrequent arrivals → more exploration
_IAT_CATEGORY_MAP = {
    "low": 50,       # short inter-arrival time → rapid-fire (< 100 ms gaps)
    "medium": 250,   # moderate inter-arrival time (~250 ms gaps)
    "high": 750,     # long inter-arrival time → infrequent (> 500 ms gaps)
}


def _parse_int_or_category(value, category_map: dict, default: int) -> int:
    """Parse a value that may be an int, float, numeric string, or category name.

    Supports: 250, 250.0, "250", "MEDIUM", "medium".
    Returns an integer suitable for the feature vector.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    if s in category_map:
        return category_map[s]
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def extract_hints(request: dict) -> dict:
    """Extract Thompson routing hints from a PreprocessedRequest dict.

    Returns dict with: prefix_id, osl, iat, total_requests, reuse_budget, tokens_in

    osl and iat accept both continuous integer values (preferred, e.g. from a
    trie model) and categorical strings ("LOW"/"MEDIUM"/"HIGH" from NeMo Agent
    Toolkit).  Categories are mapped to representative integers so the router
    always operates on a continuous scale.
    """
    routing = request.get("routing") or {}
    annotations = request.get("annotations", [])

    ann: dict[str, str] = {}
    for a in annotations:
        if ":" in a:
            k, v = a.split(":", 1)
            ann[k] = v

    # OSL: routing.expected_output_tokens > annotations "osl:<value>" > default
    osl_from_routing = routing.get("expected_output_tokens")
    if osl_from_routing is not None:
        osl = _parse_int_or_category(osl_from_routing, _OSL_CATEGORY_MAP, 250)
    else:
        osl = _parse_int_or_category(ann.get("osl"), _OSL_CATEGORY_MAP, 250)

    prefix_id = ann.get("prefix_id", "")

    total_requests = int(ann.get("total_requests", 1))

    # IAT: annotations "iat:<value>" (ms, from trie or client) > default.
    #
    # NOTE: routing.priority_jump is NOT used for IAT. It carries
    # latency_sensitivity (a scheduling priority in seconds), which is
    # semantically different from inter-arrival time. The Dynamo frontend
    # maps agent_hints.latency_sensitivity → routing.priority_jump; using
    # it as IAT would conflate "this request is latency-critical" with
    # "requests arrive rapidly", producing IAT values of 1000-5000ms when
    # the actual IAT is 90-250ms.
    iat = _parse_int_or_category(ann.get("iat"), _IAT_CATEGORY_MAP, 250)

    # Latency sensitivity: separate signal for scheduling priority (1-5 scale).
    # Preserved independently so it can be used as a feature or for routing.
    raw_ls = routing.get("priority_jump")
    if raw_ls is not None:
        try:
            latency_sensitivity = float(raw_ls)
        except (ValueError, TypeError):
            latency_sensitivity = 2.0  # default if non-numeric (e.g. categorical string)
    else:
        try:
            latency_sensitivity = float(ann.get("latency_sensitivity", 2.0))
        except (ValueError, TypeError):
            latency_sensitivity = 2.0

    token_ids = request.get("token_ids", [])

    return {
        "prefix_id": prefix_id,
        "osl": osl,
        "iat": iat,
        "total_requests": total_requests,
        "reuse_budget": max(0, total_requests - 1),
        "tokens_in": len(token_ids),
        "latency_sensitivity": latency_sensitivity,
    }
