#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and persist immutable endpoint model-discovery evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CONTEXT_WINDOW_ALIASES = ("context_window", "max_model_len")


def canonicalize_model_context(
    model_entry: dict, model: str, expected_context_window: int
) -> dict:
    aliases = {
        alias: model_entry[alias]
        for alias in CONTEXT_WINDOW_ALIASES
        if alias in model_entry and model_entry[alias] is not None
    }
    if not aliases:
        raise ValueError(
            f"served model {model!r} has no non-null context field; expected one of "
            f"{CONTEXT_WINDOW_ALIASES!r}"
        )
    values = list(aliases.values())
    first_value = values[0]
    if any(
        type(value) is not type(first_value) or value != first_value
        for value in values[1:]
    ):
        raise ValueError(
            f"served model {model!r} has conflicting context aliases: {aliases!r}"
        )
    actual_context = first_value
    if type(actual_context) is not int or actual_context != expected_context_window:
        raise ValueError(
            f"served model {model!r} context is {actual_context!r}, "
            f"expected {expected_context_window}"
        )
    canonical = dict(model_entry)
    for alias in CONTEXT_WINDOW_ALIASES:
        canonical.pop(alias, None)
    canonical["context_window"] = expected_context_window
    return canonical


def build_evidence(response: object, model: str, context_window: int) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise ValueError("/models response must contain a data list")
    matches = [
        item
        for item in response["data"]
        if isinstance(item, dict) and item.get("id") == model
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one /models entry for {model!r}, got {len(matches)}"
        )
    selected_model = canonicalize_model_context(matches[0], model, context_window)
    return {
        "schema_version": 1,
        "requested_model": model,
        "expected_context_window": context_window,
        "selected_model_response": selected_model,
        "full_response": response,
    }


def write_or_validate(path: Path, evidence: dict) -> None:
    payload = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(
                "endpoint /models evidence differs from the immutable run evidence; "
                "use a new run-name"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise SystemExit(f"endpoint evidence appeared concurrently: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-window", required=True, type=int)
    args = parser.parse_args()
    response = json.loads(args.input.read_text())
    evidence = build_evidence(response, args.model, args.context_window)
    write_or_validate(args.output, evidence)


if __name__ == "__main__":
    main()
