# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select configured source pins at an immutable Dynamo revision."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.paths import OUTPUT, ROOT
from ..common.provenance import digest
from ..common.source import git
from .pins import check_pins
from .validation import list_value, object_value, text_value

PINS = str((OUTPUT / "vllm_pins.json").relative_to(ROOT))


def selected_pin(repo: Path, revision: str, platform: str) -> dict:
    context = object_value(
        yaml.safe_load(git(repo, "show", f"{revision}:container/context.yaml")),
        "container context",
    )
    object_value(context.get("vllm"), "container context.vllm")
    pins = object_value(
        json.loads(git(repo, "show", f"{revision}:{PINS}")), "vLLM pins"
    )
    if pins.get("format_version") != 1 or pins.get("target") != "vllm":
        raise ValueError("unsupported vLLM pin schema")
    for pin in list_value(pins.get("versions"), "vLLM pin versions"):
        pin = object_value(pin, "vLLM pin")
        for key in ("version", "commit"):
            text_value(pin.get(key), f"vLLM pin {key}")
        for platform_name in list_value(pin.get("platforms"), "vLLM pin platforms"):
            text_value(platform_name, "vLLM platform")
    check_pins(context, pins)
    matching = [pin for pin in pins["versions"] if platform in pin["platforms"]]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one configured pin for platform {platform}")
    return {
        **matching[0],
        "context_sha256": digest(context),
        "pins_sha256": digest(pins),
    }
