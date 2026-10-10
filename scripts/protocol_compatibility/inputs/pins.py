# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require a native-server source pin for every configured vLLM image version.

Container versions remain authoritative. A pin is source provenance, not an
additional choice of engine version or a compatibility claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from scripts.protocol_compatibility.common.source import commit, git

from ..common.paths import OUTPUT, ROOT


def check_pins(context: dict[str, Any], pins: dict[str, Any]) -> None:
    expected = {}
    for platform, config in context["vllm"].items():
        if not isinstance(config, dict) or "runtime_image_tag" not in config:
            continue
        tag = config["runtime_image_tag"]
        match = re.fullmatch(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)(?:-ubuntu\d+)?", tag)
        if not match:
            raise ValueError(f"unrecognized vLLM image tag for {platform}: {tag}")
        expected[platform] = match[1]
    if not expected:
        raise ValueError("no configured vLLM runtime images")
    actual = {}
    versions = set()
    for pin in pins["versions"]:
        if pin["version"] in versions:
            raise ValueError(f"duplicate vLLM version: {pin['version']}")
        versions.add(pin["version"])
        if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]):
            raise ValueError("vLLM source pins must be immutable commit SHAs")
        if not pin["platforms"]:
            raise ValueError("each vLLM source pin must map to a configured platform")
        for platform in pin["platforms"]:
            if platform in actual:
                raise ValueError(f"duplicate vLLM platform: {platform}")
            actual[platform] = pin["version"]
    if expected != actual:
        raise ValueError(
            f"vLLM protocol pins do not match container/context.yaml: "
            f"expected {expected}, got {actual}. Update pins and regenerate the inventory "
            "as part of the framework-version bump."
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=ROOT / "container/context.yaml")
    parser.add_argument("--pins", type=Path, default=OUTPUT / "vllm_pins.json")
    parser.add_argument("--upstream-repo", type=Path)
    args = parser.parse_args(argv)
    pins = json.loads(args.pins.read_text())
    check_pins(yaml.safe_load(args.context.read_text()), pins)
    if args.upstream_repo:
        for pin in pins["versions"]:
            source = commit(args.upstream_repo, pin["commit"])
            tag = git(
                args.upstream_repo,
                "rev-parse",
                "--verify",
                f"refs/tags/v{pin['version']}^{{commit}}",
            ).strip()
            if source != tag:
                raise ValueError(
                    f"vLLM {pin['version']} source pin does not match release tag"
                )
    print("vLLM protocol source pins match configured runtime versions")
    return 0
