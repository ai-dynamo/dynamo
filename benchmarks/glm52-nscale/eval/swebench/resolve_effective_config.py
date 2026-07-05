#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve mini-SWE-agent config specs exactly as the pinned batch CLI does."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from minisweagent.agents.default import AgentConfig
from minisweagent.config import get_config_from_spec
from minisweagent.environments.docker import DockerEnvironmentConfig
from minisweagent.models.litellm_model import LitellmModelConfig
from minisweagent.utils.serialize import recursive_merge


def write_or_validate(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(
                "effective mini-SWE config differs from immutable run evidence; "
                "use a new run-name"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise SystemExit(f"effective config appeared concurrently: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    configs = [get_config_from_spec(spec) for spec in args.config]
    configs.append({"model": {"model_name": args.model}})
    resolved = recursive_merge(*configs)
    environment = DockerEnvironmentConfig(
        image="__TASK_IMAGE__", **resolved["environment"]
    ).model_dump(mode="json")
    environment.pop("image")
    effective = {
        "agent": AgentConfig(**resolved["agent"]).model_dump(mode="json"),
        "environment": environment,
        "model": LitellmModelConfig(**resolved["model"]).model_dump(mode="json"),
    }
    write_or_validate(args.output, effective)


if __name__ == "__main__":
    main()
