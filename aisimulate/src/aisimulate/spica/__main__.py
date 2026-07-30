# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental ``python -m aisimulate.spica`` entry point."""

from __future__ import annotations

import argparse

import yaml
from pydantic import ValidationError

from .config import SmartSearchConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate.spica",
        description=(
            "[EXPERIMENTAL] Spica is replay-runtime neutral. Use its Python API "
            "with an injected RunnerFactory to execute a sweep."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a SmartSearchConfig YAML file"
    )
    args = parser.parse_args()

    try:
        config = SmartSearchConfig.from_yaml(args.config)
    except OSError as exc:  # missing file, a directory, unreadable, etc.
        parser.error(f"could not read config {args.config}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {args.config}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {args.config}: {exc}")

    del config
    parser.error(
        "the standalone CLI has no default replay runtime; call "
        "aisimulate.spica.run_smart_search(config, runner_factory=...) from Python"
    )


if __name__ == "__main__":
    main()
