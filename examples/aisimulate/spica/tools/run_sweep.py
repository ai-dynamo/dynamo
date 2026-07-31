# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Spica with the transitional Dynamo replay composition."""

from __future__ import annotations

import argparse

from pydantic import ValidationError

from aisimulate.spica import run_smart_search
from aisimulate.spica.__main__ import (
    load_config_or_parser_error,
    print_candidates_or_exit,
)
from dynamo.replay.simulation import DynamoReplayRunnerFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Spica sweep with Dynamo Replay")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config_or_parser_error(parser, args.config)
    try:
        candidates = run_smart_search(
            config,
            runner_factory=DynamoReplayRunnerFactory(),
        )
    except ValidationError as exc:
        parser.error(f"invalid adapter search space in {args.config}: {exc}")
    print_candidates_or_exit(config, candidates)


if __name__ == "__main__":
    main()
