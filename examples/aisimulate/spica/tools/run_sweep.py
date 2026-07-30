# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Spica with the transitional Dynamo replay composition."""

from __future__ import annotations

import argparse

from aisimulate.spica import SmartSearchConfig, run_smart_search
from dynamo.replay.simulation import DynamoReplayRunnerFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Spica sweep with Dynamo Replay")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = SmartSearchConfig.from_yaml(args.config)
    candidates = run_smart_search(
        config,
        runner_factory=DynamoReplayRunnerFactory(),
    )
    for index, candidate in enumerate(candidates):
        print(f"{index}: score={candidate.score} used_gpus={candidate.used_gpus}")


if __name__ == "__main__":
    main()
