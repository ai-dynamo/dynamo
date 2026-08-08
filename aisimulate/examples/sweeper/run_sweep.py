# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a backend-neutral Sweeper example with deterministic replay metrics."""

from __future__ import annotations

import argparse

from aisimulate.sweeper import (
    ReplayReport,
    RunnerCapabilities,
    SmartSearchConfig,
    Sweeper,
)


class ExampleRunner:
    def run(self, spec):
        deployment = spec.backend_deployment
        engine_args = deployment.agg_engine_args or deployment.decode_engine_args or {}
        max_num_seqs = float(engine_args.get("max_num_seqs", 1))
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": max_num_seqs * deployment.num_workers,
                "gpu_hours": 1.0,
            }
        )

    def close(self):
        pass


class ExampleRunnerFactory:
    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("*", "*"),))

    def create(self, worker_id):
        del worker_id
        return ExampleRunner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = SmartSearchConfig.from_yaml(args.config)
    candidates = Sweeper(
        runner_factory=ExampleRunnerFactory(),
        show_progress=False,
    ).run(config)
    for index, candidate in enumerate(candidates):
        print(index, candidate.score, candidate.config)


if __name__ == "__main__":
    main()
