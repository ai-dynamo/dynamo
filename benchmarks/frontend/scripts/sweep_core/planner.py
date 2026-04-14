# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SweepPlan builder -- constructs a serializable execution plan from SweepConfig.

The planner builds the Cartesian product of deploy dimensions x aiperf dimensions,
producing a flat list of RunSpec objects. The isolation policy determines how
they are executed by the orchestrator.
"""

from __future__ import annotations

from sweep_core.models import (
    AiperfDimension,
    DeployDimension,
    RunSpec,
    SweepConfig,
    SweepPlan,
)
from sweep_core.naming import build_run_id


def build_plan(config: SweepConfig) -> SweepPlan:
    """Build a SweepPlan from a SweepConfig.

    The plan contains a flat list of RunSpecs, one per (deploy, aiperf) combination.
    The ordering is: tokenizers -> workers -> concurrencies -> ISLs -> num_requests -> rps

    This matches the grid construction order from the original sweep_runner.py.
    """
    runs: list[RunSpec] = []

    for tokenizer in config.tokenizers:
        for workers in config.worker_counts:
            for concurrency in config.concurrencies:
                for isl in config.isls:
                    for nr in config.num_requests_list:
                        for rps in config.rps_list:
                            deploy = DeployDimension(
                                backend=config.backend,
                                tokenizer=tokenizer,
                                workers=workers,
                                num_models=config.num_models,
                            )

                            aiperf = AiperfDimension(
                                concurrency=concurrency,
                                isl=isl,
                                osl=config.osl,
                                num_requests=nr,
                                benchmark_duration=config.benchmark_duration
                                if nr is None
                                else None,
                                request_rate=rps,
                            )

                            run_id = build_run_id(deploy, aiperf)

                            runs.append(
                                RunSpec(
                                    deploy=deploy,
                                    aiperf=aiperf,
                                    deploy_key=deploy.deploy_key,
                                    run_id=run_id,
                                )
                            )

    return SweepPlan(
        config=config,
        runs=runs,
        isolation_policy=config.isolation_policy,
        total_runs=len(runs),
    )


def print_plan(plan: SweepPlan) -> None:
    """Print a human-readable summary of the sweep plan."""
    config = plan.config
    print(f"Sweep plan: {plan.total_runs} runs")
    print(f"  Model:          {config.model}")
    print(f"  Mode:           {config.mode}")
    print(f"  Backend:        {config.backend}")
    print(f"  Tokenizers:     {config.tokenizers}")
    print(f"  Concurrencies:  {config.concurrencies}")
    print(f"  ISLs:           {config.isls}")
    print(f"  Workers/model:  {config.worker_counts}")
    print(f"  Models:         {config.num_models}")
    print(f"  Isolation:      {plan.isolation_policy}")
    print(f"  Benchmark dur:  {config.benchmark_duration}s")
    nr_list = [n for n in config.num_requests_list if n is not None]
    if nr_list:
        print(f"  Num requests:   {nr_list}")
    rps_list = [r for r in config.rps_list if r is not None]
    if rps_list:
        print(f"  Request rates:  {rps_list} req/s")
    print(f"  Output:         {config.output_dir}")
    if config.mode == "k8s":
        print(f"  Namespace:      {config.k8s.namespace}")
        print(f"  Endpoint:       {config.k8s.endpoint}")
        if config.k8s.frontend_replicas > 1:
            print(f"  FE replicas:    {config.k8s.frontend_replicas}")
        if config.k8s.dgd_name:
            print(f"  DGD:            {config.k8s.dgd_name}")
        if config.k8s.deploy_template:
            print(f"  Template:       {config.k8s.deploy_template}")
        print(f"  Reset strategy: {config.k8s.reset_strategy}")
    print()
