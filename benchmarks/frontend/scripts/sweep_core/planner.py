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

    The plan contains a flat list of RunSpecs, one per (deploy, aiperf)
    combination. Ordering (outer -> inner):
      images -> tokenizers -> workers -> concurrencies -> ISLs
      -> num_requests -> rps

    Image is the outermost axis so an entire per-image sub-sweep completes
    before the deploy swaps to the next image; this minimises DGD restarts
    when combined with ``reuse_by_deploy_key`` isolation.

    When ``config.images`` contains more than one entry, run_ids are
    prefixed with an 8-char image hash so sibling runs do not collide.
    """
    runs: list[RunSpec] = []

    # Normalise: an empty images list means "use k8s.image". The config layer
    # guarantees at least [""] but older callers / tests may pass empty.
    images = list(config.images) if config.images else [""]
    multi_image = len([img for img in images if img]) > 1

    for image in images:
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
                                    image=image,
                                )

                                aiperf = AiperfDimension(
                                    concurrency=concurrency,
                                    isl=isl,
                                    osl=config.osl,
                                    num_requests=nr,
                                    benchmark_duration=(
                                        config.benchmark_duration
                                        if nr is None
                                        else None
                                    ),
                                    request_rate=rps,
                                )

                                run_id = build_run_id(
                                    deploy,
                                    aiperf,
                                    include_image_tag=multi_image,
                                )

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
        non_empty_images = [img for img in config.images if img]
        if len(non_empty_images) > 1:
            print(f"  Images:         {len(non_empty_images)} (A/B sweep)")
            for img in non_empty_images:
                print(f"                    {img}")
        elif non_empty_images:
            print(f"  Image:          {non_empty_images[0]}")
        if config.k8s.deploy_template:
            print(f"  Deploy tmpl:    {config.k8s.deploy_template}")
        if config.k8s.aiperf_template:
            print(f"  aiperf tmpl:    {config.k8s.aiperf_template}")
        if config.k8s.aiperf_extra:
            print(f"  aiperf extra:   {config.k8s.aiperf_extra}")
        if config.k8s.artifact_pvc_name != "model-cache":
            print(
                f"  Artifact PVC:   {config.k8s.artifact_pvc_name} "
                f"@ {config.k8s.artifact_pvc_mount_path}"
            )
        print(f"  Reset strategy: {config.k8s.reset_strategy}")
    print()
