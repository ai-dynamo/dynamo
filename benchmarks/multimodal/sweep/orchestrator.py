# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import SweepConfig, input_file_tag, resolve_repo_root
from .runner import run_aiperf_single, run_concurrency_sweep, run_qps_sweep
from .server import ServerManager


def _resolve_workflow(workflow: str, repo_root: Path) -> str:
    p = Path(workflow)
    if p.is_absolute():
        return str(p)
    return str(repo_root / p)


def _print_banner(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}", flush=True)


def run_sweep(
    config: SweepConfig,
    repo_root: Optional[Path] = None,
) -> None:
    """Execute the full benchmark sweep: for each input file x benchmark config."""
    if repo_root is None:
        repo_root = resolve_repo_root()

    output_base = Path(config.output_dir)

    restart = config.restart_server_every_benchmark

    _print_banner("Multimodal Benchmark Sweep")
    print(f"  Model:         {config.model}")
    print(f"  Input files:   {len(config.input_files)}")
    for f in config.input_files:
        print(f"                   {f}")
    labels = [c.label for c in config.configs]
    print(f"  Configs:       {labels}")
    if config.is_qps_mode:
        print("  Mode:          QPS (fixed arrival rate)")
        print(f"  QPS rates:     {config.qps_rates}")
        print(f"  Min duration:  {config.min_duration}s")
        print(f"  Requests:      auto-scaled per QPS (min {config.request_count})")
    else:
        print("  Mode:          Concurrency (fixed in-flight)")
        print(f"  Concurrencies: {config.concurrencies}")
        print(f"  Requests:      {config.request_count} per level")
    print(f"  OSL:           {config.osl}")
    print(f"  Restart every: {restart}")
    print(f"  Output:        {output_base}")
    print(flush=True)

    server = ServerManager(port=config.port, timeout=config.timeout)
    env_overrides = dict(config.env) if config.env else {}

    # Common aiperf kwargs from config (image params for synthetic mode)
    aiperf_kwargs = {
        "isl": config.isl,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "images_per_request": config.images_per_request,
    }

    # Build iteration list: [(input_file, file_tag, file_output_dir)]
    if config.is_synthetic_mode:
        tag = f"synthetic_{config.isl}isl_{config.images_per_request}img_{config.image_width}x{config.image_height}"
        iterations = [("", tag, output_base / tag)]
    else:
        iterations = [
            (f, input_file_tag(f), output_base / input_file_tag(f))
            for f in config.input_files
        ]

    try:
        for input_file, file_tag, file_output_dir in iterations:
            _print_banner(f"Input: {file_tag}", char="#")

            for bench_cfg in config.configs:
                _print_banner(f"[{file_tag}] Config: {bench_cfg.label}", char="-")

                workflow_abs = _resolve_workflow(bench_cfg.workflow, repo_root)
                sweep_dir = file_output_dir / bench_cfg.label

                if restart:
                    _sweep_with_restart(
                        server=server,
                        workflow_script=workflow_abs,
                        config=config,
                        bench_cfg=bench_cfg,
                        env_overrides=env_overrides,
                        input_file=input_file,
                        output_dir=sweep_dir,
                        aiperf_kwargs=aiperf_kwargs,
                    )
                else:
                    server.start(
                        workflow_script=workflow_abs,
                        model=config.model,
                        extra_args=bench_cfg.extra_args,
                        env_overrides=env_overrides,
                        log_dir=sweep_dir,
                    )
                    try:
                        if config.is_qps_mode:
                            run_qps_sweep(
                                model=config.model,
                                port=config.port,
                                qps_rates=config.qps_rates,
                                request_count=config.request_count,
                                warmup_count=config.warmup_count,
                                input_file=input_file,
                                osl=config.osl,
                                output_dir=sweep_dir,
                                min_duration=config.min_duration,
                                **aiperf_kwargs,
                            )
                        else:
                            run_concurrency_sweep(
                                model=config.model,
                                port=config.port,
                                concurrencies=config.concurrencies,
                                request_count=config.request_count,
                                warmup_count=config.warmup_count,
                                input_file=input_file,
                                osl=config.osl,
                                output_dir=sweep_dir,
                                **aiperf_kwargs,
                            )
                    finally:
                        server.stop()

            if not config.skip_plots:
                _generate_plots_for_file(
                    file_output_dir,
                    [c.label for c in config.configs],
                )
    finally:
        if server.is_running:
            server.stop()

    _print_summary(config, output_base)


def _sweep_with_restart(
    server: ServerManager,
    workflow_script: str,
    config: SweepConfig,
    bench_cfg,
    env_overrides: dict,
    input_file: str,
    output_dir: Path,
    aiperf_kwargs: dict = None,
) -> None:
    """Run each load level with a fresh server to avoid warm-cache effects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = aiperf_kwargs or {}

    if config.is_qps_mode:
        for qps in config.qps_rates:
            dir_name = f"qps{qps:g}"
            artifact_dir = output_dir / dir_name
            server.start(
                workflow_script=workflow_script,
                model=config.model,
                extra_args=bench_cfg.extra_args,
                env_overrides=env_overrides,
                log_dir=artifact_dir,
            )
            try:
                scaled_count = config.request_count_for_qps(qps)
                run_aiperf_single(
                    model=config.model,
                    port=config.port,
                    qps=qps,
                    request_count=scaled_count,
                    warmup_count=config.warmup_count,
                    input_file=input_file,
                    osl=config.osl,
                    artifact_dir=artifact_dir,
                    **extra,
                )
            finally:
                server.stop()
    else:
        for c in config.concurrencies:
            artifact_dir = output_dir / f"c{c}"
            server.start(
                workflow_script=workflow_script,
                model=config.model,
                extra_args=bench_cfg.extra_args,
                env_overrides=env_overrides,
                log_dir=artifact_dir,
            )
            try:
                run_aiperf_single(
                    model=config.model,
                    port=config.port,
                    concurrency=c,
                    request_count=config.request_count,
                    warmup_count=config.warmup_count,
                    input_file=input_file,
                    osl=config.osl,
                    artifact_dir=artifact_dir,
                    **extra,
                )
            finally:
                server.stop()

    print(f"Sweep complete. Results in {output_dir}", flush=True)


def _generate_plots_for_file(
    file_output_dir: Path,
    labels: List[str],
) -> None:
    """Generate comparison plots for one input file across all configs."""
    try:
        from benchmarks.utils.plot import generate_plots

        plots_dir = file_output_dir / "plots"
        print(f"\nGenerating plots -> {plots_dir}", flush=True)
        generate_plots(
            base_output_dir=file_output_dir,
            output_dir=plots_dir,
            benchmark_names=labels,
        )
    except ImportError:
        print(
            "WARNING: benchmarks.utils.plot not importable; skipping plots.",
            flush=True,
        )
    except Exception as exc:
        print(f"WARNING: Plot generation failed: {exc}", flush=True)


def _print_summary(config: SweepConfig, output_base: Path) -> None:
    _print_banner("Sweep Complete!")
    print(f"  Results: {output_base}")
    for input_file in config.input_files:
        tag = input_file_tag(input_file)
        print(f"  [{tag}]:")
        for cfg in config.configs:
            result_dir = output_base / tag / cfg.label
            print(f"    {cfg.label}: {result_dir}")
        if not config.skip_plots:
            plots_dir = output_base / tag / "plots"
            print(f"    plots:  {plots_dir}")
    print(flush=True)
