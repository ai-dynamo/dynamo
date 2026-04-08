# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import BenchmarkConfig, SweepConfig, input_file_tag, resolve_repo_root
from .runner import run_aiperf_single
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
    """Execute the full benchmark sweep: for each config x input file x sweep value."""
    if repo_root is None:
        repo_root = resolve_repo_root()

    output_base = Path(config.output_dir)
    sweep_mode = config.sweep_mode
    sweep_values = config.sweep_values

    _print_banner("Multimodal Benchmark Sweep")
    print(f"  Model:         {config.model}")
    print(f"  Input files:   {len(config.input_files)}")
    for f in config.input_files:
        print(f"                   {f}")
    labels = [c.label for c in config.configs]
    print(f"  Configs:       {labels}")
    print(f"  Sweep mode:    {sweep_mode}")
    print(f"  Sweep values:  {sweep_values}")
    print(f"  OSL:           {config.osl}")
    print(f"  Requests:      {config.request_count} per {sweep_mode}")
    print(
        f"  Restart:       {'every run' if config.restart_server_every_benchmark else 'per config'}"
    )
    print(f"  Output:        {output_base}")
    print(flush=True)

    server = ServerManager(port=config.port, timeout=config.timeout)
    env_overrides = dict(config.env) if config.env else {}

    try:
        for bench_cfg in config.configs:
            _run_config(
                bench_cfg=bench_cfg,
                config=config,
                server=server,
                output_base=output_base,
                sweep_mode=sweep_mode,
                sweep_values=sweep_values,
                env_overrides=env_overrides,
                repo_root=repo_root,
            )

        if not config.skip_plots:
            for input_file in config.input_files:
                file_tag = input_file_tag(input_file)
                _generate_plots_for_file(
                    output_base / file_tag,
                    [c.label for c in config.configs],
                )
    finally:
        if server.is_running:
            server.stop()

    _print_summary(config, output_base)


def _run_config(
    bench_cfg: BenchmarkConfig,
    config: SweepConfig,
    server: ServerManager,
    output_base: Path,
    sweep_mode: str,
    sweep_values: List[int],
    env_overrides: dict,
    repo_root: Path,
) -> None:
    """Run all sweep values for a single benchmark config."""
    workflow_abs = _resolve_workflow(bench_cfg.workflow, repo_root)
    _print_banner(f"Config: {bench_cfg.label}", char="#")

    # Collect pending runs, skipping those with existing results.
    pending_runs: List[tuple[str, str, int, Path]] = []
    for input_file in config.input_files:
        file_tag = input_file_tag(input_file)
        sweep_dir = output_base / file_tag / bench_cfg.label

        for value in sorted(sweep_values):
            artifact_dir = sweep_dir / f"{sweep_mode}{value}"

            if (artifact_dir / "profile_export_aiperf.json").exists():
                print(
                    f"  SKIP {bench_cfg.label} {sweep_mode}={value} "
                    f"(results exist in {artifact_dir})",
                    flush=True,
                )
            else:
                pending_runs.append((input_file, file_tag, value, artifact_dir))

    if not pending_runs:
        print(f"  All runs skipped for {bench_cfg.label}", flush=True)
        return

    # For non-restart mode, use the first run's parent dir for server logs
    config_log_dir = pending_runs[0][3].parent if pending_runs else output_base

    if not config.restart_server_every_benchmark:
        server.start(
            workflow_script=workflow_abs,
            model=config.model,
            extra_args=bench_cfg.extra_args,
            env_overrides=env_overrides,
            artifact_dir=config_log_dir,
        )

    try:
        for input_file, file_tag, value, artifact_dir in pending_runs:
            _print_banner(
                f"[{file_tag}] Config: {bench_cfg.label}  " f"{sweep_mode}={value}",
                char="-",
            )

            if config.restart_server_every_benchmark:
                server.start(
                    workflow_script=workflow_abs,
                    model=config.model,
                    extra_args=bench_cfg.extra_args,
                    env_overrides=env_overrides,
                    artifact_dir=artifact_dir,
                )

            try:
                run_aiperf_single(
                    model=config.model,
                    port=config.port,
                    sweep_mode=sweep_mode,
                    sweep_value=value,
                    request_count=config.request_count,
                    warmup_count=config.warmup_count,
                    input_file=input_file,
                    osl=config.osl,
                    artifact_dir=artifact_dir,
                    cleanup_inputs_json=config.cleanup_inputs_json,
                )
            finally:
                if config.restart_server_every_benchmark:
                    server.stop()
    finally:
        if not config.restart_server_every_benchmark:
            server.stop()


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
