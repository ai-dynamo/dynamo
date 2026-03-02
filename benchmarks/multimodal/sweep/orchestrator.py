# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import SweepConfig, input_file_tag, resolve_repo_root
from .runner import run_concurrency_sweep
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

    _print_banner("Multimodal Benchmark Sweep")
    print(f"  Model:         {config.model}")
    print(f"  Input files:   {len(config.input_files)}")
    for f in config.input_files:
        print(f"                   {f}")
    labels = [c.label for c in config.configs]
    print(f"  Configs:       {labels}")
    print(f"  Concurrencies: {config.concurrencies}")
    print(f"  OSL:           {config.osl}")
    print(f"  Requests:      {config.request_count} per concurrency")
    print(f"  Output:        {output_base}")
    print(flush=True)

    server = ServerManager(port=config.port, timeout=config.timeout)
    env_overrides = dict(config.env) if config.env else {}

    try:
        for input_file in config.input_files:
            file_tag = input_file_tag(input_file)
            file_output_dir = output_base / file_tag

            _print_banner(
                f"Input: {Path(input_file).name}  ({file_tag})", char="#"
            )

            for bench_cfg in config.configs:
                _print_banner(
                    f"[{file_tag}] Config: {bench_cfg.label}", char="-"
                )

                workflow_abs = _resolve_workflow(bench_cfg.workflow, repo_root)

                server.start(
                    workflow_script=workflow_abs,
                    model=config.model,
                    extra_args=bench_cfg.extra_args,
                    env_overrides=env_overrides,
                )

                try:
                    sweep_dir = file_output_dir / bench_cfg.label
                    run_concurrency_sweep(
                        model=config.model,
                        port=config.port,
                        concurrencies=config.concurrencies,
                        request_count=config.request_count,
                        warmup_count=config.warmup_count,
                        input_file=input_file,
                        osl=config.osl,
                        output_dir=sweep_dir,
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
