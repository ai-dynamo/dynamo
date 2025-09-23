# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List

from benchmarks.utils.genai import run_concurrency_sweep
from benchmarks.utils.plot import generate_plots
from deploy.utils.kubernetes import is_running_in_cluster


def has_http_scheme(url: str) -> bool:
    """Check if URL has HTTP or HTTPS scheme."""
    return url.lower().startswith(("http://", "https://"))


def normalize_service_url(endpoint: str) -> str:
    e = endpoint.strip()
    if has_http_scheme(e):
        return e
    if is_running_in_cluster():
        return f"http://{e}"
    return e  # Outside cluster, validation will have ensured scheme is present


def print_concurrency_start(
    label: str, model: str, isl: int, osl: int, std: int
) -> None:
    """Print concurrency sweep start messages"""
    print(f"⚙️  Starting {label} concurrency sweep!", flush=True)
    print(
        "⏱️  This may take several minutes - running through multiple concurrency levels...",
        flush=True,
    )
    print(f"🎯 Model: {model} | ISL: {isl} | OSL: {osl} | StdDev: {std}")


def run_endpoint_benchmark(
    label: str,
    endpoint: str,
    model: str,
    isl: int,
    osl: int,
    std: int,
    output_dir: Path,
) -> None:
    """Run benchmark for an existing endpoint with custom label"""
    # Normalize endpoint to a usable URL (handles in-cluster scheme-less inputs)
    service_url = normalize_service_url(endpoint)

    print(f"🚀 Starting benchmark of endpoint '{label}': {service_url}")
    print(f"📁 Results will be saved to: {output_dir / label}")
    print_concurrency_start(label, model, isl, osl, std)

    # Create output directory
    (output_dir / label).mkdir(parents=True, exist_ok=True)

    run_concurrency_sweep(
        service_url=service_url,
        model_name=model,
        isl=isl,
        osl=osl,
        stddev=std,
        output_dir=output_dir / label,
    )
    print("✅ Endpoint benchmark completed successfully!")


def print_final_summary(output_dir: Path, labels: List[str]) -> None:
    """Print final benchmark summary"""
    print("📊 Generating performance plots...")
    generate_plots(base_output_dir=output_dir, output_dir=output_dir / "plots")
    print(f"📈 Plots saved to: {output_dir / 'plots'}")
    print(f"📋 Summary saved to: {output_dir / 'plots' / 'SUMMARY.txt'}")

    print()
    print("🎉 Benchmark workflow completed successfully!")
    print(f"📁 All results available at: {output_dir}")

    if labels:
        print(f"🚀 Benchmarked: {', '.join(labels)}")

    print(f"📊 View plots at: {output_dir / 'plots'}")


def run_benchmark_workflow(
    inputs: Dict[str, str],
    isl: int = 2000,
    std: int = 10,
    osl: int = 256,
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "benchmarks/results",
) -> None:
    """Main benchmark workflow orchestrator for HTTP endpoints (and in-cluster internal service URLs)"""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Run endpoint benchmarks
    benchmarked_labels = []
    for label, endpoint in inputs.items():
        run_endpoint_benchmark(label, endpoint, model, isl, osl, std, output_dir_path)
        benchmarked_labels.append(label)

    # Generate final summary
    print_final_summary(output_dir_path, benchmarked_labels)
