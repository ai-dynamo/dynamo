# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Chart generation functions for benchmark comparison plots.

Each public function creates one figure and saves it as a PNG.
All comparison charts overlay the 3 scenarios on a shared log2 x-axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Scenario display metadata
# ---------------------------------------------------------------------------
SCENARIO_META: dict[str, dict[str, str]] = {
    "scenario_1_vllm_serve": {
        "label": "vLLM Serve",
        "color": "#1f77b4",
        "marker": "o",
    },
    "scenario_2_dynamo_rust": {
        "label": "Dynamo Rust Frontend",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "scenario_3_dynamo_vllm_processor": {
        "label": "Dynamo vLLM Processor",
        "color": "#2ca02c",
        "marker": "^",
    },
}

DPI = 300


def _setup_axes(
    ax: plt.Axes,
    concurrencies: list[int],
    ylabel: str,
    title: str,
) -> None:
    """Apply common formatting to an axes object."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies], fontsize=8)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)


def _subtitle(config: dict[str, Any]) -> str:
    """Generate a subtitle string from run config."""
    model = config.get("model", "")
    isl = config.get("isl", "")
    osl = config.get("osl", "")
    return f"{model}  |  ISL={isl}, OSL={osl}"


# ---------------------------------------------------------------------------
# Graph 1: TTFT (avg) vs Concurrency
# ---------------------------------------------------------------------------
def plot_ttft(
    data: dict[str, dict[int, dict[str, Any]]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Time to First Token (avg) vs Concurrency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    for scenario_key, meta in SCENARIO_META.items():
        if scenario_key not in data:
            continue
        scenario_data = data[scenario_key]
        xs, ys = [], []
        for c in concurrencies:
            if c in scenario_data and "time_to_first_token" in scenario_data[c]:
                val = scenario_data[c]["time_to_first_token"].get("avg")
                if val is not None:
                    xs.append(c)
                    ys.append(val)
        if xs:
            ax.plot(
                xs,
                ys,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(ax, concurrencies, "TTFT (ms)", "Time to First Token vs Concurrency")
    fig.tight_layout()
    out = output_dir / "ttft_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 2: ITL (avg) vs Concurrency
# ---------------------------------------------------------------------------
def plot_itl(
    data: dict[str, dict[int, dict[str, Any]]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Inter-Token Latency (avg) vs Concurrency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    for scenario_key, meta in SCENARIO_META.items():
        if scenario_key not in data:
            continue
        scenario_data = data[scenario_key]
        xs, ys = [], []
        for c in concurrencies:
            if c in scenario_data and "inter_token_latency" in scenario_data[c]:
                val = scenario_data[c]["inter_token_latency"].get("avg")
                if val is not None:
                    xs.append(c)
                    ys.append(val)
        if xs:
            ax.plot(
                xs,
                ys,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(ax, concurrencies, "ITL (ms)", "Inter-Token Latency vs Concurrency")
    fig.tight_layout()
    out = output_dir / "itl_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 3: Output Token Throughput (total, tokens/sec) vs Concurrency
# ---------------------------------------------------------------------------
def plot_throughput(
    data: dict[str, dict[int, dict[str, Any]]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Output Token Throughput (tokens/sec) vs Concurrency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    for scenario_key, meta in SCENARIO_META.items():
        if scenario_key not in data:
            continue
        scenario_data = data[scenario_key]
        xs, ys = [], []
        for c in concurrencies:
            if c in scenario_data and "output_token_throughput" in scenario_data[c]:
                val = scenario_data[c]["output_token_throughput"].get("avg")
                if val is not None:
                    xs.append(c)
                    ys.append(val)
        if xs:
            ax.plot(
                xs,
                ys,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(
        ax,
        concurrencies,
        "Throughput (tokens/sec)",
        "Output Token Throughput vs Concurrency",
    )
    fig.tight_layout()
    out = output_dir / "throughput_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 4: Request Latency (avg) vs Concurrency
# ---------------------------------------------------------------------------
def plot_request_latency(
    data: dict[str, dict[int, dict[str, Any]]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Request Latency (avg, ms) vs Concurrency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    for scenario_key, meta in SCENARIO_META.items():
        if scenario_key not in data:
            continue
        scenario_data = data[scenario_key]
        xs, ys = [], []
        for c in concurrencies:
            if c in scenario_data and "request_latency" in scenario_data[c]:
                val = scenario_data[c]["request_latency"].get("avg")
                if val is not None:
                    xs.append(c)
                    ys.append(val)
        if xs:
            ax.plot(
                xs,
                ys,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(ax, concurrencies, "Latency (ms)", "Request Latency vs Concurrency")
    fig.tight_layout()
    out = output_dir / "request_latency_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 5: Request Throughput (req/sec) vs Concurrency
# ---------------------------------------------------------------------------
def plot_request_throughput(
    data: dict[str, dict[int, dict[str, Any]]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Request Throughput (requests/sec) vs Concurrency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    for scenario_key, meta in SCENARIO_META.items():
        if scenario_key not in data:
            continue
        scenario_data = data[scenario_key]
        xs, ys = [], []
        for c in concurrencies:
            if c in scenario_data and "request_throughput" in scenario_data[c]:
                val = scenario_data[c]["request_throughput"].get("avg")
                if val is not None:
                    xs.append(c)
                    ys.append(val)
        if xs:
            ax.plot(
                xs,
                ys,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(
        ax, concurrencies, "Throughput (req/sec)", "Request Throughput vs Concurrency"
    )
    fig.tight_layout()
    out = output_dir / "request_throughput_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 6: Tokenizer Latency vs Concurrency (Scenario 2 only)
# ---------------------------------------------------------------------------
def plot_tokenizer_latency(
    tokenizer_data: dict[str, dict[int, float]],
    concurrencies: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Avg tokenizer latency (ms) vs Concurrency for Dynamo Rust Frontend.

    Parameters
    ----------
    tokenizer_data : dict
        Keys "tokenize" and "detokenize", each mapping concurrency -> avg ms.
    """
    if not tokenizer_data.get("tokenize") and not tokenizer_data.get("detokenize"):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(_subtitle(config), fontsize=9, color="gray")

    colors = {"tokenize": "#d62728", "detokenize": "#9467bd"}
    markers = {"tokenize": "o", "detokenize": "s"}

    for op in ("tokenize", "detokenize"):
        op_data = tokenizer_data.get(op, {})
        xs = sorted(c for c in op_data if c in concurrencies)
        ys = [op_data[c] for c in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                label=f"{op}",
                color=colors[op],
                marker=markers[op],
                markersize=5,
                linewidth=1.5,
            )

    _setup_axes(
        ax,
        concurrencies,
        "Latency (ms)",
        "Tokenizer Latency vs Concurrency (Dynamo Rust Frontend)",
    )
    fig.tight_layout()
    out = output_dir / "tokenizer_latency_vs_concurrency.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
