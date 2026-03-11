# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BenchmarkConfig:
    """A single benchmark configuration: a workflow script + arguments."""

    label: str
    workflow: str
    extra_args: List[str] = field(default_factory=list)


@dataclass
class SweepConfig:
    """Top-level sweep configuration loaded from YAML with optional CLI overrides.

    Load mode is determined by which field is set:
      - concurrencies: fixed in-flight requests (default)
      - qps_rates: fixed arrival rate (QPS-based, like vLLM bench serve)

    When using qps_rates, request_count is auto-scaled to
    max(200, qps * min_duration) unless explicitly set.
    """

    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    concurrencies: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    qps_rates: Optional[List[float]] = None
    min_duration: int = 60
    isl: int = 400
    osl: int = 150
    request_count: int = 1000
    warmup_count: int = 5
    port: int = 8000
    timeout: int = 600
    image_width: int = 0
    image_height: int = 0
    images_per_request: int = 0
    input_files: List[str] = field(default_factory=list)
    configs: List[BenchmarkConfig] = field(default_factory=list)
    output_dir: str = "benchmarks/results/multimodal_default"
    skip_plots: bool = False
    restart_server_every_benchmark: bool = True
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def is_qps_mode(self) -> bool:
        return self.qps_rates is not None and len(self.qps_rates) > 0

    @property
    def load_levels(self) -> List:
        """Return the list of load levels (concurrencies or QPS rates)."""
        return self.qps_rates if self.is_qps_mode else self.concurrencies

    def request_count_for_qps(self, qps: float) -> int:
        """Auto-scale request count for a QPS level: max(200, qps * min_duration)."""
        return max(200, int(qps * self.min_duration))

    @property
    def is_synthetic_mode(self) -> bool:
        return not self.input_files

    def validate(self, repo_root: Optional[Path] = None) -> None:
        """Validate that all referenced files and scripts exist."""
        if not self.input_files and self.images_per_request <= 0:
            raise ValueError(
                "Either input_files or images_per_request (synthetic mode) is required."
            )
        if not self.configs:
            raise ValueError("At least one benchmark config is required.")

        for f in self.input_files:
            if not Path(f).is_file():
                raise FileNotFoundError(f"Input file not found: {f}")

        for cfg in self.configs:
            script = Path(cfg.workflow)
            if repo_root and not script.is_absolute():
                script = repo_root / script
            if not script.is_file():
                raise FileNotFoundError(
                    f"Workflow script not found: {script} (config '{cfg.label}')"
                )

        if self.is_qps_mode:
            if not self.qps_rates:
                raise ValueError("At least one QPS rate is required.")
        else:
            if not self.concurrencies:
                raise ValueError("At least one concurrency level is required.")


def _parse_benchmark_config(raw: Dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig(
        label=raw["label"],
        workflow=raw["workflow"],
        extra_args=[str(a) for a in raw.get("extra_args", [])],
    )


def load_config(
    yaml_path: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> SweepConfig:
    """Load a SweepConfig from a YAML file, applying optional CLI overrides."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    defaults = SweepConfig()
    configs = [_parse_benchmark_config(c) for c in raw.get("configs", [])]

    # If qps_rates is set, use QPS mode; otherwise use concurrency mode
    qps_rates = raw.get("qps_rates", None)
    if qps_rates is not None:
        qps_rates = [float(q) for q in qps_rates]

    cfg = SweepConfig(
        model=raw.get("model", defaults.model),
        concurrencies=raw.get("concurrencies", defaults.concurrencies),
        qps_rates=qps_rates,
        min_duration=raw.get("min_duration", defaults.min_duration),
        isl=raw.get("isl", defaults.isl),
        osl=raw.get("osl", defaults.osl),
        request_count=raw.get("request_count", defaults.request_count),
        warmup_count=raw.get("warmup_count", defaults.warmup_count),
        port=raw.get("port", defaults.port),
        timeout=raw.get("timeout", defaults.timeout),
        image_width=raw.get("image_width", defaults.image_width),
        image_height=raw.get("image_height", defaults.image_height),
        images_per_request=raw.get("images_per_request", defaults.images_per_request),
        input_files=raw.get("input_files", []),
        configs=configs,
        output_dir=raw.get("output_dir", defaults.output_dir),
        skip_plots=raw.get("skip_plots", False),
        restart_server_every_benchmark=raw.get(
            "restart_server_every_benchmark",
            defaults.restart_server_every_benchmark,
        ),
        env=raw.get("env", {}),
    )

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return cfg


def input_file_tag(path: str) -> str:
    """Derive a short directory-safe tag from a JSONL filename."""
    return Path(path).stem.replace(" ", "_")


def resolve_repo_root() -> Path:
    """Walk up from CWD looking for pyproject.toml to find the repo root."""
    candidate = Path(os.getcwd()).resolve()
    while candidate != candidate.parent:
        if (candidate / "pyproject.toml").is_file():
            return candidate
        candidate = candidate.parent
    return Path(os.getcwd()).resolve()
