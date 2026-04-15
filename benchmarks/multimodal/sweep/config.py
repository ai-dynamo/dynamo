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

    Exactly one of ``request_rates`` or ``concurrencies`` must be set.
    The active mode is exposed via ``sweep_mode`` and ``sweep_values``.
    """

    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    request_rates: Optional[List[int]] = None
    concurrencies: Optional[List[int]] = None
    osl: int = 150
    request_count: int = 1000
    warmup_count: int = 5
    port: int = 8000
    timeout: int = 600
    input_files: List[str] = field(default_factory=list)
    configs: List[BenchmarkConfig] = field(default_factory=list)
    output_dir: str = "benchmarks/results/multimodal_default"
    skip_plots: bool = False
    restart_server_every_benchmark: bool = True
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def sweep_mode(self) -> str:
        """Return ``'request_rate'`` or ``'concurrency'``."""
        if self.concurrencies:
            return "concurrency"
        return "request_rate"

    @property
    def sweep_values(self) -> List[int]:
        """Return the active sweep values (request_rates or concurrencies)."""
        if self.concurrencies:
            return self.concurrencies
        return self.request_rates or []

    def validate(self, repo_root: Optional[Path] = None) -> None:
        """Validate that all referenced files and scripts exist."""
        if not self.input_files:
            raise ValueError("At least one input_file is required.")
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

        if self.request_rates and self.concurrencies:
            raise ValueError(
                "Cannot set both request_rates and concurrencies. Pick one."
            )
        if not self.request_rates and not self.concurrencies:
            raise ValueError(
                "At least one of request_rates or concurrencies is required."
            )


_DEFAULT_REQUEST_RATES: List[int] = [4, 8, 16, 32, 64]


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

    configs = [_parse_benchmark_config(c) for c in raw.get("configs", [])]

    # Resolve sweep mode from YAML — support both keys, default to request_rates.
    yaml_request_rates = raw.get("request_rates")
    yaml_concurrencies = raw.get("concurrencies")

    if yaml_request_rates and yaml_concurrencies:
        raise ValueError(
            f"YAML config {yaml_path} sets both request_rates and concurrencies. "
            "Pick one."
        )

    # Default to request_rates if neither is specified.
    if not yaml_request_rates and not yaml_concurrencies:
        yaml_request_rates = _DEFAULT_REQUEST_RATES

    cfg = SweepConfig(
        model=raw.get("model", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"),
        request_rates=yaml_request_rates,
        concurrencies=yaml_concurrencies,
        osl=raw.get("osl", 150),
        request_count=raw.get("request_count", 1000),
        warmup_count=raw.get("warmup_count", 5),
        port=raw.get("port", 8000),
        timeout=raw.get("timeout", 600),
        input_files=raw.get("input_files", []),
        configs=configs,
        output_dir=raw.get("output_dir", "benchmarks/results/multimodal_default"),
        skip_plots=raw.get("skip_plots", False),
        restart_server_every_benchmark=raw.get("restart_server_every_benchmark", True),
        env=raw.get("env", {}),
    )

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        # CLI sweep mode override clears the other (mutually exclusive) field.
        if (
            "request_rates" in cli_overrides
            and cli_overrides["request_rates"] is not None
        ):
            cfg.concurrencies = None
        elif (
            "concurrencies" in cli_overrides
            and cli_overrides["concurrencies"] is not None
        ):
            cfg.request_rates = None

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
