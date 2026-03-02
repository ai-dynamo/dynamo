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
    """Top-level sweep configuration loaded from YAML with optional CLI overrides."""

    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    concurrencies: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    osl: int = 150
    request_count: int = 1000
    warmup_count: int = 5
    port: int = 8000
    timeout: int = 600
    input_files: List[str] = field(default_factory=list)
    configs: List[BenchmarkConfig] = field(default_factory=list)
    output_dir: str = "benchmarks/results/multimodal_default"
    skip_plots: bool = False
    env: Dict[str, str] = field(default_factory=dict)

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

    cfg = SweepConfig(
        model=raw.get("model", defaults.model),
        concurrencies=raw.get("concurrencies", defaults.concurrencies),
        osl=raw.get("osl", defaults.osl),
        request_count=raw.get("request_count", defaults.request_count),
        warmup_count=raw.get("warmup_count", defaults.warmup_count),
        port=raw.get("port", defaults.port),
        timeout=raw.get("timeout", defaults.timeout),
        input_files=raw.get("input_files", []),
        configs=configs,
        output_dir=raw.get("output_dir", defaults.output_dir),
        skip_plots=raw.get("skip_plots", False),
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
