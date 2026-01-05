# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoadLevelMetrics:
    """Metrics at a single load level."""

    load_level: str  # idle, medium, saturation, overload
    concurrency: int = 0
    concurrency_pct: float = 0.0
    actual_ttft_p50: Optional[float] = None
    actual_ttft_p99: Optional[float] = None
    actual_itl_p50: Optional[float] = None
    actual_itl_p99: Optional[float] = None
    ttft_error_pct: Optional[float] = None
    itl_error_pct: Optional[float] = None
    sla_hit_rate: Optional[float] = None
    meets_sla: bool = False


@dataclass
class ProfilingMetrics:
    """Metrics for a single profiling method."""

    method_name: str
    method_description: str = ""
    source_dir: str = ""

    # Profiling cost
    total_duration_seconds: float = 0.0
    num_deployments_created: int = 0
    gpu_hours_consumed: float = 0.0
    num_prefill_configs_tested: int = 0
    num_decode_configs_tested: int = 0

    # Recommended configuration
    recommended_prefill_gpus: int = 0
    recommended_decode_gpus: int = 0
    recommended_prefill_mapping: str = ""
    recommended_decode_mapping: str = ""
    predicted_ttft: float = 0.0
    predicted_itl: float = 0.0
    predicted_prefill_thpt_per_gpu: float = 0.0
    predicted_decode_thpt_per_gpu: float = 0.0

    # Predictive accuracy (at multiple load levels)
    validated: bool = False
    load_level_metrics: list[LoadLevelMetrics] = field(default_factory=list)
    ttft_error_at_idle: Optional[float] = None
    ttft_error_at_medium: Optional[float] = None
    ttft_error_at_saturation: Optional[float] = None
    ttft_error_at_overload: Optional[float] = None

    # Optimization accuracy
    optimization_validated: bool = False
    actual_goodput: Optional[float] = None
    actual_sla_hit_rate: Optional[float] = None
    optimization_regret: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["load_level_metrics"] = [asdict(m) for m in self.load_level_metrics]
        return data

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, input_path: Path) -> "ProfilingMetrics":
        with open(input_path, "r") as f:
            data = json.load(f)
        load_metrics = data.pop("load_level_metrics", [])
        metrics = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        for lm in load_metrics:
            metrics.load_level_metrics.append(LoadLevelMetrics(**lm))
        return metrics


@dataclass
class ComparisonResult:
    """Comparison results across multiple profiling methods."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    backend: str = ""
    target_ttft: float = 0.0
    target_itl: float = 0.0
    isl: int = 0
    osl: int = 0

    method_metrics: list[ProfilingMetrics] = field(default_factory=list)
    ground_truth_method: str = ""
    ground_truth_goodput: float = 0.0

    # Summary
    fastest_method: str = ""
    lowest_cost_method: str = ""
    most_accurate_at_medium: str = ""
    best_optimization_method: str = ""

    def add_method_result(self, metrics: ProfilingMetrics):
        self.method_metrics.append(metrics)

    def set_ground_truth(self, method_name: str, goodput: float):
        self.ground_truth_method = method_name
        self.ground_truth_goodput = goodput
        for m in self.method_metrics:
            if m.actual_goodput is not None and goodput > 0:
                m.optimization_regret = (goodput - m.actual_goodput) / goodput * 100

    def compute_comparison(self):
        if not self.method_metrics:
            return

        # Set ground truth from online_aiperf (it tested all configs)
        online = next((m for m in self.method_metrics if m.method_name == "online_aiperf"), None)
        if online and online.actual_goodput and online.actual_goodput > 0:
            self.set_ground_truth("online_aiperf", online.actual_goodput)

        durations = [(m.method_name, m.total_duration_seconds)
                     for m in self.method_metrics if m.total_duration_seconds > 0]
        if durations:
            self.fastest_method = min(durations, key=lambda x: x[1])[0]

        costs = [(m.method_name, m.gpu_hours_consumed) for m in self.method_metrics]
        if costs:
            self.lowest_cost_method = min(costs, key=lambda x: x[1])[0]

        errors = [(m.method_name, abs(m.ttft_error_at_medium))
                  for m in self.method_metrics if m.ttft_error_at_medium is not None]
        if errors:
            self.most_accurate_at_medium = min(errors, key=lambda x: x[1])[0]

        regrets = [(m.method_name, m.optimization_regret)
                   for m in self.method_metrics if m.optimization_regret is not None]
        if regrets:
            self.best_optimization_method = min(regrets, key=lambda x: x[1])[0]

    def generate_summary_table(self) -> str:
        lines = [
            "=" * 120,
            "PROFILING METHOD COMPARISON",
            "=" * 120,
            f"Model: {self.model}",
            f"SLA: TTFT={self.target_ttft}ms, ITL={self.target_itl}ms, ISL={self.isl}, OSL={self.osl}",
            "",
            "--- PROFILING COST ---",
            "-" * 80,
            f"{'Method':<20} {'Duration':<12} {'Deploys':<10} {'GPU-Hrs':<10} {'Pred TTFT':<12}",
            "-" * 80,
        ]

        for m in self.method_metrics:
            dur = f"{m.total_duration_seconds/60:.1f}min" if m.total_duration_seconds else "N/A"
            ttft = f"{m.predicted_ttft:.1f}ms" if m.predicted_ttft else "N/A"
            lines.append(f"{m.method_name:<20} {dur:<12} {m.num_deployments_created:<10} "
                        f"{m.gpu_hours_consumed:.2f}{'':>6} {ttft:<12}")

        # Predictive accuracy section
        has_validation = any(m.validated for m in self.method_metrics)
        if has_validation:
            lines.extend(["", "--- PREDICTIVE ACCURACY (TTFT error vs predicted) ---", "-" * 100,
                          f"{'Method':<20} {'Idle':<12} {'Medium':<12} {'Saturation':<12} {'Overload':<12}", "-" * 100])
            for m in self.method_metrics:
                idle = f"{m.ttft_error_at_idle:+.1f}%" if m.ttft_error_at_idle is not None else "N/A"
                med = f"{m.ttft_error_at_medium:+.1f}%" if m.ttft_error_at_medium is not None else "N/A"
                sat = f"{m.ttft_error_at_saturation:+.1f}%" if m.ttft_error_at_saturation is not None else "N/A"
                ovl = f"{m.ttft_error_at_overload:+.1f}%" if m.ttft_error_at_overload is not None else "N/A"
                lines.append(f"{m.method_name:<20} {idle:<12} {med:<12} {sat:<12} {ovl:<12}")

        # Optimization accuracy section
        has_optimization = any(m.optimization_validated for m in self.method_metrics)
        if has_optimization:
            lines.extend(["", "--- OPTIMIZATION ACCURACY ---", "-" * 80,
                          f"{'Method':<20} {'Goodput':<15} {'SLA Hit %':<12} {'Regret':<12}", "-" * 80])
            for m in self.method_metrics:
                gp = f"{m.actual_goodput:.1f} tok/s" if m.actual_goodput else "N/A"
                sla = f"{m.actual_sla_hit_rate:.1f}%" if m.actual_sla_hit_rate is not None else "N/A"
                reg = f"{m.optimization_regret:.1f}%" if m.optimization_regret is not None else "N/A"
                lines.append(f"{m.method_name:<20} {gp:<15} {sla:<12} {reg:<12}")

        # Summary
        lines.extend(["-" * 80, "SUMMARY:"])
        lines.append(f"  Fastest profiling: {self.fastest_method}")
        lines.append(f"  Lowest cost: {self.lowest_cost_method}")
        if self.most_accurate_at_medium:
            lines.append(f"  Most accurate (medium load): {self.most_accurate_at_medium}")
        if self.best_optimization_method:
            lines.append(f"  Best optimization: {self.best_optimization_method}")
        lines.append("=" * 120)
        return "\n".join(lines)

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": self.timestamp, "model": self.model, "backend": self.backend,
                "target_ttft": self.target_ttft, "target_itl": self.target_itl,
                "isl": self.isl, "osl": self.osl,
                "ground_truth_method": self.ground_truth_method,
                "ground_truth_goodput": self.ground_truth_goodput,
                "fastest_method": self.fastest_method,
                "lowest_cost_method": self.lowest_cost_method,
                "most_accurate_at_medium": self.most_accurate_at_medium,
                "best_optimization_method": self.best_optimization_method,
                "methods": [m.to_dict() for m in self.method_metrics],
            }, f, indent=2, default=str)

    @classmethod
    def load(cls, input_path: Path) -> "ComparisonResult":
        with open(input_path, "r") as f:
            data = json.load(f)
        methods_data = data.pop("methods", [])
        result = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        for md in methods_data:
            load_metrics = md.pop("load_level_metrics", [])
            m = ProfilingMetrics(**{k: v for k, v in md.items() if k in ProfilingMetrics.__dataclass_fields__})
            for lm in load_metrics:
                m.load_level_metrics.append(LoadLevelMetrics(**lm))
            result.method_metrics.append(m)
        return result


def load_profiling_results(results_dir: Path) -> dict:
    """Load profiling results from profile_sla.py output directory."""
    results = {}
    prefill_path = results_dir / "selected_prefill_interpolation" / "raw_data.npz"
    if prefill_path.exists():
        try:
            with np.load(prefill_path) as data:
                results["prefill"] = {
                    "isl": data["prefill_isl"].tolist(),
                    "ttft": data["prefill_ttft"].tolist(),
                    "thpt_per_gpu": data["prefill_thpt_per_gpu"].tolist(),
                }
        except Exception as e:
            logger.warning(f"Failed to load prefill data: {e}")

    decode_path = results_dir / "selected_decode_interpolation" / "raw_data.npz"
    if decode_path.exists():
        try:
            with np.load(decode_path) as data:
                results["decode"] = {
                    "kv_usage": data["x_kv_usage"].tolist(),
                    "context_length": data["y_context_length"].tolist(),
                    "itl": data["z_itl"].tolist(),
                    "thpt_per_gpu": data["z_thpt_per_gpu"].tolist(),
                }
        except Exception as e:
            logger.warning(f"Failed to load decode data: {e}")
    return results
