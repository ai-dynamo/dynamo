# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TickRecorder / TickHistory — per-tick snapshot store.

Records one TickSnapshot per tick; serialises to CSV and Prometheus textfile.
Optional plot output requires ``matplotlib`` (install dynamo[testbed-plot]).
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TickSnapshot:
    """One tick's worth of recorded data."""

    tick: int
    # Applied state:
    n_p: int
    n_d: int
    cap_p: int
    cap_d: int
    # Observed (post-noise / post-overlay):
    observed_ttft_s: float
    observed_itl_s: float
    observed_power_w_p: float
    observed_power_w_d: float
    observed_capacity_tps: float
    # Controller state:
    c_ttft: float
    c_itl: float
    c_power_p: float
    c_power_d: float
    estimated_throughput: float
    consecutive_violation_ticks: int
    # Aggregate:
    projected_w: float
    budget_w: float
    # Events fired this tick:
    sweep_fired: bool
    sla_violated: bool
    capacity_exceeded: bool
    # Counters delta:
    cap_clamped_min: int
    cap_clamped_max: int
    optimizer_exceptions: int
    correction_pegged: dict[str, int] = dataclasses.field(default_factory=dict)
    admission_partial_failures: int = 0
    # Cumulative count of replica direction flips (n_p sign change OR n_d sign
    # change relative to the previous tick's delta). Useful for asserting
    # "no oscillation" in scale-up/scale-down scenarios.
    n_oscillations: int = 0
    # γ-only columns (None in α):
    mocker_active_p: Optional[int] = None
    mocker_active_d: Optional[int] = None
    mocker_kv_hit_rate: Optional[float] = None

    # Alias properties so assertions can use short names
    @property
    def c_power_prefill(self) -> float:
        return self.c_power_p

    @property
    def c_power_decode(self) -> float:
        return self.c_power_d


# Fields visible to assertion DSL (validated at load time)
TICK_SNAPSHOT_FIELDS = {f.name for f in dataclasses.fields(TickSnapshot)}


class TickHistory:
    """Container for all tick snapshots from a scenario run."""

    def __init__(self) -> None:
        self.snapshots: list[TickSnapshot] = []

    def append(self, snap: TickSnapshot) -> None:
        self.snapshots.append(snap)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> TickSnapshot:
        return self.snapshots[idx]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            f.name
            for f in dataclasses.fields(TickSnapshot)
            if f.name not in ("correction_pegged",)
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields + ["correction_pegged_json"])
            writer.writeheader()
            import json

            for snap in self.snapshots:
                row = {k: getattr(snap, k) for k in fields}
                row["correction_pegged_json"] = json.dumps(snap.correction_pegged)
                writer.writerow(row)

    def to_prom_textfile(self, path: Path, scenario_name: str = "unknown") -> None:
        """Emit a Prometheus textfile-collector-compatible .prom file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = f'scenario="{scenario_name}"'
        lines = []
        if self.snapshots:
            last = self.snapshots[-1]
            lines += [
                f"testbed_c_ttft{{{labels}}} {last.c_ttft}",
                f"testbed_c_itl{{{labels}}} {last.c_itl}",
                f"testbed_c_power_p{{{labels}}} {last.c_power_p}",
                f"testbed_c_power_d{{{labels}}} {last.c_power_d}",
                f"testbed_cap_p_watts{{{labels}}} {last.cap_p}",
                f"testbed_cap_d_watts{{{labels}}} {last.cap_d}",
                f"testbed_projected_w{{{labels}}} {last.projected_w}",
                f"testbed_budget_w{{{labels}}} {last.budget_w}",
                f"testbed_n_p{{{labels}}} {last.n_p}",
                f"testbed_n_d{{{labels}}} {last.n_d}",
            ]
        path.write_text("\n".join(lines) + "\n")

    def plot(self, path: Path, scenario_name: str = "unknown") -> None:
        """Emit a multi-panel PNG plot (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot output. "
                "Install with: pip install dynamo[testbed-plot]"
            )
        ticks = [s.tick for s in self.snapshots]
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f"Testbed scenario: {scenario_name}", fontsize=12)

        axes[0, 0].plot(ticks, [s.c_power_p for s in self.snapshots], label="c_power_p")
        axes[0, 0].plot(ticks, [s.c_power_d for s in self.snapshots], label="c_power_d")
        axes[0, 0].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        axes[0, 0].set_title("Power correction coefficients")
        axes[0, 0].legend()

        axes[0, 1].plot(ticks, [s.cap_p for s in self.snapshots], label="cap_p (W)")
        axes[0, 1].plot(ticks, [s.cap_d for s in self.snapshots], label="cap_d (W)")
        axes[0, 1].set_title("Applied caps (W/GPU)")
        axes[0, 1].legend()

        axes[1, 0].plot(
            ticks, [s.projected_w for s in self.snapshots], label="projected_w"
        )
        axes[1, 0].plot(
            ticks,
            [s.budget_w for s in self.snapshots],
            label="budget_w",
            linestyle="--",
        )
        axes[1, 0].set_title("Power budget utilization")
        axes[1, 0].legend()

        axes[1, 1].plot(
            ticks, [s.observed_ttft_s * 1000 for s in self.snapshots], label="TTFT (ms)"
        )
        axes[1, 1].plot(
            ticks, [s.observed_itl_s * 1000 for s in self.snapshots], label="ITL (ms)"
        )
        axes[1, 1].set_title("Observed latency")
        axes[1, 1].legend()

        axes[2, 0].plot(ticks, [s.n_p for s in self.snapshots], label="n_p")
        axes[2, 0].plot(ticks, [s.n_d for s in self.snapshots], label="n_d")
        axes[2, 0].set_title("Replica counts")
        axes[2, 0].legend()

        axes[2, 1].plot(ticks, [s.observed_capacity_tps for s in self.snapshots])
        axes[2, 1].set_title("Observed capacity (tok/s)")

        plt.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=100)
        plt.close(fig)
