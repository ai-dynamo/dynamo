# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt


def create_dryrun_plot(
    time: list,
    rr: list,
    est_rr: list,
    isl: list,
    est_isl: list,
    osl: list,
    est_osl: list,
    num_p: list,
    p_thpt: list,
    safe_p_thpt: list,
    num_d: list,
    d_thpt: list,
    safe_d_thpt: list,
    output_path: str,
    warmup_time: list | None = None,
    warmup_rr: list | None = None,
    warmup_isl: list | None = None,
    warmup_osl: list | None = None,
) -> None:
    """
    Create a comprehensive dryrun plot with 4 subplots showing various metrics over time.

    Args:
        time: List of time points
        rr: List of actual request rates
        est_rr: List of estimated request rates
        isl: List of actual input sequence lengths
        est_isl: List of estimated input sequence lengths
        osl: List of actual output sequence lengths
        est_osl: List of estimated output sequence lengths
        num_p: List of prefill worker counts
        p_thpt: List of actual prefill throughputs
        safe_p_thpt: List of safe prefill throughput limits
        num_d: List of decode worker counts
        d_thpt: List of actual decode throughputs
        safe_d_thpt: List of safe decode throughput limits
        output_path: Path where the plot should be saved
        warmup_time: Optional list of warmup time points (negative seconds)
        warmup_rr: Optional list of warmup request rates (same units as rr)
        warmup_isl: Optional list of warmup input sequence lengths
        warmup_osl: Optional list of warmup output sequence lengths
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Request Rate
    if warmup_time is not None and warmup_rr is not None and len(warmup_time) > 0:
        ax1.plot(
            warmup_time,
            warmup_rr,
            "b-",
            alpha=0.35,
            linewidth=2,
            label="Warmup Request Rate",
        )
    ax1.plot(time, rr, "b-", label="Actual Request Rate", linewidth=2)
    ax1.plot(time, est_rr, "r--", label="Predicted Request Rate", linewidth=2)
    if warmup_time is not None and warmup_rr is not None:
        ax1.axvline(0, color="k", linestyle=":", linewidth=2, label="Warmup Boundary")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Request Rate")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Request Rate Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sequence Lengths
    if (
        warmup_time is not None
        and warmup_isl is not None
        and warmup_osl is not None
        and len(warmup_time) > 0
    ):
        ax2.plot(
            warmup_time,
            warmup_isl,
            "g-",
            alpha=0.35,
            linewidth=2,
            label="Warmup ISL",
        )
        ax2.plot(
            warmup_time,
            warmup_osl,
            "m-",
            alpha=0.35,
            linewidth=2,
            label="Warmup OSL",
        )
    ax2.plot(time, isl, "g-", label="Actual ISL", linewidth=2)
    ax2.plot(time, est_isl, "g--", label="Predicted ISL", linewidth=2)
    ax2.plot(time, osl, "m-", label="Actual OSL", linewidth=2)
    ax2.plot(time, est_osl, "m--", label="Predicted OSL", linewidth=2)
    if warmup_time is not None and warmup_isl is not None and warmup_osl is not None:
        ax2.axvline(0, color="k", linestyle=":", linewidth=2, label="Warmup Boundary")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Num Tokens")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Input/Output Sequence Lengths Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Worker Counts
    ax3.plot(time, p_thpt, "b-", label="Actual Prefill Throughput", linewidth=2)
    ax3.plot(
        time, safe_p_thpt, "b--", label="Safe Prefill Throughput Limit", linewidth=2
    )
    ax3_right = ax3.twinx()
    ax3_right.plot(time, num_p, "c-", label="Prefill Workers", linewidth=2, marker="o")
    ax3_right.set_ylabel("Number of Workers")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Throughput (tok/adjustment_interval)")
    ax3.set_ylim(bottom=0)
    ax3_right.set_ylabel("Number of Workers")
    ax3_right.set_ylim(bottom=0)
    ax3.set_title("Prefill Load and Workers")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Throughput Comparison
    ax4.plot(time, d_thpt, "r-", label="Actual Decode Throughput", linewidth=2)
    ax4.plot(
        time, safe_d_thpt, "r--", label="Safe Decode Throughput Limit", linewidth=2
    )
    ax4_right = ax4.twinx()
    ax4_right.plot(
        time, num_d, "orange", label="Decode Workers", linewidth=2, marker="o"
    )
    ax4_right.set_ylabel("Number of Workers")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_right.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Throughput (tok/adjustment_interval)")
    ax4.set_ylim(bottom=0)
    ax4_right.set_ylabel("Number of Workers")
    ax4_right.set_ylim(bottom=0)
    ax4.set_title("Decode Load and Workers")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
