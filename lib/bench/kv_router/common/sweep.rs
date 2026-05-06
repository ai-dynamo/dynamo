// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::results::BenchmarkResults;

/// Compute logarithmically spaced benchmark durations for sweep mode.
pub fn compute_sweep_durations(min_ms: u64, max_ms: u64, steps: usize) -> Vec<u64> {
    let log_min = (min_ms as f64).ln();
    let log_max = (max_ms as f64).ln();
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            (log_max * (1.0 - t) + log_min * t).exp().round() as u64
        })
        .collect()
}

/// Print a formatted sweep summary table.
pub fn print_sweep_summary(name: &str, results: &[(u64, BenchmarkResults)]) {
    println!("\n=== Sweep Summary: {} ===", name);
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "duration_ms", "ops/s_off", "ops/s", "blk_ops/s_off", "blk_ops/s", "p99(us)"
    );
    for (dur, r) in results {
        println!(
            "{:>12} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>10.1}",
            dur,
            r.offered_ops_throughput,
            r.ops_throughput,
            r.offered_block_throughput,
            r.block_throughput,
            r.latency_p99_us,
        );
    }
}
