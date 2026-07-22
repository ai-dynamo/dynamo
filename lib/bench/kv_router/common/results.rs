// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::time::Duration;

use serde::Serialize;

/// Results from a single benchmark run.
#[derive(Clone, Copy, Serialize)]
pub struct BenchmarkResults {
    pub offered_ops_throughput: f32,
    pub ops_throughput: f32,
    pub offered_block_throughput: f32,
    pub block_throughput: f32,
    pub latency_p99_us: f32,
}

#[derive(Clone, Copy)]
pub struct BenchmarkRun {
    pub results: BenchmarkResults,
    pub kept_up: bool,
}

pub fn compute_benchmark_run(
    total_ops: usize,
    total_blocks: usize,
    benchmark_duration_ms: u64,
    total_duration: Duration,
    mut latencies_ns: Vec<u64>,
) -> BenchmarkRun {
    let kept_up = total_duration <= Duration::from_millis(benchmark_duration_ms * 11 / 10);
    let benchmark_duration_secs = (benchmark_duration_ms as f32 / 1000.0).max(1e-6);
    let total_duration_secs = total_duration.as_secs_f32().max(1e-6);
    let offered_ops_throughput = total_ops as f32 / benchmark_duration_secs;
    let ops_throughput = total_ops as f32 / total_duration_secs;
    let offered_block_throughput = total_blocks as f32 / benchmark_duration_secs;
    let block_throughput = total_blocks as f32 / total_duration_secs;

    latencies_ns.sort_unstable();
    let latency_p99_us = if latencies_ns.is_empty() {
        0.0
    } else {
        let p99_idx = latencies_ns.len().saturating_sub(1) * 99 / 100;
        latencies_ns[p99_idx] as f32 / 1000.0
    };

    BenchmarkRun {
        results: BenchmarkResults {
            offered_ops_throughput,
            ops_throughput,
            offered_block_throughput,
            block_throughput,
            latency_p99_us,
        },
        kept_up,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PercentileStats<T> {
    pub min: T,
    pub p50: T,
    pub p90: T,
    pub p99: T,
    pub max: T,
    pub mean: f64,
}

impl<T> PercentileStats<T>
where
    T: Copy + PartialOrd,
{
    pub fn from_values(mut values: Vec<T>, to_f64: impl Fn(T) -> f64) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let percentile = |pct: usize| -> T {
            let idx = (values.len().saturating_sub(1) * pct).div_ceil(100);
            values[idx]
        };
        let mean = values.iter().map(|value| to_f64(*value)).sum::<f64>() / values.len() as f64;

        Some(Self {
            min: values[0],
            p50: percentile(50),
            p90: percentile(90),
            p99: percentile(99),
            max: values[values.len() - 1],
            mean,
        })
    }
}

pub fn print_benchmark_results_percentiles(title: &str, results: &[BenchmarkResults]) {
    if results.len() <= 1 {
        return;
    }

    println!("\n=== {title} ({} runs) ===", results.len());
    println!(
        "{:>18} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "metric", "min", "p50", "p90", "p99", "max", "mean",
    );

    print_percentile_stats_row(
        "ops/s_off",
        PercentileStats::from_values(
            results
                .iter()
                .map(|result| result.offered_ops_throughput)
                .collect(),
            f64::from,
        )
        .expect("non-empty benchmark results"),
    );
    print_percentile_stats_row(
        "ops/s",
        PercentileStats::from_values(
            results.iter().map(|result| result.ops_throughput).collect(),
            f64::from,
        )
        .expect("non-empty benchmark results"),
    );
    print_percentile_stats_row(
        "blk_ops/s_off",
        PercentileStats::from_values(
            results
                .iter()
                .map(|result| result.offered_block_throughput)
                .collect(),
            f64::from,
        )
        .expect("non-empty benchmark results"),
    );
    print_percentile_stats_row(
        "blk_ops/s",
        PercentileStats::from_values(
            results
                .iter()
                .map(|result| result.block_throughput)
                .collect(),
            f64::from,
        )
        .expect("non-empty benchmark results"),
    );
    print_percentile_stats_row(
        "p99(us)",
        PercentileStats::from_values(
            results.iter().map(|result| result.latency_p99_us).collect(),
            f64::from,
        )
        .expect("non-empty benchmark results"),
    );
}

fn print_percentile_stats_row(label: &str, stats: PercentileStats<f32>) {
    println!(
        "{:>18} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>14.1}",
        label, stats.min, stats.p50, stats.p90, stats.p99, stats.max, stats.mean,
    );
}
