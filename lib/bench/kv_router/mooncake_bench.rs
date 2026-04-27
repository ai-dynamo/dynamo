// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/mod.rs"]
mod common;
use common::*;

#[path = "mooncake_shared.rs"]
mod mooncake_shared;

use clap::{Parser, Subcommand};
use dynamo_kv_router::indexer::{KvIndexerInterface, KvIndexerMetrics, ShardSizeSnapshot};
use mooncake_shared::{MooncakeBenchmarkConfig, MooncakeIndexerConfig, run_benchmark};
use serde::Serialize;
use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Indexer backend selection and its backend-specific parameters.
#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    /// Single-threaded radix tree indexer.
    RadixTree {},

    /// Position-based nested map indexer with jump search.
    NestedMap {
        /// Number of positions to skip during jump search before scanning back.
        #[clap(long, default_value = "8")]
        jump_size: usize,

        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Lock-based concurrent radix tree indexer.
    ConcurrentRadixTree {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Compressed concurrent radix tree indexer (compressed edges).
    ConcurrentRadixTreeCompressed {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Branch-sharded CRTC: N independent CRTC shards assigned via an explicit routing
    /// table keyed on the first K block hashes. New branches are assigned to the
    /// least-loaded shard. find_matches touches exactly ONE shard (no scatter-gather).
    /// Unknown branch keys return empty scores immediately without any dispatch.
    BranchShardedCrtc {
        /// Number of independent CRTC shards.
        #[clap(long, default_value = "2")]
        num_shards: usize,

        /// Number of OS event-worker threads per shard.
        #[clap(long, default_value = "4")]
        num_event_workers_per_shard: usize,

        /// Number of prefix blocks hashed to identify a branch. K=2 is the
        /// recommended default: depth=1 often produces too few distinct branch
        /// keys, while depth=2 gives a much larger set of distinguishable branches.
        #[clap(long, default_value = "2")]
        prefix_depth: usize,

        /// Number of OS threads per shard dedicated to find_matches (read isolation).
        /// 0 (default): reads run inline on the calling tokio thread.
        #[clap(long, default_value = "0")]
        num_read_threads_per_shard: usize,
    },
}

impl IndexerArgs {
    fn to_config(&self) -> MooncakeIndexerConfig {
        match self {
            IndexerArgs::RadixTree {} => MooncakeIndexerConfig::radix_tree(),
            IndexerArgs::NestedMap {
                jump_size,
                num_event_workers,
            } => MooncakeIndexerConfig::nested_map(*jump_size, *num_event_workers),
            IndexerArgs::ConcurrentRadixTree { num_event_workers } => {
                MooncakeIndexerConfig::concurrent_radix_tree(*num_event_workers)
            }
            IndexerArgs::ConcurrentRadixTreeCompressed { num_event_workers } => {
                MooncakeIndexerConfig::concurrent_radix_tree_compressed(*num_event_workers)
            }
            IndexerArgs::BranchShardedCrtc {
                num_shards,
                num_event_workers_per_shard,
                prefix_depth,
                num_read_threads_per_shard: _,
            } => MooncakeIndexerConfig::branch_sharded_crtc(
                *num_shards,
                *num_event_workers_per_shard,
                *prefix_depth,
            ),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Output path for the sweep plot SVG.
    #[clap(long, default_value = "sweep_plot.svg")]
    sweep_output: String,

    /// Comma-separated list of indexer names to benchmark and compare on the
    /// same plot. Overrides the subcommand indexer when present. Valid names:
    /// radix-tree, nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed.
    #[clap(long, value_delimiter = ',')]
    compare: Vec<String>,

    /// Number of OS threads for event processing in compare mode. Applies to
    /// indexers that use a thread pool (nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed).
    /// Ignored by radix-tree.
    #[clap(long, default_value = "16")]
    num_event_workers: usize,

    /// Number of additional concurrent tokio tasks that issue find_matches in a
    /// tight loop to stress the read path.  These tasks run alongside the normal
    /// trace-replay workers.  Set to 0 (default) to disable.
    #[clap(long, default_value = "0")]
    find_matches_concurrency: usize,

    /// Output path for the shard-size CSV produced when `shard-metrics` feature
    /// is enabled.  Rows: `elapsed_ms,shard_idx,worker_count,block_count,node_count`.
    /// An SVG plot is written alongside it (<path>.svg).
    /// Omit or leave empty to disable shard-size sampling.
    #[clap(long, default_value = "")]
    shard_metrics_csv: String,

    /// How often (ms) to sample shard sizes when `--shard-metrics-csv` is set.
    #[clap(long, default_value = "200")]
    shard_metrics_interval_ms: u64,

    /// Indexer backend to benchmark (defaults to radix-tree if not specified).
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,
}

impl Args {
    /// Return the indexer config, falling back to RadixTree if none was specified.
    fn get_indexer(&self) -> IndexerArgs {
        self.indexer.clone().unwrap_or(IndexerArgs::RadixTree {})
    }
}

#[derive(Serialize)]
struct SweepStepResult {
    duration_ms: u64,
    #[serde(flatten)]
    results: BenchmarkResults,
}

// ---------------------------------------------------------------------------
// Shard-size sampling (always compiled; only called when a CSV path is given)
// ---------------------------------------------------------------------------

/// A single row in the shard-size time-series CSV.
#[derive(Clone)]
struct ShardSampleRow {
    elapsed_ms: u64,
    snapshot: ShardSizeSnapshot,
}

/// Spawn a background tokio task that samples `indexer.shard_sizes()` every
/// `interval_ms` milliseconds until `cancel` is triggered.
///
/// Returns a `JoinHandle` that resolves to all collected samples.
fn start_shard_sampler(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    interval_ms: u64,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<Vec<ShardSampleRow>> {
    tokio::spawn(async move {
        let mut rows = Vec::new();
        let start = Instant::now();
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let elapsed_ms = start.elapsed().as_millis() as u64;
                    for snap in indexer.shard_sizes() {
                        rows.push(ShardSampleRow { elapsed_ms, snapshot: snap });
                    }
                }
                _ = cancel.cancelled() => break,
            }
        }
        rows
    })
}

/// Write the collected shard-size samples to a CSV file.
fn write_shard_metrics_csv(rows: &[ShardSampleRow], path: &str) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "elapsed_ms,shard_idx,worker_count,block_count,node_count"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{}",
            r.elapsed_ms,
            r.snapshot.shard_idx,
            r.snapshot.worker_count,
            r.snapshot.block_count,
            r.snapshot.node_count,
        )?;
    }
    println!("Shard metrics CSV written to {path}");
    Ok(())
}

/// Plot per-shard `worker_count` and `block_count` over time and write an SVG.
///
/// Draws two panels stacked vertically:
/// - Top: workers per shard over time
/// - Bottom: blocks per shard over time
///
/// Each shard gets a distinct colour; shards are identified by their `shard_idx`.
fn plot_shard_metrics(rows: &[ShardSampleRow], svg_path: &str) -> anyhow::Result<()> {
    use plotters::prelude::*;

    if rows.is_empty() {
        return Ok(());
    }

    // Collect the set of shard indices present.
    let mut shard_indices: Vec<usize> = rows.iter().map(|r| r.snapshot.shard_idx).collect();
    shard_indices.sort_unstable();
    shard_indices.dedup();

    let max_elapsed = rows.iter().map(|r| r.elapsed_ms).max().unwrap_or(1);
    let max_workers = rows
        .iter()
        .map(|r| r.snapshot.worker_count)
        .max()
        .unwrap_or(1);
    let max_blocks = rows
        .iter()
        .map(|r| r.snapshot.block_count)
        .max()
        .unwrap_or(1);

    let colors: Vec<RGBColor> = vec![
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
    ];

    let root = SVGBackend::new(svg_path, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(350);

    // --- Top panel: workers per shard ---
    let mut chart = ChartBuilder::on(&upper)
        .caption("Workers per shard over time", ("sans-serif", 18))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0u64..max_elapsed, 0usize..max_workers + 1)?;
    chart
        .configure_mesh()
        .x_desc("Elapsed (ms)")
        .y_desc("Workers")
        .draw()?;

    for (i, &shard_idx) in shard_indices.iter().enumerate() {
        let color = colors[i % colors.len()];
        let points: Vec<(u64, usize)> = rows
            .iter()
            .filter(|r| r.snapshot.shard_idx == shard_idx)
            .map(|r| (r.elapsed_ms, r.snapshot.worker_count))
            .collect();
        let label = format!("shard {shard_idx}");
        chart
            .draw_series(LineSeries::new(points, &color))?
            .label(label)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    color.stroke_width(2),
                )
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // --- Bottom panel: blocks per shard ---
    let mut chart2 = ChartBuilder::on(&lower)
        .caption("Blocks per shard over time", ("sans-serif", 18))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0u64..max_elapsed, 0usize..max_blocks + 1)?;
    chart2
        .configure_mesh()
        .x_desc("Elapsed (ms)")
        .y_desc("Cached blocks")
        .draw()?;

    for (i, &shard_idx) in shard_indices.iter().enumerate() {
        let color = colors[i % colors.len()];
        let points: Vec<(u64, usize)> = rows
            .iter()
            .filter(|r| r.snapshot.shard_idx == shard_idx)
            .map(|r| (r.elapsed_ms, r.snapshot.block_count))
            .collect();
        let label = format!("shard {shard_idx}");
        chart2
            .draw_series(LineSeries::new(points, &color))?
            .label(label)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    color.stroke_width(2),
                )
            });
    }
    chart2
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Shard metrics plot written to {svg_path}");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.common.test {
        anyhow::bail!(
            "mooncake_bench no longer supports --test; run `cargo test --package dynamo-bench --test mooncake_trace` instead"
        );
    }

    let path = match args.common.mooncake_trace_path.as_deref() {
        Some(p) => p,
        None => {
            eprintln!("No mooncake_trace_path provided, skipping benchmark");
            return Ok(());
        }
    };
    let traces = process_mooncake_trace(
        path,
        args.common.block_size,
        args.common.trace_length_factor,
        args.common.trace_duplication_factor,
        args.common.num_unique_inference_workers,
        args.common.seed,
    )?;
    let artifacts = generate_replay_artifacts(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;

    let indexer_names: Vec<String> = if args.compare.is_empty() {
        vec![args.get_indexer().to_config().short_name().to_string()]
    } else {
        args.compare.clone()
    };

    if args.common.sweep {
        let durations_low_to_high = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        );
        let durations_high_to_low: Vec<u64> = durations_low_to_high.iter().copied().rev().collect();

        let mut all_results: Vec<(&str, Vec<(u64, BenchmarkResults)>)> = Vec::new();

        for name in &indexer_names {
            println!("\n{}", "=".repeat(60));
            println!("Benchmarking indexer: {}", name);
            println!("{}", "=".repeat(60));

            let config = if args.compare.is_empty() {
                args.get_indexer().to_config()
            } else {
                MooncakeIndexerConfig::from_short_name(name, args.num_event_workers)?
            };

            let multi_threaded = config.is_multi_threaded();
            let durations = if multi_threaded {
                &durations_high_to_low
            } else {
                &durations_low_to_high
            };

            let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();
            let mut consecutive_keeping_up = 0u32;

            for &dur_ms in durations {
                println!("\n=== Sweep: benchmark_duration_ms = {} ===", dur_ms);
                let indexer = config.build(
                    args.common.block_size,
                    Arc::new(KvIndexerMetrics::new_unregistered()),
                );
                let run = run_benchmark(
                    indexer,
                    artifacts.clone(),
                    MooncakeBenchmarkConfig {
                        benchmark_duration_ms: dur_ms,
                        inference_worker_duplication_factor: args
                            .common
                            .inference_worker_duplication_factor,
                        count_events: config.supports_remove(),
                        find_matches_concurrency: args.find_matches_concurrency,
                    },
                )
                .await?;
                if !run.kept_up {
                    eprintln!(
                        "WARNING: The benchmarker is unable to keep up with the request/event generation rate. Rerun with a larger --benchmark-duration-ms."
                    );
                }
                let result = run.results;

                if multi_threaded {
                    if result.block_throughput >= result.offered_block_throughput * 0.95 {
                        consecutive_keeping_up += 1;
                    } else {
                        consecutive_keeping_up = 0;
                    }
                    results.push((dur_ms, result));
                    if consecutive_keeping_up >= 5 {
                        println!("Early stop: achieved >= 95% offered for 5 consecutive steps");
                        break;
                    }
                } else {
                    let saturated = result.offered_block_throughput > result.block_throughput * 5.0;
                    results.push((dur_ms, result));
                    if saturated {
                        println!("Early stop: offered throughput >5x achieved throughput");
                        break;
                    }
                }
            }

            results.sort_by_key(|(dur, _)| std::cmp::Reverse(*dur));
            print_sweep_summary(name, &results);

            all_results.push((name, results));
        }

        plot_sweep(&all_results, &args.sweep_output)?;

        let json_path = args
            .sweep_output
            .replace(".png", ".json")
            .replace(".svg", ".json");
        let json_map: std::collections::BTreeMap<&str, Vec<SweepStepResult>> = all_results
            .iter()
            .map(|(name, results)| {
                let steps = results
                    .iter()
                    .map(|(dur, r)| SweepStepResult {
                        duration_ms: *dur,
                        results: BenchmarkResults {
                            offered_ops_throughput: r.offered_ops_throughput,
                            ops_throughput: r.ops_throughput,
                            offered_block_throughput: r.offered_block_throughput,
                            block_throughput: r.block_throughput,
                            latency_p99_us: r.latency_p99_us,
                        },
                    })
                    .collect();
                (*name, steps)
            })
            .collect();
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_map)?)?;
        println!("Sweep results saved to {}", json_path);
    } else {
        drop(traces);

        for name in &indexer_names {
            println!("\nBenchmarking indexer: {}", name);
            let config = if args.compare.is_empty() {
                args.get_indexer().to_config()
            } else {
                MooncakeIndexerConfig::from_short_name(name, args.num_event_workers)?
            };
            let indexer = config.build(
                args.common.block_size,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            );

            // Start shard-size sampler if a CSV path was provided.
            let shard_cancel = CancellationToken::new();
            let shard_sampler = if !args.shard_metrics_csv.is_empty() {
                Some(start_shard_sampler(
                    indexer.clone(),
                    args.shard_metrics_interval_ms,
                    shard_cancel.clone(),
                ))
            } else {
                None
            };

            let run = run_benchmark(
                indexer.clone(),
                artifacts.clone(),
                MooncakeBenchmarkConfig {
                    benchmark_duration_ms: args.common.benchmark_duration_ms,
                    inference_worker_duplication_factor: args
                        .common
                        .inference_worker_duplication_factor,
                    count_events: config.supports_remove(),
                    find_matches_concurrency: args.find_matches_concurrency,
                },
            )
            .await?;
            if !run.kept_up {
                eprintln!(
                    "WARNING: The benchmarker is unable to keep up with the request/event generation rate. Rerun with a larger --benchmark-duration-ms."
                );
            }

            // Stop sampler and write CSV + plot.
            shard_cancel.cancel();
            if let Some(handle) = shard_sampler {
                let rows = handle.await?;
                // In compare mode, prefix the indexer name to distinguish outputs.
                let csv_path = if args.compare.len() > 1 {
                    let stem = args.shard_metrics_csv.trim_end_matches(".csv");
                    format!("{stem}_{name}.csv")
                } else {
                    args.shard_metrics_csv.clone()
                };
                write_shard_metrics_csv(&rows, &csv_path)?;
                let svg = format!("{}.svg", csv_path.trim_end_matches(".csv"));
                plot_shard_metrics(&rows, &svg)?;
            }

            let report = indexer.timing_report();
            if !report.is_empty() {
                println!("{}", report);
            }
            let sizes = indexer.shard_sizes();
            if sizes.len() > 1 {
                let total_blocks: usize = sizes.iter().map(|s| s.block_count).sum();
                let total_nodes: usize = sizes.iter().map(|s| s.node_count).sum();
                println!("Shard block distribution:");
                for s in &sizes {
                    let pct = if total_blocks > 0 {
                        100.0 * s.block_count as f64 / total_blocks as f64
                    } else {
                        0.0
                    };
                    println!(
                        "  shard {}: {} blocks ({:.1}%), {} workers, {} nodes",
                        s.shard_idx, s.block_count, pct, s.worker_count, s.node_count
                    );
                }
                if total_nodes > 0 {
                    println!("  total nodes across shards: {}", total_nodes);
                }
            }

            let mut edge_lengths = indexer.node_edge_lengths();
            if !edge_lengths.is_empty() {
                let avg = edge_lengths.iter().sum::<usize>() as f64 / edge_lengths.len() as f64;
                edge_lengths.sort_unstable();
                let p99 = edge_lengths[edge_lengths.len() * 99 / 100];
                println!(
                    "Node edge lengths ({} nodes): avg={:.1} hashes/node, p99={} hashes/node",
                    edge_lengths.len(),
                    avg,
                    p99,
                );
            }
        }
    }

    Ok(())
}
