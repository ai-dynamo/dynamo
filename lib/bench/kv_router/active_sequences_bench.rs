// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "active_sequences_shared.rs"]
mod active_sequences_shared;

use active_sequences_shared::{generate_sequence_events, run_benchmark};
use clap::Parser;
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::replay::process_mooncake_trace;
use dynamo_bench::kv_router_common::results::BenchmarkResults;
use dynamo_bench::kv_router_common::sweep::{compute_sweep_durations, print_sweep_summary};
use tracing_subscriber::EnvFilter;

fn init_sequence_logging(enabled: bool) {
    if !enabled {
        return;
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "error,dynamo_kv_router::sequences=warn,dynamo_mocker=warn",
        ))
        .with_writer(std::io::stderr)
        .try_init();
}

#[derive(Parser, Debug)]
#[clap(
    version,
    about = "ActiveSequences add_request/free throughput benchmark"
)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_sequence_logging(args.common.sequence_logs);

    if args.common.test {
        anyhow::bail!(
            "active_sequences_bench no longer supports --test; run `cargo test --package dynamo-bench --test active_sequences_trace` instead"
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

    let seq_traces = generate_sequence_events(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;

    if args.common.sweep {
        let durations = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        );

        let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();
        for &dur_ms in &durations {
            println!("\n=== Sweep: benchmark_duration_ms = {} ===", dur_ms);
            let run = run_benchmark(
                &seq_traces,
                args.common.block_size,
                dur_ms,
                args.common.inference_worker_duplication_factor,
            )
            .await?;
            if !run.kept_up {
                eprintln!(
                    "WARNING: Benchmarker could not keep up. Rerun with a larger --benchmark-duration-ms."
                );
            }
            results.push((dur_ms, run.results));
        }

        print_sweep_summary("active-sequences", &results);
    } else {
        let run = run_benchmark(
            &seq_traces,
            args.common.block_size,
            args.common.benchmark_duration_ms,
            args.common.inference_worker_duplication_factor,
        )
        .await?;
        if !run.kept_up {
            eprintln!(
                "WARNING: Benchmarker could not keep up. Rerun with a larger --benchmark-duration-ms."
            );
        }
    }

    Ok(())
}
