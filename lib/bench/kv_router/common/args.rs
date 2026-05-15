// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Shared CLI arguments for trace-based benchmarks.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Path to a JSONL mooncake trace file.
    pub mooncake_trace_path: Option<String>,

    /// Deprecated compatibility flag. Use `cargo test --package dynamo-bench --test ...`
    /// for the fixture-backed integration tests instead.
    #[clap(long)]
    pub test: bool,

    /// Number of GPU blocks available in the mock engine's KV cache.
    #[clap(long, default_value = "16384")]
    pub num_gpu_blocks: usize,

    /// Number of tokens per KV cache block.
    #[clap(long, default_value = "128")]
    pub block_size: u32,

    /// Optional wall-clock duration (ms) used to rescale the trace during event generation.
    /// Omit to preserve the original Mooncake timestamps.
    #[clap(long)]
    pub trace_simulation_duration_ms: Option<u64>,

    /// Wall-clock duration (ms) over which the benchmark replays operations.
    #[clap(long, default_value = "60000")]
    pub benchmark_duration_ms: u64,

    /// Number of unique simulated inference workers.
    #[clap(short, long, default_value = "1000")]
    pub num_unique_inference_workers: usize,

    /// How many times to duplicate unique workers during the benchmark phase.
    #[clap(short = 'd', long, default_value = "1")]
    pub inference_worker_duplication_factor: usize,

    /// Factor by which to stretch each request's hash sequence length.
    #[clap(long, default_value = "1")]
    pub trace_length_factor: usize,

    /// How many times to duplicate the raw trace data with offset hash_ids.
    #[clap(long, default_value = "1")]
    pub trace_duplication_factor: usize,

    /// RNG seed for reproducible worker-to-trace assignment.
    #[clap(long, default_value = "42")]
    pub seed: u64,

    /// Enable throughput vs p99 latency sweep mode.
    #[clap(long)]
    pub sweep: bool,

    /// Minimum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "1000")]
    pub sweep_min_ms: u64,

    /// Maximum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "50000")]
    pub sweep_max_ms: u64,

    /// Number of logarithmically spaced sweep steps between min and max.
    #[clap(long, default_value = "10")]
    pub sweep_steps: usize,

    /// Ignored - passed by cargo bench harness.
    #[arg(long, hide = true, global = true)]
    pub bench: bool,

    /// Opt in to runtime warn/error logs from the mocker and sequence tracker.
    #[clap(long)]
    pub sequence_logs: bool,
}
