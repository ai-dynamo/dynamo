// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use clap::Parser;
use dynamo_bench::coding::codex::discovery::discover_rollout_files;
use dynamo_bench::coding::codex::export::{
    ExportConfig, ExportReport, load_threads, write_request_trace,
};
use dynamo_bench::coding::common::{DEFAULT_BLOCK_SIZE, expand_user_path};

#[derive(Parser, Debug)]
#[command(name = "codex_trace_export")]
#[command(about = "Export local Codex rollouts into Dynamo request-trace JSONL")]
struct Args {
    /// Rollout file or directory; defaults to `$CODEX_HOME/sessions` and `archived_sessions`.
    #[arg(long, action = clap::ArgAction::Append)]
    input_path: Vec<String>,

    #[arg(long, default_value = "codex_request_trace.jsonl")]
    output_file: String,

    #[arg(long, default_value_t = DEFAULT_BLOCK_SIZE)]
    block_size: usize,

    #[arg(long)]
    anonymize_session_id: bool,

    #[arg(long, default_value_t = default_workers())]
    workers: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.block_size == 0 {
        bail!("--block-size must be positive");
    }

    let rollout_files = discover_rollout_files(&args.input_path)?;
    if rollout_files.is_empty() {
        bail!("no Codex rollout files found");
    }
    let mut report = ExportReport::default();
    let threads = load_threads(&rollout_files, args.workers, &mut report)?;
    if threads.is_empty() {
        bail!(
            "no rollout recorded token usage; Codex writes usage records from 0.154 onward\n{}",
            report.render()
        );
    }

    let output_path = expand_user_path(&args.output_file);
    write_request_trace(
        &output_path,
        &threads,
        ExportConfig {
            block_size: args.block_size,
            preserve_session_ids: !args.anonymize_session_id,
        },
        &mut report,
    )?;
    println!(
        "Wrote {} request and {} tool rows to {}",
        report.requests,
        report.tools,
        output_path.display()
    );
    println!("{}", report.render());
    Ok(())
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(8))
        .unwrap_or(1)
}
