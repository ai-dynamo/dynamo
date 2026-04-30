// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use clap::Parser;
use dynamo_bench::coding::claude::discovery::discover_trace_files;
use dynamo_bench::coding::claude::parser::load_trace_records;
use dynamo_bench::coding::common::{
    DEFAULT_BLOCK_SIZE, DEFAULT_OUTPUT_NAME, DEFAULT_TOKENIZER, expand_user_path, sidecar_path_for,
};
use dynamo_bench::coding::mooncake::{ExportConfig, write_streamed_mooncake_rows};
use dynamo_bench::coding::tokenizer::HfTokenizerFactory;

#[derive(Parser, Debug)]
#[command(name = "claude_trace_export")]
#[command(
    about = "Export local Claude session traces into privacy-preserving Mooncake JSONL plus a sidecar"
)]
struct Args {
    #[arg(long, action = clap::ArgAction::Append)]
    input_path: Vec<String>,

    #[arg(long, default_value = DEFAULT_OUTPUT_NAME)]
    output_file: String,

    #[arg(long, default_value = DEFAULT_TOKENIZER)]
    tokenizer: String,

    #[arg(long, default_value_t = DEFAULT_BLOCK_SIZE)]
    block_size: usize,

    #[arg(long)]
    anonymize_session_id: bool,

    #[arg(long, default_value_t = 50)]
    delta_overlap_words: usize,

    #[arg(long, default_value_t = default_tokenizer_workers())]
    tokenizer_workers: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.block_size == 0 {
        bail!("--block-size must be positive");
    }
    if args.tokenizer_workers == 0 {
        bail!("--tokenizer-workers must be positive");
    }

    let current_dir = std::env::current_dir()?;
    let trace_files = discover_trace_files(&args.input_path, &current_dir)?;
    if trace_files.is_empty() {
        bail!("no Claude session traces found");
    }

    let tokenizer_factory = HfTokenizerFactory::resolve(&args.tokenizer)?;
    let sessions = load_trace_records(&trace_files)?;
    if sessions.is_empty() {
        bail!("no parseable Claude session rows were found in the discovered files");
    }

    let output_path = expand_user_path(&args.output_file);
    let sidecar_path = sidecar_path_for(&output_path);
    let stats = write_streamed_mooncake_rows(
        &output_path,
        &sidecar_path,
        sessions,
        !args.anonymize_session_id,
        tokenizer_factory,
        ExportConfig {
            block_size: args.block_size,
            delta_overlap_words: args.delta_overlap_words,
            tokenizer_workers: args.tokenizer_workers,
        },
    )?;

    if stats.row_count == 0 {
        bail!("no assistant turns were reconstructed from the discovered traces");
    }

    println!(
        "Wrote {} Mooncake rows to {}",
        stats.row_count,
        output_path.display()
    );
    println!(
        "Wrote {} sidecar rows to {}",
        stats.sidecar_count,
        sidecar_path.display()
    );
    println!("Discovered {} trace files", trace_files.len());
    Ok(())
}

fn default_tokenizer_workers() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(8))
        .unwrap_or(1)
}
