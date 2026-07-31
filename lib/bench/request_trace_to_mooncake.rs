// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convert Dynamo request traces to Mooncake replay JSONL.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use dynamo_data_gen::{
    MooncakeJsonlWriter,
    request_trace::{
        agentic::lower_agentic_mooncake_rows, load::load_request_trace_records,
        mooncake::lower_mooncake_rows,
    },
};

#[derive(Parser, Debug)]
#[command(name = "request_trace_to_mooncake")]
#[command(about = "Convert Dynamo request-trace JSONL shards to Mooncake replay JSONL")]
struct Args {
    #[arg(long, action = clap::ArgAction::Append, required = true, num_args = 1..)]
    input_path: Vec<PathBuf>,

    #[arg(long)]
    output_file: PathBuf,

    #[arg(long)]
    agentic: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let loaded = load_request_trace_records(&args.input_path)?;
    let mut writer = MooncakeJsonlWriter::create(&args.output_file, None)?;
    let (kind, trace_block_size) = if args.agentic {
        (
            "Agentic Mooncake",
            lower_agentic_mooncake_rows(loaded, |_, row| writer.write_agentic_row(&row))?,
        )
    } else {
        (
            "Mooncake",
            lower_mooncake_rows(loaded.requests, |_, row| writer.write_row(&row))?,
        )
    };
    let stats = writer.finish()?;

    println!(
        "Wrote {} {kind} rows to {}",
        stats.row_count,
        args.output_file.display()
    );
    println!("Trace block size: {trace_block_size}");
    Ok(())
}
