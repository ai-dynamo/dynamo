// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint over the modules in [`dynamo_bench::request_trace`].

use anyhow::Result;
use clap::Parser;
use dynamo_bench::coding::common::expand_user_path;
use dynamo_bench::request_trace::{
    agentic::{build_agentic_mooncake_rows, summarize_tools},
    load::load_request_trace_records,
    mooncake::build_mooncake_rows,
};
use dynamo_data_gen::MooncakeJsonlWriter;

#[derive(Parser, Debug)]
#[command(name = "request_trace_to_mooncake")]
#[command(about = "Convert Dynamo request trace JSONL/JSONL.GZ records to Mooncake replay JSONL")]
struct Args {
    #[arg(long, action = clap::ArgAction::Append, required = true, num_args = 1..)]
    input_path: Vec<String>,

    #[arg(long)]
    output_file: String,

    #[arg(long)]
    agentic: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_paths = args
        .input_path
        .iter()
        .map(|path| expand_user_path(path))
        .collect::<Vec<_>>();
    let output_path = expand_user_path(&args.output_file);

    let loaded = load_request_trace_records(&input_paths)?;
    let tool_summary = summarize_tools(&loaded.tools);

    let (kind, trace_block_size, stats) = if args.agentic {
        let (trace_block_size, rows) = build_agentic_mooncake_rows(loaded)?;
        let mut writer = MooncakeJsonlWriter::create(&output_path, None)?;
        for row in &rows {
            writer.write_agentic_row(row)?;
        }
        ("Agentic Mooncake", trace_block_size, writer.finish()?)
    } else {
        let (trace_block_size, rows) = build_mooncake_rows(loaded.requests)?;
        let mut writer = MooncakeJsonlWriter::create(&output_path, None)?;
        for row in &rows {
            writer.write_row(row)?;
        }
        ("Mooncake", trace_block_size, writer.finish()?)
    };

    println!(
        "Wrote {} {kind} rows to {}",
        stats.row_count,
        output_path.display()
    );
    println!("Trace block size: {trace_block_size}");
    if tool_summary.total_spans > 0 {
        println!();
        print!("{}", tool_summary.render());
    }
    Ok(())
}
