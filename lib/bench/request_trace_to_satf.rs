// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use dynamo_data_gen::satf::{SatfNode, convert_request_trace_to_satf, write_satf};

#[derive(Parser, Debug)]
#[command(name = "request_trace_to_satf")]
#[command(about = "Convert Dynamo request trace JSONL shards to SATF 2.0")]
struct Args {
    #[arg(required = true)]
    input: Vec<PathBuf>,

    #[arg(short, long, default_value = "satf.json")]
    output: PathBuf,

    #[arg(long)]
    pretty: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let trace = convert_request_trace_to_satf(&args.input)?;

    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = File::create(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    write_satf(&mut writer, &trace, args.pretty)?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let llm_nodes = trace
        .sessions
        .iter()
        .flat_map(|session| &session.nodes)
        .filter(|node| matches!(node, SatfNode::LlmInfer { .. }))
        .count();
    let tool_nodes = trace
        .sessions
        .iter()
        .flat_map(|session| &session.nodes)
        .filter(|node| !matches!(node, SatfNode::LlmInfer { .. }))
        .count();
    println!(
        "Wrote {} sessions, {} LLM nodes, and {} tool nodes to {}",
        trace.sessions.len(),
        llm_nodes,
        tool_nodes,
        args.output.display()
    );
    Ok(())
}
