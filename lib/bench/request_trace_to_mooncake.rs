// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convert Dynamo request traces to Mooncake replay JSONL.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Parser;
use dynamo_data_gen::{
    MooncakeJsonlWriter,
    request_trace::{
        agentic::lower_agentic_mooncake_rows,
        load::{RequestTraceMode, load_request_trace_records},
        mooncake::lower_mooncake_rows,
    },
};
use tempfile::TempDir;

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

struct TemporaryOutput {
    _directory: TempDir,
    path: PathBuf,
}

impl AsRef<Path> for TemporaryOutput {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl TemporaryOutput {
    fn persist(self, output_path: &Path) -> Result<()> {
        std::fs::rename(&self.path, output_path)?;
        Ok(())
    }
}

fn temporary_output_path(output_path: &Path) -> Result<TemporaryOutput> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let directory = tempfile::tempdir_in(parent)?;
    let path = directory.path().join(
        output_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("mooncake.jsonl")),
    );
    Ok(TemporaryOutput {
        _directory: directory,
        path,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let loaded = load_request_trace_records(&args.input_path)?;
    let temporary_output = temporary_output_path(&args.output_file)?;
    let mut writer = MooncakeJsonlWriter::create(temporary_output.as_ref(), None)?;
    if let Ok(metadata) = std::fs::metadata(&args.output_file) {
        std::fs::set_permissions(temporary_output.as_ref(), metadata.permissions())?;
    }
    let (kind, trace_block_size) = match (loaded.mode()?, args.agentic) {
        (RequestTraceMode::Agentic, true) => (
            "Agentic Mooncake",
            lower_agentic_mooncake_rows(loaded, |_, row| writer.write_agentic_row(&row))?,
        ),
        (RequestTraceMode::Standard, false) => (
            "Mooncake",
            lower_mooncake_rows(loaded.requests, |_, row| writer.write_row(&row))?,
        ),
        (RequestTraceMode::Agentic, false) => {
            bail!("agentic request traces require --agentic")
        }
        (RequestTraceMode::Standard, true) => {
            bail!("--agentic requires request traces with agent_context")
        }
    };
    let stats = writer.finish()?;
    temporary_output.persist(&args.output_file)?;

    println!(
        "Wrote {} {kind} rows to {}",
        stats.row_count,
        args.output_file.display()
    );
    println!("Trace block size: {trace_block_size}");
    Ok(())
}
