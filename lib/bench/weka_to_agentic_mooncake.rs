// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialize a local Weka file or directory as canonical Agentic Mooncake v2.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use dynamo_data_gen::{MooncakeJsonlWriter, WekaImporter};

#[derive(Debug, Parser)]
#[command(name = "weka_to_agentic_mooncake")]
#[command(about = "Convert local Weka/AgentX traces to Agentic Mooncake v2 JSONL")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.output.exists() {
        bail!("refusing to overwrite {}", args.output.display());
    }
    let parent = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    let temporary_path = temporary.into_temp_path();

    let importer = WekaImporter::open(&args.input)?;
    let mut writer = MooncakeJsonlWriter::create(&temporary_path, None)?;
    writer.write_agentic_header(importer.header())?;
    let summary = importer.for_each_row(|row| writer.write_agentic_row(&row))?;
    let stats = writer.finish()?;
    temporary_path
        .persist(&args.output)
        .with_context(|| format!("failed to publish {}", args.output.display()))?;

    println!(
        "Wrote {} requests from {} plays ({} files) to {}",
        stats.row_count,
        summary.plays,
        summary.files,
        args.output.display()
    );
    println!("Raw Weka digest: {}", summary.header.source.digest);
    println!(
        "Raw zero-output requests normalized to one: {}",
        summary.raw_zero_outputs
    );
    Ok(())
}
