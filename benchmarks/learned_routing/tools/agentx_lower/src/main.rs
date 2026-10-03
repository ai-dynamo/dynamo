// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin CLI over aisimulate-core's public Weka importer.
//!
//! `lower` writes exactly the header and rows that native `trace_format="weka"` replay compiles
//! (`WekaImporter::open` with the default nested-timestamp basis, the same call
//! `load_weka_agentic_graph` makes). `digest-weka` and `digest-agentic` print the canonical
//! graph digest of a Weka source and of an Agentic Mooncake v2 file, so equality of the two
//! proves the compiled replay graphs are identical node for node.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use aisimulate_core::replay::loadgen::{AgenticTrace, WekaImporter, load_weka_agentic_graph};
use anyhow::{Context, Result, bail};
use serde_json::json;

fn lower(source: &Path, out: &Path) -> Result<()> {
    let importer =
        WekaImporter::open(source).with_context(|| format!("importing {}", source.display()))?;
    let mut writer =
        BufWriter::new(File::create(out).with_context(|| format!("creating {}", out.display()))?);
    serde_json::to_writer(&mut writer, importer.header())?;
    writer.write_all(b"\n")?;
    let summary = importer.for_each_row(|row| {
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        Ok(())
    })?;
    writer.flush()?;
    println!(
        "{}",
        json!({
            "files": summary.files,
            "plays": summary.plays,
            "requests": summary.requests,
            "raw_zero_outputs": summary.raw_zero_outputs,
            "nested_timestamp_basis": summary.nested_timestamp_basis.as_str(),
            "header": summary.header,
        })
    );
    Ok(())
}

fn print_digest(graph: &AgenticTrace) {
    println!(
        "{}",
        json!({
            "graph_digest": graph.graph_digest(),
            "block_size": graph.block_size(),
            "nodes": graph.node_count(),
            "plays": graph.play_count(),
            "source": graph.source(),
        })
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [_, "lower", source, out] => lower(Path::new(source), Path::new(out)),
        [_, "digest-weka", source] => {
            print_digest(&load_weka_agentic_graph(source, None)?);
            Ok(())
        }
        [_, "digest-agentic", path] => {
            print_digest(&AgenticTrace::from_agentic_mooncake(Path::new(path))?);
            Ok(())
        }
        _ => bail!(
            "usage: agentx-lower lower <weka file|dir> <out.jsonl> | digest-weka <weka file|dir> | digest-agentic <file.jsonl>"
        ),
    }
}
