// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: apply the delivery rule (`delivery` module) to one arm of a load point.
//!
//! Prints the verdict as JSON and exits 1 when the arm is invalid. A load point is invalid when
//! either arm is.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use dynamo_e2e_indexer_tools::delivery::{DEFAULT_MIN_DELIVERED_FRACTION, check};

#[derive(Parser, Debug)]
#[command(about = "EXPERIMENT ONLY: check phantom delivery for one arm of a load point")]
struct Args {
    /// A publisher summary (`phantom_publisher --summary-out`); repeat for every process.
    #[arg(long = "publisher-summary", required = true)]
    publisher_summaries: Vec<PathBuf>,
    /// The serving indexer's accounting (`DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT`), read after
    /// every publisher exited.
    #[arg(long)]
    indexer_accounting: PathBuf,
    #[arg(long, default_value_t = DEFAULT_MIN_DELIVERED_FRACTION)]
    min_delivered_fraction: f64,
    /// Also require delivered to equal sent exactly (low-load plumbing smokes).
    #[arg(long)]
    require_exact: bool,
    /// Arm label echoed in the output.
    #[arg(long)]
    label: Option<String>,
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let publishers = args
        .publisher_summaries
        .iter()
        .map(|path| read_json(path))
        .collect::<Result<Vec<_>>>()?;
    let indexer = read_json(&args.indexer_accounting)?;
    let mut verdict = check(&publishers, &indexer, args.min_delivered_fraction)?;
    if args.require_exact && !verdict.exact {
        verdict.valid = false;
        verdict.reasons.push(format!(
            "delivered {:?} differs from sent {:?} (--require-exact)",
            verdict.delivered, verdict.sent
        ));
    }
    let mut output = serde_json::to_value(&verdict)?;
    output["label"] = serde_json::json!(args.label);
    println!("{}", serde_json::to_string_pretty(&output)?);
    if !verdict.valid {
        std::process::exit(1);
    }
    Ok(())
}
