// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: apply the delivery rule (`delivery` module) to one arm of a load point.
//!
//! Prints the verdict as JSON and exits 1 when the arm is invalid. A load point is invalid when
//! either arm is.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use dynamo_e2e_indexer_tools::delivery::{Rule, check};

#[derive(Parser, Debug)]
#[command(about = "EXPERIMENT ONLY: check phantom delivery for one arm of a load point")]
struct Args {
    /// A publisher summary (`phantom_publisher --summary-out`); repeat for every process.
    #[arg(long = "publisher-summary", required = true)]
    publisher_summaries: Vec<PathBuf>,
    /// The serving indexer's accounting (`DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT`), read at
    /// least three report intervals after every publisher exited.
    #[arg(long)]
    indexer_accounting: PathBuf,
    /// Admitted timed write blocks between the window marks, as a fraction of planned.
    #[arg(long, default_value_t = Rule::default().min_window_fraction)]
    min_window_fraction: f64,
    /// Longest drain of the indexer's event queues allowed at either window mark.
    #[arg(long, default_value_t = Rule::default().max_drain_ms)]
    max_drain_ms: f64,
    #[arg(long, default_value_t = Rule::default().max_lag_p99_ms)]
    max_lag_p99_ms: f64,
    #[arg(long, default_value_t = Rule::default().max_lag_ms)]
    max_lag_ms: f64,
    /// Latest a publisher's last timed send may follow its stop.
    #[arg(long, default_value_t = Rule::default().max_send_overrun_ms)]
    max_send_overrun_ms: u64,
    /// Latest the indexer's end mark may follow the publishers' stop.
    #[arg(long, default_value_t = Rule::default().max_end_grace_ms)]
    max_end_grace_ms: u64,
    /// The DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB both arms must report (phantom_plan's value).
    #[arg(long)]
    expect_endpoints_per_sub: Option<u64>,
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
    let rule = Rule {
        min_window_fraction: args.min_window_fraction,
        max_drain_ms: args.max_drain_ms,
        max_lag_p99_ms: args.max_lag_p99_ms,
        max_lag_ms: args.max_lag_ms,
        max_send_overrun_ms: args.max_send_overrun_ms,
        max_end_grace_ms: args.max_end_grace_ms,
        expect_endpoints_per_sub: args.expect_endpoints_per_sub,
    };
    let verdict = check(&publishers, &indexer, rule)?;
    let mut output = serde_json::to_value(&verdict)?;
    output["label"] = serde_json::json!(args.label);
    println!("{}", serde_json::to_string_pretty(&output)?);
    if !verdict.valid {
        std::process::exit(1);
    }
    Ok(())
}
