// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: plan a phantom load point.
//!
//! Prints the stream's natural rates, the resolved speedup, the resulting aggregate rates, and
//! whether the timed section covers the requested duration. With `--publishers`, splits the
//! phantoms into contiguous per-process ranges, prints each process's arguments, and writes the
//! serving indexer's `DYN_EXPERIMENT_STATIC_KV_SOURCES` file.

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use dynamo_e2e_indexer_tools::plan::{
    DEFAULT_WORKER_ID_BASE, PhantomLayout, TimeMap, natural_rates, resolve_speedup,
};
use dynamo_e2e_indexer_tools::stream::Manifest;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(about = "EXPERIMENT ONLY: plan phantom publishers for one load point")]
struct Args {
    #[arg(long)]
    streams: PathBuf,
    #[arg(long)]
    total_phantoms: u64,
    #[arg(long, default_value_t = DEFAULT_WORKER_ID_BASE, value_parser = parse_u64)]
    worker_id_base: u64,
    #[arg(long, default_value_t = 0)]
    salt_seed: u64,
    #[arg(long)]
    speedup: Option<f64>,
    #[arg(long)]
    target_write_blocks_per_sec: Option<f64>,
    /// Required timed coverage (wall seconds, including --start-spread-ms).
    #[arg(long)]
    duration_s: Option<f64>,
    #[arg(long, default_value_t = 0)]
    start_spread_ms: u64,
    /// Publisher processes as `host:base_port:count`, comma separated; counts must sum to
    /// --total-phantoms.
    #[arg(long, value_delimiter = ',')]
    publishers: Vec<String>,
    /// Where to write the static-source list for the serving indexer.
    #[arg(long)]
    sources_out: Option<PathBuf>,
}

fn parse_u64(value: &str) -> Result<u64, String> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|error| error.to_string())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest = Manifest::read(&args.streams)?;
    let layout = PhantomLayout {
        total: args.total_phantoms,
        worker_id_base: args.worker_id_base,
        salt_seed: args.salt_seed,
    };
    layout.validate(&manifest)?;
    let speedup = resolve_speedup(
        &manifest,
        &layout,
        args.speedup,
        args.target_write_blocks_per_sec,
    )?;
    let map = TimeMap {
        start_at_unix_us: 0,
        t0_us: manifest.t0_us,
        speedup,
    };
    let natural = natural_rates(&manifest, &layout, 0..layout.total);
    let coverage_s = map.coverage_s(&manifest);
    let required_s = args
        .duration_s
        .map(|duration| duration + args.start_spread_ms as f64 / 1e3);
    let wall = |rate: f64| rate * speedup;
    let plan = json!({
        "bases": manifest.bases.len(),
        "block_size": manifest.block_size,
        "timed_span_virtual_s": manifest.span_us() as f64 / 1e6,
        "total_phantoms": layout.total,
        "phantoms_per_base": layout.total as f64 / manifest.bases.len() as f64,
        "speedup": speedup,
        "coverage_s": coverage_s,
        "required_s": required_s,
        "natural_per_virtual_s": natural,
        "aggregate_per_wall_s": {
            "write_blocks": wall(natural.write_blocks),
            "stored_blocks": wall(natural.stored_blocks),
            "removed_blocks": wall(natural.removed_blocks),
            "events": wall(natural.events),
            "queries": wall(natural.queries),
            "query_blocks": wall(natural.query_blocks),
            "queries_per_phantom": wall(natural.queries) / layout.total as f64,
        },
        "warmup_write_blocks": natural.warmup_write_blocks,
    });
    println!("{}", serde_json::to_string_pretty(&plan)?);
    if let Some(required) = required_s
        && coverage_s < required
    {
        bail!(
            "the timed section covers {coverage_s:.1} s at speedup {speedup:.3}, less than the required {required:.1} s; capture a longer corpus, lower the speedup, or add phantoms"
        );
    }

    if args.publishers.is_empty() {
        return Ok(());
    }
    let mut first = 0u64;
    let mut entries = Vec::new();
    for spec in &args.publishers {
        let mut parts = spec.split(':');
        let (Some(host), Some(port), Some(count), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            bail!("publisher {spec:?} is not host:base_port:count");
        };
        let port: u16 = port
            .parse()
            .with_context(|| format!("publisher {spec:?} port"))?;
        let count: u64 = count
            .parse()
            .with_context(|| format!("publisher {spec:?} count"))?;
        ensure!(
            count > 0 && count <= 1000,
            "publisher {spec:?} count must be in 1..=1000"
        );
        ensure!(
            u64::from(port) + count - 1 <= u64::from(u16::MAX),
            "publisher {spec:?} ports run past 65535"
        );
        let entry = format!("{}+{count}@tcp://{host}:{port}", layout.worker_id(first));
        println!(
            "{}",
            json!({
                "publisher": spec,
                "args": [
                    "--total-phantoms", layout.total.to_string(),
                    "--first-phantom", first.to_string(),
                    "--count", count.to_string(),
                    "--worker-id-base", format!("{:#x}", layout.worker_id_base),
                    "--salt-seed", layout.salt_seed.to_string(),
                    "--advertise-host", host,
                    "--base-port", port.to_string(),
                ],
                "sources_entry": entry,
            })
        );
        entries.push(entry);
        first += count;
    }
    ensure!(
        first == layout.total,
        "publisher counts sum to {first}, not --total-phantoms {}",
        layout.total
    );
    if let Some(path) = &args.sources_out {
        std::fs::write(path, entries.join("\n") + "\n")
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}
