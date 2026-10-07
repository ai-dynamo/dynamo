// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: plan a phantom load point.
//!
//! With `--duration-s`, the speedup is solved on the planned window itself: the streams are not
//! stationary (fresh agentic sessions write more per request), so a whole-span average would
//! mislabel a window that plays only part of the span. The plan reads every base's timeline
//! (list and query timestamps with block counts, no hashes) and applies the publisher's and
//! driver's own stop rule. Pass the printed `speedup` to every publisher and driver.
//!
//! Prints the stream's natural rates, the resolved speedup, the resulting aggregate rates,
//! whether the timed section covers the requested duration, whether the capture reached
//! steady-state eviction, the serving indexer's ZMQ socket budget (and so the mandatory
//! `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB`), and the delivery rule a run must pass. With
//! `--publishers`, splits the phantoms into contiguous per-process ranges, prints each process's
//! arguments, and writes the serving indexer's `DYN_EXPERIMENT_STATIC_KV_SOURCES` file.

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use dynamo_e2e_indexer_tools::delivery::Rule;
use dynamo_e2e_indexer_tools::plan::{
    DEFAULT_SOCKET_RESERVE, DEFAULT_SOCKETS_PER_LIVE_SOURCE, DEFAULT_WORKER_ID_BASE, PhantomLayout,
    TimeMap, WindowTotals, ZMQ_MAX_SOCKETS, indexer_sockets, natural_rates, resolve_speedup,
    solve_window_speedup, window_totals,
};
use dynamo_e2e_indexer_tools::stream::{
    BaseTimeline, DEFAULT_MIN_TIMED_REMOVE_RATIO, Manifest, eviction_report, read_timeline,
};
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
    /// Live KV sources (mocker workers x DP ranks) the serving indexer also subscribes to.
    #[arg(long)]
    live_sources: u64,
    /// Ungrouped direct-ZMQ SUB sockets the indexer opens per live source besides KV events.
    #[arg(long, default_value_t = DEFAULT_SOCKETS_PER_LIVE_SOURCE)]
    sockets_per_live_source: u64,
    /// Sockets kept free for everything else the indexer opens.
    #[arg(long, default_value_t = DEFAULT_SOCKET_RESERVE)]
    socket_reserve: u64,
    #[arg(long, default_value_t = ZMQ_MAX_SOCKETS)]
    zmq_max_sockets: u64,
    /// Pin DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB instead of the smallest value that fits. Pin one
    /// value (at least 2, so every point takes the grouped SUB path) for a whole sweep, in both
    /// arms; it must fit the largest point.
    #[arg(long)]
    endpoints_per_sub: Option<u64>,
    /// Write the indexer's experiment environment here (`export` lines). Source the same file
    /// in both arms.
    #[arg(long)]
    indexer_env_out: Option<PathBuf>,
    /// Refuse streams whose timed section removes fewer blocks than this fraction of the blocks
    /// it stores.
    #[arg(long, default_value_t = DEFAULT_MIN_TIMED_REMOVE_RATIO)]
    min_timed_remove_ratio: f64,
    /// Plan despite streams below --min-timed-remove-ratio (smoke tests only).
    #[arg(long)]
    allow_low_eviction: bool,
}

fn publisher_args(
    layout: &PhantomLayout,
    first: u64,
    count: u64,
    host: &str,
    port: u16,
    speedup: f64,
    args: &Args,
) -> Vec<String> {
    let mut out = vec![
        "--speedup".to_string(),
        speedup.to_string(),
        "--total-phantoms".to_string(),
        layout.total.to_string(),
        "--first-phantom".to_string(),
        first.to_string(),
        "--count".to_string(),
        count.to_string(),
        "--worker-id-base".to_string(),
        format!("{:#x}", layout.worker_id_base),
        "--salt-seed".to_string(),
        layout.salt_seed.to_string(),
        "--advertise-host".to_string(),
        host.to_string(),
        "--base-port".to_string(),
        port.to_string(),
        "--min-timed-remove-ratio".to_string(),
        args.min_timed_remove_ratio.to_string(),
    ];
    if args.allow_low_eviction {
        out.push("--allow-low-eviction".to_string());
    }
    out
}

fn parse_u64(value: &str) -> Result<u64, String> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|error| error.to_string())
}

/// Every base's timeline, read in parallel.
fn read_timelines(dir: &std::path::Path, manifest: &Manifest) -> Result<Vec<BaseTimeline>> {
    let paths: Vec<PathBuf> = manifest
        .bases
        .iter()
        .map(|base| dir.join(&base.file))
        .collect();
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
    let chunk = paths.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .map(|paths| {
                scope.spawn(move || {
                    paths
                        .iter()
                        .map(|path| read_timeline(path))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        let mut timelines = Vec::with_capacity(paths.len());
        for handle in handles {
            timelines.extend(handle.join().expect("timeline reader panicked")?);
        }
        Ok(timelines)
    })
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
    let span_average_speedup = resolve_speedup(
        &manifest,
        &layout,
        args.speedup,
        args.target_write_blocks_per_sec,
    )?;
    // With a duration, solve (or evaluate) the speedup on the planned window itself.
    let window: Option<WindowTotals> = match args.duration_s {
        None => None,
        Some(duration_s) => {
            let timelines = read_timelines(&args.streams, &manifest)?;
            let duration_us = (duration_s * 1e6) as u64;
            let spread_us = args.start_spread_ms * 1000;
            Some(match args.target_write_blocks_per_sec {
                Some(target) => solve_window_speedup(
                    &timelines,
                    &layout,
                    manifest.t0_us,
                    target,
                    duration_us,
                    spread_us,
                )?,
                None => window_totals(
                    &timelines,
                    &layout,
                    manifest.t0_us,
                    span_average_speedup,
                    duration_us,
                    spread_us,
                ),
            })
        }
    };
    let speedup = window.map_or(span_average_speedup, |window| window.speedup);
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
    let aggregate = match &window {
        Some(window) => {
            let mut per_s = window.per_s();
            per_s["queries_per_phantom"] =
                json!(window.queries as f64 / window.duration_s / layout.total as f64);
            per_s["basis"] = json!("planned window");
            per_s
        }
        None => json!({
            "write_blocks": wall(natural.write_blocks),
            "stored_blocks": wall(natural.stored_blocks),
            "removed_blocks": wall(natural.removed_blocks),
            "events": wall(natural.events),
            "queries": wall(natural.queries),
            "query_blocks": wall(natural.query_blocks),
            "queries_per_phantom": wall(natural.queries) / layout.total as f64,
            "basis": "whole-span average",
        }),
    };
    let sockets = indexer_sockets(
        layout.total,
        args.live_sources,
        args.sockets_per_live_source,
        args.socket_reserve,
        args.zmq_max_sockets,
        args.endpoints_per_sub,
    )?;
    let eviction = eviction_report(&manifest, args.min_timed_remove_ratio);
    let flagged_rows: Vec<_> = eviction
        .rows
        .iter()
        .filter(|row| eviction.bases_below_min.contains(&row.base))
        .collect();
    let plan = json!({
        "bases": manifest.bases.len(),
        "block_size": manifest.block_size,
        "timed_span_virtual_s": manifest.span_us() as f64 / 1e6,
        "total_phantoms": layout.total,
        "phantoms_per_base": layout.total as f64 / manifest.bases.len() as f64,
        "speedup": speedup,
        "speedup_span_average": span_average_speedup,
        "coverage_s": coverage_s,
        "required_s": required_s,
        "natural_per_virtual_s": natural,
        "window": window,
        "aggregate_per_wall_s": aggregate,
        "warmup_write_blocks": natural.warmup_write_blocks,
        "eviction": {
            "min_timed_remove_ratio": eviction.min_timed_remove_ratio,
            "timed_remove_ratio": eviction.timed_remove_ratio,
            "capacity_blocks": eviction.capacity_blocks,
            "bases_below_min": flagged_rows,
        },
        "indexer": {
            "sockets": sockets,
            "env": { "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB": sockets.endpoints_per_sub },
        },
        "acceptance": {
            "defaults": Rule::default(),
            "expect_endpoints_per_sub": sockets.endpoints_per_sub,
            "rule": "per arm, run delivery_check --expect-endpoints-per-sub <this plan's value> on every publisher summary and the indexer accounting (read three report intervals after the last publisher exits); the load point is invalid if either arm is: any phantom not delivered exactly as planned (events, write blocks, first and last event ID); the timed start or end mark missing; warm-up not exactly admitted at the start mark; window delivery below min_window_fraction of planned timed or above it; event queues not drained within max_drain_ms at either mark; timed send lag p99 or max over bounds or a send past the stop; any gap or rank reset; a late warm-up, send error or interruption; a source-count or endpoints_per_sub mismatch; or a stale file",
            "indexer_env": "launch the indexer with DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS = --start-at-unix-ms and DYN_EXPERIMENT_STATIC_KV_TIMED_END_UNIX_MS = start + duration + 1-10 s",
        },
    });
    println!("{}", serde_json::to_string_pretty(&plan)?);
    if let Some(required) = required_s
        && coverage_s < required
    {
        bail!(
            "the timed section covers {coverage_s:.1} s at speedup {speedup:.3}, less than the required {required:.1} s; capture a longer corpus, lower the speedup, or add phantoms"
        );
    }

    if !eviction.steady() {
        let message = format!(
            "{} of {} streams remove fewer than {} of the blocks they store in the timed section (aggregate {:.3}): the capture had not reached steady-state eviction; capture with a smaller --num-gpu-blocks or a longer --agentic-warmup-sim-ms",
            eviction.bases_below_min.len(),
            eviction.rows.len(),
            args.min_timed_remove_ratio,
            eviction.timed_remove_ratio
        );
        if !args.allow_low_eviction {
            bail!(message);
        }
        eprintln!("warning: {message}; continuing because of --allow-low-eviction");
    }
    if let Some(path) = &args.indexer_env_out {
        let mut env = format!(
            "# EXPERIMENT ONLY: generated by phantom_plan; source this same file in both arms.\nexport DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB={}\n",
            sockets.endpoints_per_sub
        );
        if let Some(sources) = &args.sources_out {
            let sources = std::path::absolute(sources)?;
            env.push_str(&format!(
                "export DYN_EXPERIMENT_STATIC_KV_SOURCES=@{}\n",
                sources.display()
            ));
        }
        std::fs::write(path, env).with_context(|| format!("writing {}", path.display()))?;
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
                "args": publisher_args(&layout, first, count, host, port, speedup, &args),
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
