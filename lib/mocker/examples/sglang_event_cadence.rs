// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Campaign scratch: replay a Mooncake-style request schedule through the SGLang-mode mocker
//! (offline, one worker) and summarize its KV-event cadence: blocks per Stored/Removed event,
//! stored blocks per request split into uncached-prompt vs output, and events per request.
//!
//! Usage: sglang_event_cadence <trace.jsonl> <trace_block_size> <num_gpu_blocks> <page_size>

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, SglangArgs};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::generate_trace_worker_artifacts_offline;

fn pct(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((q * sorted.len() as f64) as usize).min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "usage: {} <trace.jsonl> <trace_block_size> <num_gpu_blocks> <page_size>", args[0]);
    let trace_block_size: usize = args[2].parse()?;
    let num_gpu_blocks: usize = args[3].parse()?;
    let page_size: usize = args[4].parse()?;
    let trace = Trace::from_mooncake(Path::new(&args[1]), trace_block_size).context("load trace")?;
    let engine_args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .sglang(Some(SglangArgs {
            page_size: Some(page_size),
            ..SglangArgs::default()
        }))
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(page_size)
        .enable_prefix_caching(true)
        .max_num_batched_tokens(None)
        .max_num_seqs(None)
        .build()?;
    let started = std::time::Instant::now();
    let artifacts = generate_trace_worker_artifacts_offline(engine_args, trace)?;
    let wall = started.elapsed().as_secs_f64();

    let mut stored_sizes = Vec::new();
    let mut removed_sizes = Vec::new();
    let mut cleared = 0usize;
    for event in &artifacts.kv_events {
        match &event.event.data {
            KvCacheEventData::Stored(data) => stored_sizes.push(data.blocks.len()),
            KvCacheEventData::Removed(data) => removed_sizes.push(data.block_hashes.len()),
            KvCacheEventData::Cleared => cleared += 1,
        }
    }
    let n = artifacts.requests.len();
    let input: usize = artifacts.requests.iter().map(|r| r.input_length).sum();
    let mut cached: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut output_tokens = 0usize;
    let mut rejected = 0usize;
    for s in &artifacts.output_signals {
        if let Some(c) = s.signal.cached_tokens {
            cached.entry(s.signal.uuid).or_insert(c as usize);
        }
        if s.signal.token_id.is_some() {
            output_tokens += 1;
        }
        if s.signal.rejected {
            rejected += 1;
        }
    }
    let cached_total: usize = cached.values().sum();
    let stored_blocks: usize = stored_sizes.iter().sum();
    let removed_blocks: usize = removed_sizes.iter().sum();
    let one_block = stored_sizes.iter().filter(|&&s| s == 1).count();
    // Histogram buckets of blocks per Stored event: 1, 2-8, 9-64, 65-1024, 1025-8192, >8192.
    let bounds = [1usize, 8, 64, 1024, 8192, usize::MAX];
    let mut hist_events = [0usize; 6];
    let mut hist_blocks = [0usize; 6];
    for &s in &stored_sizes {
        let b = bounds.iter().position(|&ub| s <= ub).unwrap();
        hist_events[b] += 1;
        hist_blocks[b] += s;
    }
    let t_first = artifacts.requests.iter().map(|r| r.timestamp_us).min().unwrap_or(0);
    let t_last = artifacts
        .kv_events
        .iter()
        .map(|e| e.timestamp_us)
        .chain(artifacts.output_signals.iter().map(|s| s.timestamp_us))
        .max()
        .unwrap_or(0);
    let sim_s = (t_last.saturating_sub(t_first)) as f64 / 1e6;
    let uncached = input.saturating_sub(cached_total);
    let page_blocks = |tokens: usize| tokens as f64 / page_size as f64;
    let mut ss = stored_sizes.clone();
    ss.sort_unstable();
    let mut rs = removed_sizes.clone();
    rs.sort_unstable();
    let mean = |v: &[usize]| if v.is_empty() { 0.0 } else { v.iter().sum::<usize>() as f64 / v.len() as f64 };
    let summary = serde_json::json!({
        "wall_s": wall,
        "sim_s": sim_s,
        "requests": n,
        "rejected": rejected,
        "page_size": page_size,
        "num_gpu_blocks": num_gpu_blocks,
        "stored_events": stored_sizes.len(),
        "removed_events": removed_sizes.len(),
        "cleared_events": cleared,
        "stored_blocks": stored_blocks,
        "removed_blocks": removed_blocks,
        "stored_blocks_per_event": {"mean": mean(&stored_sizes), "p10": pct(&ss, 0.1), "p50": pct(&ss, 0.5), "p90": pct(&ss, 0.9), "p99": pct(&ss, 0.99), "max": ss.last().copied().unwrap_or(0), "frac_events_1_block": one_block as f64 / stored_sizes.len().max(1) as f64},
        "removed_blocks_per_event": {"mean": mean(&removed_sizes), "p50": pct(&rs, 0.5), "p99": pct(&rs, 0.99), "max": rs.last().copied().unwrap_or(0)},
        "totals": {"prompt_tokens": input, "cached_tokens": cached_total, "uncached_prompt_tokens": uncached, "completion_tokens": output_tokens},
        "per_request": {
            "prompt_tokens": input as f64 / n as f64,
            "cached_tokens": cached_total as f64 / n as f64,
            "uncached_prompt_tokens": uncached as f64 / n as f64,
            "completion_tokens": output_tokens as f64 / n as f64,
            "stored_events": stored_sizes.len() as f64 / n as f64,
            "stored_blocks": stored_blocks as f64 / n as f64,
            "accounting_prompt_blocks": page_blocks(uncached) / n as f64,
            "accounting_output_blocks": (stored_blocks as f64 - page_blocks(uncached)) / n as f64,
            "removed_events": removed_sizes.len() as f64 / n as f64,
            "removed_blocks": removed_blocks as f64 / n as f64,
            "lookup_blocks": page_blocks(input) / n as f64,
        },
        "stored_size_hist": {"buckets": ["1", "2-8", "9-64", "65-1024", "1025-8192", ">8192"], "events": hist_events, "blocks": hist_blocks},
        "ratios": {
            "stored_events_per_completion_token": stored_sizes.len() as f64 / output_tokens.max(1) as f64,
            "one_block_stored_events_per_completion_token": one_block as f64 / output_tokens.max(1) as f64,
        },
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
