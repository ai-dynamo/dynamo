// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufRead, BufReader};

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::XXH3_SEED;
use dynamo_mocker::loadgen::{SessionPartitionSpec, Trace};
use dynamo_tokens::compute_hash_v2;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::make_progress_bar;

/// Load, transform, and partition the mooncake trace into per-worker request lists.
pub fn process_mooncake_trace(
    path: &str,
    block_size: u32,
    trace_length_factor: usize,
    trace_duplication_factor: usize,
    num_workers: usize,
    seed: u64,
) -> anyhow::Result<Vec<Trace>> {
    let trace = Trace::from_mooncake(std::path::Path::new(path), block_size as usize)?
        .expand_hash_prefix_depth(trace_length_factor)
        .duplicate_hash_space(trace_duplication_factor);
    Ok(trace.partition_by_session(SessionPartitionSpec::Random {
        num_partitions: num_workers,
        seed,
    }))
}

/// A single request deserialized from the mooncake trace JSONL.
#[derive(Serialize, Deserialize, Clone)]
pub struct MooncakeRequest {
    #[serde(default = "Uuid::new_v4")]
    pub uuid: uuid::Uuid,
    pub timestamp: u64,
    #[serde(default)]
    pub input_length: usize,
    pub hash_ids: Vec<u64>,
    #[serde(alias = "output_length", alias = "osl")]
    pub output_length: u64,
}

#[derive(Deserialize)]
struct RawMooncakeRecord {
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    delay: Option<f64>,
    hash_ids: Vec<u64>,
    #[serde(alias = "output_length", alias = "osl")]
    output_length: u64,
}

/// Load the mooncake trace from disk into a flat list of requests.
///
/// Supports two JSONL formats:
///   - Legacy: every record has an integer `timestamp` field (absolute ms).
///   - aiperf: first record has `timestamp` (float), subsequent records have
///     `delay` (float ms since previous). Absolute timestamps are reconstructed
///     by accumulating delays.
pub fn load_mooncake_trace(path: &str) -> anyhow::Result<Vec<MooncakeRequest>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    println!("Loading trace...");
    let progress = make_progress_bar(None);

    let mut requests = Vec::new();
    let mut cursor_ms: f64 = 0.0;

    for line in reader.lines() {
        let raw: RawMooncakeRecord = serde_json::from_str(&line?)?;

        if let Some(ts) = raw.timestamp {
            cursor_ms = ts;
        } else if let Some(d) = raw.delay {
            cursor_ms += d;
        }

        requests.push(MooncakeRequest {
            uuid: Uuid::new_v4(),
            timestamp: cursor_ms as u64,
            input_length: 0,
            hash_ids: raw.hash_ids,
            output_length: raw.output_length,
        });
        progress.inc(1);
    }

    Ok(requests)
}

/// Randomly partition a flat request list across `num_workers` worker buckets.
pub fn partition_trace(
    requests: Vec<MooncakeRequest>,
    num_workers: usize,
    seed: u64,
) -> Vec<Vec<MooncakeRequest>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut traces: Vec<Vec<MooncakeRequest>> = (0..num_workers).map(|_| Vec::new()).collect();
    for request in requests {
        traces[rng.random_range(0..num_workers)].push(request);
    }

    for trace in &mut traces {
        trace.sort_by_key(|r| r.timestamp);
    }
    traces
}

/// Linearly rescale all timestamps in a worker's trace so the total span equals
/// `duration` milliseconds.
pub fn scale_mooncake_trace(trace: &[MooncakeRequest], duration: u64) -> Vec<MooncakeRequest> {
    let Some(first) = trace.first() else {
        return Vec::new();
    };
    let total_duration = trace.last().unwrap().timestamp - first.timestamp;
    if total_duration == 0 {
        return trace
            .iter()
            .map(|r| MooncakeRequest {
                timestamp: 0,
                ..r.clone()
            })
            .collect();
    }
    trace
        .iter()
        .map(|request| MooncakeRequest {
            timestamp: (request.timestamp - first.timestamp) * duration / total_duration,
            ..request.clone()
        })
        .collect()
}

/// Stretch each request's hash sequence by the given factor, simulating longer
/// prefix chains with the same tree structure.
pub fn expand_trace_lengths(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }

    println!("Expanding trace lengths by {}x", factor);

    requests
        .into_iter()
        .map(|mut request| {
            request.hash_ids = request
                .hash_ids
                .iter()
                .flat_map(|&h| {
                    let base = h * factor as u64;
                    (0..factor as u64).map(move |offset| base + offset)
                })
                .collect();
            request
        })
        .collect()
}

/// Duplicate all worker traces with offset hash_ids, creating `factor`
/// structurally identical copies of the prefix tree with disjoint hash spaces.
pub fn duplicate_traces(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }

    let max_hash_id = requests
        .iter()
        .flat_map(|r| r.hash_ids.iter().copied())
        .max()
        .unwrap_or(0);
    let offset_base = max_hash_id + 1;

    println!(
        "Duplicating traces: {}x (hash offset base: {})",
        factor, offset_base
    );

    let mut out = Vec::with_capacity(requests.len() * factor);
    for r in &requests {
        for d in 0..factor {
            let offset = offset_base * d as u64;
            out.push(MooncakeRequest {
                uuid: Uuid::new_v4(),
                hash_ids: r.hash_ids.iter().map(|&h| h + offset).collect(),
                ..r.clone()
            });
        }
    }
    out
}

/// Expand a request's block-level hash_ids into per-token IDs by repeating each
/// hash_id `block_size` times.
pub fn tokens_from_request(request: &MooncakeRequest, block_size: u32) -> Vec<u32> {
    let mut tokens = request
        .hash_ids
        .iter()
        .flat_map(|id| (0..block_size).map(|_| *id as u32))
        .collect::<Vec<_>>();
    if request.input_length > 0 && request.input_length < tokens.len() {
        tokens.truncate(request.input_length);
    }
    tokens
}

/// Compute the LocalBlockHash for a block-level hash_id the same way the mock
/// engine does: expand to `block_size` repeated u32 tokens, then XXH3 hash.
pub fn local_block_hash_from_id(hash_id: u64, block_size: u32) -> LocalBlockHash {
    let tokens: Vec<u32> = (0..block_size).map(|_| hash_id as u32).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
    LocalBlockHash(compute_hash_v2(bytes, XXH3_SEED))
}
