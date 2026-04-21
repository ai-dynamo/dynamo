// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;

#[derive(Debug, Deserialize)]
struct RawTraceRecord {
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    created_time: Option<f64>,
    #[serde(default, alias = "input_tokens")]
    input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    output_length: Option<usize>,
    #[serde(default)]
    hash_ids: Option<Vec<u64>>,
}

pub(super) fn load_trace_requests(
    trace_path: &Path,
    trace_block_size: usize,
    timestamps_required: bool,
) -> Result<Vec<DirectRequest>> {
    let file = File::open(trace_path)
        .with_context(|| format!("failed to open trace file {}", trace_path.display()))?;
    let reader = BufReader::new(file);
    let mut requests = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_idx + 1,
                trace_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let raw: RawTraceRecord = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse line {} from {} as JSON",
                line_idx + 1,
                trace_path.display()
            )
        })?;

        let input_length = raw
            .input_length
            .ok_or_else(|| anyhow!("trace line {} is missing input_length", line_idx + 1))?;
        let output_length = raw
            .output_length
            .ok_or_else(|| anyhow!("trace line {} is missing output_length", line_idx + 1))?;
        let hash_ids = raw
            .hash_ids
            .ok_or_else(|| anyhow!("trace line {} is missing hash_ids", line_idx + 1))?;
        let arrival_timestamp_ms = if timestamps_required {
            match raw.timestamp.or(raw.created_time) {
                Some(timestamp_ms) => Some(timestamp_ms),
                None => return Err(anyhow!("trace line {} is missing timestamp", line_idx + 1)),
            }
        } else {
            None
        };
        let tokens = synthesize_tokens_from_hash_ids(&hash_ids, input_length, trace_block_size)
            .with_context(|| {
                format!(
                    "failed to synthesize tokens from hash_ids on line {}",
                    line_idx + 1
                )
            })?;

        requests.push(DirectRequest {
            tokens,
            max_output_tokens: output_length,
            uuid: Some(Uuid::new_v4()),
            dp_rank: 0,
            arrival_timestamp_ms,
        });
    }

    if requests.is_empty() {
        bail!(
            "trace file {} did not contain any requests",
            trace_path.display()
        );
    }

    Ok(requests)
}

fn synthesize_tokens_from_hash_ids(
    hash_ids: &[u64],
    input_length: usize,
    trace_block_size: usize,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(input_length);

    for &hash_id in hash_ids {
        let token_id = u32::try_from(hash_id)
            .map_err(|_| anyhow!("hash_id {hash_id} exceeds u32::MAX for token synthesis"))?;
        // TODO: Replace this repeated-token expansion with a hash-native prompt representation.
        tokens.extend((0..trace_block_size).map(|_| token_id));
        if tokens.len() >= input_length {
            tokens.truncate(input_length);
            return Ok(tokens);
        }
    }

    bail!(
        "input_length {} exceeds synthesized capacity {} from {} hash_ids and block_size {}",
        input_length,
        hash_ids.len() * trace_block_size,
        hash_ids.len(),
        trace_block_size
    );
}
