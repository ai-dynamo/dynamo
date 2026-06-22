// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake JSONL primitives.
//!
//! This module is producer- and consumer-agnostic: it defines the row schema,
//! the block-hash-to-id mapping, the token-block hashing helper, and the JSONL
//! writer. Workload-specific orchestration (session scheduling, tokenization,
//! parsing) lives elsewhere -- the Claude exporter in `dynamo-bench` is one
//! such producer; the `dynamo-mocker` load generator is one such consumer.
//!
//! The [`MooncakeRow`] schema deliberately matches the externally-authored
//! Mooncake trace format: `timestamp` and `delay` are `f64` milliseconds, and
//! `input_length`/`output_length`/`timestamp`/`delay` accept the upstream
//! aliases (`input_tokens`, `output_tokens`, `created_time`, `delay_ms`) on
//! deserialization. Dynamo-produced traces always emit the canonical names.

use anyhow::{Context, Result, bail};
use dynamo_kv_hashing::{Request, compute_hash_v2, compute_next_sequence_hash};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One row of a Mooncake replay trace.
///
/// `timestamp` is an absolute request arrival offset in milliseconds. Rows
/// without a `session_id` are independent request arrivals. Rows that share a
/// `session_id` are interpreted as closed-loop turns; later turns use `delay`
/// or timestamp deltas relative to the previous row in that session.
///
/// The row type is `Serialize + Deserialize` so the same definition serves
/// producers and consumers. Field-level aliases on deserialization accept the
/// upstream Mooncake field names (`input_tokens`, `output_tokens`,
/// `created_time`, `delay_ms`) without requiring producers to emit them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeRow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "created_time"
    )]
    pub timestamp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "delay_ms")]
    pub delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
}

/// One row of an agentic Mooncake replay trace.
///
/// This format keeps the request/cache fields from [`MooncakeRow`] and adds a
/// tiny workflow layer above them. `request_id` names the row. `wait_for` names
/// request ids whose simulated completions must arrive before this row becomes
/// eligible. Once all dependencies are satisfied, replay waits `delay` plus
/// `tool_wait_ms` before dispatching the request. Rows with no dependencies
/// use `timestamp` as their open-loop start time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticMooncakeRow {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "created_time"
    )]
    pub timestamp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "delay_ms")]
    pub delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wait_for: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_reset: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_wait_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_events: Vec<AgenticToolEvent>,
}

impl AgenticMooncakeRow {
    /// Return the total wait after all dependencies complete.
    pub fn dependency_delay_ms(&self) -> f64 {
        self.delay.unwrap_or(0.0) + self.tool_wait_ms.unwrap_or(0.0)
    }
}

/// Harness tool span attributed to the LLM request that consumed it. Mirrors
/// `tool_end` / `tool_error` fields from `dynamo.request.trace.v1`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgenticToolEvent {
    pub tool_call_id: String,
    pub tool_class: String,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: u64,
    pub duration_ms: f64,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
}

/// Maps sequence-aware block hashes to compact, stable `u64` ids.
///
/// The mapper is intentionally stateful and reusable across requests/turns: a
/// block of tokens that appears at the same prefix position in two different
/// requests will be assigned the same id. Equality of leading `hash_ids`
/// between rows therefore signals shared prompt prefixes for replay purposes.
///
/// `hash_ids` here are workload identity labels, not literal Dynamo runtime
/// KV-cache hashes. Producers should not try to reconcile them with a
/// production cache.
pub struct RollingHashIdMapper {
    block_size: usize,
    hash_to_id: FxHashMap<u64, u64>,
    next_id: u64,
}

impl RollingHashIdMapper {
    /// Create a new mapper for the given block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            hash_to_id: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Block size that this mapper was constructed with.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Hash a sequence of tokens into Mooncake `hash_ids`.
    ///
    /// Tokens are chunked by `block_size`; each complete block contributes one
    /// compact id derived from Dynamo's shared KV-hashing contract. A trailing
    /// partial block also contributes one compact id so replay capacity still
    /// covers the full prompt length. Identical prefixes across requests
    /// resolve to identical leading `hash_ids` once the mapper has seen them.
    pub fn hash_token_blocks(&mut self, tokens: &[u32]) -> Vec<u64> {
        hash_token_blocks(self, tokens)
    }

    /// Fallible variant of [`Self::hash_token_blocks`].
    pub fn try_hash_token_blocks(&mut self, tokens: &[u32]) -> Result<Vec<u64>> {
        try_hash_token_blocks(self, tokens)
    }

    /// Map precomputed sequence-aware block hashes into compact Mooncake IDs.
    ///
    /// This is useful for producers that record stable block hashes in the
    /// serving path and only compact them during offline trace conversion.
    pub fn ids_for_sequence_hashes(&mut self, sequence_hashes: &[u64]) -> Vec<u64> {
        ids_for_sequence_hashes(self, sequence_hashes)
    }
}

/// Token-block hashing helper for the Mooncake replay schema.
///
/// Splits `tokens` into chunks of `mapper.block_size()`, derives sequence-aware
/// hashes for complete blocks through `dynamo-kv-hashing`, appends a sequence
/// hash for a trailing partial block when present, and returns the
/// compact ids assigned by `mapper`. Mirrors
/// [`RollingHashIdMapper::hash_token_blocks`] as a free function so callers
/// that already hold a mutable mapper reference can invoke it without
/// re-borrowing.
pub fn hash_token_blocks(mapper: &mut RollingHashIdMapper, tokens: &[u32]) -> Vec<u64> {
    try_hash_token_blocks(mapper, tokens).expect("Mooncake token-block hashing failed")
}

/// Fallible token-block hashing helper for callers that want to surface
/// invalid block-size or request-shape errors.
pub fn try_hash_token_blocks(mapper: &mut RollingHashIdMapper, tokens: &[u32]) -> Result<Vec<u64>> {
    require_positive("block size", mapper.block_size)?;
    let block_size: u32 = mapper
        .block_size
        .try_into()
        .context("block_size does not fit u32")?;
    let request = Request::builder().tokens(tokens.to_vec()).build()?;
    let salt_hash = request.salt_hash()?;
    let mut sequence_hashes = request.into_sequence_hashes(block_size)?;
    if let Some(partial_hash) =
        trailing_partial_sequence_hash(salt_hash, mapper.block_size, tokens, &sequence_hashes)
    {
        sequence_hashes.push(partial_hash);
    }
    Ok(ids_for_sequence_hashes(mapper, &sequence_hashes))
}

fn trailing_partial_sequence_hash(
    salt_hash: u64,
    block_size: usize,
    tokens: &[u32],
    complete_sequence_hashes: &[u64],
) -> Option<u64> {
    let tail_len = tokens.len() % block_size;
    if tail_len == 0 {
        return None;
    }

    let tail = &tokens[tokens.len() - tail_len..];
    let mut tail_bytes = Vec::with_capacity(std::mem::size_of_val(tail));
    for token in tail {
        tail_bytes.extend_from_slice(&token.to_ne_bytes());
    }
    let tail_block_hash = compute_hash_v2(&tail_bytes, salt_hash);
    Some(match complete_sequence_hashes.last().copied() {
        Some(parent) => compute_next_sequence_hash(parent, tail_block_hash),
        None => tail_block_hash,
    })
}

/// Map stable sequence hashes to compact Mooncake IDs with a shared mapper.
pub fn ids_for_sequence_hashes(
    mapper: &mut RollingHashIdMapper,
    sequence_hashes: &[u64],
) -> Vec<u64> {
    sequence_hashes
        .iter()
        .map(|sequence_hash| {
            *mapper.hash_to_id.entry(*sequence_hash).or_insert_with(|| {
                let next_id = mapper.next_id;
                mapper.next_id += 1;
                next_id
            })
        })
        .collect()
}

/// Counters for what a [`MooncakeJsonlWriter`] has emitted.
#[derive(Debug, Clone, Copy, Default)]
pub struct WriterStats {
    pub row_count: usize,
    pub sidecar_count: usize,
}

/// JSONL writer for Mooncake rows plus an optional sidecar stream.
///
/// The sidecar stream is configured at construction time. Producers that do
/// not emit sidecar metadata pass `None` for `sidecar_path` and never call
/// [`Self::write_sidecar`]. When a sidecar path is configured, callers are
/// responsible for choosing the path -- this writer does not enforce a naming
/// convention.
pub struct MooncakeJsonlWriter {
    output: BufWriter<File>,
    sidecar: Option<BufWriter<File>>,
    stats: WriterStats,
}

impl MooncakeJsonlWriter {
    /// Create a writer at `output_path`, optionally with a paired sidecar
    /// JSONL file at `sidecar_path`. Parent directories are created as needed.
    pub fn create(output_path: &Path, sidecar_path: Option<&Path>) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let output = BufWriter::new(
            File::create(output_path)
                .with_context(|| format!("failed to create {}", output_path.display()))?,
        );
        let sidecar = if let Some(path) = sidecar_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Some(BufWriter::new(File::create(path).with_context(|| {
                format!("failed to create {}", path.display())
            })?))
        } else {
            None
        };
        Ok(Self {
            output,
            sidecar,
            stats: WriterStats::default(),
        })
    }

    /// Append one Mooncake row.
    pub fn write_row(&mut self, row: &MooncakeRow) -> Result<()> {
        serde_json::to_writer(&mut self.output, row)?;
        self.output.write_all(b"\n")?;
        self.stats.row_count += 1;
        Ok(())
    }

    /// Append one agentic Mooncake row.
    pub fn write_agentic_row(&mut self, row: &AgenticMooncakeRow) -> Result<()> {
        serde_json::to_writer(&mut self.output, row)?;
        self.output.write_all(b"\n")?;
        self.stats.row_count += 1;
        Ok(())
    }

    /// Append one sidecar entry. Errors if no sidecar was configured.
    pub fn write_sidecar<S: Serialize>(&mut self, sidecar: &S) -> Result<()> {
        let writer = self
            .sidecar
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("sidecar was not configured for this writer"))?;
        serde_json::to_writer(writer, sidecar)?;
        let writer = self.sidecar.as_mut().unwrap();
        writer.write_all(b"\n")?;
        self.stats.sidecar_count += 1;
        Ok(())
    }

    /// True if a sidecar stream is configured.
    pub fn has_sidecar(&self) -> bool {
        self.sidecar.is_some()
    }

    /// Snapshot of how many rows and sidecar entries have been written so far.
    pub fn stats(&self) -> WriterStats {
        self.stats
    }

    /// Flush both streams and return the final stats.
    pub fn finish(mut self) -> Result<WriterStats> {
        self.output.flush()?;
        if let Some(sidecar) = self.sidecar.as_mut() {
            sidecar.flush()?;
        }
        Ok(self.stats)
    }
}

/// Create both files empty (touch-equivalent), preserving directory creation
/// semantics for callers that want a "no rows produced" outcome to still emit
/// well-formed (empty) JSONL files.
pub fn write_empty_files(output_path: &Path, sidecar_path: Option<&Path>) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    if let Some(path) = sidecar_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    }
    Ok(())
}

/// Sentinel used by callers that want to bail when neither block_size nor
/// worker count is allowed to be zero. Producers may also enforce this on
/// their own configuration types.
pub fn require_positive(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        bail!("{name} must be greater than 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use tempfile::TempDir;

    #[test]
    fn shared_prefix_yields_shared_leading_hash_ids() {
        let mut mapper = RollingHashIdMapper::new(2);
        let prefix = vec![1u32, 2, 3, 4];
        let extended = vec![1u32, 2, 3, 4, 5, 6];

        let prefix_ids = mapper.hash_token_blocks(&prefix);
        let extended_ids = mapper.hash_token_blocks(&extended);

        assert_eq!(prefix_ids.len(), 2);
        assert_eq!(extended_ids.len(), 3);
        assert_eq!(extended_ids[..2], prefix_ids[..]);
    }

    #[test]
    fn mapper_state_is_reused_across_requests() {
        let mut mapper = RollingHashIdMapper::new(4);
        let request_a = vec![10u32, 20, 30, 40, 50, 60, 70, 80];
        let request_b = vec![10u32, 20, 30, 40, 50, 60, 70, 80];
        let request_c = vec![10u32, 20, 30, 40, 99, 99, 99, 99];

        let ids_a = mapper.hash_token_blocks(&request_a);
        let ids_b = mapper.hash_token_blocks(&request_b);
        let ids_c = mapper.hash_token_blocks(&request_c);

        assert_eq!(ids_a, ids_b);
        assert_eq!(ids_c[0], ids_a[0], "shared first block should keep its id");
        assert_ne!(
            ids_c[1], ids_a[1],
            "diverging tail block must get a fresh id"
        );
    }

    #[test]
    fn free_function_and_method_agree() {
        let mut mapper_a = RollingHashIdMapper::new(2);
        let mut mapper_b = RollingHashIdMapper::new(2);
        let tokens = vec![7u32, 8, 9, 10, 11];

        let via_method = mapper_a.hash_token_blocks(&tokens);
        let via_function = hash_token_blocks(&mut mapper_b, &tokens);

        assert_eq!(via_method, via_function);
    }

    #[test]
    fn token_blocks_derive_ids_from_shared_kv_hashing_contract() {
        let tokens = vec![7u32, 8, 9, 10, 11, 12, 13, 14];
        let request = Request::builder().tokens(tokens.clone()).build().unwrap();
        let sequence_hashes = request.sequence_hashes(4).unwrap();

        let mut token_mapper = RollingHashIdMapper::new(4);
        let mut sequence_mapper = RollingHashIdMapper::new(4);

        let token_ids = token_mapper.hash_token_blocks(&tokens);
        let sequence_ids = sequence_mapper.ids_for_sequence_hashes(&sequence_hashes);

        assert_eq!(token_ids, sequence_ids);
    }

    #[test]
    fn empty_token_input_yields_empty_hash_ids() {
        let mut mapper = RollingHashIdMapper::new(4);
        assert!(mapper.hash_token_blocks(&[]).is_empty());
    }

    #[test]
    fn trailing_partial_block_preserves_replay_capacity() {
        let mut mapper = RollingHashIdMapper::new(4);

        assert_eq!(mapper.hash_token_blocks(&[1, 2, 3]), vec![0]);
        assert_eq!(mapper.hash_token_blocks(&[1, 2, 3, 4, 5, 6]), vec![1, 2]);
    }

    #[test]
    fn trailing_partial_block_uses_shared_chain_contract() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6];
        let request = Request::builder().tokens(tokens.clone()).build().unwrap();
        let complete_sequence_hashes = request.sequence_hashes(4).unwrap();
        let mut tail_bytes = Vec::new();
        for token in &tokens[4..] {
            tail_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        let tail_block_hash = compute_hash_v2(&tail_bytes, request.salt_hash().unwrap());
        let expected_tail_hash =
            compute_next_sequence_hash(complete_sequence_hashes[0], tail_block_hash);

        let mut token_mapper = RollingHashIdMapper::new(4);
        let mut sequence_mapper = RollingHashIdMapper::new(4);
        let token_ids = token_mapper.hash_token_blocks(&tokens);
        let mut expected_hashes = complete_sequence_hashes;
        expected_hashes.push(expected_tail_hash);
        let sequence_ids = sequence_mapper.ids_for_sequence_hashes(&expected_hashes);

        assert_eq!(token_ids, sequence_ids);
    }

    #[test]
    fn exact_block_boundary_does_not_add_partial_hash_id() {
        let mut mapper = RollingHashIdMapper::new(4);

        assert_eq!(mapper.hash_token_blocks(&[1, 2, 3, 4]), vec![0]);
        assert_eq!(
            mapper.hash_token_blocks(&[1, 2, 3, 4, 5, 6, 7, 8]),
            vec![0, 1]
        );
    }

    #[test]
    fn try_hash_token_blocks_rejects_zero_block_size() {
        let mut mapper = RollingHashIdMapper::new(0);
        let err = mapper.try_hash_token_blocks(&[1, 2, 3]).unwrap_err();

        assert!(err.to_string().contains("block size"));
    }

    #[test]
    fn precomputed_sequence_hashes_map_to_stable_ids() {
        let mut mapper = RollingHashIdMapper::new(64);

        let first = mapper.ids_for_sequence_hashes(&[101, 202, 303]);
        let second = mapper.ids_for_sequence_hashes(&[101, 202, 404]);

        assert_eq!(first[..2], second[..2]);
        assert_ne!(first[2], second[2]);
    }

    #[test]
    fn row_omits_timestamp_and_delay_when_absent() {
        let row = MooncakeRow {
            session_id: Some("s".to_string()),
            input_length: Some(4),
            output_length: Some(1),
            hash_ids: Some(vec![0, 1]),
            timestamp: None,
            delay: None,
            ..Default::default()
        };
        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert!(rendered.get("timestamp").is_none());
        assert!(rendered.get("delay").is_none());
        assert_eq!(rendered["hash_ids"], json!([0, 1]));
    }

    #[test]
    fn row_serializes_optional_fields_when_set() {
        let with_timestamp = MooncakeRow {
            session_id: Some("s".to_string()),
            input_length: Some(4),
            output_length: Some(1),
            hash_ids: Some(vec![]),
            timestamp: Some(0.0),
            delay: None,
            ..Default::default()
        };
        let with_delay = MooncakeRow {
            session_id: Some("s".to_string()),
            input_length: Some(4),
            output_length: Some(1),
            hash_ids: Some(vec![]),
            timestamp: None,
            delay: Some(123.0),
            ..Default::default()
        };
        let v_ts: Value = serde_json::to_value(&with_timestamp).unwrap();
        let v_dl: Value = serde_json::to_value(&with_delay).unwrap();
        assert_eq!(v_ts["timestamp"], json!(0.0));
        assert!(v_ts.get("delay").is_none());
        assert_eq!(v_dl["delay"], json!(123.0));
        assert!(v_dl.get("timestamp").is_none());
    }

    #[test]
    fn row_deserializes_canonical_field_names() {
        let raw = r#"{"session_id":"s","input_length":4,"output_length":1,"hash_ids":[0,1],"timestamp":12.5,"delay":3.0}"#;
        let row: MooncakeRow = serde_json::from_str(raw).unwrap();
        assert_eq!(row.session_id.as_deref(), Some("s"));
        assert_eq!(row.input_length, Some(4));
        assert_eq!(row.output_length, Some(1));
        assert_eq!(row.hash_ids, Some(vec![0, 1]));
        assert_eq!(row.timestamp, Some(12.5));
        assert_eq!(row.delay, Some(3.0));
    }

    #[test]
    fn row_deserializes_upstream_mooncake_aliases() {
        let raw = r#"{"input_tokens":4,"output_tokens":1,"hash_ids":[0,1],"created_time":12.5,"delay_ms":3.0}"#;
        let row: MooncakeRow = serde_json::from_str(raw).unwrap();
        assert_eq!(row.input_length, Some(4));
        assert_eq!(row.output_length, Some(1));
        assert_eq!(row.timestamp, Some(12.5));
        assert_eq!(row.delay, Some(3.0));
    }

    #[test]
    fn row_alias_input_round_trips_to_canonical_fields() {
        let raw = r#"{"input_tokens":8,"output_tokens":2,"created_time":12.5,"delay_ms":3.0}"#;
        let mut row: MooncakeRow = serde_json::from_str(raw).unwrap();

        let tokens: Vec<u32> = (0..row.input_length.unwrap() as u32).collect();
        let mut mapper = RollingHashIdMapper::new(4);
        row.hash_ids = Some(mapper.hash_token_blocks(&tokens));

        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert_eq!(rendered["input_length"], json!(8));
        assert_eq!(rendered["output_length"], json!(2));
        assert_eq!(rendered["timestamp"], json!(12.5));
        assert_eq!(rendered["delay"], json!(3.0));
        assert_eq!(rendered["hash_ids"], json!([0, 1]));
        assert!(rendered.get("input_tokens").is_none());
        assert!(rendered.get("output_tokens").is_none());
        assert!(rendered.get("created_time").is_none());
        assert!(rendered.get("delay_ms").is_none());
    }

    #[test]
    fn row_canonical_input_round_trips_without_renaming() {
        let raw = r#"{"input_length":8,"output_length":2,"timestamp":12.5,"delay":3.0}"#;
        let mut row: MooncakeRow = serde_json::from_str(raw).unwrap();

        let tokens: Vec<u32> = (0..row.input_length.unwrap() as u32).collect();
        let mut mapper = RollingHashIdMapper::new(4);
        row.hash_ids = Some(mapper.hash_token_blocks(&tokens));

        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert_eq!(rendered["input_length"], json!(8));
        assert_eq!(rendered["output_length"], json!(2));
        assert_eq!(rendered["timestamp"], json!(12.5));
        assert_eq!(rendered["delay"], json!(3.0));
        assert_eq!(rendered["hash_ids"], json!([0, 1]));
        assert!(rendered.get("input_tokens").is_none());
        assert!(rendered.get("output_tokens").is_none());
        assert!(rendered.get("created_time").is_none());
        assert!(rendered.get("delay_ms").is_none());
    }

    #[test]
    fn row_deserializes_with_missing_optional_fields() {
        let raw = r#"{"output_length":2}"#;
        let row: MooncakeRow = serde_json::from_str(raw).unwrap();
        assert_eq!(row.session_id, None);
        assert_eq!(row.input_length, None);
        assert_eq!(row.output_length, Some(2));
        assert_eq!(row.hash_ids, None);
        assert_eq!(row.timestamp, None);
        assert_eq!(row.delay, None);
        assert_eq!(row.priority, None);
        assert_eq!(row.strict_priority, None);
        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert!(rendered.get("priority").is_none());
        assert!(rendered.get("strict_priority").is_none());
    }

    #[test]
    fn row_round_trips_priorities() {
        for priority in [Some(7), Some(0), Some(-3)] {
            let raw = json!({
                "output_length": 2,
                "priority": priority,
                "strict_priority": 9
            });
            let row: MooncakeRow = serde_json::from_value(raw).unwrap();
            assert_eq!(row.priority, priority);
            assert_eq!(row.strict_priority, Some(9));

            let rendered: Value = serde_json::to_value(&row).unwrap();
            assert_eq!(rendered["priority"], json!(priority.unwrap()));
            assert_eq!(rendered["strict_priority"], json!(9));
        }
    }

    #[test]
    fn agentic_row_defaults_workflow_fields() {
        let raw = r#"{"request_id":"r1","input_length":4,"output_length":1,"hash_ids":[0,1],"timestamp":10.0}"#;
        let row: AgenticMooncakeRow = serde_json::from_str(raw).unwrap();

        assert_eq!(row.request_id, "r1");
        assert!(row.wait_for.is_empty());
        assert!(row.branches.is_empty());
        assert_eq!(row.prefix_reset, None);
        assert_eq!(row.dependency_delay_ms(), 0.0);
        assert_eq!(row.priority, None);
        assert_eq!(row.strict_priority, None);
        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert!(rendered.get("priority").is_none());
        assert!(rendered.get("strict_priority").is_none());
    }

    #[test]
    fn agentic_row_round_trips_priorities() {
        let raw = r#"{"request_id":"r1","priority":-2,"strict_priority":4}"#;
        let row: AgenticMooncakeRow = serde_json::from_str(raw).unwrap();
        assert_eq!(row.priority, Some(-2));
        assert_eq!(row.strict_priority, Some(4));

        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert_eq!(rendered["priority"], json!(-2));
        assert_eq!(rendered["strict_priority"], json!(4));
    }

    #[test]
    fn agentic_row_delay_includes_tool_wait() {
        let row = AgenticMooncakeRow {
            request_id: "r2".to_string(),
            session_id: Some("trajectory-a".to_string()),
            input_length: Some(4),
            output_length: Some(1),
            hash_ids: Some(vec![0, 1]),
            timestamp: Some(20.0),
            delay: Some(3.0),
            wait_for: vec!["r1".to_string()],
            branches: vec!["r3".to_string()],
            prefix_reset: Some(false),
            tool_wait_ms: Some(7.0),
            ..Default::default()
        };

        assert_eq!(row.dependency_delay_ms(), 10.0);
        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert_eq!(rendered["request_id"], json!("r2"));
        assert_eq!(rendered["wait_for"], json!(["r1"]));
        assert_eq!(rendered["branches"], json!(["r3"]));
        assert_eq!(rendered["tool_wait_ms"], json!(7.0));
        assert!(rendered.get("tool_events").is_none());
    }

    #[test]
    fn agentic_row_round_trips_tool_events() {
        let row = AgenticMooncakeRow {
            request_id: "r1".to_string(),
            session_id: Some("trajectory-a".to_string()),
            input_length: Some(4),
            output_length: Some(1),
            hash_ids: Some(vec![0, 1]),
            timestamp: Some(0.0),
            delay: Some(0.0),
            priority: Some(5),
            strict_priority: Some(6),
            prefix_reset: Some(true),
            tool_wait_ms: Some(8.0),
            tool_events: vec![AgenticToolEvent {
                tool_call_id: "call-1".to_string(),
                tool_class: "web_search".to_string(),
                started_at_unix_ms: 1_000,
                ended_at_unix_ms: 1_008,
                duration_ms: 8.0,
                status: "succeeded".to_string(),
                output_bytes: Some(512),
                output_tokens: None,
                error_type: None,
            }],
            ..Default::default()
        };

        let rendered = serde_json::to_string(&row).unwrap();
        let decoded: AgenticMooncakeRow = serde_json::from_str(&rendered).unwrap();
        assert_eq!(decoded.tool_events.len(), 1);
        assert_eq!(decoded.tool_events[0].tool_class, "web_search");
        assert_eq!(decoded.tool_events[0].output_bytes, Some(512));
        assert_eq!(decoded.priority, Some(5));
        assert_eq!(decoded.strict_priority, Some(6));
    }

    #[test]
    fn writer_writes_rows_and_sidecar_jsonl() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("trace.jsonl");
        let sidecar = temp.path().join("trace.sidecar.jsonl");

        let mut writer = MooncakeJsonlWriter::create(&output, Some(&sidecar)).unwrap();
        writer
            .write_row(&MooncakeRow {
                session_id: Some("s".to_string()),
                input_length: Some(2),
                output_length: Some(1),
                hash_ids: Some(vec![0]),
                timestamp: Some(0.0),
                delay: None,
                ..Default::default()
            })
            .unwrap();
        writer.write_sidecar(&json!({"k": "v"})).unwrap();
        let stats = writer.finish().unwrap();

        assert_eq!(stats.row_count, 1);
        assert_eq!(stats.sidecar_count, 1);

        let row_lines: Vec<Value> = std::fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let sidecar_lines: Vec<Value> = std::fs::read_to_string(&sidecar)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(row_lines.len(), 1);
        assert_eq!(sidecar_lines, vec![json!({"k": "v"})]);
        assert_eq!(row_lines[0]["session_id"], json!("s"));
        assert!(row_lines[0].get("delay").is_none());
    }

    #[test]
    fn writer_writes_agentic_rows() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("agentic.jsonl");
        let mut writer = MooncakeJsonlWriter::create(&output, None).unwrap();
        writer
            .write_agentic_row(&AgenticMooncakeRow {
                request_id: "r1".to_string(),
                session_id: None,
                input_length: Some(2),
                output_length: Some(1),
                hash_ids: Some(vec![0]),
                timestamp: Some(0.0),
                delay: None,
                prefix_reset: Some(true),
                ..Default::default()
            })
            .unwrap();
        let stats = writer.finish().unwrap();

        assert_eq!(stats.row_count, 1);
        let row_lines: Vec<Value> = std::fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(row_lines[0]["request_id"], json!("r1"));
        assert_eq!(row_lines[0]["prefix_reset"], json!(true));
    }

    #[test]
    fn writer_without_sidecar_rejects_sidecar_writes() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("trace.jsonl");
        let mut writer = MooncakeJsonlWriter::create(&output, None).unwrap();
        assert!(!writer.has_sidecar());
        let err = writer.write_sidecar(&json!({})).unwrap_err();
        assert!(err.to_string().contains("sidecar was not configured"));
    }
}
