// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic Mooncake JSONL primitives.
//!
//! This module is intentionally producer-agnostic: it defines the row schema,
//! the block-hash-to-id mapping, the token-block hashing helper, and the JSONL
//! writer. Workload-specific orchestration (session scheduling, tokenization,
//! parsing) lives elsewhere -- the Claude exporter in
//! [`crate::coding::claude::export`] is one such producer; future producers
//! (e.g. a Dynamo agent trace exporter) consume the same primitives.

use anyhow::{Context, Result, bail};
use bytemuck::cast_slice;
use dynamo_tokens::compute_hash_v2;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One row of a Mooncake replay trace.
///
/// `timestamp` and `delay` are mutually exclusive in the typical convention
/// (the first row in a session carries `timestamp`, subsequent rows carry
/// `delay`), but neither is required by the schema -- both fields are skipped
/// during serialization when `None`.
#[derive(Debug, Clone, Serialize)]
pub struct MooncakeRow {
    pub session_id: String,
    pub input_length: usize,
    pub output_length: usize,
    pub hash_ids: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delay: Option<i64>,
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
    /// Tokens are chunked by `block_size`; each block contributes one id. The
    /// chained hash mixes the prior block's combined hash, so identical
    /// prefixes across requests resolve to identical leading `hash_ids` once
    /// the mapper has seen them.
    pub fn hash_token_blocks(&mut self, tokens: &[u32]) -> Vec<u64> {
        hash_token_blocks(self, tokens)
    }
}

/// Token-block hashing helper for the Mooncake replay schema.
///
/// Splits `tokens` into chunks of `mapper.block_size()`, computes a chained
/// hash per block, and returns the compact ids assigned by `mapper`. Mirrors
/// [`RollingHashIdMapper::hash_token_blocks`] as a free function so callers
/// that already hold a mutable mapper reference can invoke it without
/// re-borrowing.
pub fn hash_token_blocks(mapper: &mut RollingHashIdMapper, tokens: &[u32]) -> Vec<u64> {
    let block_size = mapper.block_size;
    let mut hash_ids = Vec::with_capacity(tokens.len().div_ceil(block_size));
    let mut parent_hash = 0_u64;
    for block in tokens.chunks(block_size) {
        let block_hash = compute_hash_v2(cast_slice(block), 0);
        let combined_hash = compute_hash_v2(&block_hash.to_be_bytes(), parent_hash);
        let id = *mapper.hash_to_id.entry(combined_hash).or_insert_with(|| {
            let next_id = mapper.next_id;
            mapper.next_id += 1;
            next_id
        });
        hash_ids.push(id);
        parent_hash = combined_hash;
    }
    hash_ids
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
/// [`Self::write_sidecar`]. When a sidecar path is configured, the path
/// convention from [`crate::coding::common::sidecar_path_for`] is the typical
/// choice but not enforced here -- callers pass the sidecar path explicitly.
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
    use crate::coding::common::sidecar_path_for;
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
    fn empty_token_input_yields_empty_hash_ids() {
        let mut mapper = RollingHashIdMapper::new(4);
        assert!(mapper.hash_token_blocks(&[]).is_empty());
    }

    #[test]
    fn row_omits_timestamp_and_delay_when_absent() {
        let row = MooncakeRow {
            session_id: "s".to_string(),
            input_length: 4,
            output_length: 1,
            hash_ids: vec![0, 1],
            timestamp: None,
            delay: None,
        };
        let rendered: Value = serde_json::to_value(&row).unwrap();
        assert!(rendered.get("timestamp").is_none());
        assert!(rendered.get("delay").is_none());
        assert_eq!(rendered["hash_ids"], json!([0, 1]));
    }

    #[test]
    fn row_serializes_optional_fields_when_set() {
        let with_timestamp = MooncakeRow {
            session_id: "s".to_string(),
            input_length: 4,
            output_length: 1,
            hash_ids: vec![],
            timestamp: Some(0),
            delay: None,
        };
        let with_delay = MooncakeRow {
            session_id: "s".to_string(),
            input_length: 4,
            output_length: 1,
            hash_ids: vec![],
            timestamp: None,
            delay: Some(123),
        };
        let v_ts: Value = serde_json::to_value(&with_timestamp).unwrap();
        let v_dl: Value = serde_json::to_value(&with_delay).unwrap();
        assert_eq!(v_ts["timestamp"], json!(0));
        assert!(v_ts.get("delay").is_none());
        assert_eq!(v_dl["delay"], json!(123));
        assert!(v_dl.get("timestamp").is_none());
    }

    #[test]
    fn writer_writes_rows_and_sidecar_jsonl() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("trace.jsonl");
        let sidecar = sidecar_path_for(&output);

        let mut writer = MooncakeJsonlWriter::create(&output, Some(&sidecar)).unwrap();
        writer
            .write_row(&MooncakeRow {
                session_id: "s".to_string(),
                input_length: 2,
                output_length: 1,
                hash_ids: vec![0],
                timestamp: Some(0),
                delay: None,
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
    fn writer_without_sidecar_rejects_sidecar_writes() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("trace.jsonl");
        let mut writer = MooncakeJsonlWriter::create(&output, None).unwrap();
        assert!(!writer.has_sidecar());
        let err = writer.write_sidecar(&json!({})).unwrap_err();
        assert!(err.to_string().contains("sidecar was not configured"));
    }

    #[test]
    fn sidecar_path_convention_is_preserved() {
        let path = std::path::Path::new("/tmp/example/trace.jsonl");
        assert_eq!(
            sidecar_path_for(path),
            std::path::PathBuf::from("/tmp/example/trace.sidecar.jsonl")
        );
    }
}
