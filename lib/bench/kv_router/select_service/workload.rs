// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake-format session traces replayed as token-id prompts: rows grouped by
//! `session_id`, copied per session slot with shifted hash ids so copies share only
//! the declared system-prompt prefix.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, bail, ensure};
use serde::Deserialize;

/// Token ids are drawn below this (Qwen3's vocabulary size).
pub const VOCAB_SIZE: u64 = 151_936;

#[derive(Debug)]
pub struct Turn {
    pub hash_ids: Vec<u64>,
    pub input_length: usize,
    pub output_length: usize,
}

#[derive(Debug)]
pub struct SessionTemplate {
    /// Read by `select_service_bench` (session ids), not by every binary.
    #[allow(dead_code)]
    pub id: String,
    pub turns: Vec<Turn>,
}

#[derive(Debug)]
pub struct Workload {
    pub sessions: Vec<SessionTemplate>,
    /// Hash-id shift per session copy; larger than every id in the trace.
    pub copy_stride: u64,
    pub shared_prefix_blocks: usize,
    pub trace_block_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct TraceRow {
    pub session_id: Option<String>,
    pub input_length: usize,
    pub output_length: usize,
    pub hash_ids: Vec<u64>,
}

impl Workload {
    pub fn load(
        path: &Path,
        trace_block_size: usize,
        shared_prefix_blocks: usize,
    ) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read trace {}", path.display()))?;
        let rows = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(idx, line)| {
                serde_json::from_str::<TraceRow>(line)
                    .with_context(|| format!("trace line {} is not a Mooncake row", idx + 1))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Self::from_rows(rows, trace_block_size, shared_prefix_blocks)
    }

    pub fn from_rows(
        rows: Vec<TraceRow>,
        trace_block_size: usize,
        shared_prefix_blocks: usize,
    ) -> anyhow::Result<Self> {
        ensure!(trace_block_size > 0, "trace_block_size must be positive");
        ensure!(!rows.is_empty(), "trace has no rows");
        let shared_prefix = rows[0]
            .hash_ids
            .get(..shared_prefix_blocks)
            .map(<[u64]>::to_vec);
        let Some(shared_prefix) = shared_prefix else {
            bail!("first row is shorter than --shared-prefix-blocks {shared_prefix_blocks}");
        };
        let mut max_id = 0u64;
        let mut order = Vec::new();
        let mut sessions: HashMap<String, Vec<Turn>> = HashMap::new();
        for (idx, row) in rows.into_iter().enumerate() {
            ensure!(
                row.hash_ids.starts_with(&shared_prefix),
                "row {} does not start with the {shared_prefix_blocks} shared prefix ids",
                idx + 1
            );
            max_id = max_id.max(row.hash_ids.iter().copied().max().unwrap_or(0));
            let session = row.session_id.unwrap_or_else(|| format!("row-{idx}"));
            let turns = sessions.entry(session.clone()).or_insert_with(|| {
                order.push(session);
                Vec::new()
            });
            turns.push(Turn {
                hash_ids: row.hash_ids,
                input_length: row.input_length,
                output_length: row.output_length.max(1),
            });
        }
        let sessions = order
            .into_iter()
            .map(|id| {
                let turns = sessions.remove(&id).expect("session recorded in order");
                SessionTemplate { id, turns }
            })
            .collect();
        Ok(Self {
            sessions,
            copy_stride: max_id + 1,
            shared_prefix_blocks,
            trace_block_size,
        })
    }

    /// Prompt tokens of `turn` in session copy `copy`. A token block depends only on its
    /// (shifted) hash id, so equal ids give equal blocks across sessions and turns.
    pub fn tokens(&self, turn: &Turn, copy: u64) -> Vec<u32> {
        let len = turn
            .input_length
            .min(turn.hash_ids.len() * self.trace_block_size);
        let mut tokens = Vec::with_capacity(len);
        'blocks: for (position, &id) in turn.hash_ids.iter().enumerate() {
            let id = if position < self.shared_prefix_blocks {
                id
            } else {
                id + copy * self.copy_stride
            };
            let mut state = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..self.trace_block_size {
                if tokens.len() == len {
                    break 'blocks;
                }
                tokens.push((splitmix64(&mut state) % VOCAB_SIZE) as u32);
            }
        }
        tokens
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(session: &str, hash_ids: Vec<u64>, input_length: usize) -> TraceRow {
        TraceRow {
            session_id: Some(session.to_string()),
            input_length,
            output_length: 4,
            hash_ids,
        }
    }

    #[test]
    fn copies_share_only_the_prefix() {
        let workload = Workload::from_rows(
            vec![row("a", vec![100, 1, 2], 10), row("b", vec![100, 3], 8)],
            4,
            1,
        )
        .unwrap();
        let turn = &workload.sessions[0].turns[0];
        let copy0 = workload.tokens(turn, 0);
        let copy1 = workload.tokens(turn, 1);
        assert_eq!(copy0.len(), 10, "truncated to input_length");
        assert_eq!(copy0[..4], copy1[..4], "shared prefix block is identical");
        assert_ne!(copy0[4..8], copy1[4..8], "per-copy blocks differ");

        let mismatched = Workload::from_rows(
            vec![row("a", vec![100, 1], 8), row("b", vec![101, 1], 8)],
            4,
            1,
        );
        assert!(mismatched.is_err(), "rows must share the declared prefix");
    }
}
