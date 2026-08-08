// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion from neutral replay metadata to Dynamo Router wire identities.

use aisimulate_replay::loadgen::Trace;
use anyhow::{Context, Result};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, WorkerId, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};

#[derive(Debug, Clone, Copy)]
pub enum SequenceHashMode {
    Raw,
    Cumulative,
}

#[derive(Debug, Clone)]
pub struct RouterSequence {
    pub worker_id: WorkerId,
    pub local_hashes: Vec<LocalBlockHash>,
    pub external_hashes: Vec<ExternalSequenceBlockHash>,
}

/// Restore the legacy method-style API while keeping Router types out of the
/// neutral replay crate.
pub trait DynamoTraceRouterExt {
    fn to_router_sequences(
        &self,
        worker_id: WorkerId,
        hash_mode: SequenceHashMode,
    ) -> Result<Vec<RouterSequence>>;
}

impl DynamoTraceRouterExt for Trace {
    fn to_router_sequences(
        &self,
        worker_id: WorkerId,
        hash_mode: SequenceHashMode,
    ) -> Result<Vec<RouterSequence>> {
        to_router_sequences(self, worker_id, hash_mode)
    }
}

pub fn to_router_sequences(
    trace: &Trace,
    worker_id: WorkerId,
    hash_mode: SequenceHashMode,
) -> Result<Vec<RouterSequence>> {
    let mut sequences = Vec::new();
    for session in &trace.sessions {
        for turn in &session.turns {
            let local_hashes = turn
                .hash_ids
                .iter()
                .map(|&hash_id| local_block_hash_from_id(hash_id, trace.block_size))
                .collect::<Result<Vec<_>>>()?;
            let external_hashes = match hash_mode {
                SequenceHashMode::Raw => local_hashes
                    .iter()
                    .map(|hash| ExternalSequenceBlockHash(hash.0))
                    .collect(),
                SequenceHashMode::Cumulative => compute_seq_hash_for_block(&local_hashes)
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            };
            sequences.push(RouterSequence {
                worker_id,
                local_hashes,
                external_hashes,
            });
        }
    }
    Ok(sequences)
}

/// Convert neutral replay block identities once, at the Dynamo placement
/// boundary.
pub(crate) fn local_block_hashes(hashes: Vec<u64>) -> Vec<LocalBlockHash> {
    hashes.into_iter().map(LocalBlockHash).collect()
}

fn local_block_hash_from_id(hash_id: u32, block_size: usize) -> Result<LocalBlockHash> {
    let block_size = u32::try_from(block_size).context("trace block size does not fit in u32")?;
    let tokens = vec![hash_id; block_size as usize];
    compute_block_hash_for_seq(&tokens, block_size, BlockHashOptions::default())
        .into_iter()
        .next()
        .context("trace block size must be greater than zero")
}
