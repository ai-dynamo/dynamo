// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use dynamo_tokens::SequenceHash;
use serde::Deserialize;

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, LocalBlockHash, compute_block_hash_for_seq,
};
use crate::tracking_hash::{TrackingHashContext, TrackingHashScope};

use super::error::SelectionError;

type RoutingTokensAndMmInfos<'a> = (&'a [u32], Option<&'a [Option<BlockExtraInfo>]>);

pub(super) struct TrackingHashInput<'a> {
    pub(super) context: &'a TrackingHashContext,
    pub(super) scope: TrackingHashScope<'a>,
    pub(super) assume_kv_reuse: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MmRoutingInfoRequest {
    pub routing_token_ids: Vec<u32>,
    #[serde(default)]
    pub block_mm_infos: Vec<Option<BlockExtraInfo>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PromptRequest {
    pub token_ids: Option<Vec<u32>>,
    pub mm_routing_info: Option<MmRoutingInfoRequest>,
    pub block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    pub block_hashes: Option<Vec<i64>>,
    pub sequence_hashes: Option<Vec<i64>>,
    pub isl_tokens: Option<usize>,
    pub lora_name: Option<String>,
    #[serde(default, rename = "cache_salt")]
    pub cache_namespace: Option<String>,
    pub is_eagle: Option<bool>,
}

impl PromptRequest {
    /// Borrow the request as the shape selection consumes; hosts that already
    /// hold the pieces build a [`PromptView`] directly instead of allocating.
    pub fn view(&self) -> PromptView<'_> {
        PromptView {
            token_ids: self.token_ids.as_deref(),
            mm_routing_info: self.mm_routing_info.as_ref(),
            block_mm_infos: self.block_mm_infos.as_deref(),
            block_hashes: self.block_hashes.as_deref(),
            sequence_hashes: self.sequence_hashes.as_deref(),
            isl_tokens: self.isl_tokens,
            lora_name: self.lora_name.as_deref(),
            cache_namespace: self.cache_namespace.as_deref(),
            is_eagle: self.is_eagle,
        }
    }
}

/// A prompt borrowed for one selection: the same fields as [`PromptRequest`]
/// with the same precedence (multimodal routing tokens, then raw tokens, then
/// the hash-only trio).
#[derive(Debug, Clone, Copy)]
pub struct PromptView<'a> {
    pub token_ids: Option<&'a [u32]>,
    pub mm_routing_info: Option<&'a MmRoutingInfoRequest>,
    pub block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
    pub block_hashes: Option<&'a [i64]>,
    pub sequence_hashes: Option<&'a [i64]>,
    pub isl_tokens: Option<usize>,
    pub lora_name: Option<&'a str>,
    pub cache_namespace: Option<&'a str>,
    pub is_eagle: Option<bool>,
}

impl PromptView<'_> {
    /// `tracking` is `None` when the caller does not track active blocks; the
    /// tracking hashes are then left empty instead of computed and discarded.
    pub(super) fn normalize_for_selection(
        &self,
        block_size: u32,
        default_is_eagle: bool,
        tracking: Option<TrackingHashInput<'_>>,
    ) -> Result<NormalizedPrompt, SelectionError> {
        if let Some((token_ids, block_mm_infos)) = self.routing_tokens_and_mm_infos() {
            return Ok(normalize_tokens_for_selection(
                token_ids,
                block_size,
                self.lora_name,
                self.cache_namespace,
                block_mm_infos,
                self.is_eagle.unwrap_or(default_is_eagle),
                tracking,
            ));
        }

        let block_hashes = self.block_hashes.ok_or_else(|| {
            SelectionError::BadRequest("block_hashes is required without token_ids".to_string())
        })?;
        let sequence_hashes = self.sequence_hashes.ok_or_else(|| {
            SelectionError::BadRequest("sequence_hashes is required without token_ids".to_string())
        })?;
        let isl_tokens = self.isl_tokens.ok_or_else(|| {
            SelectionError::BadRequest("isl_tokens is required without token_ids".to_string())
        })?;
        normalize_hashes(block_hashes, sequence_hashes, isl_tokens)
    }

    pub(super) fn normalize_for_reservation(
        &self,
        default_is_eagle: bool,
        tracking: TrackingHashInput<'_>,
    ) -> Result<NormalizedReservation, SelectionError> {
        if let Some((token_ids, block_mm_infos)) = self.routing_tokens_and_mm_infos() {
            return Ok(normalize_tokens_for_reservation(
                token_ids,
                self.lora_name,
                self.cache_namespace,
                block_mm_infos,
                self.is_eagle.unwrap_or(default_is_eagle),
                tracking,
            ));
        }

        let sequence_hashes = self.sequence_hashes.ok_or_else(|| {
            SelectionError::BadRequest("sequence_hashes is required without token_ids".to_string())
        })?;
        if self.isl_tokens.is_none() {
            return Err(SelectionError::BadRequest(
                "isl_tokens is required without token_ids".to_string(),
            ));
        }
        Ok(NormalizedReservation {
            sequence_hashes: signed_sequence_hashes(sequence_hashes),
            isl_tokens: self.isl_tokens.expect("validated above"),
        })
    }

    pub(super) fn block_hashes_for_indexer(
        &self,
        block_size: u32,
        default_is_eagle: bool,
    ) -> Result<Vec<LocalBlockHash>, SelectionError> {
        if let Some((token_ids, block_mm_infos)) = self.routing_tokens_and_mm_infos() {
            return Ok(compute_block_hash_for_seq(
                token_ids,
                block_size,
                BlockHashOptions {
                    block_mm_infos,
                    lora_name: self.lora_name,
                    cache_namespace: self.cache_namespace,
                    is_eagle: Some(self.is_eagle.unwrap_or(default_is_eagle)),
                },
            ));
        }

        let block_hashes = self.block_hashes.ok_or_else(|| {
            SelectionError::BadRequest("block_hashes is required without token_ids".to_string())
        })?;
        let sequence_hashes = self.sequence_hashes.ok_or_else(|| {
            SelectionError::BadRequest("sequence_hashes is required without token_ids".to_string())
        })?;
        let isl_tokens = self.isl_tokens.ok_or_else(|| {
            SelectionError::BadRequest("isl_tokens is required without token_ids".to_string())
        })?;
        Ok(normalize_hashes(block_hashes, sequence_hashes, isl_tokens)?.block_hashes)
    }

    fn routing_tokens_and_mm_infos(&self) -> Option<RoutingTokensAndMmInfos<'_>> {
        if let Some(mm_routing_info) = self.mm_routing_info
            && !mm_routing_info.routing_token_ids.is_empty()
        {
            return Some((
                &mm_routing_info.routing_token_ids,
                Some(mm_routing_info.block_mm_infos.as_slice()),
            ));
        }

        self.token_ids
            .map(|token_ids| (token_ids, self.block_mm_infos))
    }
}

fn normalize_tokens_for_selection(
    token_ids: &[u32],
    block_size: u32,
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: bool,
    tracking: Option<TrackingHashInput<'_>>,
) -> NormalizedPrompt {
    let hash_options = BlockHashOptions {
        block_mm_infos,
        lora_name,
        cache_namespace,
        is_eagle: Some(is_eagle),
    };
    let started = Instant::now();
    let block_hashes = tracing::info_span!("kv_router.compute_block_hashes")
        .in_scope(|| compute_block_hash_for_seq(token_ids, block_size, hash_options));
    let block_hashing = started.elapsed();
    let sequence_hashes = tracing::info_span!("kv_router.compute_seq_hashes").in_scope(|| {
        tracking.map_or_else(Vec::new, |tracking| {
            tracking.context.compute_sequence_hashes_for_tracking(
                tracking.scope,
                token_ids,
                hash_options,
                tracking.assume_kv_reuse,
                Some(&block_hashes),
            )
        })
    });
    NormalizedPrompt {
        block_hashes,
        sequence_hashes,
        isl_tokens: token_ids.len(),
        block_hashing,
        seq_hashing: started.elapsed().saturating_sub(block_hashing),
    }
}

fn normalize_tokens_for_reservation(
    token_ids: &[u32],
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: bool,
    tracking: TrackingHashInput<'_>,
) -> NormalizedReservation {
    let hash_options = BlockHashOptions {
        block_mm_infos,
        lora_name,
        cache_namespace,
        is_eagle: Some(is_eagle),
    };
    let sequence_hashes = tracking.context.compute_sequence_hashes_for_tracking(
        tracking.scope,
        token_ids,
        hash_options,
        tracking.assume_kv_reuse,
        None,
    );
    NormalizedReservation {
        sequence_hashes,
        isl_tokens: token_ids.len(),
    }
}

fn normalize_hashes(
    block_hashes: &[i64],
    sequence_hashes: &[i64],
    isl_tokens: usize,
) -> Result<NormalizedPrompt, SelectionError> {
    if isl_tokens == 0 {
        return Err(SelectionError::BadRequest(
            "isl_tokens must be greater than 0".to_string(),
        ));
    }
    if block_hashes.len() != sequence_hashes.len() {
        return Err(SelectionError::BadRequest(format!(
            "block_hashes length {} must match sequence_hashes length {}",
            block_hashes.len(),
            sequence_hashes.len()
        )));
    }
    Ok(NormalizedPrompt {
        block_hashes: block_hashes
            .iter()
            .map(|hash| LocalBlockHash(*hash as u64))
            .collect(),
        sequence_hashes: signed_sequence_hashes(sequence_hashes),
        isl_tokens,
        block_hashing: Duration::ZERO,
        seq_hashing: Duration::ZERO,
    })
}

fn signed_sequence_hashes(sequence_hashes: &[i64]) -> Vec<SequenceHash> {
    sequence_hashes.iter().map(|hash| *hash as u64).collect()
}

pub(super) struct NormalizedPrompt {
    pub(super) block_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
    pub(super) isl_tokens: usize,
    /// Zero for hash-only inputs.
    pub(super) block_hashing: Duration,
    pub(super) seq_hashing: Duration,
}

pub(super) struct NormalizedReservation {
    pub(super) sequence_hashes: Vec<SequenceHash>,
    pub(super) isl_tokens: usize,
}
