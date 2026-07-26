// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    #[serde(default)]
    pub token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub mm_routing_info: Option<MmRoutingInfoRequest>,
    #[serde(default)]
    pub block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    #[serde(default)]
    pub block_hashes: Option<Vec<i64>>,
    #[serde(default)]
    pub sequence_hashes: Option<Vec<i64>>,
    #[serde(default)]
    pub isl_tokens: Option<usize>,
    #[serde(default)]
    pub lora_name: Option<String>,
    #[serde(default, rename = "cache_salt")]
    pub cache_namespace: Option<String>,
    #[serde(default)]
    pub is_eagle: Option<bool>,
}

impl PromptRequest {
    pub(super) fn normalize_for_selection(
        &self,
        default_is_eagle: bool,
        tracking: TrackingHashInput<'_>,
    ) -> Result<NormalizedPrompt, SelectionError> {
        if let Some((token_ids, block_mm_infos)) = self.routing_tokens_and_mm_infos() {
            return Ok(normalize_tokens_for_selection(
                token_ids,
                self.lora_name.as_deref(),
                self.cache_namespace.as_deref(),
                block_mm_infos,
                self.is_eagle.unwrap_or(default_is_eagle),
                tracking,
            ));
        }

        let block_hashes = self.block_hashes.as_ref().ok_or_else(|| {
            SelectionError::BadRequest("block_hashes is required without token_ids".to_string())
        })?;
        let sequence_hashes = self.sequence_hashes.as_ref().ok_or_else(|| {
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
                self.lora_name.as_deref(),
                self.cache_namespace.as_deref(),
                block_mm_infos,
                self.is_eagle.unwrap_or(default_is_eagle),
                tracking,
            ));
        }

        let sequence_hashes = self.sequence_hashes.as_ref().ok_or_else(|| {
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
                    lora_name: self.lora_name.as_deref(),
                    cache_namespace: self.cache_namespace.as_deref(),
                    is_eagle: Some(self.is_eagle.unwrap_or(default_is_eagle)),
                },
            ));
        }

        let block_hashes = self.block_hashes.as_ref().ok_or_else(|| {
            SelectionError::BadRequest("block_hashes is required without token_ids".to_string())
        })?;
        let sequence_hashes = self.sequence_hashes.as_ref().ok_or_else(|| {
            SelectionError::BadRequest("sequence_hashes is required without token_ids".to_string())
        })?;
        let isl_tokens = self.isl_tokens.ok_or_else(|| {
            SelectionError::BadRequest("isl_tokens is required without token_ids".to_string())
        })?;
        Ok(normalize_hashes(block_hashes, sequence_hashes, isl_tokens)?.block_hashes)
    }

    fn routing_tokens_and_mm_infos(&self) -> Option<RoutingTokensAndMmInfos<'_>> {
        if let Some(mm_routing_info) = &self.mm_routing_info
            && !mm_routing_info.routing_token_ids.is_empty()
        {
            return Some((
                &mm_routing_info.routing_token_ids,
                Some(mm_routing_info.block_mm_infos.as_slice()),
            ));
        }

        self.token_ids
            .as_deref()
            .map(|token_ids| (token_ids, self.block_mm_infos.as_deref()))
    }
}

fn normalize_tokens_for_selection(
    token_ids: &[u32],
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: bool,
    tracking: TrackingHashInput<'_>,
) -> NormalizedPrompt {
    let hash_options = BlockHashOptions {
        block_mm_infos,
        lora_name,
        cache_namespace,
        is_eagle: Some(is_eagle),
    };
    let block_hashes =
        compute_block_hash_for_seq(token_ids, tracking.scope.block_size, hash_options);
    let sequence_hashes = tracking.context.compute_sequence_hashes_for_tracking(
        tracking.scope,
        token_ids,
        hash_options,
        tracking.assume_kv_reuse,
        Some(&block_hashes),
    );
    NormalizedPrompt {
        block_hashes,
        sequence_hashes,
        isl_tokens: token_ids.len(),
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
    })
}

fn signed_sequence_hashes(sequence_hashes: &[i64]) -> Vec<SequenceHash> {
    sequence_hashes.iter().map(|hash| *hash as u64).collect()
}

pub(super) struct NormalizedPrompt {
    pub(super) block_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
    pub(super) isl_tokens: usize,
}

pub(super) struct NormalizedReservation {
    pub(super) sequence_hashes: Vec<SequenceHash>,
    pub(super) isl_tokens: usize,
}
