// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use serde::Deserialize;

use crate::protocols::{
    BlockHashOptions, LocalBlockHash, compute_block_hash_for_seq, compute_seq_hash_for_block,
};

use super::error::SelectionError;

#[derive(Debug, Clone, Deserialize)]
pub struct PromptRequest {
    #[serde(default)]
    pub token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub block_hashes: Option<Vec<i64>>,
    #[serde(default)]
    pub sequence_hashes: Option<Vec<i64>>,
    #[serde(default)]
    pub isl_tokens: Option<usize>,
    #[serde(default)]
    pub lora_name: Option<String>,
}

impl PromptRequest {
    pub(super) fn normalize_for_selection(
        &self,
        block_size: u32,
    ) -> Result<NormalizedPrompt, SelectionError> {
        if let Some(token_ids) = &self.token_ids {
            return Ok(normalize_tokens(
                token_ids,
                block_size,
                self.lora_name.as_deref(),
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
        block_size: u32,
    ) -> Result<NormalizedReservation, SelectionError> {
        if let Some(token_ids) = &self.token_ids {
            let normalized = normalize_tokens(token_ids, block_size, self.lora_name.as_deref());
            return Ok(NormalizedReservation {
                sequence_hashes: normalized.sequence_hashes,
            });
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
        })
    }
}

fn normalize_tokens(
    token_ids: &[u32],
    block_size: u32,
    lora_name: Option<&str>,
) -> NormalizedPrompt {
    let block_hashes = compute_block_hash_for_seq(
        token_ids,
        block_size,
        BlockHashOptions {
            lora_name,
            ..Default::default()
        },
    );
    let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
    NormalizedPrompt {
        block_hashes,
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
}
