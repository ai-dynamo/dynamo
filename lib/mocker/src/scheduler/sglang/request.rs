// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::kv_manager::sglang_backend::ActiveKvLease;

#[derive(Debug)]
pub(super) struct SglangRequest {
    pub(super) uuid: Uuid,
    pub(super) sequence_tokens: Vec<u32>,
    pub(super) prompt_len: usize,
    pub(super) max_output_tokens: usize,
    pub(super) planned_output_ids: Option<Vec<u32>>,
    pub(super) kv_lease: ActiveKvLease,
    pub(super) materialized_tokens: usize,
    pub(super) allocated_tokens: usize,
}

impl SglangRequest {
    pub(super) fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub(super) fn output_len(&self) -> usize {
        self.sequence_tokens.len() - self.prompt_len
    }

    pub(super) fn current_sequence_len(&self) -> usize {
        self.sequence_tokens.len()
    }

    pub(super) fn extend_input_len(&self) -> usize {
        self.current_sequence_len()
            .saturating_sub(self.materialized_tokens)
    }

    pub(super) fn remaining_output_tokens(&self) -> usize {
        self.max_output_tokens.saturating_sub(self.output_len())
    }

    #[cfg(debug_assertions)]
    pub(super) fn extra_reserved_tokens(&self) -> usize {
        self.allocated_tokens.saturating_sub(self.kv_len())
    }

    #[cfg(test)]
    pub(super) fn kv_indices(&self) -> &[usize] {
        self.kv_lease.indices()
    }

    #[cfg(debug_assertions)]
    pub(super) fn kv_len(&self) -> usize {
        self.kv_lease.len()
    }

    pub(super) fn cached_tokens(&self) -> usize {
        self.kv_lease.cached_tokens()
    }

    pub(super) fn page_aligned_materialized_tokens(&self, block_size: usize) -> usize {
        self.materialized_tokens / block_size * block_size
    }

    pub(super) fn prompt_tokens(&self) -> &[u32] {
        &self.sequence_tokens[..self.prompt_len]
    }

    pub(super) fn sequence_tokens(&self) -> &[u32] {
        &self.sequence_tokens
    }

    pub(super) fn sequence_prefix(&self, len: usize) -> &[u32] {
        &self.sequence_tokens[..len]
    }

    #[cfg(test)]
    pub(super) fn output_tokens(&self) -> &[u32] {
        &self.sequence_tokens[self.prompt_len..]
    }

    pub(super) fn next_output_token(&self) -> u32 {
        if let Some(token_id) = self
            .planned_output_ids
            .as_ref()
            .and_then(|ids| ids.get(self.output_len()))
        {
            return *token_id;
        }

        let mut hasher = DefaultHasher::new();
        self.uuid.hash(&mut hasher);
        self.output_len().hash(&mut hasher);
        hasher.finish() as u32
    }

    pub(super) fn append_output_token(&mut self, token: u32) {
        self.sequence_tokens.push(token);
        self.materialized_tokens += 1;
    }

    pub(super) fn debug_assert_invariants(&self, _block_size: usize) {
        #[cfg(debug_assertions)]
        {
            let block_size = _block_size;
            let sequence_len = self.current_sequence_len();
            debug_assert!(
                self.prompt_len <= sequence_len,
                "request {} has prompt_len={} but only {sequence_len} sequence tokens",
                self.uuid,
                self.prompt_len
            );
            debug_assert!(
                self.cached_tokens() <= self.materialized_tokens,
                "request {} cached {} tokens but materialized {}",
                self.uuid,
                self.cached_tokens(),
                self.materialized_tokens
            );
            debug_assert!(
                self.materialized_tokens <= sequence_len,
                "request {} materialized {} tokens but sequence length is {sequence_len}",
                self.uuid,
                self.materialized_tokens
            );
            debug_assert_eq!(
                self.kv_len(),
                self.materialized_tokens,
                "request {} has {} kv indices but {} materialized tokens",
                self.uuid,
                self.kv_len(),
                self.materialized_tokens
            );
            debug_assert!(
                self.allocated_tokens >= self.materialized_tokens,
                "request {} allocated {} tokens but materialized {}",
                self.uuid,
                self.allocated_tokens,
                self.materialized_tokens
            );
            debug_assert_eq!(
                self.cached_tokens() % block_size,
                0,
                "request {} cached tokens {} are not page-aligned to block size {block_size}",
                self.uuid,
                self.cached_tokens()
            );
            debug_assert!(
                self.allocated_tokens == 0 || self.allocated_tokens.is_multiple_of(block_size),
                "request {} allocated tokens {} are not page-aligned to block size {block_size}",
                self.uuid,
                self.allocated_tokens
            );
            debug_assert!(
                self.extra_reserved_tokens() < block_size,
                "request {} reserves {} extra tokens with block size {block_size}",
                self.uuid,
                self.extra_reserved_tokens()
            );
            debug_assert_eq!(
                self.kv_lease.is_active(),
                self.materialized_tokens > 0,
                "request {} has active_kv={} but materialized_tokens={}",
                self.uuid,
                self.kv_lease.is_active(),
                self.materialized_tokens
            );
        }
    }

    pub(super) fn reset_for_retract(&mut self) {
        debug_assert!(!self.kv_lease.is_active());
        self.materialized_tokens = 0;
        self.allocated_tokens = 0;
    }
}

impl From<DirectRequest> for SglangRequest {
    fn from(req: DirectRequest) -> Self {
        let prompt_len = req.tokens.len();
        let max_output_tokens = req
            .output_token_ids
            .as_ref()
            .map_or(req.max_output_tokens, Vec::len);
        let sequence_tokens = req.tokens;
        Self {
            uuid: req.uuid.unwrap_or_else(Uuid::new_v4),
            sequence_tokens,
            prompt_len,
            max_output_tokens,
            planned_output_ids: req.output_token_ids,
            kv_lease: ActiveKvLease::default(),
            materialized_tokens: 0,
            allocated_tokens: 0,
        }
    }
}
