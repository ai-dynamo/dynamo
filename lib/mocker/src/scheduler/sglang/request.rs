// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::kv_manager::sglang_backend::ActiveKvLease;

#[derive(Debug)]
pub(super) struct SglangRequest {
    pub(super) uuid: Uuid,
    pub(super) prompt_tokens: Vec<u64>,
    pub(super) max_output_tokens: usize,
    pub(super) planned_output_ids: Option<Vec<u32>>,
    pub(super) output_ids: Vec<u32>,
    pub(super) kv_lease: ActiveKvLease,
    pub(super) materialized_tokens: usize,
    pub(super) allocated_tokens: usize,
}

impl SglangRequest {
    pub(super) fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub(super) fn output_len(&self) -> usize {
        self.output_ids.len()
    }

    pub(super) fn current_sequence_len(&self) -> usize {
        self.prompt_len() + self.output_len()
    }

    pub(super) fn extend_input_len(&self) -> usize {
        self.current_sequence_len()
            .saturating_sub(self.materialized_tokens)
    }

    pub(super) fn remaining_output_tokens(&self) -> usize {
        self.max_output_tokens.saturating_sub(self.output_len())
    }

    pub(super) fn extra_reserved_tokens(&self) -> usize {
        self.allocated_tokens.saturating_sub(self.kv_len())
    }

    #[cfg(test)]
    pub(super) fn kv_indices(&self) -> &[usize] {
        self.kv_lease.indices()
    }

    pub(super) fn kv_len(&self) -> usize {
        self.kv_lease.len()
    }

    pub(super) fn cached_tokens(&self) -> usize {
        self.kv_lease.cached_tokens()
    }

    pub(super) fn page_aligned_materialized_tokens(&self, block_size: usize) -> usize {
        self.materialized_tokens / block_size * block_size
    }

    pub(super) fn sequence_tokens(&self) -> Cow<'_, [u64]> {
        if self.output_ids.is_empty() {
            return Cow::Borrowed(&self.prompt_tokens);
        }

        let mut sequence = self.prompt_tokens.clone();
        sequence.extend(self.output_ids.iter().map(|&token| token as u64));
        Cow::Owned(sequence)
    }

    pub(super) fn sequence_prefix(&self, len: usize) -> Cow<'_, [u64]> {
        let prompt_len = self.prompt_len();
        if len <= prompt_len {
            return Cow::Borrowed(&self.prompt_tokens[..len]);
        }

        let mut prefix = self.prompt_tokens.clone();
        prefix.extend(
            self.output_ids[..len - prompt_len]
                .iter()
                .map(|&token| token as u64),
        );
        Cow::Owned(prefix)
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
        self.output_ids.push(token);
        self.materialized_tokens += 1;
    }

    pub(super) fn debug_assert_invariants(&self, _block_size: usize) {
        #[cfg(debug_assertions)]
        {
            let block_size = _block_size;
            let sequence_len = self.current_sequence_len();
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
        Self {
            uuid: req.uuid.unwrap_or_else(Uuid::new_v4),
            prompt_tokens: req.tokens.iter().map(|&t| t as u64).collect(),
            max_output_tokens: req
                .output_token_ids
                .as_ref()
                .map_or(req.max_output_tokens, Vec::len),
            planned_output_ids: req.output_token_ids,
            output_ids: Vec::new(),
            kv_lease: ActiveKvLease::default(),
            materialized_tokens: 0,
            allocated_tokens: 0,
        }
    }
}
