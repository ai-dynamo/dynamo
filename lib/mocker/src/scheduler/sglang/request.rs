// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use uuid::Uuid;

use crate::cache::radix_cache::NodeId;
use crate::common::protocols::DirectRequest;

#[derive(Clone, Debug)]
pub(super) struct SglangRequest {
    pub(super) uuid: Uuid,
    pub(super) prompt_tokens: Vec<u64>,
    pub(super) max_output_tokens: usize,
    pub(super) output_ids: Vec<u64>,
    pub(super) last_node: Option<NodeId>,
    pub(super) kv_indices: Vec<usize>,
    pub(super) materialized_tokens: usize,
    pub(super) cached_tokens: usize,
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

    pub(super) fn total_tokens_needed(&self, clip_max_new_tokens: usize) -> usize {
        let remaining_input = self.extend_input_len();
        let remaining_output = self.remaining_output_tokens().min(clip_max_new_tokens);
        remaining_input + remaining_output
    }

    pub(super) fn remaining_output_tokens(&self) -> usize {
        self.max_output_tokens.saturating_sub(self.output_len())
    }

    pub(super) fn extra_reserved_tokens(&self) -> usize {
        self.allocated_tokens.saturating_sub(self.kv_indices.len())
    }

    pub(super) fn page_aligned_materialized_tokens(&self, block_size: usize) -> usize {
        self.materialized_tokens / block_size * block_size
    }

    pub(super) fn sequence_tokens(&self) -> Vec<u64> {
        let mut sequence = self.prompt_tokens.clone();
        sequence.extend_from_slice(&self.output_ids);
        sequence
    }

    pub(super) fn sequence_prefix(&self, len: usize) -> Vec<u64> {
        let prompt_len = self.prompt_len();
        if len <= prompt_len {
            return self.prompt_tokens[..len].to_vec();
        }

        let mut prefix = self.prompt_tokens.clone();
        prefix.extend_from_slice(&self.output_ids[..len - prompt_len]);
        prefix
    }

    pub(super) fn next_output_token(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.uuid.hash(&mut hasher);
        self.output_len().hash(&mut hasher);
        hasher.finish()
    }

    pub(super) fn append_output_token(&mut self, token: u64) {
        self.output_ids.push(token);
        self.materialized_tokens += 1;
    }

    pub(super) fn reset_for_retract(&mut self) {
        self.last_node = None;
        self.kv_indices.clear();
        self.materialized_tokens = 0;
        self.cached_tokens = 0;
        self.allocated_tokens = 0;
    }
}

pub(super) fn direct_to_sglang(req: DirectRequest) -> SglangRequest {
    SglangRequest {
        uuid: req.uuid.unwrap_or_else(Uuid::new_v4),
        prompt_tokens: req.tokens.iter().map(|&t| t as u64).collect(),
        max_output_tokens: req.max_output_tokens,
        output_ids: Vec::new(),
        last_node: None,
        kv_indices: Vec::new(),
        materialized_tokens: 0,
        cached_tokens: 0,
        allocated_tokens: 0,
    }
}
