// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::SglangRequest;
use crate::kv_manager::SglangKvManager;

/// Per-request prefill data needed for FPM snapshot construction.
pub(super) struct PrefillFpmItem {
    pub(super) prompt_len: usize,
    pub(super) tokens_computed: usize,
    pub(super) prefix_tokens: usize,
}

pub(super) struct AdmitResult {
    pub(super) can_run: Vec<SglangRequest>,
    pub(super) admissions: Vec<AdmissionEvent>,
    pub(super) total_isl: usize,
    pub(super) total_prefix: usize,
    pub(super) oom: bool,
    /// Per-request prefill info for building FPM snapshots.
    pub(super) prefill_fpm: Vec<PrefillFpmItem>,
}

pub(super) fn get_new_batch_prefill(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
) -> AdmitResult {
    let cache = kv_manager.cache();
    let reserved_decode_output: f64 = running
        .iter()
        .map(|req| {
            let remaining_output = req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();
    let reserved_page_overhead = waiting
        .iter()
        .map(SglangRequest::extra_reserved_tokens)
        .sum::<usize>()
        + running
            .iter()
            .map(SglangRequest::extra_reserved_tokens)
            .sum::<usize>();

    let mut rem_total_tokens = (cache.available_tokens() + cache.evictable_size)
        .saturating_sub(reserved_page_overhead) as f64
        - reserved_decode_output;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();
    let mut rejected = VecDeque::new();
    let mut oom = false;
    let mut total_isl = 0usize;
    let mut total_prefix = 0usize;

    while let Some(mut req) = waiting.pop_front() {
        let extend_input = req.extend_input_len();
        if extend_input == 0 {
            rejected.push_back(req);
            break;
        }

        let total_needed = req.total_tokens_needed(config.clip_max_new_tokens) as f64;
        if total_needed >= rem_total_tokens {
            rejected.push_back(req);
            break;
        }

        let chunk_tokens = if extend_input <= config.chunked_prefill_size {
            extend_input
        } else {
            let chunk = (rem_chunk_tokens as usize / config.block_size) * config.block_size;
            if chunk == 0 {
                rejected.push_back(req);
                break;
            }
            chunk.min(extend_input)
        };

        let charged_input_tokens = ceil_to_block(chunk_tokens, config.block_size) as f64;
        if charged_input_tokens > rem_input_tokens || charged_input_tokens > rem_chunk_tokens {
            rejected.push_back(req);
            break;
        }

        let chunk_end = req.materialized_tokens + chunk_tokens;
        let old_allocated_tokens = req.allocated_tokens;
        let prev_node = req.last_node.take();
        let alloc_tokens = req.sequence_prefix(chunk_end);
        let actual_new_tokens = alloc_tokens.len().saturating_sub(req.materialized_tokens);
        let available = kv_manager.cache().token_pool.available();
        if available < actual_new_tokens {
            kv_manager.evict(actual_new_tokens - available);
        }

        let alloc = if req.materialized_tokens > 0 {
            let last_node = prev_node.unwrap_or_else(|| {
                panic!(
                    "prefill: request {} has materialized_tokens={} but last_node is None",
                    req.uuid, req.materialized_tokens
                )
            });
            kv_manager.allocate_after_prefix(
                &alloc_tokens,
                req.materialized_tokens,
                &req.kv_indices[..req.materialized_tokens],
                last_node,
            )
        } else {
            kv_manager.allocate_for_request(&alloc_tokens)
        };

        let Some(alloc) = alloc else {
            req.last_node = prev_node;
            rejected.push_back(req);
            oom = true;
            break;
        };

        if let Some(node) = prev_node {
            kv_manager.free_request(node);
        }

        req.last_node = Some(alloc.last_node);
        req.kv_indices = alloc.kv_indices;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        let is_truncated = chunk_end < req.current_sequence_len();
        let output_reserve = if is_truncated {
            0
        } else {
            req.remaining_output_tokens()
                .min(config.clip_max_new_tokens)
        };

        admissions.push(AdmissionEvent {
            uuid: req.uuid,
            reused_input_tokens: alloc.prefix_len,
        });
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed: chunk_tokens,
            prefix_tokens: alloc.prefix_len,
        });

        total_isl += chunk_end;
        total_prefix += alloc.prefix_len;
        rem_total_tokens -= (req.allocated_tokens - old_allocated_tokens + output_reserve) as f64;
        rem_input_tokens -= charged_input_tokens;
        rem_chunk_tokens -= charged_input_tokens;
        can_run.push(req);

        if rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    while let Some(req) = rejected.pop_back() {
        waiting.push_front(req);
    }

    AdmitResult {
        can_run,
        admissions,
        total_isl,
        total_prefix,
        oom,
        prefill_fpm,
    }
}
