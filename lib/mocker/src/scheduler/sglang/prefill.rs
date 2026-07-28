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

#[derive(Default)]
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
    // PagePool already removes the full physical footprint of every active
    // partial page from available capacity, so page slack must not be charged
    // a second time here.
    let mut rem_total_tokens =
        (cache.available_tokens() + cache.evictable_size) as f64 - reserved_decode_output;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();
    let mut rejected = VecDeque::new();
    let mut oom = false;
    let mut total_isl = 0usize;
    let mut total_prefix = 0usize;

    let available_running_slots = config.max_running_requests.saturating_sub(running.len());
    while can_run.len() < available_running_slots
        && let Some(mut req) = waiting.pop_front()
    {
        let extend_input = req.extend_input_len();
        if extend_input == 0 {
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
        let output_reserve = if chunk_tokens < extend_input {
            0
        } else {
            req.remaining_output_tokens()
                .min(config.clip_max_new_tokens)
        };
        if charged_input_tokens + output_reserve as f64 >= rem_total_tokens {
            rejected.push_back(req);
            break;
        }
        if charged_input_tokens > rem_input_tokens || charged_input_tokens > rem_chunk_tokens {
            rejected.push_back(req);
            break;
        }

        let chunk_end = req.materialized_tokens + chunk_tokens;
        let old_allocated_tokens = req.allocated_tokens;
        let mut lease = std::mem::take(&mut req.kv_lease);
        let alloc_tokens = req.sequence_prefix(chunk_end);
        if req.materialized_tokens > 0 {
            let target_allocated_tokens = ceil_to_block(chunk_end, config.block_size);
            let required_capacity = target_allocated_tokens.saturating_sub(old_allocated_tokens);
            let available = kv_manager.cache().available_tokens();
            if available < required_capacity {
                kv_manager.evict(required_capacity - available);
            }
        }

        let prefix_len = if req.materialized_tokens > 0 {
            if !lease.is_active() {
                panic!(
                    "prefill: request {} has materialized_tokens={} but no active KV lease",
                    req.uuid, req.materialized_tokens
                );
            }
            kv_manager
                .extend_allocation(alloc_tokens, &mut lease)
                .then_some(req.materialized_tokens)
        } else {
            kv_manager.allocate_for_request(alloc_tokens).map(|alloc| {
                lease = alloc.lease;
                alloc.prefix_len
            })
        };

        let Some(prefix_len) = prefix_len else {
            req.kv_lease = lease;
            rejected.push_back(req);
            oom = true;
            break;
        };
        let tokens_computed = chunk_end.saturating_sub(prefix_len);

        req.kv_lease = lease;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        admissions.push(AdmissionEvent {
            uuid: req.uuid,
            reused_input_tokens: prefix_len,
        });
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed,
            prefix_tokens: prefix_len,
        });

        total_isl += chunk_end;
        total_prefix += prefix_len;
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
