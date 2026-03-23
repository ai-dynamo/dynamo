// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::common::protocols::OutputSignal;
use crate::kv_manager::SglangKvManager;

use super::config::{SglangConfig, floor_to_block};
use super::request::SglangRequest;

#[derive(Default)]
pub(super) struct DecodeResult {
    pub(super) requests: Vec<SglangRequest>,
    pub(super) output_signals: Vec<OutputSignal>,
    pub(super) retracted_any: bool,
    pub(super) end_ms: f64,
}

fn decode_capacity_state(
    running: &[SglangRequest],
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
) -> (usize, usize) {
    let actual_available =
        kv_manager.cache().available_tokens() + kv_manager.cache().evictable_size;
    let reserved_tokens = running
        .iter()
        .map(SglangRequest::extra_reserved_tokens)
        .sum::<usize>();
    let logical_available = actual_available.saturating_sub(reserved_tokens);
    let page_growth_needed = running
        .iter()
        .map(|req| {
            if req.current_sequence_len() + 1 > req.allocated_tokens {
                config.block_size
            } else {
                0
            }
        })
        .sum();

    (
        actual_available,
        logical_available.saturating_sub(page_growth_needed),
    )
}

pub(super) fn cache_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) {
    let aligned_tokens = req.page_aligned_materialized_tokens(config.block_size);
    if aligned_tokens == 0 || aligned_tokens <= req.cached_tokens {
        return;
    }

    let Some(last_node) = req.last_node else {
        return;
    };

    let sequence = req.sequence_prefix(aligned_tokens);
    let new_last =
        kv_manager.cache_unfinished_req(&sequence, &req.kv_indices[..aligned_tokens], last_node);
    req.last_node = Some(new_last);
    req.cached_tokens = aligned_tokens;
    req.debug_assert_invariants(config.block_size);
}

pub(super) fn check_decode_mem(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) -> Vec<SglangRequest> {
    let mut retracted = Vec::new();

    loop {
        let (actual_available, logical_available_after_growth) =
            decode_capacity_state(running, kv_manager, config);
        if actual_available >= running.len() && logical_available_after_growth > 0 {
            break;
        }
        if running.len() <= 1 {
            break;
        }

        let Some((idx, _)) = running
            .iter()
            .enumerate()
            .min_by_key(|(_, req)| req.output_len())
        else {
            break;
        };

        let mut req = running.swap_remove(idx);
        kv_manager.free_indices(&req.kv_indices[req.cached_tokens..]);
        if let Some(last_node) = req.last_node.take() {
            kv_manager.free_request(last_node);
        }
        req.reset_for_retract();
        req.debug_assert_invariants(config.block_size);
        retracted.push(req);
    }

    let available = kv_manager.cache().token_pool.available();
    let needed = running.len();
    if available < needed {
        kv_manager.evict(needed - available);
    }

    if !retracted.is_empty() {
        tracing::warn!(
            num_retracted = retracted.len(),
            remaining = running.len(),
            "SGLang decode retract requests because KV pool is full"
        );
    }

    retracted
}

pub(super) fn simulate_decode_step(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    current_time_ms: f64,
    apply_speedup: bool,
) -> DecodeResult {
    if running.is_empty() {
        return DecodeResult {
            end_ms: current_time_ms,
            ..DecodeResult::default()
        };
    }

    let total_context: usize = running
        .iter()
        .map(SglangRequest::current_sequence_len)
        .sum();
    let avg_context = total_context / running.len();
    let decode_time =
        config
            .perf_model
            .predict_decode_time(running.len(), total_context, avg_context);
    let unscaled_time = Duration::from_secs_f64(decode_time / 1000.0);
    let effective_ratio = config.speedup_ratio * config.decode_speedup_ratio;
    let total_time = if apply_speedup && effective_ratio > 0.0 && unscaled_time > Duration::ZERO {
        Duration::from_secs_f64(unscaled_time.as_secs_f64() / effective_ratio)
    } else {
        unscaled_time
    };

    let retracted = check_decode_mem(running, kv_manager, config);
    let retracted_any = !retracted.is_empty();
    let mut output_signals = Vec::with_capacity(running.len());
    let mut completed_indices = Vec::new();

    for (idx, req) in running.iter_mut().enumerate() {
        if kv_manager.cache().token_pool.available() == 0 {
            kv_manager.evict(1);
        }

        let crossing_page_boundary = req.current_sequence_len() + 1 > req.allocated_tokens;
        let last_idx = req.kv_indices.last().copied();
        let Some(new_idx) = kv_manager.allocate_decode_token(last_idx) else {
            tracing::warn!(uuid = %req.uuid, "Failed to allocate decode token, skipping output");
            continue;
        };

        req.kv_indices.push(new_idx);
        if crossing_page_boundary {
            req.allocated_tokens += config.block_size;
        }
        req.append_output_token(req.next_output_token());
        req.debug_assert_invariants(config.block_size);

        let is_complete = req.output_len() >= req.max_output_tokens;
        output_signals.push(OutputSignal {
            uuid: req.uuid,
            completed: is_complete,
        });

        if is_complete {
            let sequence = req.sequence_tokens();
            let tokens_to_cache = floor_to_block(sequence.len(), config.block_size);
            if req.kv_indices.len() > tokens_to_cache {
                kv_manager.free_indices(&req.kv_indices[tokens_to_cache..]);
            }

            if let Some(last_node) = req.last_node.take() {
                if tokens_to_cache > 0 {
                    kv_manager.cache_finished_req(
                        &sequence[..tokens_to_cache],
                        &req.kv_indices[..tokens_to_cache],
                        last_node,
                    );
                } else {
                    kv_manager.free_request(last_node);
                }
            }

            completed_indices.push(idx);
            continue;
        }

        cache_materialized_prefix(req, kv_manager, config);
        req.debug_assert_invariants(config.block_size);
    }

    for &idx in completed_indices.iter().rev() {
        running.swap_remove(idx);
    }

    DecodeResult {
        requests: retracted,
        output_signals,
        retracted_any,
        end_ms: current_time_ms + total_time.as_secs_f64() * 1000.0,
    }
}
