// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::common::protocols::OutputSignal;
use crate::common::speculative::SpeculativeDecodeSampler;
use crate::common::utils::compute_prefill_handoff_delay_ms;
use crate::kv_manager::SglangKvManager;

use super::config::{SglangConfig, floor_to_block};
use super::request::SglangRequest;

#[derive(Default)]
pub(super) struct DecodeResult {
    pub(super) requests: Vec<SglangRequest>,
    pub(super) completed_requests: Vec<SglangRequest>,
    pub(super) output_signals: Vec<OutputSignal>,
    pub(super) retracted_any: bool,
    pub(super) end_ms: f64,
}

fn decode_capacity_state(
    running: &[SglangRequest],
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
) -> (usize, usize, usize) {
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
            let burst = max_burst.min(req.remaining_output_tokens());
            let target =
                super::config::ceil_to_block(req.current_sequence_len() + burst, config.block_size);
            target.saturating_sub(req.allocated_tokens)
        })
        .sum();

    (actual_available, logical_available, page_growth_needed)
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

    let last_node = req.last_node.unwrap_or_else(|| {
        panic!(
            "cache_materialized_prefix: request {} has aligned_tokens={aligned_tokens} but last_node is None",
            req.uuid
        )
    });

    let sequence = req.sequence_prefix(aligned_tokens);
    let new_last = kv_manager.cache_unfinished_req(
        &sequence,
        &req.kv_indices[..aligned_tokens],
        last_node,
        req.cached_tokens,
    );
    req.last_node = Some(new_last);
    req.cached_tokens = aligned_tokens;
    req.debug_assert_invariants(config.block_size);
}

#[cfg(test)]
pub(super) fn check_decode_mem(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) -> Vec<SglangRequest> {
    check_decode_mem_for_burst(running, kv_manager, config, 1)
}

fn check_decode_mem_for_burst(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
) -> Vec<SglangRequest> {
    let mut retracted = Vec::new();

    loop {
        let (actual_available, logical_available, page_growth_needed) =
            decode_capacity_state(running, kv_manager, config, max_burst);
        let needed = running
            .iter()
            .map(|req| max_burst.min(req.remaining_output_tokens()))
            .sum::<usize>();
        if actual_available >= needed && logical_available >= page_growth_needed {
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

        let mut req = running.remove(idx);
        kv_manager.free_indices(&req.kv_indices[req.cached_tokens..]);
        if let Some(last_node) = req.last_node.take() {
            kv_manager.free_request(last_node);
        }
        req.reset_for_retract();
        req.debug_assert_invariants(config.block_size);
        retracted.push(req);
    }

    let available = kv_manager.cache().token_pool.available();
    let needed = running
        .iter()
        .map(|req| max_burst.min(req.remaining_output_tokens()))
        .sum::<usize>();
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

#[cfg(test)]
pub(super) fn simulate_decode_step(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    current_time_ms: f64,
    apply_speedup: bool,
) -> DecodeResult {
    let mut result = simulate_decode_step_with_sampler(
        running,
        kv_manager,
        config,
        None,
        current_time_ms,
        apply_speedup,
    );
    for mut request in result.completed_requests.drain(..) {
        cleanup_completed_request(&mut request, kv_manager, config.block_size);
    }
    result
}

pub(super) fn cleanup_completed_request(
    request: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    block_size: usize,
) {
    let sequence = request.sequence_tokens();
    let tokens_to_cache = floor_to_block(sequence.len(), block_size);
    if request.kv_indices.len() > tokens_to_cache {
        kv_manager.free_indices(&request.kv_indices[tokens_to_cache..]);
    }

    let Some(last_node) = request.last_node.take() else {
        return;
    };
    if tokens_to_cache > 0 {
        kv_manager.cache_finished_req(
            &sequence[..tokens_to_cache],
            &request.kv_indices[..tokens_to_cache],
            last_node,
            request.cached_tokens,
        );
    } else {
        kv_manager.free_request(last_node);
    }
}

pub(super) fn simulate_decode_step_with_sampler(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    mut sampler: Option<&mut SpeculativeDecodeSampler>,
    current_time_ms: f64,
    apply_speedup: bool,
) -> DecodeResult {
    if running.is_empty() {
        return DecodeResult {
            end_ms: current_time_ms,
            ..DecodeResult::default()
        };
    }

    let max_burst = if config.worker_type == crate::common::protocols::WorkerType::Prefill {
        1
    } else {
        config.speculative_max_tokens.unwrap_or(1)
    };
    let retracted = check_decode_mem_for_burst(running, kv_manager, config, max_burst);
    let retracted_any = !retracted.is_empty();
    if running.is_empty() {
        return DecodeResult {
            requests: retracted,
            retracted_any,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        };
    }

    let total_context: usize = running
        .iter()
        .map(SglangRequest::current_sequence_len)
        .sum();
    let avg_context = total_context / running.len();
    let active_kv_tokens = total_context.min(config.total_kv_tokens);
    let decode_time = config.perf_model.predict_decode_time(
        running.len(),
        active_kv_tokens,
        avg_context,
        config.total_kv_tokens,
    );
    let unscaled_time = Duration::from_secs_f64(decode_time / 1000.0);
    let effective_ratio = config.speedup_ratio * config.decode_speedup_ratio;
    let total_time = if apply_speedup && effective_ratio > 0.0 && unscaled_time > Duration::ZERO {
        Duration::from_secs_f64(unscaled_time.as_secs_f64() / effective_ratio)
    } else {
        unscaled_time
    };

    let reserved_tokens = running
        .iter()
        .map(|req| max_burst.min(req.remaining_output_tokens()))
        .sum();
    let Some(mut reservation) = kv_manager.reserve_decode_tokens(reserved_tokens) else {
        tracing::warn!(
            reserved_tokens,
            "Failed to reserve speculative decode tokens after capacity preflight"
        );
        return DecodeResult {
            requests: retracted,
            retracted_any,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        };
    };

    let sampled_bursts = running
        .iter()
        .map(|req| {
            let remaining = req.remaining_output_tokens();
            if config.worker_type == crate::common::protocols::WorkerType::Prefill {
                remaining.min(1)
            } else if let Some(sampler) = sampler.as_deref_mut() {
                sampler.sample_output_tokens(remaining)
            } else {
                remaining.min(1)
            }
        })
        .collect::<Vec<_>>();
    let mut output_signals = Vec::with_capacity(sampled_bursts.iter().copied().sum::<usize>());
    let mut completed_indices = Vec::new();

    for (idx, (req, burst)) in running
        .iter_mut()
        .zip(sampled_bursts.into_iter())
        .enumerate()
    {
        for _ in 0..burst {
            let crossing_page_boundary = req.current_sequence_len() + 1 > req.allocated_tokens;
            let last_idx = req.kv_indices.last().copied();
            let new_idx = reservation.take();
            kv_manager.publish_decode_token(new_idx, last_idx);

            req.kv_indices.push(new_idx);
            if crossing_page_boundary {
                req.allocated_tokens += config.block_size;
            }
            let token_id = req.next_output_token();
            req.append_output_token(token_id);
            req.debug_assert_invariants(config.block_size);

            let is_complete = req.output_len() >= req.max_output_tokens;
            output_signals.push(OutputSignal {
                uuid: req.uuid,
                token_id: Some(token_id),
                completed: is_complete,
                rejected: false,
                handoff_delay_ms: compute_prefill_handoff_delay_ms(
                    config.worker_type,
                    is_complete,
                    req.prompt_len(),
                    config.kv_transfer_bandwidth,
                    config.kv_bytes_per_token,
                ),
            });

            if is_complete {
                completed_indices.push(idx);
                break;
            }

            cache_materialized_prefix(req, kv_manager, config);
            req.debug_assert_invariants(config.block_size);
        }
    }

    debug_assert_eq!(
        reservation.len(),
        reserved_tokens.saturating_sub(output_signals.len())
    );
    kv_manager.release_decode_reservation(reservation);

    let mut completed_requests = Vec::with_capacity(completed_indices.len());
    for &idx in completed_indices.iter().rev() {
        completed_requests.push(running.remove(idx));
    }
    completed_requests.reverse();

    DecodeResult {
        requests: retracted,
        completed_requests,
        output_signals,
        retracted_any,
        end_ms: current_time_ms + total_time.as_secs_f64() * 1000.0,
    }
}
