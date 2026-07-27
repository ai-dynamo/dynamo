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

fn decode_page_growth_needed(
    running: &[SglangRequest],
    block_size: usize,
    max_burst: usize,
) -> usize {
    running
        .iter()
        .map(|req| {
            let burst = max_burst.min(req.remaining_output_tokens());
            let target =
                super::config::ceil_to_block(req.current_sequence_len() + burst, block_size);
            target.saturating_sub(req.allocated_tokens)
        })
        .sum()
}

fn decode_capacity_state(
    running: &[SglangRequest],
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
) -> (usize, usize, usize) {
    let actual_available =
        kv_manager.cache().available_tokens() + kv_manager.cache().evictable_size;
    // Full partial pages are already owned by PagePool and excluded from
    // `actual_available`; subtracting their slack again would double-charge it.
    let logical_available = actual_available;
    let page_growth_needed = decode_page_growth_needed(running, config.block_size, max_burst);

    (actual_available, logical_available, page_growth_needed)
}

pub(super) fn cache_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) {
    let aligned_tokens = req.page_aligned_materialized_tokens(config.block_size);
    if aligned_tokens == 0 || aligned_tokens <= req.cached_tokens() {
        return;
    }

    if !req.kv_lease.is_active() {
        panic!(
            "cache_materialized_prefix: request {} has aligned_tokens={aligned_tokens} but no active KV lease",
            req.uuid
        );
    }

    let sequence = &req.sequence_tokens[..aligned_tokens];
    kv_manager.extend_cached_prefix(sequence, &mut req.kv_lease);
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
        let (_actual_available, logical_available, page_growth_needed) =
            decode_capacity_state(running, kv_manager, config, max_burst);
        if logical_available >= page_growth_needed {
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
        kv_manager.retract(std::mem::take(&mut req.kv_lease));
        req.reset_for_retract();
        req.debug_assert_invariants(config.block_size);
        retracted.push(req);
    }

    let available = kv_manager.cache().available_tokens();
    let page_growth_needed = decode_page_growth_needed(running, config.block_size, max_burst);
    if available < page_growth_needed {
        kv_manager.evict(page_growth_needed - available);
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
    let tokens_to_cache = floor_to_block(request.current_sequence_len(), block_size);
    if !request.kv_lease.is_active() {
        return;
    }
    let lease = std::mem::take(&mut request.kv_lease);
    kv_manager.finish(request.sequence_prefix(tokens_to_cache), lease);
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

    // Terminal requests have no decode work and otherwise remain in `running` forever.
    let already_completed_indices = running
        .iter()
        .enumerate()
        .filter_map(|(idx, req)| (req.remaining_output_tokens() == 0).then_some(idx))
        .collect::<Vec<_>>();
    let mut output_signals = already_completed_indices
        .iter()
        .map(|&idx| {
            let req = &running[idx];
            OutputSignal {
                uuid: req.uuid,
                token_id: None,
                completed: true,
                rejected: false,
                handoff_delay_ms: compute_prefill_handoff_delay_ms(
                    config.worker_type,
                    true,
                    req.prompt_len(),
                    config.kv_transfer_bandwidth,
                    config.kv_bytes_per_token,
                ),
            }
        })
        .collect::<Vec<_>>();
    let mut completed_requests = already_completed_indices
        .iter()
        .rev()
        .map(|&idx| running.remove(idx))
        .collect::<Vec<_>>();
    completed_requests.reverse();

    if running.is_empty() {
        return DecodeResult {
            completed_requests,
            output_signals,
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
            completed_requests,
            output_signals,
            requests: retracted,
            retracted_any,
            end_ms: current_time_ms,
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

    let reserved_page_tokens = decode_page_growth_needed(running, config.block_size, max_burst);
    let reserved_pages = reserved_page_tokens / config.block_size;
    let Some(mut reservation) = kv_manager.reserve_decode_pages(reserved_pages) else {
        tracing::warn!(
            reserved_pages,
            "Failed to reserve speculative decode pages after capacity preflight"
        );
        return DecodeResult {
            completed_requests,
            output_signals,
            requests: retracted,
            retracted_any,
            end_ms: current_time_ms,
        };
    };

    output_signals.reserve(running.len());
    let mut completed_indices = Vec::new();

    for (idx, req) in running.iter_mut().enumerate() {
        let remaining = req.remaining_output_tokens();
        let burst = if config.worker_type == crate::common::protocols::WorkerType::Prefill {
            remaining.min(1)
        } else if let Some(sampler) = sampler.as_deref_mut() {
            sampler.sample_output_tokens(remaining)
        } else {
            remaining.min(1)
        };
        for _ in 0..burst {
            let crossing_page_boundary = req.current_sequence_len() + 1 > req.allocated_tokens;
            kv_manager.extend_decode(&mut req.kv_lease, &mut reservation);
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

    debug_assert!(reservation.len() <= reserved_pages);
    kv_manager.release_decode_reservation(reservation);

    let mut newly_completed_requests = Vec::with_capacity(completed_indices.len());
    for &idx in completed_indices.iter().rev() {
        newly_completed_requests.push(running.remove(idx));
    }
    newly_completed_requests.reverse();
    completed_requests.extend(newly_completed_requests);

    DecodeResult {
        requests: retracted,
        completed_requests,
        output_signals,
        retracted_any,
        end_ms: current_time_ms + total_time.as_secs_f64() * 1000.0,
    }
}
