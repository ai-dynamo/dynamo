// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific policy for the shared vLLM/TRT-LLM scheduler core.

use crate::common::protocols::{PrefillCost, SchedulingPolicy};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::G1Manager;

#[derive(Debug)]
pub(super) enum AdmissionDecision {
    Admit {
        prefill_cost: PrefillCost,
        g1_cached_tokens: usize,
    },
    Wait,
    Reject,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WaitingAdmissionConfig {
    pub(super) policy: SchedulingPolicy,
    pub(super) num_gpu_blocks: usize,
    pub(super) block_size: usize,
    pub(super) mtp_enabled: bool,
}

pub(super) fn should_reject_for_model_len(
    policy: SchedulingPolicy,
    sequence: &ActiveSequence,
    max_model_len: Option<usize>,
) -> bool {
    policy == SchedulingPolicy::Vllm
        && max_model_len.is_some_and(|limit| sequence.num_input_tokens() >= limit)
}

/// Number of additional tokens the request may generate before reaching
/// either its requested output length or the model sequence-length limit.
pub(super) fn remaining_generation_tokens(
    sequence: &ActiveSequence,
    max_model_len: Option<usize>,
) -> usize {
    let requested_remaining = sequence
        .max_output_tokens()
        .saturating_sub(sequence.generated_tokens());
    let context_remaining = max_model_len
        .map(|limit| limit.saturating_sub(sequence.len()))
        .unwrap_or(usize::MAX);
    requested_remaining.min(context_remaining)
}

pub(super) fn generation_complete(sequence: &ActiveSequence, max_model_len: Option<usize>) -> bool {
    remaining_generation_tokens(sequence, max_model_len) == 0
}

/// Apply vLLM's EAGLE/MTP prefix-cache rule.
///
/// The drafter needs hidden states from the final matched block, so vLLM
/// removes one block from every non-empty prefix-cache hit and recomputes it
/// during prefill. Keep that backend-specific accounting here rather than in
/// the shared scheduler core.
pub(super) fn apply_mtp_prefix_recompute(
    policy: SchedulingPolicy,
    block_size: usize,
    mtp_enabled: bool,
    mut prefill_cost: PrefillCost,
) -> PrefillCost {
    if policy != SchedulingPolicy::Vllm || !mtp_enabled || prefill_cost.cached_tokens < block_size {
        return prefill_cost;
    }

    prefill_cost.cached_tokens -= block_size;
    prefill_cost.new_tokens += block_size;
    prefill_cost.new_blocks += 1;
    prefill_cost.active_cached_tokens = prefill_cost
        .active_cached_tokens
        .min(prefill_cost.cached_tokens);
    prefill_cost
}

/// Apply the ordinary vLLM prefix-cache rule before any speculative-decoding
/// adjustment. A request whose complete known context is cached must still
/// recompute its final token to produce logits. For a preempted request this
/// context includes retained generated tokens, matching vLLM's
/// `request.num_tokens - 1` lookup bound. Because the shared scheduler
/// allocates whole blocks, an exactly block-aligned context recomputes its
/// final block.
///
/// TensorRT-LLM uses the same physical G1 manager but owns its compute policy,
/// so this adjustment is deliberately selected by [`SchedulingPolicy`] rather
/// than embedded in the block manager.
pub(super) fn apply_prefix_recompute(
    policy: SchedulingPolicy,
    known_tokens: usize,
    block_size: usize,
    mtp_enabled: bool,
    requires_logits: bool,
    mut prefill_cost: PrefillCost,
) -> PrefillCost {
    if !requires_logits {
        return prefill_cost;
    }

    if policy == SchedulingPolicy::Vllm {
        let max_cached_tokens = known_tokens
            .saturating_sub(1)
            .checked_div(block_size)
            .unwrap_or(0)
            .saturating_mul(block_size);
        if prefill_cost.cached_tokens > max_cached_tokens {
            let recompute_tokens = prefill_cost.cached_tokens - max_cached_tokens;
            debug_assert_eq!(recompute_tokens % block_size, 0);
            prefill_cost.cached_tokens = max_cached_tokens;
            prefill_cost.active_cached_tokens =
                prefill_cost.active_cached_tokens.min(max_cached_tokens);
            prefill_cost.new_tokens = prefill_cost.new_tokens.saturating_add(recompute_tokens);
            prefill_cost.new_blocks = prefill_cost
                .new_blocks
                .saturating_add(recompute_tokens / block_size);
        }
    }

    apply_mtp_prefix_recompute(policy, block_size, mtp_enabled, prefill_cost)
}

/// Decide whether the FIFO head can enter the shared scheduler core.
///
/// vLLM reserves only the current known sequence. TRT-LLM
/// `GUARANTEED_NO_EVICT` reserves the request through its maximum completion
/// and accounts for the completion reservations of running requests.
pub(super) fn decide_waiting_admission<'a>(
    config: WaitingAdmissionConfig,
    sequence: &ActiveSequence,
    is_fresh: bool,
    running: impl Iterator<Item = &'a ActiveSequence>,
    kv_manager: &G1Manager,
) -> AdmissionDecision {
    let WaitingAdmissionConfig {
        policy,
        num_gpu_blocks,
        block_size,
        mtp_enabled,
    } = config;

    if is_fresh {
        match policy {
            SchedulingPolicy::Vllm => {
                // Total worker KV remains a fallback one-time admission cap
                // when max_model_len is unset or larger than the KV pool.
                if sequence.current_known_blocks() > num_gpu_blocks {
                    return AdmissionDecision::Reject;
                }
            }
            SchedulingPolicy::TrtllmGuaranteedNoEvict => {
                if sequence.to_completion_blocks() > num_gpu_blocks {
                    return AdmissionDecision::Reject;
                }
            }
        }
    }

    let raw_prefill_cost = kv_manager.get_prefill_cost(sequence);
    let g1_cached_tokens = raw_prefill_cost.cached_tokens;
    let prefill_cost = apply_prefix_recompute(
        policy,
        sequence.len(),
        block_size,
        mtp_enabled,
        !generation_complete(sequence, None),
        raw_prefill_cost,
    );
    let available = match policy {
        SchedulingPolicy::Vllm => num_gpu_blocks.saturating_sub(kv_manager.num_active_blocks()),
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            available_blocks(running, num_gpu_blocks, block_size, kv_manager)
        }
    };
    let needed = match policy {
        SchedulingPolicy::Vllm => sequence
            .current_known_blocks()
            .saturating_sub(prefill_cost.active_cached_tokens / block_size),
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            blocks_needed_to_finish(sequence, block_size, kv_manager, Some(&prefill_cost))
        }
    };

    if needed > available {
        AdmissionDecision::Wait
    } else {
        AdmissionDecision::Admit {
            prefill_cost,
            g1_cached_tokens,
        }
    }
}

/// Blocks a request still needs to reserve to run to completion under the
/// TRT-LLM `GUARANTEED_NO_EVICT` policy.
///
/// ```text
/// needed = ceil((prompt_len + max_output_tokens) / block_size)
///          - blocks_already_held
///          - active_cached_prefix_blocks   (waiting candidates only)
/// ```
///
/// For a running request, the blocks it already holds are physical (counted in
/// the KV manager's active blocks), so only the remaining footprint is reserved.
/// For a waiting candidate, only the active cached prefix is discounted
/// (`active_cached_tokens`).
fn blocks_needed_to_finish(
    sequence: &ActiveSequence,
    block_size: usize,
    kv_manager: &G1Manager,
    prefill_cost: Option<&PrefillCost>,
) -> usize {
    let full_blocks = sequence.to_completion_blocks();
    if sequence.num_allocated_tokens() == 0 {
        let reusable_blocks = prefill_cost
            .map(|cost| cost.active_cached_tokens)
            .unwrap_or_else(|| kv_manager.get_prefill_cost(sequence).active_cached_tokens)
            / block_size;
        full_blocks.saturating_sub(reusable_blocks)
    } else {
        let allocated_blocks = sequence.num_allocated_tokens().div_ceil(block_size);
        full_blocks.saturating_sub(allocated_blocks)
    }
}

/// Free blocks remaining after reserving every running request's to-completion
/// footprint. A waiting candidate may be admitted iff its
/// [`blocks_needed_to_finish`] is `<=` this value.
///
/// `running` yields the active sequence of each currently-running request;
/// `num_gpu_blocks` is the KV pool size and `kv_manager` supplies the count of
/// physically allocated blocks.
fn available_blocks<'a>(
    running: impl Iterator<Item = &'a ActiveSequence>,
    num_gpu_blocks: usize,
    block_size: usize,
    kv_manager: &G1Manager,
) -> usize {
    let reserved: usize = running
        .map(|sequence| blocks_needed_to_finish(sequence, block_size, kv_manager, None))
        .sum();
    let free = num_gpu_blocks.saturating_sub(kv_manager.num_active_blocks());
    free.saturating_sub(reserved)
}

pub(super) fn allows_preemption(policy: SchedulingPolicy) -> bool {
    policy == SchedulingPolicy::Vllm
}

pub(super) fn supports_destination_reservation(policy: SchedulingPolicy) -> bool {
    policy == SchedulingPolicy::Vllm
}

/// TRT-LLM enqueue normalization: a no-evict request's `prompt + output` can
/// reserve at most the whole KV pool. Returns `max_output_tokens` clamped to the
/// room left after the prompt, or `None` if the prompt alone leaves no decode
/// room (the request can never run and should be rejected).
pub(super) fn normalize_max_output_tokens(
    policy: SchedulingPolicy,
    prompt_len: usize,
    max_output_tokens: usize,
    num_gpu_blocks: usize,
    block_size: usize,
) -> Option<usize> {
    if policy == SchedulingPolicy::Vllm {
        return Some(max_output_tokens);
    }
    let capacity_tokens = num_gpu_blocks.saturating_mul(block_size);
    if prompt_len >= capacity_tokens {
        return None;
    }
    Some(max_output_tokens.min(capacity_tokens - prompt_len))
}

/// Fail loudly when the no-evict invariant is violated.
///
/// Under `GUARANTEED_NO_EVICT` the capacity gate reserves blocks for every
/// admitted request up front, so a preemption should never be required.
/// Reaching the preemption path means the reservation under-counted physical
/// KV demand (e.g. a reusable prefix block was evicted before the request
/// claimed it). A silent preempt would still produce output but no longer
/// represent TRT-LLM, degrading timing fidelity undetectably — so debug builds
/// assert and release builds log and decline to preempt.
pub(super) fn report_no_preemption_violation() {
    debug_assert!(
        false,
        "no-evict invariant violated: trtllm GUARANTEED_NO_EVICT required preemption"
    );
    tracing::error!(
        "trtllm GUARANTEED_NO_EVICT required preemption; reservation under-counted physical KV demand"
    );
}

#[cfg(test)]
mod tests;
