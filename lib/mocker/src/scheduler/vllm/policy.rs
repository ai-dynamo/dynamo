// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific policy for the shared vLLM/TRT-LLM scheduler core.

use crate::common::protocols::PrefillCost;
use crate::common::protocols::SchedulingPolicy;
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::KvManager;

#[derive(Debug)]
pub(super) enum AdmissionDecision {
    Admit { prefill_cost: PrefillCost },
    Wait,
    Reject,
}

/// Decide whether the FIFO head can enter the shared scheduler core.
///
/// vLLM reserves only the current known sequence. TRT-LLM
/// `GUARANTEED_NO_EVICT` reserves the request through its maximum completion
/// and accounts for the completion reservations of running requests.
pub(super) fn decide_waiting_admission<'a>(
    policy: SchedulingPolicy,
    sequence: &ActiveSequence,
    is_fresh: bool,
    running: impl Iterator<Item = &'a ActiveSequence>,
    num_gpu_blocks: usize,
    block_size: usize,
    kv_manager: &KvManager,
) -> AdmissionDecision {
    if is_fresh {
        match policy {
            SchedulingPolicy::Vllm => {
                // TODO: Carry vLLM's max_model_len explicitly. Upstream bounds prompt
                // plus generated tokens by max_model_len and sizes KV for one
                // max-length sequence. Until the mocker models that value, total
                // worker KV is only a proxy for the one-time fresh-sequence admission
                // cap; it does not cap future output length.
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

    let prefill_cost = kv_manager.get_prefill_cost(sequence);
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
        AdmissionDecision::Admit { prefill_cost }
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
    kv_manager: &KvManager,
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
    kv_manager: &KvManager,
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
