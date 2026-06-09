// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TRT-LLM `GUARANTEED_NO_EVICT` capacity policy: the scheduling decisions that
//! differ from vLLM, expressed as free functions over public scheduler types so
//! they need none of the vLLM core's internals.

use crate::common::protocols::SchedulingPolicy;
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::KvManager;

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
pub(crate) fn blocks_needed_to_finish(
    sequence: &ActiveSequence,
    block_size: usize,
    kv_manager: &KvManager,
) -> usize {
    let full_blocks = sequence.to_completion_blocks();
    if sequence.num_allocated_tokens() == 0 {
        let reusable_blocks =
            kv_manager.get_prefill_cost(sequence).active_cached_tokens / block_size;
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
pub(crate) fn available_blocks<'a>(
    running: impl Iterator<Item = &'a ActiveSequence>,
    num_gpu_blocks: usize,
    block_size: usize,
    kv_manager: &KvManager,
) -> usize {
    let reserved: usize = running
        .map(|sequence| blocks_needed_to_finish(sequence, block_size, kv_manager))
        .sum();
    let free = num_gpu_blocks.saturating_sub(kv_manager.num_active_blocks());
    free.saturating_sub(reserved)
}

/// Whether the policy is TRT-LLM `GUARANTEED_NO_EVICT` — the mode that both
/// reservation-gates admission and forbids preemption.
pub(crate) fn is_no_evict(policy: SchedulingPolicy) -> bool {
    policy == SchedulingPolicy::TrtllmGuaranteedNoEvict
}

/// TRT-LLM enqueue normalization: a no-evict request's `prompt + output` can
/// reserve at most the whole KV pool. Returns `max_output_tokens` clamped to the
/// room left after the prompt, or `None` if the prompt alone leaves no decode
/// room (the request can never run and should be rejected).
pub(crate) fn normalize_max_output_tokens(
    prompt_len: usize,
    max_output_tokens: usize,
    num_gpu_blocks: usize,
    block_size: usize,
) -> Option<usize> {
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
pub(crate) fn report_no_evict_violation() {
    debug_assert!(
        false,
        "no-evict invariant violated: trtllm GUARANTEED_NO_EVICT required preemption"
    );
    tracing::error!(
        "trtllm GUARANTEED_NO_EVICT required preemption; reservation under-counted physical KV demand"
    );
}
