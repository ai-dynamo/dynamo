// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};

use uuid::Uuid;

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::SglangRequest;
use crate::cache::radix_cache::NodeId;
use crate::common::semantic::{
    FallbackReason, SegmentMode, SemanticOutcome, SemanticReusePlan, SemanticSimConfig,
    trim_and_gate,
};
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
    /// Blended-reuse accounting for this admission pass.
    pub(super) total_copied: usize,
    pub(super) total_repaired: usize,
    pub(super) accepted_plans: usize,
    /// Semantic outcomes newly decided this pass, in admission order.
    pub(super) plan_outcomes: Vec<(Uuid, SemanticOutcome)>,
}

/// Donor span that must stay resident: the furthest donor token any
/// Copied/Repaired segment reads from, independent of exact-prefix trim.
fn plan_donor_extent(plan: &SemanticReusePlan) -> usize {
    plan.segments
        .iter()
        .filter(|s| s.mode != SegmentMode::Recomputed)
        .map(|s| s.donor_range.1)
        .max()
        .unwrap_or(0)
}

/// Gate the plan against current cache state and pin the donor's radix path
/// so recipient allocation cannot evict it mid-"copy". The returned node must
/// be unlocked by the caller after allocation (success or OOM).
fn resolve_and_pin(
    plan: &SemanticReusePlan,
    recipient_tokens: &[u64],
    donor_registry: &HashMap<Uuid, Vec<u64>>,
    kv_manager: &mut SglangKvManager,
    sem: &SemanticSimConfig,
) -> Result<NodeId, FallbackReason> {
    if plan.donor_worker.is_some() {
        // v1 simulation resolves same-worker donors only.
        return Err(FallbackReason::DonorUnknown);
    }
    let donor_tokens = donor_registry
        .get(&plan.donor_uuid)
        .ok_or(FallbackReason::DonorUnknown)?;
    // Gate with the pre-eviction exact-prefix estimate. Final accounting
    // re-trims against the allocation's actual prefix length, which can only
    // be <= this estimate (eviction shrinks the tree), so gating here is the
    // conservative side.
    let exact_estimate = kv_manager.cache().prefix_match_len(recipient_tokens);
    trim_and_gate(
        plan,
        exact_estimate,
        recipient_tokens.len(),
        donor_tokens.len(),
        sem,
    )?;
    let pin_len = plan_donor_extent(plan).min(donor_tokens.len());
    if pin_len == 0 {
        return Err(FallbackReason::InvalidPlan);
    }
    let live = kv_manager
        .cache()
        .prefix_match_len(&donor_tokens[..pin_len]);
    if live < pin_len {
        return Err(FallbackReason::DonorEvicted);
    }
    let (_, donor_node) = kv_manager
        .cache_mut()
        .match_prefix(&donor_tokens[..pin_len]);
    kv_manager.cache_mut().inc_lock_ref(donor_node);
    Ok(donor_node)
}

pub(super) fn get_new_batch_prefill(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
    donor_registry: &HashMap<Uuid, Vec<u64>>,
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
    let mut total_copied = 0usize;
    let mut total_repaired = 0usize;
    let mut accepted_plans = 0usize;
    let mut plan_outcomes: Vec<(Uuid, SemanticOutcome)> = Vec::new();

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

        // Blended reuse: gate the plan and pin the donor BEFORE eviction and
        // allocation so the recipient's own allocation cannot evict the donor
        // KV it is about to "copy". Resolution happens once, at a first
        // admission that covers the whole input; chunked admissions fall back.
        let first_full_chunk = req.materialized_tokens == 0 && chunk_tokens == extend_input;
        let mut donor_pin: Option<NodeId> = None;
        let mut pre_fallback: Option<FallbackReason> = None;
        let mut plan_gated = false;
        if req.semantic_outcome.is_none()
            && let (Some(sem), Some(plan)) = (config.semantic.as_ref(), req.reuse_plan.as_ref())
        {
            if !first_full_chunk {
                pre_fallback = Some(FallbackReason::ChunkedUnsupported);
            } else {
                match resolve_and_pin(plan, &alloc_tokens, donor_registry, kv_manager, sem) {
                    Ok(node) => {
                        donor_pin = Some(node);
                        plan_gated = true;
                    }
                    Err(reason) => pre_fallback = Some(reason),
                }
            }
        }

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

        // The donor pin only needs to outlive eviction + allocation.
        if let Some(node) = donor_pin.take() {
            kv_manager.cache_mut().dec_lock_ref(node);
        }

        let Some(alloc) = alloc else {
            req.last_node = prev_node;
            rejected.push_back(req);
            oom = true;
            break;
        };

        if let Some(node) = prev_node {
            kv_manager.free_request(node);
        }

        // Finalize the semantic outcome against the allocation's actual exact
        // prefix (eviction between the gate and the allocation can only have
        // shrunk it, which re-trimming handles conservatively).
        if req.semantic_outcome.is_none()
            && let (Some(sem), Some(plan)) = (config.semantic.as_ref(), req.reuse_plan.as_ref())
        {
            let outcome = if let Some(reason) = pre_fallback {
                SemanticOutcome::Fallback { reason }
            } else if plan_gated {
                let donor_len = donor_registry
                    .get(&plan.donor_uuid)
                    .map(Vec::len)
                    .unwrap_or(0);
                match trim_and_gate(plan, alloc.prefix_len, req.prompt_len(), donor_len, sem) {
                    Ok(resolved) => {
                        total_copied += resolved.copied_tokens;
                        total_repaired += resolved.repaired_tokens;
                        accepted_plans += 1;
                        SemanticOutcome::Accepted {
                            copied_tokens: resolved.copied_tokens,
                            repaired_tokens: resolved.repaired_tokens,
                            recomputed_halo_tokens: resolved.recomputed_halo_tokens,
                        }
                    }
                    Err(reason) => SemanticOutcome::Fallback { reason },
                }
            } else {
                SemanticOutcome::Fallback {
                    reason: FallbackReason::Disabled,
                }
            };
            req.semantic_outcome = Some(outcome);
            plan_outcomes.push((req.uuid, outcome));
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
        total_copied,
        total_repaired,
        accepted_plans,
        plan_outcomes,
    }
}
