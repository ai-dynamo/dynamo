// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_kv_router::conditional_disagg::ConditionalDisaggDecisionInput;
use dynamo_kv_router::protocols::WorkerWithDpRank;

use super::PrefillRouter;
use crate::protocols::common::llm_backend::PreprocessedRequest;
use crate::session_affinity::AffinityTarget;

/// Conditional-disagg decision: which decode worker to pin the request to,
/// plus diagnostic counts for logging.
pub(super) struct ConditionalDisaggDecodeDecision {
    pub worker: WorkerWithDpRank,
    pub overlap_tokens: usize,
    pub net_new_tokens: usize,
}

impl PrefillRouter {
    /// Peek the decode router to see which decode worker would be picked, then
    /// consult the configured conditional-disagg policy.
    pub(super) async fn select_decode_worker_for_conditional_disagg(
        &self,
        req: &PreprocessedRequest,
        request_id: &str,
        decode_affinity_target: Option<AffinityTarget>,
    ) -> Result<Option<ConditionalDisaggDecodeDecision>> {
        if !self.router_mode.is_kv_routing() {
            return Ok(None);
        }

        let has_explicit_prefill_pin = req
            .routing
            .as_ref()
            .is_some_and(|routing| routing.prefill_worker_id.is_some());
        if has_explicit_prefill_pin {
            tracing::debug!(
                request_id,
                "Skipping conditional disagg because request has a preselected prefill worker"
            );
            return Ok(None);
        }

        let Some(decode_router) = self.decode_router.as_ref() else {
            tracing::debug!(
                request_id,
                "Skipping conditional disagg because decode router is unavailable"
            );
            return Ok(None);
        };

        let (routing_token_ids, block_mm_infos) = req.block_mm_routing_info();
        if routing_token_ids.is_empty() {
            return Ok(None);
        }

        let lora_name = req
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.clone());
        let cache_namespace = req
            .routing
            .as_ref()
            .and_then(|routing| routing.cache_namespace.clone());
        let priority_jump = req
            .routing
            .as_ref()
            .and_then(|routing| routing.priority_jump)
            .unwrap_or(0.0);
        let strict_priority = req
            .routing
            .as_ref()
            .and_then(|routing| routing.strict_priority)
            .unwrap_or(0);
        let expected_output_tokens = req
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        let allowed_worker_ids = req
            .routing
            .as_ref()
            .and_then(|routing| routing.allowed_worker_ids.clone());
        let affinity_pinned_worker = match decode_affinity_target {
            Some(target) => {
                let Some(dp_rank) = target
                    .dp_rank
                    .or_else(|| decode_router.unique_dp_rank_for_worker(target.worker_id))
                else {
                    tracing::debug!(
                        request_id,
                        worker_id = target.worker_id,
                        "Skipping conditional disagg because decode affinity target has no resolved DP rank"
                    );
                    return Ok(None);
                };
                Some(WorkerWithDpRank::new(target.worker_id, dp_rank))
            }
            None => None,
        };
        let request_pinned_worker = req.routing.as_ref().and_then(|routing| {
            let worker_id = routing.decode_worker_id.or(routing.backend_instance_id)?;
            let dp_rank = routing
                .dp_rank
                .or_else(|| decode_router.unique_dp_rank_for_worker(worker_id))?;
            Some(WorkerWithDpRank::new(worker_id, dp_rank))
        });
        let pinned_worker = request_pinned_worker.or(affinity_pinned_worker);
        let routing_constraints = req
            .routing
            .as_ref()
            .and_then(|routing| routing.routing_constraints.clone())
            .unwrap_or_default();

        let outcome = decode_router
            .find_best_match_details_without_admission(
                Some(request_id),
                routing_token_ids,
                block_mm_infos,
                req.router_config_override.as_ref(),
                false,
                lora_name,
                cache_namespace,
                priority_jump,
                strict_priority,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
            )
            .await?;
        let (worker, overlap_blocks, cached_tokens) = match outcome {
            crate::kv_router::FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                cached_tokens,
                ..
            } => (worker, overlap_blocks, cached_tokens),
            crate::kv_router::FindBestMatchOutcome::QueueRejected { .. } => {
                return Ok(None);
            }
        };

        let block_size = decode_router.block_size() as usize;
        let prompt_tokens = routing_token_ids.len();

        let input = ConditionalDisaggDecisionInput::new(prompt_tokens, cached_tokens);
        let net_new_tokens = input.net_new_tokens();
        let overlap_tokens = (overlap_blocks as usize) * block_size;

        let bypass = self
            .conditional_disagg_policy
            .should_bypass_remote_prefill(input)
            .await;

        tracing::debug!(
            request_id,
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            prompt_tokens,
            net_new_tokens,
            overlap_tokens,
            cached_tokens,
            bypass,
            "Conditional disagg decision"
        );

        if bypass {
            return Ok(Some(ConditionalDisaggDecodeDecision {
                worker,
                overlap_tokens,
                net_new_tokens,
            }));
        }

        Ok(None)
    }
}
