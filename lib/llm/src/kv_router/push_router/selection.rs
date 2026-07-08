// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use dynamo_kv_router::{
    RouterConfigOverride,
    indexer::RoutingDecisionHashes,
    protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank},
    scheduling::RoutingEligibility,
};
use dynamo_runtime::{dynamo_nvtx_range, pipeline::Error};

use crate::{
    kv_router::{FindBestMatchOutcome, push_router::KvPushRouter},
    preprocessor::PreprocessedRequest,
    protocols::{
        TokenIdType,
        common::{preprocessor::RoutingHints, timing::RequestPhase},
    },
};

pub(super) struct WorkerSelection {
    pub(super) instance_id: u64,
    pub(super) dp_rank: u32,
    pub(super) overlap_amount: u32,
    pub(super) effective_overlap_blocks: f64,
    pub(super) cached_tokens: usize,
    pub(super) routing_hashes: Option<RoutingDecisionHashes>,
    pub(super) scheduler_tracked: bool,
}

#[derive(Clone, Copy)]
pub(super) struct RoutingRequestParts<'a> {
    pub(super) token_ids: &'a [TokenIdType],
    pub(super) block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
}

impl<'a> RoutingRequestParts<'a> {
    pub(super) fn new(request: &'a PreprocessedRequest) -> Self {
        let (token_ids, block_mm_infos) = request.block_mm_routing_info();
        Self {
            token_ids,
            block_mm_infos,
        }
    }
}

pub(super) struct SelectionOptions {
    pub(super) affinity_worker: Option<WorkerWithDpRank>,
    pub(super) policy_class: Option<String>,
    pub(super) session_id: Option<String>,
}

struct BestMatchArgs<'a> {
    context_id: &'a str,
    routing_parts: RoutingRequestParts<'a>,
    router_config_override: Option<&'a RouterConfigOverride>,
    update_states: bool,
    return_routing_hashes: bool,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    priority_jump: f64,
    strict_priority: u32,
    policy_class: Option<String>,
    session_id: Option<String>,
    expected_output_tokens: Option<u32>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    routing_constraints: RoutingConstraints,
    scheduler_tracked: bool,
}

impl KvPushRouter {
    async fn select_best_match(&self, args: BestMatchArgs<'_>) -> Result<WorkerSelection, Error> {
        let outcome = self
            .chooser
            .find_best_match_details_with_policy_class(
                Some(args.context_id),
                args.routing_parts.token_ids,
                args.routing_parts.block_mm_infos,
                args.router_config_override,
                args.update_states,
                args.return_routing_hashes,
                args.lora_name,
                args.cache_namespace,
                args.priority_jump,
                args.strict_priority,
                args.policy_class,
                args.session_id,
                args.expected_output_tokens,
                args.pinned_worker,
                args.allowed_worker_ids,
                args.routing_constraints,
            )
            .await?;

        match outcome {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                effective_overlap_blocks,
                cached_tokens,
                routing_hashes,
            } => Ok(WorkerSelection {
                instance_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                overlap_amount: overlap_blocks,
                effective_overlap_blocks,
                cached_tokens,
                routing_hashes,
                scheduler_tracked: args.scheduler_tracked,
            }),
            FindBestMatchOutcome::QueueRejected { rejection } => Err(rejection.into()),
        }
    }

    /// Select a worker using either a phase-specific pin or KV overlap.
    pub(super) async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        routing_parts: RoutingRequestParts<'_>,
        phase: RequestPhase,
        is_query_only: bool,
        options: SelectionOptions,
    ) -> Result<WorkerSelection, Error> {
        let _nvtx_select = dynamo_nvtx_range!("route.select_worker");
        let routing = request.routing.as_ref();
        let lora_name = routing.and_then(|routing| routing.lora_name.clone());
        let cache_namespace = routing.and_then(|routing| routing.cache_namespace.clone());
        let priority_jump = routing
            .and_then(|routing| routing.priority_jump)
            .unwrap_or(0.0);
        let strict_priority = routing
            .and_then(|routing| routing.strict_priority)
            .unwrap_or(0);
        let expected_output_tokens = routing.and_then(|routing| routing.expected_output_tokens);
        let allowed_worker_ids = routing.and_then(|routing| routing.allowed_worker_ids.clone());
        let return_routing_hashes =
            !is_query_only && self.chooser.indexer().records_routing_decisions();
        let routing_constraints = routing
            .and_then(|routing| routing.routing_constraints.clone())
            .unwrap_or_default();
        let explicit_pin = pinned_worker_hint(phase, routing);
        let SelectionOptions {
            affinity_worker,
            policy_class,
            session_id,
        } = options;
        let affinity_pin = affinity_worker.map(|worker| (worker.worker_id, Some(worker.dp_rank)));
        let Some((pinned_worker_id, requested_dp_rank)) =
            merge_affinity_pin(explicit_pin, affinity_pin)
        else {
            let _nvtx_kv = dynamo_nvtx_range!("route.kv_match");
            let selection = self
                .select_best_match(BestMatchArgs {
                    context_id,
                    routing_parts,
                    router_config_override: request.router_config_override.as_ref(),
                    update_states: !is_query_only,
                    return_routing_hashes,
                    lora_name,
                    cache_namespace,
                    priority_jump,
                    strict_priority,
                    policy_class,
                    session_id,
                    expected_output_tokens,
                    pinned_worker: None,
                    allowed_worker_ids,
                    routing_constraints: routing_constraints.clone(),
                    scheduler_tracked: !is_query_only,
                })
                .await?;

            if !is_query_only {
                let total_blocks = routing_parts
                    .token_ids
                    .len()
                    .div_ceil(self.chooser.block_size() as usize);
                // tests/utils/router_logs.py parses the structured fields on this event.
                tracing::debug!(
                    request_id = %context_id,
                    worker_id = selection.instance_id,
                    dp_rank = selection.dp_rank,
                    overlap_blocks = selection.overlap_amount,
                    total_blocks,
                    "[ROUTING] Best: worker_{} dp_rank={} with {}/{} blocks overlap",
                    selection.instance_id,
                    selection.dp_rank,
                    selection.overlap_amount,
                    total_blocks,
                );
            }

            return Ok(selection);
        };
        let cache_namespace = routing.and_then(|routing| routing.cache_namespace.clone());

        let pinned_worker = resolve_pinned_worker_rank(
            pinned_worker_id,
            requested_dp_rank,
            self.chooser.unique_dp_rank_for_worker(pinned_worker_id),
        )?;
        {
            let configs = self.chooser.workers_with_configs.borrow();
            let eligibility = RoutingEligibility::new(
                allowed_worker_ids.as_ref(),
                None,
                Some(pinned_worker),
                &routing_constraints,
            );
            if let Err(error) = eligibility.validate_worker_rank(&configs, pinned_worker) {
                return Err(anyhow::anyhow!(
                    "Pinned worker {} dp_rank {} is not eligible: {error}",
                    pinned_worker.worker_id,
                    pinned_worker.dp_rank
                ));
            }
        }

        tracing::debug!(
            worker_id = pinned_worker.worker_id,
            dp_rank = pinned_worker.dp_rank,
            ?phase,
            "Routing to specified worker"
        );

        self.select_best_match(BestMatchArgs {
            context_id,
            routing_parts,
            router_config_override: request.router_config_override.as_ref(),
            update_states: !is_query_only,
            return_routing_hashes,
            lora_name,
            cache_namespace,
            priority_jump,
            strict_priority,
            policy_class,
            session_id,
            expected_output_tokens,
            pinned_worker: Some(pinned_worker),
            allowed_worker_ids,
            routing_constraints,
            scheduler_tracked: !is_query_only,
        })
        .await
    }
}

fn merge_affinity_pin(
    explicit: Option<(u64, Option<u32>)>,
    affinity: Option<(u64, Option<u32>)>,
) -> Option<(u64, Option<u32>)> {
    match (explicit, affinity) {
        (Some((worker_id, None)), Some((affinity_worker_id, affinity_rank)))
            if worker_id == affinity_worker_id =>
        {
            Some((worker_id, affinity_rank))
        }
        (Some(explicit), _) => Some(explicit),
        (None, affinity) => affinity,
    }
}

fn resolve_pinned_worker_rank(
    worker_id: WorkerId,
    requested_dp_rank: Option<u32>,
    unique_dp_rank: Option<u32>,
) -> Result<WorkerWithDpRank, Error> {
    let Some(dp_rank) = requested_dp_rank.or(unique_dp_rank) else {
        return Err(anyhow::anyhow!(
            "Pinned worker {worker_id} requires an explicit dp_rank because it has multiple or unknown DP ranks"
        ));
    };

    Ok(WorkerWithDpRank::new(worker_id, dp_rank))
}

fn pinned_worker_hint(
    phase: RequestPhase,
    routing: Option<&RoutingHints>,
) -> Option<(u64, Option<u32>)> {
    let routing = routing?;
    match phase {
        RequestPhase::Prefill => {
            let worker_id = routing.prefill_worker_id.or(routing.backend_instance_id)?;
            let dp_rank = routing.prefill_dp_rank.or(routing.dp_rank);
            Some((worker_id, dp_rank))
        }
        RequestPhase::Decode => {
            let worker_id = routing.decode_worker_id.or(routing.backend_instance_id)?;
            Some((worker_id, routing.dp_rank))
        }
        RequestPhase::Aggregated => {
            let worker_id = routing.decode_worker_id.or(routing.backend_instance_id)?;
            Some((worker_id, routing.dp_rank))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use dynamo_kv_router::{
        protocols::{RoutingConstraints, WorkerWithDpRank},
        scheduling::{RoutingEligibility, WorkerEligibilityError},
    };

    use super::{merge_affinity_pin, pinned_worker_hint, resolve_pinned_worker_rank};
    use crate::{
        local_model::runtime_config::ModelRuntimeConfig,
        protocols::common::{preprocessor::RoutingHints, timing::RequestPhase},
    };

    #[test]
    fn resolve_pinned_worker_rank_uses_explicit_rank_including_zero() {
        let worker = resolve_pinned_worker_rank(7, Some(0), Some(3)).unwrap();
        assert_eq!(worker.worker_id, 7);
        assert_eq!(worker.dp_rank, 0);
    }

    #[test]
    fn resolve_pinned_worker_rank_uses_unique_rank_when_unset() {
        let worker = resolve_pinned_worker_rank(7, None, Some(3)).unwrap();
        assert_eq!(worker.worker_id, 7);
        assert_eq!(worker.dp_rank, 3);
    }

    #[test]
    fn resolve_pinned_worker_rank_rejects_unresolved_rank() {
        let error = resolve_pinned_worker_rank(7, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires an explicit dp_rank"));
    }

    #[test]
    fn affinity_pin_supplies_rank_for_matching_explicit_worker() {
        assert_eq!(
            merge_affinity_pin(Some((7, None)), Some((7, Some(0)))),
            Some((7, Some(0)))
        );
        assert_eq!(
            merge_affinity_pin(Some((7, Some(2))), Some((7, Some(3)))),
            Some((7, Some(2)))
        );
    }

    #[test]
    fn pinned_worker_hint_prefill_uses_prefill_worker_before_backend() {
        let routing = RoutingHints {
            backend_instance_id: Some(1),
            prefill_worker_id: Some(2),
            dp_rank: Some(3),
            prefill_dp_rank: Some(4),
            ..Default::default()
        };

        assert_eq!(
            pinned_worker_hint(RequestPhase::Prefill, Some(&routing)),
            Some((2, Some(4)))
        );
    }

    #[test]
    fn pinned_worker_hint_decode_uses_decode_worker_before_backend() {
        let routing = RoutingHints {
            backend_instance_id: Some(1),
            decode_worker_id: Some(5),
            dp_rank: Some(6),
            ..Default::default()
        };

        assert_eq!(
            pinned_worker_hint(RequestPhase::Decode, Some(&routing)),
            Some((5, Some(6)))
        );
    }

    #[test]
    fn pinned_worker_hint_aggregated_uses_decode_worker_before_backend() {
        let routing = RoutingHints {
            backend_instance_id: Some(9),
            decode_worker_id: Some(5),
            dp_rank: Some(7),
            ..Default::default()
        };

        assert_eq!(
            pinned_worker_hint(RequestPhase::Aggregated, Some(&routing)),
            Some((5, Some(7)))
        );
    }

    #[test]
    fn affinity_validation_ignores_transient_overload() {
        let worker = WorkerWithDpRank::new(7, 0);
        let configs = HashMap::from([(7, ModelRuntimeConfig::default())]);
        let constraints = RoutingConstraints::default();
        let overloaded = HashSet::from([7]);
        let scheduling_eligibility =
            RoutingEligibility::new(None, Some(&overloaded), Some(worker), &constraints);
        let affinity_eligibility = RoutingEligibility::new(None, None, Some(worker), &constraints);

        assert_eq!(
            scheduling_eligibility
                .validate_worker_rank(&configs, worker)
                .err(),
            Some(WorkerEligibilityError::WorkerOverloaded { worker_id: 7 })
        );
        assert!(
            affinity_eligibility
                .validate_worker_rank(&configs, worker)
                .is_ok()
        );
    }
}
