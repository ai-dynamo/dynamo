// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicU8;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::{
    PrefillLoadEstimator, config::RouterConfigOverride, protocols::RoutingConstraints,
    scheduling::QueueRejection,
};
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, Operator, RouterMode, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated},
};

use crate::{
    discovery::ModelManager,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{BootstrapInfo, PrefillResult, TraceLink},
        timing::{RequestPhase, RequestTracker},
    },
};

mod activation;
mod admission;
mod query;

use admission::InnerPrefillRouter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum PrefillLifecycleState {
    Pending = 0,
    Active = 1,
    Unavailable = 2,
}

impl TryFrom<u8> for PrefillLifecycleState {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            value if value == Self::Pending as u8 => Ok(Self::Pending),
            value if value == Self::Active as u8 => Ok(Self::Active),
            value if value == Self::Unavailable as u8 => Ok(Self::Unavailable),
            value => Err(value),
        }
    }
}

impl PrefillLifecycleState {
    fn from_atomic(value: u8) -> Self {
        Self::try_from(value)
            .unwrap_or_else(|value| panic!("invalid prefill lifecycle state: {value}"))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("Prefill router not yet activated")]
    NotActivated,

    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

enum PrefillOutcome {
    Bootstrap {
        bootstrap_info: BootstrapInfo,
        worker_id: u64,
    },
    Completed {
        result: PrefillResult,
        worker_id: u64,
        worker_link: Option<TraceLink>,
    },
}

struct PreparedPrefill {
    worker_id: u64,
    bootstrap_info: Option<BootstrapInfo>,
    topology_constraints: Option<RoutingConstraints>,
}

/// Advisory prefill worker selection result.
pub enum PrefillQueryOutcome {
    Routed {
        worker_id: u64,
        dp_rank: Option<u32>,
    },
    QueueRejected {
        rejection: QueueRejection,
    },
}

struct PrefillCompletion {
    result: PrefillResult,
    worker_link: Option<TraceLink>,
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: `query_instance_id` annotation present → returns worker IDs without execution
/// - Pre-routed: `prefill_worker_id`/`decode_worker_id` set → routes to specified workers
/// - Normal: Worker IDs determined by router based on KV cache state
pub struct PrefillRouter {
    prefill_router: OnceLock<InnerPrefillRouter>,
    model_manager: Arc<ModelManager>,
    endpoint_id: OnceLock<EndpointId>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    enforce_disagg: bool,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    /// Model name (used for logging / lifecycle messages).
    model_name: String,
    /// Namespace (used for logging / lifecycle messages).
    namespace: String,
    is_eagle: bool,
    /// Initialization and worker availability state.
    lifecycle: AtomicU8,
}

impl Drop for PrefillRouter {
    fn drop(&mut self) {
        tracing::debug!("Dropping PrefillRouter, cancelling background activation task");
        self.cancel_token.cancel();
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for PrefillRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        // Extract request data while preserving context
        let (mut req, context) = request.into_parts();
        let request_id = context.id().to_string();
        let metadata = context.metadata().clone();
        let engine_ctx = context.context();

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If prefill router is not activated (no prefill workers discovered) or has been
        // deactivated (all prefill workers died), this is aggregated mode -- route directly
        // to decode. With --enforce-disagg, fail instead of falling back.
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            if self.enforce_disagg {
                return Err(anyhow::anyhow!(PrefillError::NotActivated));
            }
            return next.generate(context.map(|_| req)).await;
        }

        // Ensure tracker exists for routing decisions in disaggregated mode.
        // Create one if not provided by the upstream DeltaGenerator.
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.as_ref().unwrap();
        let prefill_phase_barrier = tracker.set_phase(RequestPhase::Prefill).await;

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set)
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // Try to resolve prefill worker upfront: if we can get bootstrap info early,
        // spawn prefill in background and proceed to decode immediately.
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id);

        if self.router_mode.is_direct_routing() && preselected_worker.is_none() {
            return Err(anyhow::anyhow!(
                "Prefill worker ID required in Direct routing mode but none found in request. \
                 Expected prefill_worker_id to be set via x-prefill-instance-id header by external router (e.g., EPP)."
            ));
        }

        let tracker = prefill_req.tracker.clone();
        let prefill_context =
            Context::with_id_and_metadata(prefill_req, request_id.clone(), metadata.clone());
        let router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;
        let prefill_result: Result<(PrefillOutcome, Option<RoutingConstraints>)> = async {
            let (prepared, prefill_stream) = router
                .select_and_dispatch_prefill(
                    prefill_context,
                    preselected_worker,
                    |request, worker_id, dp_rank| {
                        self.prepare_prefill_dispatch(request, worker_id, dp_rank)
                    },
                )
                .await?;
            let topology_constraints = prepared.topology_constraints;
            let outcome = if let Some(bootstrap_info) = prepared.bootstrap_info {
                self.spawn_prefill_task(prefill_stream, tracker, prefill_phase_barrier);
                PrefillOutcome::Bootstrap {
                    bootstrap_info,
                    worker_id: prepared.worker_id,
                }
            } else {
                drop(prefill_phase_barrier);
                let completion = Self::consume_prefill_stream(prefill_stream, tracker).await?;
                PrefillOutcome::Completed {
                    result: completion.result,
                    worker_id: prepared.worker_id,
                    worker_link: completion.worker_link,
                }
            };
            Ok((outcome, topology_constraints))
        }
        .await;
        let (outcome, topology_constraints) = match prefill_result {
            Ok(result) => result,
            Err(error) => {
                use dynamo_runtime::error::{ErrorType, match_error_chain};
                if match_error_chain(error.as_ref(), &[ErrorType::ResourceExhausted], &[]) {
                    tracing::warn!(
                        error = %error,
                        "request rejected by prefill worker (at capacity)"
                    );
                } else {
                    tracing::error!(error = %error, "Remote prefill failed, failing request");
                }
                return Err(error);
            }
        };

        // NVBugs 5969206: Do NOT abort decode routing when context is killed.
        // In disaggregated serving, the prefill may have completed and KV transfer
        // is in flight. Blocking decode here orphans the transfer (no receiver)
        // and leaks KV blocks permanently. The decode handler's
        // kv_transfer_complete_event guard will clean up after KV is received.
        // Log-only; decode routing must proceed for KV transfer cleanup.
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!(
                "Context {} killed/stopped after prefill, allowing decode routing for KV transfer",
                engine_ctx.id()
            );
        }

        tracing::debug!("Prefill completed, proceeding to decode");

        // Set phase to Decode for the decode request.
        // In bootstrap path, this blocks until the spawned prefill task releases its
        // phase barrier after routing completes, ensuring correct worker attribution.
        if let Some(ref tracker) = req.tracker {
            let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
        }

        let mut decode_req = req;
        match outcome {
            PrefillOutcome::Bootstrap {
                bootstrap_info,
                worker_id,
            } => {
                decode_req.bootstrap_info = Some(bootstrap_info);
                decode_req.routing_mut().prefill_worker_id = Some(worker_id);
            }
            PrefillOutcome::Completed {
                result,
                worker_id,
                worker_link,
            } => {
                decode_req.prefill_result = Some(result);
                decode_req.migration_link = worker_link;
                decode_req.routing_mut().prefill_worker_id = Some(worker_id);
            }
        };

        if let Some(topology_constraints) = topology_constraints {
            merge_decode_topology_constraints(&mut decode_req, topology_constraints);
        }

        decode_req.stop_conditions.max_tokens = original_max_tokens;

        // Decode should not score prompt overlap or account prompt-side load.
        let existing_override = decode_req.router_config_override.take();
        decode_req.router_config_override = Some(build_decode_router_override(existing_override));

        next.generate(context.map(|_| decode_req)).await
    }
}

impl PrefillRouter {
    pub fn enforce_disagg(&self) -> bool {
        self.enforce_disagg
    }

    fn prepare_prefill_dispatch(
        &self,
        request: &mut PreprocessedRequest,
        worker_id: u64,
        dp_rank: Option<u32>,
    ) -> anyhow::Result<PreparedPrefill> {
        let endpoint_id = self.endpoint_id.get();
        let topology_constraints =
            self.preflight_kv_transfer_constraints(endpoint_id, worker_id)?;

        let bootstrap_info = endpoint_id
            .and_then(|endpoint_id| {
                self.model_manager
                    .get_disaggregated_endpoint(endpoint_id, worker_id)
                    .map(|endpoint| (endpoint_id, endpoint))
            })
            .and_then(|(endpoint_id, endpoint)| {
                let host = endpoint.bootstrap_host?;
                let port = endpoint.bootstrap_port?;
                let dp_size = self
                    .model_manager
                    .get_data_parallel_size(endpoint_id, worker_id);
                let random_room = rand::random_range(0..=i64::MAX.cast_unsigned());
                let bootstrap_room = compute_bootstrap_room(dp_rank, dp_size, random_room);
                Some(BootstrapInfo {
                    bootstrap_host: host,
                    bootstrap_port: port,
                    bootstrap_room,
                })
            });

        let routing = request.routing_mut();
        routing.prefill_worker_id = Some(worker_id);
        routing.prefill_dp_rank = dp_rank;
        request.bootstrap_info = bootstrap_info.clone();

        Ok(PreparedPrefill {
            worker_id,
            bootstrap_info,
            topology_constraints,
        })
    }

    fn preflight_kv_transfer_constraints(
        &self,
        endpoint_id: Option<&EndpointId>,
        worker_id: u64,
    ) -> anyhow::Result<Option<RoutingConstraints>> {
        let Some(endpoint_id) = endpoint_id else {
            return Ok(None);
        };

        self.model_manager
            .get_kv_transfer_routing_constraints(endpoint_id, worker_id)
    }
}

fn compute_bootstrap_room(dp_rank: Option<u32>, dp_size: Option<u32>, random_room: u64) -> u64 {
    let max_room = i64::MAX.cast_unsigned();
    debug_assert!(random_room <= max_room);
    match (dp_rank, dp_size) {
        (Some(rank), Some(size)) if size > 0 => {
            let size = size as u64;
            let rank = rank as u64;
            let max_quotient = (max_room - rank) / size;
            let quotient = random_room % (max_quotient + 1);
            quotient * size + rank
        }
        _ => random_room,
    }
}

fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    }
}

fn merge_decode_topology_constraints(
    request: &mut PreprocessedRequest,
    topology_constraints: RoutingConstraints,
) {
    if topology_constraints.is_empty() {
        return;
    }

    let routing_constraints = request
        .routing_mut()
        .routing_constraints
        .get_or_insert_with(RoutingConstraints::default);
    routing_constraints
        .required_taints
        .extend(topology_constraints.required_taints);
    routing_constraints
        .preferred_taints
        .extend(topology_constraints.preferred_taints);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::RouterConfigOverride;
    use std::collections::{HashMap, HashSet};

    use crate::protocols::common::preprocessor::{PreprocessedRequest, RoutingHints};

    const MAX_ROOM: u64 = i64::MAX as u64;

    #[test]
    fn decode_router_override_disables_overlap_and_prefill_tracking() {
        let override_config = build_decode_router_override(Some(RouterConfigOverride {
            router_temperature: Some(0.7),
            ..Default::default()
        }));

        assert_eq!(override_config.overlap_score_credit, Some(0.0));
        assert_eq!(override_config.assume_kv_reuse, Some(false));
        assert_eq!(override_config.track_prefill_tokens, Some(false));
        assert_eq!(override_config.router_temperature, Some(0.7));
    }

    #[test]
    fn bootstrap_room_falls_back_when_dp_unavailable() {
        assert_eq!(compute_bootstrap_room(None, None, 12345), 12345);
        assert_eq!(compute_bootstrap_room(Some(3), None, 12345), 12345);
        assert_eq!(compute_bootstrap_room(None, Some(8), 12345), 12345);
        assert_eq!(compute_bootstrap_room(Some(0), Some(0), 12345), 12345);
    }

    #[test]
    fn bootstrap_room_respects_modulo_and_cap() {
        let random_rooms = [0u64, 1, 49, 1_000_000, 1u64 << 62, MAX_ROOM - 1, MAX_ROOM];
        for size in [3u32, 7, 48, 49, 128] {
            for rank in [0u32, 1, size / 2, size - 1] {
                for random_room in random_rooms {
                    let room = compute_bootstrap_room(Some(rank), Some(size), random_room);
                    assert!(room <= MAX_ROOM);
                    assert_eq!(room % size as u64, rank as u64);
                }
            }
        }
    }

    #[test]
    fn bootstrap_room_is_deterministic_in_random_input() {
        let room_a = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
        let room_b = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
        assert_eq!(room_a, room_b);
        assert_eq!(room_a % 48, 7);
    }

    fn request_with_constraints(
        routing_constraints: Option<RoutingConstraints>,
    ) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .routing(Some(RoutingHints {
                routing_constraints,
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn merge_decode_topology_constraints_creates_and_preserves_constraints() {
        for (mut request, expect_user_constraints) in [
            (request_with_constraints(None), false),
            (
                request_with_constraints(Some(RoutingConstraints {
                    required_taints: HashSet::from(["user.required".to_string()]),
                    preferred_taints: HashMap::from([("user.preferred".to_string(), 0.25)]),
                })),
                true,
            ),
        ] {
            merge_decode_topology_constraints(
                &mut request,
                RoutingConstraints {
                    required_taints: HashSet::from(["dynamo.topology/zone=us-east-1a".to_string()]),
                    preferred_taints: HashMap::from([(
                        "dynamo.topology/rack=rack-7".to_string(),
                        0.85,
                    )]),
                },
            );

            let constraints = request
                .routing
                .as_ref()
                .and_then(|routing| routing.routing_constraints.as_ref())
                .unwrap();
            assert!(
                constraints
                    .required_taints
                    .contains("dynamo.topology/zone=us-east-1a")
            );
            assert_eq!(
                constraints.preferred_taints["dynamo.topology/rack=rack-7"],
                0.85
            );

            if expect_user_constraints {
                assert!(constraints.required_taints.contains("user.required"));
                assert_eq!(constraints.preferred_taints["user.preferred"], 0.25);
            }
        }
    }

    fn make_test_router(enforce_disagg: bool) -> Arc<PrefillRouter> {
        PrefillRouter::disabled(
            Arc::new(crate::discovery::ModelManager::new()),
            RouterMode::RoundRobin,
            enforce_disagg,
        )
    }

    #[test]
    fn pending_state_uses_aggregated_fallback_only_when_allowed() {
        let strict = make_test_router(true);
        let fallback = make_test_router(false);

        for router in [&strict, &fallback] {
            assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Pending);
            assert!(!router.is_activated());
            assert!(!router.is_deactivated());
        }
        assert!(!strict.can_serve_requests());
        assert!(fallback.can_serve_requests());
    }

    #[test]
    fn active_state_serves_strict_and_fallback_routers() {
        for enforce_disagg in [true, false] {
            let router = make_test_router(enforce_disagg);
            router.mark_active_for_test();

            assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Active);
            assert!(!router.is_deactivated());
            assert!(router.can_serve_requests());
        }
    }

    #[test]
    fn unavailable_state_blocks_strict_and_allows_aggregated_fallback() {
        let strict = make_test_router(true);
        let fallback = make_test_router(false);
        strict.mark_active_for_test();
        fallback.mark_active_for_test();
        strict.deactivate();
        fallback.deactivate();

        for router in [&strict, &fallback] {
            assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Unavailable);
            assert!(router.is_deactivated());
        }
        assert!(!strict.can_serve_requests());
        assert!(fallback.can_serve_requests());
    }

    #[test]
    fn deactivation_is_idempotent() {
        let router = make_test_router(true);
        router.mark_active_for_test();
        router.deactivate();
        router.deactivate();
        assert!(router.is_deactivated());
    }

    #[test]
    fn pending_router_latches_worker_availability_transitions() {
        let router = make_test_router(true);
        router.deactivate();
        assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Unavailable);

        router.reactivate();
        router.reactivate();

        assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Pending);
    }

    #[test]
    fn activation_does_not_overwrite_latched_deactivation() {
        let router = make_test_router(true);
        router.deactivate();

        assert_eq!(
            router.complete_activation(),
            PrefillLifecycleState::Unavailable
        );
        assert_eq!(router.lifecycle_state(), PrefillLifecycleState::Unavailable);
    }

    #[test]
    fn lifecycle_state_conversion_rejects_invalid_values() {
        assert_eq!(
            PrefillLifecycleState::try_from(0),
            Ok(PrefillLifecycleState::Pending)
        );
        assert_eq!(
            PrefillLifecycleState::try_from(1),
            Ok(PrefillLifecycleState::Active)
        );
        assert_eq!(
            PrefillLifecycleState::try_from(2),
            Ok(PrefillLifecycleState::Unavailable)
        );
        assert_eq!(PrefillLifecycleState::try_from(3), Err(3));
    }
}
