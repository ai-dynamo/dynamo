// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::{PrefillLoadEstimator, protocols::RoutingConstraints};
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
        timing::{RequestPhase, RequestTracker},
    },
};

mod activation;
mod execution;
mod inner;
mod types;

use inner::InnerPrefillRouter;
pub use types::{PrefillError, PrefillQueryOutcome};
use types::{PrefillOutcome, PrefillResolveDecision, build_decode_router_override};

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
    /// Model name used to look up the worker monitor for prefill client registration
    model_name: String,
    /// Namespace used to look up the correct WorkerSet's worker monitor
    namespace: String,
    is_eagle: bool,
    /// Set to true when all prefill workers die. Checked in generate() to prevent
    /// routing to dead workers. Cleared on reactivation when workers rejoin.
    deactivated: AtomicBool,
    /// Set to true when the prefill router has been activated (inner router populated).
    /// Used by `can_serve_requests()` to gate enforce_disagg readiness so a cold-started
    /// strict-disagg model isn't listed before the prefill has rendezvoused.
    activated: AtomicBool,
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
        if self.prefill_router.get().is_none() || self.deactivated.load(Ordering::Relaxed) {
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

        let endpoint_id = self.endpoint_id.get();
        let (prefill_result, topology_constraints) = match self
            .resolve_prefill_worker(&request_id, &prefill_req, preselected_worker)
            .await
        {
            PrefillResolveDecision::Resolved {
                worker_id,
                dp_rank,
                bootstrap_info,
            } => {
                let topology_constraints =
                    self.preflight_kv_transfer_constraints(endpoint_id, Some(worker_id))?;

                // Bootstrap optimization path: spawn prefill in background
                self.commit_selected_prefill_worker(
                    &mut prefill_req,
                    worker_id,
                    dp_rank,
                    preselected_worker,
                );
                prefill_req.bootstrap_info = Some(bootstrap_info.clone());

                // NVBugs 5969206: Do NOT link prefill as child of engine context.
                // Kill propagation tears down the RPC transport, interrupting NIXL
                // KV cache transfers and leaking blocks permanently. The prefill
                // runs to completion independently; blocks are freed via the normal
                // completion path (state 21→22).
                // NOTE: This means prefill runs to completion even if the client
                // disconnects, wasting prefill compute. This is an accepted
                // trade-off (wasted compute vs permanent KV block leak). Future
                // work: add NIXL-level cancellation that properly frees blocks.
                let prefill_context = Context::with_id_and_metadata(
                    prefill_req,
                    request_id.clone(),
                    metadata.clone(),
                );

                // Pass the phase barrier to the spawned task. It is released after routing
                // completes so worker recording finishes before phase changes to Decode.
                self.spawn_prefill_task(prefill_context, Some(worker_id), prefill_phase_barrier);

                (
                    Ok(PrefillOutcome::Bootstrap {
                        bootstrap_info,
                        worker_id,
                    }),
                    topology_constraints,
                )
            }
            PrefillResolveDecision::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => {
                // Quick-reject: bubble up as ResourceExhausted so the caller
                // can return a retryable signal upstream instead of falling
                // back to the synchronous prefill path (which would re-enter
                // the saturated queue).
                //
                // TODO(ai-dynamo#8189): once the shared rejection
                // layer lands, classify queue-depth saturation distinctly
                // from generic resource exhaustion (operator-facing 429 vs
                // 503) instead of stringifying through ResourceExhausted.
                drop(prefill_phase_barrier);
                return Err(dynamo_runtime::error::DynamoError::builder()
                    .error_type(dynamo_runtime::error::ErrorType::ResourceExhausted)
                    .message(format!(
                        "router backpressure during prefill resolve: {reason:?} (queued_isl_tokens={queued_isl_tokens}, max_queued_isl_tokens={max_queued_isl_tokens:?})"
                    ))
                    .build()
                    .into());
            }
            PrefillResolveDecision::NoBootstrapEndpoint {
                worker_id: resolved_wid,
                dp_rank: resolved_dp_rank,
            } => {
                let topology_constraints =
                    self.preflight_kv_transfer_constraints(endpoint_id, Some(resolved_wid))?;

                // Bootstrap unavailable after resolve_prefill_worker selected a worker.
                // Commit the same selection in the synchronous path
                tracing::debug!(
                    worker_id = resolved_wid,
                    "Using original prefill path (no bootstrap endpoint), routing to resolved worker"
                );
                self.commit_selected_prefill_worker(
                    &mut prefill_req,
                    resolved_wid,
                    resolved_dp_rank,
                    preselected_worker,
                );

                drop(prefill_phase_barrier);
                let prefill_context = Context::with_id_and_metadata(
                    prefill_req,
                    request_id.clone(),
                    metadata.clone(),
                );
                let completion = Self::execute_prefill(
                    self.prefill_router.get().cloned(),
                    prefill_context,
                    Some(resolved_wid),
                    None,
                )
                .await?;
                (
                    Ok(PrefillOutcome::Completed {
                        result: completion.result,
                        worker_id: Some(resolved_wid),
                        worker_link: completion.worker_link,
                    }),
                    topology_constraints,
                )
            }
            PrefillResolveDecision::Unavailable | PrefillResolveDecision::NotActivated => {
                let topology_constraints =
                    self.preflight_kv_transfer_constraints(endpoint_id, None)?;

                // No worker resolved; fall back to router-selected prefill.
                tracing::debug!("Using original prefill path (no resolved worker)");

                // Drop the phase barrier because we wait for prefill completion in this task,
                // so there is no race with set_phase(Decode) below.
                drop(prefill_phase_barrier);

                // NVBugs 5969206: Do NOT link prefill as child (same rationale as bootstrap path).
                let prefill_context = Context::with_id_and_metadata(
                    prefill_req,
                    request_id.clone(),
                    metadata.clone(),
                );

                // In Direct mode, pass preselected_worker so execute_prefill uses
                // router.direct() instead of router.generate() (which bails in Direct mode).
                let completion = Self::execute_prefill(
                    self.prefill_router.get().cloned(),
                    prefill_context,
                    preselected_worker,
                    None,
                )
                .await?;
                let prefill_worker_id = completion
                    .worker_info
                    .map(|(wid, _)| wid)
                    .or(preselected_worker);
                (
                    Ok(PrefillOutcome::Completed {
                        result: completion.result,
                        worker_id: prefill_worker_id,
                        worker_link: completion.worker_link,
                    }),
                    topology_constraints,
                )
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

        // Handle prefill result
        match prefill_result {
            Ok(outcome) => {
                tracing::debug!("Prefill completed, proceeding to decode");

                // Set phase to Decode for the decode request.
                // In bootstrap path, this blocks until the spawned prefill task releases its
                // phase barrier after routing completes, ensuring correct worker attribution.
                if let Some(ref tracker) = req.tracker {
                    let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    // Permit is dropped immediately - decode proceeds, no need to hold it
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
                        if let Some(wid) = worker_id {
                            decode_req.routing_mut().prefill_worker_id = Some(wid);
                        }
                    }
                };

                if let Some(topology_constraints) = topology_constraints {
                    merge_decode_topology_constraints(&mut decode_req, topology_constraints);
                }

                // Restore original max_tokens for decode
                decode_req.stop_conditions.max_tokens = original_max_tokens;

                // Set router_config_override for decode:
                // - overlap_score_credit = 0 (no KV cache overlap scoring for decode)
                // - assume_kv_reuse = false (generate random hashes since decode workers
                //   may already have blocks cached from prefill transfer)
                // - track_prefill_tokens = false (decode router should ignore prompt-side load)
                let existing_override = decode_req.router_config_override.take();
                decode_req.router_config_override =
                    Some(build_decode_router_override(existing_override));

                // Map the modified request through with preserved context
                let decode_request = context.map(|_| decode_req);
                next.generate(decode_request).await
            }
            Err(PrefillError::NotActivated) => {
                tracing::error!("Prefill router not activated, failing request");
                Err(anyhow::anyhow!(PrefillError::NotActivated))
            }
            Err(e) => {
                use dynamo_runtime::error::{ErrorType, match_error_chain};
                if match_error_chain(&e, &[ErrorType::ResourceExhausted], &[]) {
                    tracing::warn!(error = %e, "request rejected by prefill worker (at capacity)");
                } else {
                    tracing::error!(error = %e, "Remote prefill failed, failing request");
                }
                Err(anyhow::anyhow!(e))
            }
        }
    }
}

impl PrefillRouter {
    fn preflight_kv_transfer_constraints(
        &self,
        endpoint_id: Option<&EndpointId>,
        worker_id: Option<u64>,
    ) -> anyhow::Result<Option<RoutingConstraints>> {
        let Some(endpoint_id) = endpoint_id else {
            return Ok(None);
        };

        if let Some(worker_id) = worker_id {
            return self
                .model_manager
                .get_kv_transfer_routing_constraints(endpoint_id, worker_id);
        }

        // TODO: Make synchronous prefill completion always report the exact
        // prefill worker id. Required KV-transfer policy needs that id to derive
        // decode constraints, so fail closed until attribution is authoritative.
        if self
            .model_manager
            .has_kv_transfer_required_routing_policy(endpoint_id)
        {
            anyhow::bail!(
                "prefill worker id unavailable before prefill; cannot derive KV transfer topology constraints for endpoint {endpoint_id}"
            );
        }

        Ok(None)
    }

    fn commit_selected_prefill_worker(
        &self,
        prefill_req: &mut PreprocessedRequest,
        worker_id: u64,
        dp_rank: Option<u32>,
        preselected_worker: Option<u64>,
    ) {
        // SimpleRouter workers selected by resolve_prefill_worker are peeked first,
        // so advance once when committing that router-selected worker. Externally
        // preselected workers did not come from the router cursor and must not
        // advance round-robin state.
        if preselected_worker.is_none()
            && !self.router_mode.is_kv_routing()
            && let Some(router) = self.prefill_router.get()
        {
            router.select_next_worker();
        }

        let routing = prefill_req.routing_mut();
        routing.prefill_worker_id = Some(worker_id);
        routing.dp_rank = dp_rank;
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

    // -- Prefill death handling tests --

    /// Helper: create a disabled PrefillRouter for testing deactivation behavior.
    fn make_test_router(enforce_disagg: bool) -> Arc<PrefillRouter> {
        PrefillRouter::disabled(
            Arc::new(crate::discovery::ModelManager::new()),
            RouterMode::RoundRobin,
            enforce_disagg,
        )
    }

    #[test]
    fn test_deactivated_flag_blocks_when_enforce_disagg() {
        let router = make_test_router(true);
        // Not activated, so enforce_disagg blocks even before deactivation
        assert!(
            !router.can_serve_requests(),
            "enforce_disagg must block before prefill activation"
        );

        router.deactivate();
        assert!(router.is_deactivated());
        assert!(
            !router.can_serve_requests(),
            "deactivated + enforce_disagg must block"
        );
    }

    #[test]
    fn test_deactivated_flag_allows_fallback_no_enforce() {
        let router = make_test_router(false);
        router.deactivate();
        assert!(router.is_deactivated());
        assert!(
            router.can_serve_requests(),
            "deactivated + !enforce_disagg must allow fallback"
        );
    }

    #[test]
    fn test_reactivate_clears_deactivated_no_enforce() {
        let router = make_test_router(false);
        router.deactivate();
        // !enforce_disagg allows fallback even while deactivated
        assert!(router.can_serve_requests());

        router.reactivate();
        assert!(!router.is_deactivated());
        assert!(
            router.can_serve_requests(),
            "reactivated non-enforce router must serve requests"
        );
    }

    #[test]
    fn test_reactivate_clears_deactivated_enforce_needs_activation() {
        // disabled() never sets the activated flag, so enforce_disagg stays blocked.
        // In a real deployment, activate() sets the flag before the first
        // deactivate/reactivate cycle, so this only exercises the flag reset.
        let router = make_test_router(true);
        router.deactivate();
        assert!(!router.can_serve_requests());

        router.reactivate();
        assert!(!router.is_deactivated());
        assert!(
            !router.can_serve_requests(),
            "enforce_disagg without activation still can't serve"
        );
    }

    #[test]
    fn test_fresh_router_not_deactivated() {
        let router = make_test_router(true);
        assert!(!router.is_deactivated());
        // enforce_disagg + no prefill activation => not servable
        assert!(!router.can_serve_requests());
    }

    #[test]
    fn test_fresh_router_no_enforce_disagg_can_serve() {
        let router = make_test_router(false);
        assert!(!router.is_deactivated());
        assert!(
            router.can_serve_requests(),
            "non-enforce_disagg router must be servable even without prefill activation"
        );
    }

    #[test]
    fn test_deactivate_is_idempotent() {
        let router = make_test_router(true);
        router.deactivate();
        router.deactivate();
        assert!(router.is_deactivated());
        assert!(!router.can_serve_requests());
    }
}
