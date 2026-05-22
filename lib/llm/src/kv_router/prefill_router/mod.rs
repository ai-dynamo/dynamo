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

        let prefill_result = match self
            .resolve_prefill_worker(&prefill_req, preselected_worker)
            .await
        {
            PrefillResolveDecision::Resolved {
                worker_id,
                dp_rank,
                bootstrap_info,
            } => {
                // Bootstrap optimization path: spawn prefill in background
                self.commit_selected_prefill_worker(&mut prefill_req, worker_id, dp_rank);
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

                Ok(PrefillOutcome::Bootstrap {
                    bootstrap_info,
                    worker_id,
                })
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
                // TODO(DEP-8189 / ai-dynamo#8189): once the shared rejection
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
                Ok(PrefillOutcome::Completed {
                    result: completion.result,
                    worker_id: Some(resolved_wid),
                    worker_link: completion.worker_link,
                })
            }
            PrefillResolveDecision::Unavailable | PrefillResolveDecision::NotActivated => {
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
                Ok(PrefillOutcome::Completed {
                    result: completion.result,
                    worker_id: prefill_worker_id,
                    worker_link: completion.worker_link,
                })
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

                let selected_prefill_worker_id = match outcome {
                    PrefillOutcome::Bootstrap {
                        bootstrap_info,
                        worker_id,
                    } => {
                        decode_req.bootstrap_info = Some(bootstrap_info);
                        decode_req.routing_mut().prefill_worker_id = Some(worker_id);
                        Some(worker_id)
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
                        worker_id
                    }
                };

                // Resolve prefill worker topology from the prefill endpoint's worker configs,
                // then express it through the standard decode routing constraints.
                let endpoint_id = self.endpoint_id.get();
                let topology_constraints = if let Some((wid, eid)) =
                    selected_prefill_worker_id.zip(endpoint_id)
                {
                    self.model_manager
                        .get_kv_transfer_routing_constraints(eid, wid)?
                } else {
                    // TODO: Make synchronous prefill completion always report the exact
                    // prefill worker id. Required KV-transfer policy needs that id to derive
                    // decode constraints, so fail closed until attribution is authoritative.
                    if let Some(eid) = endpoint_id
                        && self
                            .model_manager
                            .has_kv_transfer_required_routing_policy(eid)
                    {
                        return Err(anyhow::anyhow!(
                            "prefill worker id unavailable after prefill; cannot derive KV transfer topology constraints for endpoint {eid}"
                        ));
                    }
                    None
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
                tracing::error!(error = %e, "Remote prefill failed, failing request");
                Err(anyhow::anyhow!(e))
            }
        }
    }
}

impl PrefillRouter {
    fn commit_selected_prefill_worker(
        &self,
        prefill_req: &mut PreprocessedRequest,
        worker_id: u64,
        dp_rank: Option<u32>,
    ) {
        // SimpleRouter selection was peeked during resolve_prefill_worker, so
        // advance once before direct routing to preserve router state semantics.
        if !self.router_mode.is_kv_routing()
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
    use dynamo_kv_router::{
        config::RouterConfigOverride,
        protocols::{KvTransferEnforcement, WorkerId},
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{AsyncEngine, PushRouter, ResponseStream, network::Ingress},
        stream,
    };
    use futures::StreamExt;
    use serde_json::json;
    use std::{
        collections::{HashMap, HashSet},
        sync::Mutex,
    };

    use crate::local_model::runtime_config::ModelRuntimeConfig;
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

    struct PrefillTestEngine;

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for PrefillTestEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let response = Annotated::from_data(LLMEngineOutput {
                disaggregated_params: Some(json!({"kv_transfer": "prefill-complete"})),
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![response])),
                request.context(),
            ))
        }
    }

    struct CapturingDecodeEngine {
        request: Arc<Mutex<Option<PreprocessedRequest>>>,
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for CapturingDecodeEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
            *self.request.lock().unwrap() = Some(request.content().clone());
            Ok(ResponseStream::new(
                Box::pin(stream::iter(Vec::new())),
                request.context(),
            ))
        }
    }

    fn required_topology_runtime_config() -> ModelRuntimeConfig {
        let mut config = ModelRuntimeConfig {
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
            ..Default::default()
        };
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());
        config
    }

    fn prefill_request_with_user_constraints() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3, 4])
            .stop_conditions(crate::protocols::common::StopConditions {
                max_tokens: Some(16),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .routing(Some(RoutingHints {
                routing_constraints: Some(RoutingConstraints {
                    required_taints: HashSet::from(["user.required".to_string()]),
                    preferred_taints: HashMap::from([("user.preferred".to_string(), 0.25)]),
                }),
                ..Default::default()
            }))
            .router_config_override(Some(RouterConfigOverride {
                router_temperature: Some(0.7),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn generate_sync_prefill_adds_topology_constraints_to_decode_request() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt
            .namespace(format!(
                "prefill_router_generate_{}",
                uuid::Uuid::new_v4().simple()
            ))
            .unwrap();
        let component = namespace
            .component("prefill_component".to_string())
            .unwrap();
        let endpoint = component.endpoint("generate".to_string());

        let ingress = Ingress::for_engine(Arc::new(PrefillTestEngine)).unwrap();
        let endpoint_task = {
            let endpoint = endpoint.clone();
            tokio::spawn(async move { endpoint.endpoint_builder().handler(ingress).start().await })
        };

        let client = endpoint.client().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();
        let push_router =
            PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
                client,
                RouterMode::RoundRobin,
            )
            .await
            .unwrap();

        let model_manager = Arc::new(ModelManager::new());
        model_manager.insert_runtime_configs_for_test(
            endpoint.id(),
            HashMap::<WorkerId, ModelRuntimeConfig>::from([(
                worker_id,
                required_topology_runtime_config(),
            )]),
        );

        let prefill_router = PrefillRouter::disabled(model_manager, RouterMode::RoundRobin, true);
        prefill_router.endpoint_id.set(endpoint.id()).ok().unwrap();
        prefill_router
            .prefill_router
            .set(InnerPrefillRouter::SimpleRouter(Arc::new(push_router)))
            .ok()
            .unwrap();
        prefill_router.activated.store(true, Ordering::Release);

        let captured_decode_request = Arc::new(Mutex::new(None));
        let decode_engine = Arc::new(CapturingDecodeEngine {
            request: captured_decode_request.clone(),
        });
        let request = Context::with_id(
            prefill_request_with_user_constraints(),
            "generate-sync-prefill-topology".to_string(),
        );

        let mut response = prefill_router
            .generate(request, decode_engine)
            .await
            .unwrap();
        while response.next().await.is_some() {}

        let decode_request = captured_decode_request.lock().unwrap().take().unwrap();
        let routing = decode_request.routing.as_ref().unwrap();
        assert_eq!(routing.prefill_worker_id, Some(worker_id));
        assert_eq!(decode_request.stop_conditions.max_tokens, Some(16));

        let prefill_result = decode_request.prefill_result.as_ref().unwrap();
        assert_eq!(
            prefill_result.disaggregated_params,
            json!({"kv_transfer": "prefill-complete"})
        );

        let constraints = routing.routing_constraints.as_ref().unwrap();
        assert!(constraints.required_taints.contains("user.required"));
        assert!(
            constraints
                .required_taints
                .contains("dynamo.topology/zone=us-east-1a")
        );
        assert_eq!(constraints.preferred_taints["user.preferred"], 0.25);

        let override_config = decode_request.router_config_override.as_ref().unwrap();
        assert_eq!(override_config.overlap_score_credit, Some(0.0));
        assert_eq!(override_config.assume_kv_reuse, Some(false));
        assert_eq!(override_config.track_prefill_tokens, Some(false));
        assert_eq!(override_config.router_temperature, Some(0.7));

        rt.shutdown();
        endpoint_task.await.unwrap().unwrap();
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
