// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicU8;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use arc_swap::ArcSwapOption;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_kv_router::{
    PrefillLoadEstimator,
    conditional_disagg::ConditionalDisaggPolicy,
    config::RouterConfigOverride,
    protocols::RoutingConstraints,
    scheduling::QueueRejection,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, Operator, ResponseStream, RouterMode,
        ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated},
};
use futures::stream::{self, StreamExt};

use crate::{
    discovery::ModelManager,
    kv_router::WorkerSelectorFactory,
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{BootstrapInfo, PrefillResult, TraceLink},
        timing::{RequestPhase, RequestTracker},
    },
    session_affinity::{AffinityCoordinator, AffinityTarget},
};

mod activation;
mod admission;
mod conditional_bypass;
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
    Terminal {
        output: Box<Annotated<LLMEngineOutput>>,
    },
}

fn extract_bootstrap_info(params: &serde_json::Value) -> Option<BootstrapInfo> {
    let bootstrap_host = params.get("bootstrap_host")?.as_str()?.to_string();
    let bootstrap_port = u16::try_from(params.get("bootstrap_port")?.as_u64()?).ok()?;
    let bootstrap_room = params.get("bootstrap_room")?.as_u64()?;
    Some(BootstrapInfo {
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
        handoff_id: Some(Uuid::new_v4()),
    })
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

enum PrefillCompletion {
    Handoff {
        result: PrefillResult,
        worker_link: Option<TraceLink>,
    },
    Terminal {
        output: Box<Annotated<LLMEngineOutput>>,
    },
}

fn strip_terminal_disaggregated_params(
    mut output: Annotated<LLMEngineOutput>,
) -> Annotated<LLMEngineOutput> {
    if let Some(data) = output.data.as_mut() {
        data.disaggregated_params = None;
    }
    output
}

/// Annotation marker set when conditional disagg routes a request directly to
/// a DECODE-mode worker to run prefill+decode locally.
pub(crate) const BYPASS_REMOTE_PREFILL_ANNOTATION: &str = "x-bypass-remote-prefill";

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: `query_instance_id` annotation present → returns worker IDs without execution
/// - Pre-routed: `prefill_worker_id`/`decode_worker_id` set → routes to specified workers
/// - Normal: Worker IDs determined by router based on KV cache state
pub struct PrefillRouter<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    binding: ArcSwapOption<PrefillBinding<Sel>>,
    target: Mutex<Option<EndpointId>>,
    target_tx: Option<watch::Sender<Option<dynamo_runtime::component::Endpoint>>>,
    /// Reference to the decode-side `KvRouter` so conditional disagg can peek
    /// the cache-hot decode worker. `None` for non-KV routing and disabled routers.
    decode_router: Option<Arc<super::KvRouter<Sel>>>,
    worker_selector_factory: Option<WorkerSelectorFactory<Sel>>,
    decode_session_affinity: OnceLock<AffinityCoordinator>,
    model_manager: Arc<ModelManager>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    session_affinity_ttl: Option<std::time::Duration>,
    conditional_disagg_policy: Box<dyn ConditionalDisaggPolicy>,
    /// Resolved once at construction: dedicated threshold if set, otherwise
    /// `router_queue_threshold`. `None` means the prefill-load condition is disabled.
    conditional_disagg_prefill_busy_threshold: Option<f64>,
    /// Dedicated decode-busy guard threshold. `None` means disabled.
    conditional_disagg_decode_busy_threshold: Option<f64>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    /// Model name (used for logging / lifecycle messages).
    model_name: String,
    /// Namespace (used for logging / lifecycle messages).
    namespace: String,
    is_eagle: bool,
    /// Initialization and worker availability state.
    lifecycle: AtomicU8,
}

struct PrefillBinding<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    endpoint_id: EndpointId,
    router: InnerPrefillRouter<Sel>,
}

struct PrefillBuildContext<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    model_manager: Arc<ModelManager>,
    router_mode: RouterMode,
    worker_selector_factory: WorkerSelectorFactory<Sel>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    session_affinity_ttl: Option<std::time::Duration>,
    model_name: String,
    is_eagle: bool,
}

pub(crate) trait PrefillRouterLifecycle: Send + Sync {
    fn set_target(&self, target: Option<dynamo_runtime::component::Endpoint>);
}

impl<Sel> PrefillRouterLifecycle for PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn set_target(&self, target: Option<dynamo_runtime::component::Endpoint>) {
        self.set_target(target);
    }
}

impl<Sel> Drop for PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        tracing::debug!("Dropping PrefillRouter, cancelling background activation task");
        self.cancel_token.cancel();
    }
}

#[async_trait]
impl<Sel>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
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
        let policy_class = context.metadata().get("policy-class").cloned();
        let engine_ctx = context.context();

        // Conditional-disagg bypass is a router-owned decision. Drop any
        // client-supplied marker before the policy runs so normal disagg
        // requests cannot accidentally or maliciously skip remote prefill.
        req.annotations
            .retain(|annotation| annotation != BYPASS_REMOTE_PREFILL_ANNOTATION);

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If the prefill router is not activated (no prefill workers discovered) or has been
        // deactivated (all prefill workers died), route directly to the backend. Model admission
        // remains gated by the registered worker topology before the request reaches this stage.
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return next.generate(context.map(|_| req)).await;
        }

        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| anyhow::anyhow!("invalid session affinity context: {message}"))?;

        let decode_affinity_target =
            self.decode_session_affinity_target(session_affinity.as_deref())?;

        if self.conditional_disagg_policy.is_enabled() {
            match self
                .select_decode_worker_for_conditional_disagg(
                    &req,
                    &request_id,
                    policy_class.clone(),
                    session_affinity.as_deref(),
                    decode_affinity_target,
                )
                .await
            {
                Ok(Some(decision)) => {
                    tracing::info!(
                        request_id = %request_id,
                        worker_id = decision.worker.worker_id,
                        dp_rank = decision.worker.dp_rank,
                        net_new_tokens = decision.net_new_tokens,
                        overlap_tokens = decision.overlap_tokens,
                        "Conditional disagg routing to decode worker"
                    );

                    if req.tracker.is_none() {
                        req.tracker = Some(Arc::new(RequestTracker::new()));
                    }
                    if let Some(ref tracker) = req.tracker {
                        let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    }

                    let routing = req.routing_mut();
                    routing.decode_worker_id = Some(decision.worker.worker_id);
                    routing.dp_rank = Some(decision.worker.dp_rank);

                    req.annotations
                        .push(BYPASS_REMOTE_PREFILL_ANNOTATION.to_string());

                    // TODO: This advisory selection does not reserve decode capacity. If the
                    // exact pinned admission below races and fails, the no-clone fix is a
                    // scheduler reservation handoff rather than retrying with a mutated request.
                    let response_stream = next.generate(context.map(|_| req)).await?;
                    let ctx = response_stream.context();
                    let annotation = Annotated::<LLMEngineOutput>::from_annotation(
                        BYPASS_REMOTE_PREFILL_ANNOTATION,
                        &true,
                    )?;
                    let merged = stream::once(async move { annotation }).chain(response_stream);
                    return Ok(ResponseStream::new(Box::pin(merged), ctx));
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %error,
                        "Conditional disagg decision failed; falling back to remote prefill"
                    );
                }
            }
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
                 Expected prefill_worker_id to be set via x-dynamo-prefill-instance-id header by external router (e.g., EPP)."
            ));
        }

        let tracker = prefill_req.tracker.clone();
        let mut prefill_context =
            Context::with_id_and_metadata(prefill_req, request_id.clone(), metadata.clone());
        if let Some(session_affinity) = session_affinity {
            prefill_context.insert(
                SESSION_AFFINITY_CONTEXT_KEY,
                session_affinity.as_ref().clone(),
            );
        }
        let Some(binding) = self.binding.load_full() else {
            return next.generate(context.map(|_| req)).await;
        };
        let router = &binding.router;
        let endpoint_id = &binding.endpoint_id;
        let prefill_result: Result<(PrefillOutcome, Option<RoutingConstraints>)> = async {
            let (prepared, prefill_stream) = router
                .select_and_dispatch_prefill(prefill_context, |request, target| {
                    self.prepare_prefill_dispatch(request, target, endpoint_id)
                })
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

                match completion {
                    PrefillCompletion::Handoff {
                        result,
                        worker_link,
                    } => {
                        if let Some(bootstrap_info) =
                            extract_bootstrap_info(&result.disaggregated_params)
                        {
                            PrefillOutcome::Bootstrap {
                                bootstrap_info,
                                worker_id: prepared.worker_id,
                            }
                        } else {
                            PrefillOutcome::Completed {
                                result,
                                worker_id: prepared.worker_id,
                                worker_link,
                            }
                        }
                    }
                    PrefillCompletion::Terminal { output } => PrefillOutcome::Terminal { output },
                }
            };
            Ok((outcome, topology_constraints))
        }
        .await;
        let (outcome, topology_constraints) = match prefill_result {
            Ok(result) => result,
            Err(error) => {
                use dynamo_runtime::error::{ErrorType, match_error_chain};
                if match_error_chain(
                    error.as_ref(),
                    &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
                    &[],
                ) {
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

        // A prefill request can terminate before the backend establishes a KV
        // handoff (for example, EOS on the one-token context step). Native
        // disaggregated backends return that context response directly instead
        // of launching a generation-only request with missing handoff IDs.
        let outcome = match outcome {
            PrefillOutcome::Terminal { output } => {
                let output = strip_terminal_disaggregated_params(*output);
                return Ok(dynamo_runtime::pipeline::ResponseStream::new(
                    Box::pin(stream::once(async move { output })),
                    engine_ctx,
                ));
            }
            outcome => outcome,
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
            PrefillOutcome::Terminal { .. } => {
                unreachable!("terminal prefill outcomes return before decode routing")
            }
        };

        if let Some(topology_constraints) = topology_constraints {
            merge_decode_topology_constraints(&mut decode_req, topology_constraints);
        }

        decode_req.stop_conditions.max_tokens = original_max_tokens;

        // Decode should not account prompt-side load. Normal disagg also
        // forces zero overlap credit so decode routing stays load-only.
        let existing_override = decode_req.router_config_override.take();
        decode_req.router_config_override = Some(build_decode_router_override(
            existing_override,
            self.conditional_disagg_policy.is_enabled(),
        ));

        next.generate(context.map(|_| decode_req)).await
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(crate) fn conditional_disagg_enabled(&self) -> bool {
        self.conditional_disagg_policy.is_enabled()
    }

    pub(crate) fn set_decode_session_affinity(&self, affinity: Option<AffinityCoordinator>) {
        let Some(affinity) = affinity else {
            return;
        };
        if self.decode_session_affinity.get().is_some() {
            return;
        }
        let _ = self.decode_session_affinity.set(affinity);
    }

    fn decode_session_affinity_target(
        &self,
        session_affinity: Option<&SessionAffinityId>,
    ) -> Result<Option<AffinityTarget>> {
        let Some(session_affinity) = session_affinity else {
            return Ok(None);
        };
        let Some(affinity) = self.decode_session_affinity.get() else {
            return Ok(None);
        };
        affinity.query_target(session_affinity, None)
    }

    fn prepare_prefill_dispatch(
        &self,
        request: &mut PreprocessedRequest,
        target: AffinityTarget,
        endpoint_id: &EndpointId,
    ) -> anyhow::Result<PreparedPrefill> {
        let AffinityTarget { worker_id, dp_rank } = target;
        let topology_constraints =
            self.preflight_kv_transfer_constraints(Some(endpoint_id), worker_id)?;

        let bootstrap_info = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)
            .map(|endpoint| (endpoint_id, endpoint))
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
                    handoff_id: Some(Uuid::new_v4()),
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
    allow_decode_overlap_affinity: bool,
) -> RouterConfigOverride {
    let mut override_config = existing_override.unwrap_or_default();

    // Normal disagg keeps decode routing load-only by forcing zero overlap
    // credit. Conditional disagg leaves this unset so the base router
    // `overlap_score_credit` applies, unless the request already had an
    // explicit override.
    if !allow_decode_overlap_affinity {
        override_config.overlap_score_credit = Some(0.0);
    }
    override_config.assume_kv_reuse = Some(false);
    override_config.track_prefill_tokens = Some(false);

    override_config
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

    use crate::protocols::common::{
        FinishReason,
        preprocessor::{PreprocessedRequest, RoutingHints},
    };

    const MAX_ROOM: u64 = i64::MAX as u64;

    #[test]
    fn decode_router_override_disables_overlap_and_prefill_tracking() {
        let override_config = build_decode_router_override(
            Some(RouterConfigOverride {
                overlap_score_credit: Some(0.5),
                router_temperature: Some(0.7),
                ..Default::default()
            }),
            false,
        );

        assert_eq!(override_config.overlap_score_credit, Some(0.0));
        assert_eq!(override_config.assume_kv_reuse, Some(false));
        assert_eq!(override_config.track_prefill_tokens, Some(false));
        assert_eq!(override_config.router_temperature, Some(0.7));
    }

    #[test]
    fn terminal_response_strips_disaggregated_params() {
        let output = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![2],
            finish_reason: Some(FinishReason::EoS),
            disaggregated_params: Some(serde_json::json!({
                "ctx_request_id": null,
                "request_type": "context_only",
            })),
            ..Default::default()
        });

        let output = strip_terminal_disaggregated_params(output);
        let data = output
            .data
            .expect("terminal response should retain its data");
        assert_eq!(data.token_ids, vec![2]);
        assert_eq!(data.finish_reason, Some(FinishReason::EoS));
        assert!(data.disaggregated_params.is_none());
    }

    #[test]
    fn decode_router_override_inherits_base_overlap_when_conditional_disagg_allows_it() {
        let override_config = build_decode_router_override(None, true);

        assert_eq!(override_config.overlap_score_credit, None);
        assert_eq!(override_config.assume_kv_reuse, Some(false));
        assert_eq!(override_config.track_prefill_tokens, Some(false));
    }

    #[test]
    fn decode_router_override_preserves_request_overlap_when_conditional_disagg_allows_it() {
        let override_config = build_decode_router_override(
            Some(RouterConfigOverride {
                overlap_score_credit: Some(0.25),
                router_temperature: Some(0.7),
                ..Default::default()
            }),
            true,
        );

        assert_eq!(override_config.overlap_score_credit, Some(0.25));
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

    #[test]
    fn extract_bootstrap_info_parses_valid_params() {
        let params = serde_json::json!({
            "bootstrap_host": "10.0.0.5",
            "bootstrap_port": 12345,
            "bootstrap_room": 987654321u64,
            // extra fields (e.g. worker_id) must be ignored
            "worker_id": {"prefill_worker_id": 7},
        });
        let info = extract_bootstrap_info(&params).expect("valid params should parse");
        assert_eq!(info.bootstrap_host, "10.0.0.5");
        assert_eq!(info.bootstrap_port, 12345);
        assert_eq!(info.bootstrap_room, 987654321);
    }

    #[test]
    fn extract_bootstrap_info_none_when_field_missing() {
        // Missing bootstrap_room -> not the bootstrap path (falls through to Completed).
        let missing_room = serde_json::json!({
            "bootstrap_host": "10.0.0.5",
            "bootstrap_port": 12345,
        });
        assert!(extract_bootstrap_info(&missing_room).is_none());
        // An aggregated / vLLM completed prefill carries no bootstrap fields.
        assert!(extract_bootstrap_info(&serde_json::json!({})).is_none());
    }

    #[test]
    fn extract_bootstrap_info_rejects_out_of_range_port() {
        // bootstrap_port must fit in u16 -> reject rather than silently truncating.
        let params = serde_json::json!({
            "bootstrap_host": "h",
            "bootstrap_port": 70000,
            "bootstrap_room": 1,
        });
        assert!(extract_bootstrap_info(&params).is_none());
    }
}
