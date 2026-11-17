// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use arc_swap::ArcSwap;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::{
    error::{DynamoError, ErrorClass, ErrorType, match_error_chain},
    pipeline::{
        AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, RouterMode,
        ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated},
};
use dynamo_scheduler::host::{PrefillLoadEstimator, RouterConfigOverride, RoutingConstraints};
use dynamo_worker_selection::host::ConditionalDisaggPolicy;
use futures::{StreamExt, stream};

use crate::{
    discovery::{ModelManager, WorkerSetTarget, WorkerSetTargetId},
    kv_router::{RoutingHost, SelectionPolicySource, routing_host::ReplanLifecycle},
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{BootstrapInfo, PrefillResult, TraceLink},
        timing::{RequestPhase, RequestTracker},
    },
    session_affinity::{AffinityTarget, SessionAffinityMode},
};

mod activation;
mod admission;
mod cancellation;
mod conditional_bypass;
mod dispatch;
mod handoff;
mod query;
use activation::PrefillActivation;
use dispatch::EncoderHandoff;
use handoff::PrefillTask;
pub use query::PrefillReservation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillDispatchErrorClass {
    Cancelled,
    AtCapacity,
    Failed,
}

fn classify_prefill_dispatch_error(
    error: &(dyn std::error::Error + 'static),
) -> PrefillDispatchErrorClass {
    if matches!(
        error.downcast_ref::<PrefillError>(),
        Some(PrefillError::Cancelled)
    ) || match_error_chain(error, &[ErrorType::Cancelled], &[])
    {
        PrefillDispatchErrorClass::Cancelled
    } else if match_error_chain(
        error,
        &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
        &[],
    ) {
        PrefillDispatchErrorClass::AtCapacity
    } else {
        PrefillDispatchErrorClass::Failed
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("Prefill router not yet activated")]
    NotActivated,

    // Callers must include the worker's error text in this message.
    // The frontend boundary walks the `#[source]` chain, so a classified
    // InvalidRequest with a public message reaches the client with its own
    // text.
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    #[error("Prefill execution failed: Prefill backend cancelled before completing the handoff")]
    Cancelled,

    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

enum PrefillOutcome {
    Bootstrap {
        bootstrap_info: BootstrapInfo,
        worker_id: u64,
        /// The decode engine's KV receiver needs the prefill DP rank up front;
        /// without it SGLang parks the request and asks the prefill bootstrap
        /// server over HTTP, and the prefill forward waits on that round trip.
        prefill_dp_rank: Option<u32>,
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

/// Turn a request whose prefill has finished into its decode leg.
///
/// Both surviving outcomes leave KV blocks staged on the prefill worker for one
/// specific decode worker, and only that worker's KV-transfer-complete path
/// frees them. `staged_kv_cleanup` records that obligation so routing still
/// delivers the leg after the client has disconnected.
fn into_decode_request(
    mut req: PreprocessedRequest,
    outcome: PrefillOutcome,
) -> PreprocessedRequest {
    match outcome {
        PrefillOutcome::Bootstrap {
            bootstrap_info,
            worker_id,
            prefill_dp_rank,
        } => {
            req.bootstrap_info = Some(bootstrap_info);
            let routing = req.routing_mut();
            routing.prefill_worker_id = Some(worker_id);
            routing.prefill_dp_rank = prefill_dp_rank;
        }
        PrefillOutcome::Completed {
            result,
            worker_id,
            worker_link,
        } => {
            req.prefill_result = Some(result);
            req.migration_link = worker_link;
            req.routing_mut().prefill_worker_id = Some(worker_id);
        }
        PrefillOutcome::Terminal { .. } => {
            unreachable!("terminal prefill outcomes return before decode routing")
        }
    }
    req.staged_kv_cleanup = true;
    req
}

/// The request reached a decode WorkerSet whose prefill target is known but
/// whose prefill binding is still being built. `WorkerUnavailable` is the
/// migratable sibling of `Unavailable`: it maps to HTTP 503 and lets the retry
/// layer re-dispatch rather than treating the whole model as down.
fn prefill_activation_pending_error(model_name: &str, namespace: &str) -> anyhow::Error {
    DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message(format!(
            "prefill routing for model {model_name} in namespace {namespace} is still activating"
        ))
        .build()
        .into()
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
    prefill_dp_rank: Option<u32>,
    bootstrap_info: Option<BootstrapInfo>,
    topology_constraints: Option<RoutingConstraints>,
}

enum PrefillCompletion {
    Handoff {
        result: PrefillResult,
        worker_link: Option<TraceLink>,
        completion: Option<PrefillTask>,
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

fn settle_prefill_terminal(
    request: &PreprocessedRequest,
    local: &mut Option<ReplanLifecycle>,
    output: &Annotated<LLMEngineOutput>,
) {
    let Some(mut lifecycle) = request
        .migration_state
        .as_ref()
        .and_then(|state| state.take_request_lifecycle())
        .or_else(|| local.as_mut().and_then(ReplanLifecycle::take_owned))
    else {
        return;
    };
    let cause = match output
        .data
        .as_ref()
        .and_then(|data| data.finish_reason.as_ref())
    {
        Some(crate::protocols::common::FinishReason::Error(message)) => {
            Some(DynamoError::msg(message.clone()))
        }
        Some(crate::protocols::common::FinishReason::Cancelled) => Some(
            DynamoError::builder()
                .class(ErrorClass::Cancelled)
                .message("Prefill request cancelled")
                .build(),
        ),
        _ => None,
    };
    if let Some(cause) = cause {
        lifecycle.abort(Some(Arc::new(cause)));
    } else {
        lifecycle.complete();
    }
}

/// Annotation marker set when conditional disagg routes a request directly to
/// a DECODE-mode worker to run prefill+decode locally.
pub(crate) const BYPASS_REMOTE_PREFILL_ANNOTATION: &str = "x-bypass-remote-prefill";

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: KV routing with `query_instance_id` → returns worker IDs without execution
/// - Pre-routed: `prefill_worker_id`/`decode_worker_id` set → routes to specified workers
/// - Normal: Worker IDs determined by router based on KV cache state
///
/// # Future SGLang input-token logprobs
///
/// In disaggregated SGLang serving, prompt-side logprob metadata is produced
/// during the prefill/decode handoff rather than solely by the terminal decode
/// stream. Supporting it requires retaining the prefill metadata while decode
/// runs, then concatenating the peers' raw `input_token_logprobs` and
/// `input_top_logprobs` arrays in prompt order and normalizing them once on the
/// terminal decode output. Prefill and decode must still run concurrently:
/// waiting for prefill before starting decode can deadlock the KV transfer.
/// Client-visible logprobs should not be placed in `disaggregated_params`,
/// which is an engine-owned KV handoff contract rather than a public response
/// channel.
pub struct PrefillRouter {
    activation: ArcSwap<PrefillActivation>,
    target: Mutex<Option<WorkerSetTargetId>>,
    target_tx: Option<watch::Sender<Option<WorkerSetTarget>>>,
    /// Decode routing owns conditional-disagg planning and dispatch. This is
    /// installed after the frontend constructs its one decode `RoutingHost`.
    decode_routing_host: OnceLock<Arc<RoutingHost>>,
    selection_policy: Option<SelectionPolicySource>,
    model_manager: Arc<ModelManager>,
    cancel_token: CancellationToken,
    /// Mode of the decode set that owns this router. Governs decode-side
    /// decisions, and is the fallback for the prefill hop -- not its mode. That
    /// lives on [`PrefillBinding::prefill_router_mode`].
    decode_router_mode: RouterMode,
    session_affinity_ttl: Option<std::time::Duration>,
    decode_supports_prefill_cancellation: bool,

    session_affinity_mode: SessionAffinityMode,
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
    task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
    #[cfg(test)]
    activation_task_state: Arc<()>,
}

struct PrefillBinding {
    target_id: WorkerSetTargetId,
    endpoint_id: EndpointId,
    router: Arc<RoutingHost>,
    /// Resolved at activation from the prefill card. Lives here rather than on
    /// `PrefillRouter` because it is unknowable until a target is discovered,
    /// and changes when the binding is rebuilt.
    prefill_router_mode: RouterMode,
    encoder_result_requires_prefill: bool,
}

struct PrefillBuildContext {
    model_manager: Arc<ModelManager>,
    /// Fallback mode for the prefill hop when the prefill card advertises none.
    decode_router_mode: RouterMode,
    selection_policy: SelectionPolicySource,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    session_affinity_ttl: Option<std::time::Duration>,
    session_affinity_mode: SessionAffinityMode,
    model_name: String,
    load_thresholds: crate::discovery::LoadThresholdHandle,
    parent_token: CancellationToken,
    task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
}

pub(crate) trait PrefillRouterLifecycle: Send + Sync {
    fn set_target(&self, target: Option<WorkerSetTarget>);
    /// Returns `None` when this router does not target the group, and an empty
    /// set when it targets the endpoint but its binding is not yet available.
    fn available_worker_ids_for(
        &self,
        group_id: &str,
        endpoint: &EndpointId,
    ) -> Option<std::collections::HashSet<u64>>;
}

impl PrefillRouterLifecycle for PrefillRouter {
    fn set_target(&self, target: Option<WorkerSetTarget>) {
        self.set_target(target);
    }

    fn available_worker_ids_for(
        &self,
        group_id: &str,
        endpoint: &EndpointId,
    ) -> Option<std::collections::HashSet<u64>> {
        // Target changes and activation publish binding/lifecycle under this same lock.
        let target_id = self.target.lock();
        let activation = self.activation.load();
        let binding = activation.binding();
        let binding_matches = binding.as_ref().is_some_and(|binding| {
            &binding.endpoint_id == endpoint && binding.target_id.matches_group(group_id, endpoint)
        });
        let target_matches = self.target_tx.as_ref().map(|target_tx| {
            target_tx.borrow().as_ref().is_some_and(|target| {
                &target.endpoint().id() == endpoint
                    && target_id
                        .as_ref()
                        .is_some_and(|id| id.matches_group(group_id, endpoint))
            })
        });
        if !target_matches.unwrap_or(binding_matches) {
            return None;
        }
        Some(
            if self.cancel_token.is_cancelled() || !activation.is_active() {
                std::collections::HashSet::new()
            } else {
                binding
                    .as_ref()
                    .filter(|binding| {
                        &binding.endpoint_id == endpoint
                            && binding.target_id.matches_group(group_id, endpoint)
                    })
                    .map(|binding| binding.router.available_worker_ids())
                    .unwrap_or_default()
            },
        )
    }
}

/// Prepend the `x-bypass-remote-prefill` marker to a bypassed response stream
/// only when the client listed it in `nvext.annotations` (same contract as
/// `request_id`). It is an annotation-only frame, serialised as `event:` with
/// no `data:` line; the OpenAI Python and Node SDKs abort on it, so it is
/// never sent unrequested.
fn with_bypass_marker(
    response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    client_requested: bool,
) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
    if !client_requested {
        return Ok(response_stream);
    }
    let ctx = response_stream.context();
    let annotation =
        Annotated::<LLMEngineOutput>::from_annotation(BYPASS_REMOTE_PREFILL_ANNOTATION, &true)?;
    let merged = stream::once(async move { annotation }).chain(response_stream);
    Ok(ResponseStream::new(Box::pin(merged), ctx))
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
        let handoff = EncoderHandoff::for_request(&request);
        let (mut req, mut context) = request.into_parts();
        let request_id = context.id().to_string();
        let engine_ctx = context.context();

        // Conditional-disagg bypass is a router-owned decision. Drop any
        // client-supplied marker before the policy runs so normal disagg
        // requests cannot accidentally or maliciously skip remote prefill.
        // A client that lists the marker in `nvext.annotations` is only asking
        // to be told when the bypass happens (like `request_id`), so remember
        // that before the marker is removed.
        let annotations_before = req.annotations.len();
        req.annotations
            .retain(|annotation| annotation != BYPASS_REMOTE_PREFILL_ANNOTATION);
        let client_requested_bypass_marker = req.annotations.len() != annotations_before;

        // Worker lookups never execute Prefill or consume an encoder export.
        if self.is_query_only(&req) {
            return next.generate(context.map(|_| req)).await;
        }
        handoff.check_cancellation(engine_ctx.as_ref())?;
        if self.binding_for_request(&req)?.is_none() {
            return next.generate(context.map(|_| req)).await;
        }

        crate::routing_policy::check_stage(&req, crate::routing_policy::ExecutionStage::Prefill)?;

        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| anyhow::anyhow!("invalid session affinity context: {message}"))?;

        let passthrough_phase = req.phase();
        let mut replan_lifecycle = None;
        if self.conditional_disagg_policy.is_enabled()
            && handoff != EncoderHandoff::MustConsume
            && !(req.encoder_result.is_some() && engine_ctx.is_stopped())
        {
            let conditional_request = context.map(|_| req);
            match self
                .plan_conditional_disagg_decode(&conditional_request, &request_id)
                .await
            {
                Ok(Some(decision)) => {
                    let mut conditional_request = conditional_request;
                    if conditional_request.tracker.is_none() {
                        conditional_request.tracker = Some(Arc::new(RequestTracker::new()));
                    }
                    if let Some(ref tracker) = conditional_request.tracker {
                        let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    }
                    if decision.allows_encoder_bypass(self) {
                        let worker = decision.plan.worker();
                        tracing::info!(
                            request_id = %request_id,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            net_new_tokens = decision.net_new_tokens,
                            overlap_tokens = decision.overlap_tokens,
                            "Conditional disagg routing to decode worker"
                        );

                        conditional_request
                            .annotations
                            .push(BYPASS_REMOTE_PREFILL_ANNOTATION.to_string());

                        let decode_host = self
                            .decode_routing_host
                            .get()
                            .expect("conditional plan requires a decode RoutingHost");
                        let response_stream = decode_host
                            .dispatch_kv_plan(conditional_request, decision.plan)
                            .await?;
                        return with_bypass_marker(response_stream, client_requested_bypass_marker);
                    }
                    replan_lifecycle = decision
                        .plan
                        .release_for_replan(&mut conditional_request)
                        .await;
                    if let Some(ref tracker) = conditional_request.tracker {
                        let _permit = tracker.set_phase(passthrough_phase).await;
                    }
                    (req, context) = conditional_request.into_parts();
                }
                Ok(None) => {
                    (req, context) = conditional_request.into_parts();
                }
                Err(error)
                    if match_error_chain(
                        error.as_ref(),
                        &[ErrorType::Cancelled, ErrorType::InvalidArgument],
                        &[],
                    ) =>
                {
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %error,
                        "Conditional disagg decision failed; falling back to remote prefill"
                    );
                    (req, context) = conditional_request.into_parts();
                }
            }
        }

        let result = self
            .prefill_and_decode(
                context.map(|_| req),
                next,
                handoff,
                session_affinity,
                passthrough_phase,
                &mut replan_lifecycle,
            )
            .await;
        if let Err(error) = &result
            && let Some(lifecycle) = replan_lifecycle.as_mut()
        {
            lifecycle.abort(Some(error));
        }
        result
    }
}

impl PrefillRouter {
    fn is_query_only(&self, request: &PreprocessedRequest) -> bool {
        self.decode_router_mode.is_kv_routing()
            && request.get_annotation_value("query_instance_id").is_some()
    }

    fn validate_passthrough(
        &self,
        request: &PreprocessedRequest,
        encoder_result_requires_prefill: bool,
    ) -> Result<()> {
        if encoder_result_requires_prefill
            && request.encoder_result.is_some()
            && !request.staged_kv_cleanup
            && !self.is_query_only(request)
        {
            return Err(DynamoError::builder()
                .class(ErrorClass::WorkerUnavailable)
                .message(format!(
                    "Prefill worker required to consume the encoder handoff for model {} in namespace {} is unavailable",
                    self.model_name, self.namespace
                ))
                .build()
                .into());
        }
        Ok(())
    }

    pub(crate) fn conditional_disagg_enabled(&self) -> bool {
        self.conditional_disagg_policy.is_enabled()
    }

    pub(crate) fn set_decode_routing_host(&self, routing_host: Arc<RoutingHost>) -> Result<()> {
        match self.decode_routing_host.set(routing_host) {
            Ok(()) => Ok(()),
            Err(routing_host)
                if self
                    .decode_routing_host
                    .get()
                    .is_some_and(|existing| Arc::ptr_eq(existing, &routing_host)) =>
            {
                Ok(())
            }
            Err(_) => anyhow::bail!(
                "PrefillRouter already has a different decode RoutingHost; rebuild requires a new PrefillRouter"
            ),
        }
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
            prefill_dp_rank: dp_rank,
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
mod tests;
