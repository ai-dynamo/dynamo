// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
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
    prefill_continue::{PrefillContinueDecisionInput, PrefillContinuePolicy},
    protocols::{RoutingConstraints, WorkerId},
    scheduling::QueueRejection,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, Operator, ResponseStream, RouterMode,
        ServerStreamingEngine, SingleIn, async_trait, propagate_first_response_guard,
    },
    protocols::{EndpointId, annotated::Annotated},
};
use futures::stream::{self, StreamExt};

use census::{ContinuationCensus, ContinuationPermit};

use crate::{
    discovery::{ModelManager, RuntimeConfigWatch},
    kv_router::metrics::{
        PREFILL_CONTINUE_METRICS, prefill_continue_decision, prefill_continue_demotion,
    },
    kv_router::{RoutingHost, WorkerSelectorFactory},
    local_model::runtime_config::{ModelRuntimeConfig, PREFILL_CONTINUE_CAPABILITY},
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{BootstrapInfo, PrefillResult, TraceLink},
        timing::{RequestPhase, RequestTracker, WORKER_TYPE_PREFILL},
    },
    session_affinity::AffinityTarget,
};

mod activation;
mod admission;
mod census;
mod conditional_bypass;
mod query;

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

    // Callers must include the worker's error text in this message. The
    // frontend receives this error through `to_pyerr`, which keeps only
    // `Display` and drops the source chain, so a `#[source]` is never seen.
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
    /// The prefill worker is serving the whole request, so its stream is the
    /// response. Nothing is consumed and no decode leg is dispatched.
    Continuation {
        stream: ManyOut<Annotated<LLMEngineOutput>>,
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
    /// Resolved at dispatch, so the continuation phase can attribute its
    /// per-worker metrics to the same rank the request actually ran on.
    dp_rank: Option<u32>,
    bootstrap_info: Option<BootstrapInfo>,
    topology_constraints: Option<RoutingConstraints>,
    /// Present only when the request is going out as a continuation. Its
    /// absence on a request that asked to continue means dispatch demoted it
    /// back to today's handoff.
    continuation_permit: Option<ContinuationPermit>,
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

/// Annotation marker set when the router asks a PREFILL-mode worker to keep
/// generating instead of handing the request to a decode worker. The worker will
/// read it to keep the request's own token budget and skip the decode handoff.
pub(crate) const PREFILL_CONTINUE_ANNOTATION: &str = "x-prefill-continue";

/// Drop any client-supplied copy of the router-owned routing markers.
///
/// Both markers select a routing path, so honoring a client copy would let a
/// caller pick its own. The router stamps them itself after its policies run.
///
/// Matches the bare marker and its `marker:value` form, because
/// `PreprocessedRequest::get_annotation_value` reads values off a `marker:`
/// prefix. Stripping only the bare form would leave `x-prefill-continue:1`
/// intact for any future valued read.
fn strip_router_owned_annotations(annotations: &mut Vec<String>) {
    fn is_router_owned(annotation: &str, marker: &str) -> bool {
        annotation == marker
            || (annotation.len() > marker.len()
                && annotation.starts_with(marker)
                && annotation.as_bytes()[marker.len()] == b':')
    }

    annotations.retain(|annotation| {
        !is_router_owned(annotation, BYPASS_REMOTE_PREFILL_ANNOTATION)
            && !is_router_owned(annotation, PREFILL_CONTINUE_ANNOTATION)
    });
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: `query_instance_id` annotation present → returns worker IDs without execution
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
pub struct PrefillRouter<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    binding: ArcSwapOption<PrefillBinding<Sel>>,
    target: Mutex<Option<EndpointId>>,
    target_tx: Option<watch::Sender<Option<dynamo_runtime::component::Endpoint>>>,
    /// Decode routing owns conditional-disagg planning and dispatch. This is
    /// installed after the frontend constructs its one decode `RoutingHost`.
    decode_routing_host: OnceLock<Arc<RoutingHost<Sel>>>,
    worker_selector_factory: Option<WorkerSelectorFactory<Sel>>,
    model_manager: Arc<ModelManager>,
    cancel_token: CancellationToken,
    /// Mode of the decode set that owns this router. Governs decode-side
    /// decisions, and is the fallback for the prefill hop -- not its mode. That
    /// lives on [`PrefillBinding::prefill_router_mode`].
    decode_router_mode: RouterMode,
    session_affinity_ttl: Option<std::time::Duration>,
    conditional_disagg_policy: Box<dyn ConditionalDisaggPolicy>,
    prefill_continue_policy: PrefillContinuePolicy,
    /// Continuations in flight per prefill worker. The only bound that can see
    /// a continuation after its first token.
    continuations: Arc<ContinuationCensus>,
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
    /// Initialization and worker availability state.
    lifecycle: AtomicU8,
    #[cfg(test)]
    activation_task_state: Arc<()>,
}

struct PrefillBinding<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    endpoint_id: EndpointId,
    router: Arc<RoutingHost<Sel>>,
    /// Resolved at activation from the prefill card. Lives here rather than on
    /// `PrefillRouter` because it is unknowable until a target is discovered,
    /// and changes when the binding is rebuilt.
    prefill_router_mode: RouterMode,
    /// Live per-worker runtime configuration for the prefill endpoint.
    ///
    /// Read rather than snapshotted, because the binding is only rebuilt when
    /// the endpoint itself changes: a worker joining an existing endpoint would
    /// never be seen by a value captured at activation.
    prefill_runtime_configs: RuntimeConfigWatch,
}

/// Why the router may not ask this prefill pool for a continuation.
///
/// Carries the reason rather than a bare `false`, for the same reason
/// [`PrefillContinueSkip`] does: during bring-up this gate is the likeliest
/// explanation for "the feature never fired", and an operator needs to be told
/// which worker is holding it back.
#[derive(Debug, PartialEq, Eq)]
enum PrefillPoolCapability {
    /// Every routable worker declared it understands the marker.
    Supported,
    /// Nothing to route to, so nothing declared anything.
    NoRoutableWorkers,
    /// These routable workers did not declare support. A worker with no card
    /// yet lands here too: absent is not a yes.
    Undeclared(Vec<WorkerId>),
}

/// Ask whether every worker the router could pick declared it understands the
/// continuation marker.
///
/// Unanimous, not first-wins. One worker that ignores the marker answers with a
/// handoff message and pins cache blocks, so a mixed pool turns the feature off
/// rather than gambling on which worker gets selected.
///
/// The question is asked of `routable`, not of the config map, because the two
/// are not the same set. The map holds only workers that have both registered
/// and had a card discovered, so a worker that is already selectable but whose
/// card has not arrived yet is missing from it entirely — and a check that only
/// walked the map would read unanimous while that worker took a marked request.
///
/// Both inputs are sampled before a worker is selected, and they are sampled
/// separately, so this narrows the window rather than closing it: a worker that
/// becomes routable afterwards can still be handed a marked request. Closing it
/// for good means re-checking the chosen worker at dispatch.
fn prefill_pool_capability(
    routable: &[WorkerId],
    runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> PrefillPoolCapability {
    if routable.is_empty() {
        return PrefillPoolCapability::NoRoutableWorkers;
    }
    // The same truthy vocabulary every other runtime capability uses. Being
    // stricter here would refuse a backend that spells the flag as a string,
    // and refuse it silently.
    let undeclared: Vec<WorkerId> = routable
        .iter()
        .copied()
        .filter(|worker_id| {
            !runtime_configs.get(worker_id).is_some_and(|config| {
                config.supports_runtime_capability(PREFILL_CONTINUE_CAPABILITY)
            })
        })
        .collect();
    if undeclared.is_empty() {
        PrefillPoolCapability::Supported
    } else {
        PrefillPoolCapability::Undeclared(undeclared)
    }
}

/// Read the live pool state and say whether a continuation may be asked for,
/// logging the reason when it may not.
fn prefill_pool_allows_continuation<Sel>(
    binding: &PrefillBinding<Sel>,
    routable: &[WorkerId],
    request_id: &str,
) -> bool
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    // The watch guard lives only for this statement, so no lock is held across
    // the awaits that follow.
    let capability = prefill_pool_capability(routable, &binding.prefill_runtime_configs.borrow());
    match capability {
        PrefillPoolCapability::Supported => true,
        PrefillPoolCapability::NoRoutableWorkers => {
            PREFILL_CONTINUE_METRICS
                .record_decision(prefill_continue_decision::NO_ROUTABLE_WORKERS);
            tracing::debug!(
                request_id,
                "Prefill continuation declined: no routable prefill workers"
            );
            false
        }
        PrefillPoolCapability::Undeclared(undeclared) => {
            PREFILL_CONTINUE_METRICS.record_decision(prefill_continue_decision::POOL_UNDECLARED);
            tracing::debug!(
                request_id,
                ?undeclared,
                routable = routable.len(),
                "Prefill continuation declined: these routable prefill workers did not declare \
                 support for the continuation marker, so the whole pool hands off as today"
            );
            false
        }
    }
}

/// Put a request that asked to continue back on today's handoff path.
///
/// Both halves of the ask have to come off together. Leaving the marker on
/// would have the worker generate a whole response that nothing returns;
/// leaving the budget unclamped would have it generate one that nothing reads.
fn demote_to_handoff(request: &mut PreprocessedRequest) {
    request
        .annotations
        .retain(|annotation| annotation != PREFILL_CONTINUE_ANNOTATION);
    request.stop_conditions.max_tokens = Some(1);
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Decide, before routing, whether this request should keep generating on
    /// its prefill worker.
    ///
    /// Every signal here is a property of the pool or the request, because a
    /// worker has usually not been chosen yet — the exception is a request that
    /// names its own, which is asked about directly. Either way the per-worker
    /// bound and the chosen worker's capability are settled again at dispatch,
    /// which is the authoritative check.
    async fn wants_prefill_continuation(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        binding: &PrefillBinding<Sel>,
        request_id: &str,
    ) -> bool {
        if !self.prefill_continue_policy.is_enabled() {
            return false;
        }
        // One sample of the pool, read by both the capability gate and the
        // census, so the two cannot disagree about who is routable.
        let routable = binding.router.routable_instance_ids();
        if !prefill_pool_allows_continuation(binding, &routable, request_id) {
            return false;
        }

        // Cheap gates first. Each probe below costs a scheduler selection, and
        // a request the budget or the cap already refuses must not pay for one.
        // At a cap of zero that would be every request.
        let budget = request.stop_conditions.max_tokens;
        // An externally routed request already names its worker, so ask that
        // worker rather than the pool. Otherwise the emptiest worker's count
        // is the only honest answer available before selection.
        let active = match request.routing.as_ref().and_then(|r| r.prefill_worker_id) {
            Some(worker_id) => Some(self.continuations.in_flight(worker_id)),
            None => self.continuations.min_in_flight(&routable),
        };
        if let Some(reason) = self.prefill_continue_policy.preflight(budget, active) {
            PREFILL_CONTINUE_METRICS.record_decision(reason.as_str());
            tracing::debug!(
                request_id,
                ?reason,
                "Prefill continuation declined before measuring load"
            );
            return false;
        }

        // The override continues without consulting decode load, so measuring
        // it would buy an answer nothing reads.
        let measured = if self.prefill_continue_policy.needs_decode_load() {
            self.peek_decode_headroom(request, request_id).await
        } else {
            PrefillContinueDecisionInput::new(None, None, 0)
        };
        let input = measured
            .with_prefill_worker_busy(self.peek_prefill_busy(request, binding, request_id).await)
            .with_remaining_budget_tokens(budget)
            .with_active_continuations(active);

        let decision = self.prefill_continue_policy.decide(input);
        match decision.skip_reason() {
            Some(reason) => {
                PREFILL_CONTINUE_METRICS.record_decision(reason.as_str());
                tracing::debug!(
                    request_id,
                    ?reason,
                    "Prefill continuation declined before routing"
                );
            }
            None => PREFILL_CONTINUE_METRICS.record_decision(prefill_continue_decision::CONTINUE),
        }
        decision.should_continue()
    }

    /// Read the decode pool's headroom for this request.
    ///
    /// A preview needs a KV-routed decode set, so anything else reports nothing
    /// and the policy refuses on `DecodeLoadUnknown`. That is the honest
    /// answer: without the preview there is no way to tell a full decode pool
    /// from an idle one.
    ///
    /// A cancelled request is reported as unknown like any other failure, which
    /// refuses, and the ordinary handoff it falls back to then fails on the
    /// same cancelled context. Deliberate, unlike the sibling decision, which
    /// re-raises cancellation because it is about to dispatch on it.
    async fn peek_decode_headroom(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        request_id: &str,
    ) -> PrefillContinueDecisionInput {
        let unknown = PrefillContinueDecisionInput::new(None, None, 0);
        let Some(decode_host) = self.decode_routing_host.get() else {
            return unknown;
        };
        // Ask the host, not the mode flag: `kv_router()` panics on a host with
        // no KV plane, and the two could otherwise disagree.
        let Some(kv_router) = decode_host.kv_router_if_enabled() else {
            return unknown;
        };
        let block_size = kv_router.block_size() as usize;
        match decode_host
            .preview_kv_route(request, RequestPhase::Decode)
            .await
        {
            Ok(preview) => {
                let signals = preview.signals();
                PrefillContinueDecisionInput::new(
                    Some(signals.potential_decode_blocks as usize),
                    signals.total_kv_blocks.map(|total| total as usize),
                    block_size,
                )
            }
            Err(error) => {
                tracing::debug!(
                    request_id,
                    %error,
                    "Decode headroom probe failed; treating decode load as unavailable"
                );
                PrefillContinueDecisionInput::new(None, None, block_size)
            }
        }
    }

    /// Read whether the prefill worker this request would land on is over its
    /// own busy line.
    ///
    /// Probes the caller's binding rather than re-reading it, so the interlock
    /// cannot end up asking a different pool than the capability gate did.
    async fn peek_prefill_busy(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        binding: &PrefillBinding<Sel>,
        request_id: &str,
    ) -> Option<bool> {
        // The policy owns the threshold, so the probe and the judgement cannot
        // disagree about which one is in force.
        let threshold = self.prefill_continue_policy.interlock_threshold()?;
        match binding.router.prefill_worker_busy(request, threshold).await {
            Ok(busy) => Some(busy),
            Err(error) => {
                // Cancellation lands here too, like any other failure: unknown
                // refuses, and the handoff it falls back to fails on the same
                // dead context a moment later.
                tracing::debug!(
                    request_id,
                    %error,
                    "Prefill-load probe failed; treating prefill load as unavailable"
                );
                None
            }
        }
    }
}

struct PrefillBuildContext<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    model_manager: Arc<ModelManager>,
    /// Fallback mode for the prefill hop when the prefill card advertises none.
    decode_router_mode: RouterMode,
    worker_selector_factory: WorkerSelectorFactory<Sel>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    session_affinity_ttl: Option<std::time::Duration>,
    model_name: String,
    load_thresholds: crate::discovery::LoadThresholdHandle,
    parent_token: CancellationToken,
    task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
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
        let (mut req, mut context) = request.into_parts();
        let request_id = context.id().to_string();
        let metadata = context.metadata().clone();
        let engine_ctx = context.context();

        strip_router_owned_annotations(&mut req.annotations);

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If the prefill router is not activated (no prefill workers discovered) or has been
        // deactivated (all prefill workers died), route directly to the backend. Model admission
        // remains gated by the registered worker topology before the request reaches this stage.
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return next.generate(context.map(|_| req)).await;
        }

        // Query-only requests are owned by the decode RoutingHost. In particular,
        // do not turn an advisory worker lookup into conditional local execution.
        if req.get_annotation_value("query_instance_id").is_some() {
            return next.generate(context.map(|_| req)).await;
        }

        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| anyhow::anyhow!("invalid session affinity context: {message}"))?;

        if self.conditional_disagg_policy.is_enabled() {
            let conditional_request = context.map(|_| req);
            match self
                .plan_conditional_disagg_decode(&conditional_request, &request_id)
                .await
            {
                Ok(Some(decision)) => {
                    let signals = decision.plan.signals();
                    tracing::info!(
                        request_id = %request_id,
                        worker_id = signals.worker.worker_id,
                        dp_rank = signals.worker.dp_rank,
                        net_new_tokens = decision.net_new_tokens,
                        overlap_tokens = decision.overlap_tokens,
                        "Conditional disagg routing to decode worker"
                    );

                    let mut conditional_request = conditional_request;
                    if conditional_request.tracker.is_none() {
                        conditional_request.tracker = Some(Arc::new(RequestTracker::new()));
                    }
                    if let Some(ref tracker) = conditional_request.tracker {
                        let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    }

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
                    let ctx = response_stream.context();
                    let annotation = Annotated::<LLMEngineOutput>::from_annotation(
                        BYPASS_REMOTE_PREFILL_ANNOTATION,
                        &true,
                    )?;
                    let merged = stream::once(async move { annotation }).chain(response_stream);
                    return Ok(ResponseStream::new(Box::pin(merged), ctx));
                }
                Ok(None) => {
                    (req, context) = conditional_request.into_parts();
                }
                Err(error) if crate::kv_router::routing_host::is_cancelled(&error) => {
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

        // Ensure tracker exists for routing decisions in disaggregated mode.
        // Create one if not provided by the upstream DeltaGenerator.
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.as_ref().unwrap();
        let prefill_phase_barrier = tracker.set_phase(RequestPhase::Prefill).await;

        // The binding is what makes a prefill hop possible at all, so read it
        // before building the hop. It also carries the prefill pool's declared
        // capabilities, which the continuation decision below needs.
        let Some(binding) = self.binding.load_full() else {
            return next.generate(context.map(|_| req)).await;
        };

        // The probes below want a request context, as the conditional-disagg
        // decision above did. Borrow one and take it back apart.
        let continue_request = context.map(|_| req);
        let prefill_continues = self
            .wants_prefill_continuation(&continue_request, &binding, &request_id)
            .await;
        (req, context) = continue_request.into_parts();

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set).
        // A continuation is the whole response, so it keeps the request's own
        // budget; clamping it here would silently reduce the feature to a no-op.
        let mut prefill_req = req.clone();
        if prefill_continues {
            prefill_req
                .annotations
                .push(PREFILL_CONTINUE_ANNOTATION.to_string());
        } else {
            prefill_req.stop_conditions.max_tokens = Some(1);
        }

        // Try to resolve prefill worker upfront: if we can get bootstrap info early,
        // spawn prefill in background and proceed to decode immediately.
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id);

        let tracker = prefill_req.tracker.clone();
        let mut prefill_context =
            Context::with_id_and_metadata(prefill_req, request_id.clone(), metadata.clone());
        propagate_first_response_guard(&context, &mut prefill_context)?;
        // Kept so the continuation arm can link this context to the client's
        // once dispatch confirms the request really is continuing. Linking here
        // would be too early: dispatch can still demote, and `link_child` has
        // no inverse, so the handoff path would be left carrying a cancel route
        // into the prefill leg that it does not have today.
        let prefill_ctx = prefill_context.context();
        if let Some(session_affinity) = session_affinity {
            prefill_context.insert(
                SESSION_AFFINITY_CONTEXT_KEY,
                session_affinity.as_ref().clone(),
            );
        }
        // Keyed on the prefill mode, not the decode set's.
        if binding.prefill_router_mode.is_direct_routing() && preselected_worker.is_none() {
            return Err(anyhow::anyhow!(
                "Prefill worker ID required in Direct routing mode but none found in request. \
                 Expected prefill_worker_id to be set via x-dynamo-prefill-instance-id header by external router (e.g., EPP)."
            ));
        }

        let router = &binding.router;
        let prefill_result: Result<(PrefillOutcome, Option<RoutingConstraints>)> = async {
            let (prepared, prefill_stream) = router
                .select_and_dispatch_prefill(prefill_context, |request, target| {
                    self.prepare_prefill_dispatch(request, target, &binding, prefill_continues)
                })
                .await?;
            let topology_constraints = prepared.topology_constraints;
            // Not `prefill_continues`: dispatch has the last word, because only
            // it knew which worker was chosen.
            let outcome = if let Some(permit) = prepared.continuation_permit {
                // Outranks bootstrap: that path backgrounds the prefill stream and
                // dispatches a decode leg, which would discard the response. Drop
                // the phase permit as the ordinary path does, so a migration retry
                // can set Prefill again.
                drop(prefill_phase_barrier);
                // Move the request out of its prefill phase even though the
                // worker does not change.
                //
                // The frontend latches worker attribution from the tracker on
                // the first response chunk and resolves the inter-token latency
                // gauge from the decode worker it finds there. A continuation
                // never dispatches one, so that latch saw nothing and the gauge
                // was never written: inter-token latency was emitted for every
                // arm except the ones running this feature. Recording it here,
                // before the stream can be polled, is what fixes it.
                //
                // `Continuation` records the same worker as both legs and stays
                // distinct from `Aggregated`, so the arm is still tellable apart
                // in an A/B. Note the decode worker type is recorded as
                // `prefill`; see the field's doc for why, and query it that way.
                //
                // The barrier above must be dropped first: the phase semaphore
                // holds a single permit, so setting a phase while holding it
                // would deadlock. The permit taken here is bound, not dropped,
                // so it covers the recording below.
                if let Some(tracker) = tracker.as_ref() {
                    let _continuation_phase_permit =
                        tracker.set_phase(RequestPhase::Continuation).await;
                    tracker.record_worker(
                        prepared.worker_id,
                        prepared.dp_rank,
                        WORKER_TYPE_PREFILL,
                    );
                }
                // The prefill context has its own controller, so a cancel would
                // not reach the worker on a stream we hand back to the client.
                // Link it, as Migration does for its retry children.
                engine_ctx.link_child(prefill_ctx.clone());
                // link_child does not replay state already set, so a cancel that
                // arrived before the link would be lost and the worker would
                // generate a whole response for a client that is gone. Migration
                // guards the same race the same way.
                if engine_ctx.is_stopped() || engine_ctx.is_killed() {
                    prefill_ctx.stop_generating();
                }
                PrefillOutcome::Continuation {
                    stream: permit.into_stream(prefill_stream),
                }
            } else if let Some(bootstrap_info) = prepared.bootstrap_info {
                self.spawn_prefill_task(prefill_stream, tracker, prefill_phase_barrier);
                PrefillOutcome::Bootstrap {
                    bootstrap_info,
                    worker_id: prepared.worker_id,
                }
            } else {
                drop(prefill_phase_barrier);
                let completion =
                    Self::consume_prefill_stream(prefill_stream, tracker, self.task_guard.clone())
                        .await?;

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
            PrefillOutcome::Continuation { stream } => return Ok(stream),
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
            PrefillOutcome::Continuation { .. } => {
                unreachable!("a continuation returns its stream before decode routing")
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

    /// Whether this router needs the decode set's `RoutingHost` installed.
    ///
    /// Both pre-routing decisions read decode load through it, and without it
    /// they can only ever answer "unknown", which both of them treat as a
    /// refusal. So the host has to be installed for either feature, not just
    /// the one that first needed it.
    pub(crate) fn needs_decode_routing_host(&self) -> bool {
        self.conditional_disagg_enabled() || self.prefill_continue_policy.is_enabled()
    }

    pub(crate) fn set_decode_routing_host(
        &self,
        routing_host: Arc<RoutingHost<Sel>>,
    ) -> Result<()> {
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
        binding: &PrefillBinding<Sel>,
        wants_continuation: bool,
    ) -> anyhow::Result<PreparedPrefill> {
        let AffinityTarget { worker_id, dp_rank } = target;
        let endpoint_id = &binding.endpoint_id;
        let topology_constraints =
            self.preflight_kv_transfer_constraints(Some(endpoint_id), worker_id)?;

        // The continuation decision is only final here, because only here is
        // the worker known. Everything before this ran against the pool.
        let continuation_permit = if wants_continuation {
            self.admit_continuation(request, worker_id, binding)
        } else {
            None
        };

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
            dp_rank,
            bootstrap_info,
            topology_constraints,
            continuation_permit,
        })
    }

    /// Take a place in the census for `worker_id`, or put the request back on
    /// today's handoff.
    ///
    /// Two things can refuse here, and both need the chosen worker:
    ///
    /// - the worker is already running its share of continuations, and
    /// - the worker never declared it understands the marker. The pool check
    ///   before routing reads the routable set, but that set is sampled before
    ///   selection; a worker that appeared in between reaches this point
    ///   unchecked. Asking again once it is chosen closes that window.
    ///
    /// Demoting means undoing both halves of the ask: the marker comes off, so
    /// the worker builds its usual handoff, and the one-token clamp goes back
    /// on, so it stops after the token that handoff carries.
    fn admit_continuation(
        &self,
        request: &mut PreprocessedRequest,
        worker_id: u64,
        binding: &PrefillBinding<Sel>,
    ) -> Option<ContinuationPermit> {
        let declared = binding
            .prefill_runtime_configs
            .borrow()
            .get(&worker_id)
            .is_some_and(|config| config.supports_runtime_capability(PREFILL_CONTINUE_CAPABILITY));
        if !declared {
            PREFILL_CONTINUE_METRICS.record_demotion(prefill_continue_demotion::WORKER_UNDECLARED);
            tracing::debug!(
                worker_id,
                "Prefill continuation demoted to a handoff: the selected worker never declared \
                 support for the continuation marker"
            );
            demote_to_handoff(request);
            return None;
        }

        // Refuse rather than run unbounded when no cap is configured. Startup
        // validation asks for one, but it does not run on every path a router
        // can be built from, so a config can still arrive here without one.
        let Some(cap) = self.prefill_continue_policy.max_concurrent() else {
            PREFILL_CONTINUE_METRICS.record_demotion(prefill_continue_demotion::NO_CAP_CONFIGURED);
            tracing::debug!(
                worker_id,
                "Prefill continuation demoted to a handoff: no continuation cap is configured, \
                 and the cap is the only bound on continuations that are already running"
            );
            demote_to_handoff(request);
            return None;
        };
        if let Some(permit) = self.continuations.try_admit(worker_id, cap) {
            return Some(permit);
        }

        PREFILL_CONTINUE_METRICS.record_demotion(prefill_continue_demotion::WORKER_AT_CAP);
        tracing::debug!(
            worker_id,
            in_flight = self.continuations.in_flight(worker_id),
            ?cap,
            "Prefill continuation demoted to a handoff: the selected worker has no free \
             continuation place"
        );
        demote_to_handoff(request);
        None
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
    use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
    use dynamo_runtime::{engine::AsyncEngine, pipeline::Error};
    use futures::StreamExt;
    use std::{
        collections::{HashMap, HashSet},
        sync::atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::oneshot;

    use crate::protocols::common::{
        FinishReason,
        preprocessor::{PreprocessedRequest, RoutingHints},
        timing::RoutingData,
    };

    const MAX_ROOM: u64 = i64::MAX as u64;

    #[derive(Default)]
    struct QueryOnlyDecodeHost {
        requests: AtomicUsize,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for QueryOnlyDecodeHost
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            assert!(request.get_annotation_value("query_instance_id").is_some());
            // The router owns both routing markers. Assert here, at the earliest
            // return in generate(), that a client copy never reaches a worker.
            assert!(!request.has_annotation(BYPASS_REMOTE_PREFILL_ANNOTATION));
            assert!(!request.has_annotation(PREFILL_CONTINUE_ANNOTATION));
            assert!(
                request
                    .get_annotation_value(PREFILL_CONTINUE_ANNOTATION)
                    .is_none()
            );
            self.requests.fetch_add(1, Ordering::Relaxed);
            let output = Annotated::from_data(LLMEngineOutput {
                routing_data: Some(RoutingData {
                    token_ids: Some(request.token_ids.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::once(async move { output })),
                request.context(),
            ))
        }
    }

    fn active_conditional_router() -> Arc<PrefillRouter> {
        let (_activation_tx, activation_rx) = oneshot::channel();
        let router = PrefillRouter::new(
            activation_rx,
            Arc::new(ModelManager::new()),
            RouterMode::KV,
            16,
            Some(KvRouterConfig {
                conditional_disagg_enabled: true,
                ..Default::default()
            }),
            None,
            None,
            "model".to_string(),
            "namespace".to_string(),
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            CancellationToken::new(),
        );
        assert!(router.conditional_disagg_enabled());
        router.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        router
    }

    fn query_only_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .annotations(vec!["query_instance_id:".to_string()])
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn conditional_disagg_query_only_forwards_to_decode_host() {
        let router = active_conditional_router();
        let decode_host = Arc::new(QueryOnlyDecodeHost::default());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            decode_host.clone();

        let mut response = router
            .generate(SingleIn::new(query_only_request()), next)
            .await
            .expect("query-only request should bypass conditional planning");
        let output = response
            .next()
            .await
            .expect("decode host should return routing data")
            .data
            .expect("query-only response should contain data");

        assert_eq!(decode_host.requests.load(Ordering::Relaxed), 1);
        assert_eq!(
            output.routing_data.and_then(|routing| routing.token_ids),
            Some(vec![1, 2, 3])
        );
    }

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

    #[tokio::test]
    async fn dropping_pending_router_releases_activation_tasks() {
        let (_activation_tx, activation_rx) = tokio::sync::oneshot::channel();
        let router = PrefillRouter::new(
            activation_rx,
            Arc::new(crate::discovery::ModelManager::new()),
            RouterMode::RoundRobin,
            16,
            None,
            None,
            None,
            "test-model".to_string(),
            "test-namespace".to_string(),
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            CancellationToken::new(),
        );
        let task_state = Arc::downgrade(&router.activation_task_state);
        let weak = Arc::downgrade(&router);

        drop(router);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while weak.strong_count() != 0 || task_state.strong_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pending activation tasks retained their PrefillRouter");
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

    #[tokio::test]
    async fn client_supplied_routing_markers_never_reach_a_worker() {
        // The helper being correct is the easy half. This guards the half that
        // matters: that generate() strips BEFORE its earliest return, so the
        // ordering cannot silently regress under refactor.
        let router = active_conditional_router();
        let decode_host = Arc::new(QueryOnlyDecodeHost::default());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            decode_host.clone();

        let mut request = query_only_request();
        request.annotations.extend([
            BYPASS_REMOTE_PREFILL_ANNOTATION.to_string(),
            PREFILL_CONTINUE_ANNOTATION.to_string(),
            // the valued form a prefix-blind strip would miss
            format!("{PREFILL_CONTINUE_ANNOTATION}:1"),
        ]);

        let mut response = router
            .generate(SingleIn::new(request), next)
            .await
            .expect("query-only request should bypass conditional planning");
        while response.next().await.is_some() {}

        // The mock asserts the markers are gone; reaching it at all proves the
        // strip ran before generate()'s earliest return.
        assert_eq!(decode_host.requests.load(Ordering::Relaxed), 1);
    }

    /// A pool of workers. `true` publishes the capability; `false` publishes
    /// nothing at all, which is how a worker that has never heard of the
    /// feature looks.
    fn pool(publishes: &[bool]) -> HashMap<WorkerId, ModelRuntimeConfig> {
        publishes
            .iter()
            .enumerate()
            .map(|(index, publishes)| {
                let mut config = ModelRuntimeConfig::default();
                if *publishes {
                    config
                        .set_engine_specific(PREFILL_CONTINUE_CAPABILITY, true)
                        .unwrap();
                }
                (index as WorkerId, config)
            })
            .collect()
    }

    /// A one-worker pool that published `value` under the capability key.
    fn pool_declaring<T: serde::Serialize>(value: T) -> HashMap<WorkerId, ModelRuntimeConfig> {
        let mut config = ModelRuntimeConfig::default();
        config
            .set_engine_specific(PREFILL_CONTINUE_CAPABILITY, value)
            .unwrap();
        HashMap::from([(0, config)])
    }

    /// Every worker in `configs`, which is what the router can route to once
    /// discovery has settled.
    fn all_routable(configs: &HashMap<WorkerId, ModelRuntimeConfig>) -> Vec<WorkerId> {
        let mut ids: Vec<WorkerId> = configs.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn a_unanimous_pool_supports_continuation() {
        let configs = pool(&[true, true, true]);

        assert_eq!(
            prefill_pool_capability(&all_routable(&configs), &configs),
            PrefillPoolCapability::Supported
        );
    }

    #[test]
    fn one_undeclared_worker_disables_the_whole_pool() {
        let configs = pool(&[true, true, false]);

        assert_eq!(
            prefill_pool_capability(&all_routable(&configs), &configs),
            PrefillPoolCapability::Undeclared(vec![2])
        );
    }

    #[test]
    fn a_pool_that_declared_nothing_names_every_worker() {
        let configs = pool(&[false, false]);

        assert_eq!(
            prefill_pool_capability(&all_routable(&configs), &configs),
            PrefillPoolCapability::Undeclared(vec![0, 1])
        );
    }

    #[test]
    fn a_routable_worker_with_no_card_yet_is_a_refusal() {
        // The runtime-config watch lists only workers that both registered and
        // had a card discovered, so a worker can be selectable before it
        // appears there. Walking the map alone would read this pool as
        // unanimous and send the marker to a worker of unknown capability.
        let configs = pool(&[true, true]);
        let mut routable = all_routable(&configs);
        routable.push(99);

        assert_eq!(
            prefill_pool_capability(&routable, &configs),
            PrefillPoolCapability::Undeclared(vec![99])
        );
    }

    #[test]
    fn an_empty_pool_is_not_capable() {
        assert_eq!(
            prefill_pool_capability(&[], &pool(&[])),
            PrefillPoolCapability::NoRoutableWorkers
        );
    }

    #[test]
    fn an_explicit_false_declaration_is_a_refusal() {
        assert_eq!(
            prefill_pool_capability(&[0], &pool_declaring(false)),
            PrefillPoolCapability::Undeclared(vec![0])
        );
    }

    #[test]
    fn a_string_encoded_declaration_is_accepted() {
        // Deliberately the same truthy vocabulary as every other runtime
        // capability. A backend that spells the flag as a string must not be
        // refused, because that refusal would be silent.
        for spelling in ["true", "1", "on", "yes"] {
            assert_eq!(
                prefill_pool_capability(&[0], &pool_declaring(spelling)),
                PrefillPoolCapability::Supported,
                "{spelling} should read as support"
            );
        }
    }

    #[test]
    fn a_value_that_means_nothing_is_a_refusal() {
        for value in [
            serde_json::json!("banana"),
            serde_json::json!(1),
            serde_json::json!(null),
        ] {
            assert_eq!(
                prefill_pool_capability(&[0], &pool_declaring(value.clone())),
                PrefillPoolCapability::Undeclared(vec![0]),
                "{value} must not read as support"
            );
        }
    }

    #[test]
    fn strip_router_owned_annotations_drops_the_valued_form() {
        // `get_annotation_value` reads values off a `marker:` prefix, so the
        // valued form is a live bypass if the strip only matches the bare marker.
        let mut annotations = vec![
            format!("{PREFILL_CONTINUE_ANNOTATION}:1"),
            format!("{BYPASS_REMOTE_PREFILL_ANNOTATION}:true"),
            // a different marker that merely shares a prefix must survive
            format!("{PREFILL_CONTINUE_ANNOTATION}-other"),
        ];

        strip_router_owned_annotations(&mut annotations);

        assert_eq!(
            annotations,
            vec![format!("{PREFILL_CONTINUE_ANNOTATION}-other")]
        );
    }

    #[test]
    fn strip_router_owned_annotations_drops_every_copy_of_both_markers() {
        // A client can send a marker more than once; every copy must go, and
        // everything else must survive in order.
        let mut annotations = vec![
            "keep-me".to_string(),
            BYPASS_REMOTE_PREFILL_ANNOTATION.to_string(),
            "also-keep".to_string(),
            PREFILL_CONTINUE_ANNOTATION.to_string(),
            PREFILL_CONTINUE_ANNOTATION.to_string(),
        ];

        strip_router_owned_annotations(&mut annotations);

        assert_eq!(annotations, vec!["keep-me", "also-keep"]);
    }
}
