// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PrefillRouter` as a [`StageRouter`], and `generate_v2` composed from it.
//!
//! `generate_v2` is the operator's request path rebuilt from `schedule`,
//! `preview`, `dispatch` and small helpers. It is meant to replace `generate`.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, ResponseStream, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tokio::sync::OwnedSemaphorePermit;

use super::{
    ActivationSnapshot, BYPASS_REMOTE_PREFILL_ANNOTATION, PrefillBinding, PrefillCompletion,
    PrefillLifecycleState, PrefillOutcome, PrefillRouter, PreparedPrefill, apply_prefill_dispatch,
    conditional_bypass::ConditionalDisaggDecodeDecision, extract_bootstrap_info,
    independent_prefill_context, prefill_activation_pending_error, prefill_need,
    strip_terminal_disaggregated_params,
};
use crate::{
    kv_router::{
        RequestAttemptRelease,
        routing_host::{PrefillDispatch, RoutePlan},
        stage_router::StageRouter,
    },
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::{RequestPhase, RequestTracker},
    },
    session_affinity::AffinityTarget,
};

use super::handoff::PrefillTask;

/// Where the prefill stage runs for one request.
// Built once per request and moved straight into its next step.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PrefillPlan {
    /// No prefill target: the request goes to decode without a prefill hop.
    Untargeted,
    /// A KV-routed prefill worker, admitted and booked.
    Booked {
        binding: Arc<PrefillBinding>,
        route: RoutePlan,
        prepared: PreparedPrefill,
    },
    /// A prefill binding whose router mode selects and sends in one step, so
    /// the worker is chosen at dispatch.
    SelectAtDispatch { binding: Arc<PrefillBinding> },
}

/// What the decode stage looks like for one request.
// Built once per request and moved straight into its next step.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PrefillPreview {
    /// Run prefill on a prefill worker, the ordinary disaggregated route.
    RemotePrefill,
    /// Conditional disaggregation keeps the request on this admitted decode
    /// worker, which runs prefill locally.
    LocalDecode(ConditionalDisaggDecodeDecision),
}

/// A dispatched prefill leg: where it went and its response stream.
pub(crate) struct PrefillDispatched {
    prepared: PreparedPrefill,
    stream: ManyOut<Annotated<LLMEngineOutput>>,
    /// Frees the KV booking early, so a migration retry can release it while
    /// the prefill stream still runs. `None` for built-in modes.
    booking: Option<RequestAttemptRelease>,
}

/// How far prefill got before decode can start.
struct PrefillProgress {
    outcome: PrefillOutcome,
    topology_constraints: Option<RoutingConstraints>,
    /// Prefill still running in the background; decode merges its completion.
    completion: Option<PrefillTask>,
}

#[async_trait]
impl StageRouter for PrefillRouter {
    type Plan = PrefillPlan;
    type Preview = PrefillPreview;
    type Dispatched = PrefillDispatched;

    /// Resolve the prefill binding and book a prefill worker.
    ///
    /// Fails closed while a prefill target is still activating. Returns
    /// [`PrefillPlan::Untargeted`] when there is no prefill target, so the
    /// caller can send the request to decode instead.
    async fn schedule(&self, request: &SingleIn<PreprocessedRequest>) -> Result<PrefillPlan> {
        let Some(binding) = self.resolve_binding()? else {
            return Ok(PrefillPlan::Untargeted);
        };

        // Keyed on the prefill mode, not the decode set's, and checked after
        // the binding loads because that is where the prefill mode lives.
        let preselected_worker = request.routing.as_ref().and_then(|r| r.prefill_worker_id);
        if binding.prefill_router_mode.is_direct_routing() && preselected_worker.is_none() {
            return Err(anyhow::anyhow!(
                "Prefill worker ID required in Direct routing mode but none found in request. \
                 Expected prefill_worker_id to be set via x-dynamo-prefill-instance-id header by external router (e.g., EPP)."
            ));
        }

        if binding.router.kv_router_if_enabled().is_none() {
            return Ok(PrefillPlan::SelectAtDispatch { binding });
        }
        let route = binding
            .router
            .plan_kv_route_without_preview(request, RequestPhase::Prefill)
            .await?;
        let worker = route.worker();
        let target = AffinityTarget::new(worker.worker_id, Some(worker.dp_rank));
        match self.plan_prefill_dispatch(target, &binding.endpoint_id) {
            Ok(prepared) => Ok(PrefillPlan::Booked {
                binding,
                route,
                prepared,
            }),
            Err(error) => {
                route.abort(Some(&error)).await;
                Err(error)
            }
        }
    }

    /// Decide whether conditional disaggregation keeps the request on its
    /// decode worker. When it does, the decode worker is already admitted.
    ///
    /// Cancellation and invalid arguments fail the request. Any other planning
    /// error falls back to remote prefill.
    async fn preview(&self, request: &SingleIn<PreprocessedRequest>) -> Result<PrefillPreview> {
        if !self.conditional_disagg_policy.is_enabled() {
            return Ok(PrefillPreview::RemotePrefill);
        }
        let request_id = request.id().to_string();
        match self
            .plan_conditional_disagg_decode(request, &request_id)
            .await
        {
            Ok(Some(decision)) => Ok(PrefillPreview::LocalDecode(decision)),
            Ok(None) => Ok(PrefillPreview::RemotePrefill),
            Err(error)
                if match_error_chain(
                    error.as_ref(),
                    &[ErrorType::Cancelled, ErrorType::InvalidArgument],
                    &[],
                ) =>
            {
                Err(error)
            }
            Err(error) => {
                tracing::warn!(
                    request_id = %request_id,
                    error = %error,
                    "Conditional disagg decision failed; falling back to remote prefill"
                );
                Ok(PrefillPreview::RemotePrefill)
            }
        }
    }

    /// Send the prefill request to its worker.
    async fn dispatch(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        plan: PrefillPlan,
    ) -> Result<PrefillDispatched> {
        match plan {
            PrefillPlan::Untargeted => Err(anyhow::anyhow!(
                "an untargeted prefill plan has no prefill worker to dispatch to"
            )),
            PrefillPlan::Booked {
                binding,
                route,
                prepared,
            } => {
                apply_prefill_dispatch(&mut request, &prepared);
                let booking = route.attempt_release();
                let stream = binding.router.dispatch_kv_plan(request, route).await?;
                Ok(PrefillDispatched {
                    prepared,
                    stream,
                    booking,
                })
            }
            PrefillPlan::SelectAtDispatch { binding } => {
                let endpoint_id = &binding.endpoint_id;
                let PrefillDispatch {
                    metadata: prepared,
                    stream,
                    booking,
                } = binding
                    .router
                    .select_and_dispatch_prefill(request, |request, target| {
                        self.prepare_prefill_dispatch(request, target, endpoint_id)
                    })
                    .await?;
                Ok(PrefillDispatched {
                    prepared,
                    stream,
                    booking,
                })
            }
        }
    }
}

impl PrefillRouter {
    /// `generate` composed from [`StageRouter`] calls and helpers.
    ///
    /// Behaves like `generate`, though not bit for bit: a KV-routed prefill
    /// leg books its worker before preparing the request, and schedule
    /// failures are not logged as prefill failures. It is not wired into the
    /// pipeline yet.
    #[allow(dead_code)]
    pub(crate) async fn generate_v2(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (mut req, context) = request.into_parts();
        let engine_ctx = context.context();

        strip_client_bypass_marker(&mut req);
        if is_query_only(&req) || !self.activation_gate(context.id())? {
            return next.generate(context.map(|_| req)).await;
        }
        ensure_tracker(&mut req);
        let session_affinity = session_affinity(&context)?;

        let request = context.map(|_| req);
        if let PrefillPreview::LocalDecode(decision) = self.preview(&request).await? {
            return self.dispatch_local_decode(request, decision).await;
        }
        let (req, context) = request.into_parts();

        let tracker = req.tracker.clone();
        let prefill_phase_barrier = tracker
            .as_ref()
            .expect("tracker is set above")
            .set_phase(RequestPhase::Prefill)
            .await;
        let prefill_request = prefill_request(&req, &context, session_affinity)?;
        let prefill_engine_ctx = prefill_request.context();

        let plan = self.schedule(&prefill_request).await?;
        if matches!(plan, PrefillPlan::Untargeted) {
            return next.generate(context.map(|_| req)).await;
        }
        let progress = match self.dispatch(prefill_request, plan).await {
            Ok(mut dispatched) => {
                if let (Some(state), Some(booking)) =
                    (req.migration_state.as_ref(), dispatched.booking.take())
                {
                    state.store_prefill_release(booking);
                }
                self.await_prefill(dispatched, tracker, prefill_phase_barrier)
                    .await
            }
            Err(error) => Err(error),
        }
        .inspect_err(log_prefill_failure)?;

        // A prefill request can terminate before the backend establishes a KV
        // handoff, for example on EOS at the one-token context step. That
        // response is the whole answer.
        let outcome = match progress.outcome {
            PrefillOutcome::Terminal { output } => {
                let output = strip_terminal_disaggregated_params(*output);
                return Ok(ResponseStream::new(
                    Box::pin(stream::once(async move { output })),
                    engine_ctx,
                ));
            }
            outcome => outcome,
        };

        // NVBugs 5969206: decode routing proceeds even if the client context
        // is gone, so an in-flight KV transfer still has a receiver.
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!(
                "Context {} killed/stopped after prefill, allowing decode routing for KV transfer",
                engine_ctx.id()
            );
        }

        // On the bootstrap path this waits until the prefill task has finished
        // routing, so the prefill worker is attributed correctly.
        if let Some(ref tracker) = req.tracker {
            let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
        }

        let decode_req = self.build_decode_request(req, outcome, progress.topology_constraints);
        let output = prefill_need(&decode_req.annotations);
        let decode = next.generate(context.map(|_| decode_req));
        match progress.completion {
            Some(completion) => {
                completion
                    .forward_decode(decode, engine_ctx, output, prefill_engine_ctx)
                    .await
            }
            None => decode.await,
        }
    }

    /// The first activation check, before any routing work. `Ok(true)` serves
    /// the request through prefill, `Ok(false)` passes it to decode, and an
    /// error fails it closed while a prefill target is still activating.
    fn activation_gate(&self, request_id: &str) -> Result<bool> {
        // Hot path: an active router never takes the target lock.
        if self.lifecycle_state() == PrefillLifecycleState::Active {
            return Ok(true);
        }
        match self.activation_snapshot() {
            ActivationSnapshot::Active(_) => Ok(true),
            ActivationSnapshot::Pending => {
                tracing::debug!(
                    request_id = %request_id,
                    model_name = %self.model_name,
                    namespace = %self.namespace,
                    "Prefill router target is still activating; rejecting request instead of \
                     forwarding it to decode without a prefill hop"
                );
                Err(prefill_activation_pending_error(
                    &self.model_name,
                    &self.namespace,
                ))
            }
            ActivationSnapshot::Untargeted => Ok(false),
        }
    }

    /// The binding to serve through, resolved from a locked snapshot when the
    /// lock-free load finds none because the target was replaced. `None`
    /// means there is no prefill target.
    fn resolve_binding(&self) -> Result<Option<Arc<PrefillBinding>>> {
        if let Some(binding) = self.binding.load_full() {
            return Ok(Some(binding));
        }
        match self.activation_snapshot() {
            ActivationSnapshot::Active(binding) => Ok(Some(binding)),
            ActivationSnapshot::Pending => Err(prefill_activation_pending_error(
                &self.model_name,
                &self.namespace,
            )),
            ActivationSnapshot::Untargeted => Ok(None),
        }
    }

    /// Send a request that conditional disaggregation kept local to its
    /// admitted decode worker, marked so the worker runs prefill itself.
    async fn dispatch_local_decode(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        decision: ConditionalDisaggDecodeDecision,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let worker = decision.plan.worker();
        tracing::info!(
            request_id = %request.id(),
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            net_new_tokens = decision.net_new_tokens,
            overlap_tokens = decision.overlap_tokens,
            "Conditional disagg routing to decode worker"
        );

        if let Some(ref tracker) = request.tracker {
            let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
        }
        request
            .annotations
            .push(BYPASS_REMOTE_PREFILL_ANNOTATION.to_string());

        let decode_host = self
            .decode_routing_host
            .get()
            .expect("conditional plan requires a decode RoutingHost");
        let response_stream = decode_host.dispatch_kv_plan(request, decision.plan).await?;
        let ctx = response_stream.context();
        let annotation =
            Annotated::<LLMEngineOutput>::from_annotation(BYPASS_REMOTE_PREFILL_ANNOTATION, &true)?;
        let merged = stream::once(async move { annotation }).chain(response_stream);
        Ok(ResponseStream::new(Box::pin(merged), ctx))
    }

    /// Wait for prefill as far as decode needs. With early bootstrap info the
    /// rest of prefill runs in the background; otherwise drain the stream to a
    /// handoff or a terminal output.
    async fn await_prefill(
        &self,
        dispatched: PrefillDispatched,
        tracker: Option<Arc<RequestTracker>>,
        prefill_phase_barrier: OwnedSemaphorePermit,
    ) -> Result<PrefillProgress> {
        let PrefillDispatched {
            prepared, stream, ..
        } = dispatched;
        let topology_constraints = prepared.topology_constraints;

        if let Some(bootstrap_info) = prepared.bootstrap_info {
            let completion = self.spawn_prefill_task(stream, tracker, prefill_phase_barrier);
            return Ok(PrefillProgress {
                outcome: PrefillOutcome::Bootstrap {
                    bootstrap_info,
                    worker_id: prepared.worker_id,
                    prefill_dp_rank: prepared.prefill_dp_rank,
                },
                topology_constraints,
                completion: Some(completion),
            });
        }

        drop(prefill_phase_barrier);
        let (outcome, completion) =
            match Self::consume_prefill_stream(stream, tracker, self.task_guard.clone()).await? {
                PrefillCompletion::Handoff {
                    result,
                    worker_link,
                    completion,
                } => {
                    let outcome = match extract_bootstrap_info(&result.disaggregated_params) {
                        Some(bootstrap_info) => PrefillOutcome::Bootstrap {
                            bootstrap_info,
                            worker_id: prepared.worker_id,
                            prefill_dp_rank: prepared.prefill_dp_rank,
                        },
                        None => PrefillOutcome::Completed {
                            result,
                            worker_id: prepared.worker_id,
                            worker_link,
                        },
                    };
                    (outcome, completion)
                }
                PrefillCompletion::Terminal { output } => {
                    (PrefillOutcome::Terminal { output }, None)
                }
            };
        Ok(PrefillProgress {
            outcome,
            topology_constraints,
            completion,
        })
    }
}

/// Conditional-disagg bypass is a router-owned decision. Drop any
/// client-supplied marker so a client cannot skip remote prefill.
fn strip_client_bypass_marker(req: &mut PreprocessedRequest) {
    req.annotations
        .retain(|annotation| annotation != BYPASS_REMOTE_PREFILL_ANNOTATION);
}

/// Query-only requests are advisory worker lookups owned by the decode
/// `RoutingHost` in every activation state.
fn is_query_only(req: &PreprocessedRequest) -> bool {
    req.get_annotation_value("query_instance_id").is_some()
}

/// Create a tracker for routing decisions when the upstream `DeltaGenerator`
/// did not provide one.
fn ensure_tracker(req: &mut PreprocessedRequest) {
    if req.tracker.is_none() {
        req.tracker = Some(Arc::new(RequestTracker::new()));
    }
}

fn session_affinity(context: &Context<()>) -> Result<Option<Arc<SessionAffinityId>>> {
    context
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| anyhow::anyhow!("invalid session affinity context: {message}"))
}

/// The prefill leg: a one-token copy of the request on its own cancellation
/// context, carrying the session affinity.
fn prefill_request(
    req: &PreprocessedRequest,
    context: &Context<()>,
    session_affinity: Option<Arc<SessionAffinityId>>,
) -> Result<SingleIn<PreprocessedRequest>> {
    let mut prefill_req = req.clone();
    prefill_req.stop_conditions.max_tokens = Some(1);
    let mut prefill_context = independent_prefill_context(prefill_req, context)?;
    if let Some(session_affinity) = session_affinity {
        prefill_context.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            session_affinity.as_ref().clone(),
        );
    }
    Ok(prefill_context)
}

fn log_prefill_failure(error: &anyhow::Error) {
    if match_error_chain(
        error.as_ref(),
        &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
        &[],
    ) {
        tracing::warn!(error = %error, "request rejected by prefill worker (at capacity)");
    } else {
        tracing::error!(error = %error, "Remote prefill failed, failing request");
    }
}
