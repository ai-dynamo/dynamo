// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PrefillRouter` as a [`StageRouter`], and the helpers the operator's
//! `generate` is composed from.
//!
//! `generate` calls `preview` and `dispatch` directly. In place of
//! `schedule` it calls the steps `schedule` is made of (`resolve_binding`,
//! `require_direct_prefill_worker`, `select_prefill_worker`) so that a worker
//! selection error is logged as a prefill failure and the earlier steps are
//! not.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, ResponseStream, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tokio::sync::OwnedSemaphorePermit;

use super::{
    ActivationSnapshot, BYPASS_REMOTE_PREFILL_ANNOTATION, PrefillBinding, PrefillCompletion,
    PrefillLifecycleState, PrefillOutcome, PrefillRouter, PreparedPrefill, apply_prefill_dispatch,
    conditional_bypass::ConditionalDisaggDecodeDecision, extract_bootstrap_info,
    independent_prefill_context, prefill_activation_pending_error,
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
    pub(super) prepared: PreparedPrefill,
    pub(super) stream: ManyOut<Annotated<LLMEngineOutput>>,
    /// Frees the KV booking early, so a migration retry can release it while
    /// the prefill stream still runs. `None` for built-in modes.
    pub(super) booking: Option<RequestAttemptRelease>,
}

/// How far prefill got before decode can start.
pub(super) struct PrefillProgress {
    pub(super) outcome: PrefillOutcome,
    pub(super) topology_constraints: Option<RoutingConstraints>,
    /// Prefill still running in the background; decode merges its completion.
    pub(super) completion: Option<PrefillTask>,
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

        require_direct_prefill_worker(&binding, request)?;
        self.select_prefill_worker(binding, request).await
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
    /// The prefill availability gate, run before any routing work. It decides
    /// whether the deployment currently serves this request through
    /// disaggregated prefill. Discovery sets the prefill target, so this is
    /// runtime state rather than static configuration.
    ///
    /// `Ok(true)` continues to the prefill hop. `Ok(false)` means there is no
    /// prefill target: the prefill hop is skipped and the decode worker does
    /// the whole request, aggregated. An error fails the request closed while
    /// a prefill target is still activating.
    pub(super) fn activation_gate(&self, request_id: &str) -> Result<bool> {
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
    pub(super) fn resolve_binding(&self) -> Result<Option<Arc<PrefillBinding>>> {
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

    /// Choose the prefill worker for `binding` without sending to it.
    ///
    /// KV routing books the worker now; [`StageRouter::dispatch`] sends to it.
    /// Built-in modes choose at dispatch instead, because their send can move
    /// to another worker and prepare the request for that one.
    pub(super) async fn select_prefill_worker(
        &self,
        binding: Arc<PrefillBinding>,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<PrefillPlan> {
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

    /// Send a request that conditional disaggregation kept local to its
    /// admitted decode worker, marked so the worker runs prefill itself.
    pub(super) async fn dispatch_local_decode(
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
    pub(super) async fn await_prefill(
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
pub(super) fn strip_client_bypass_marker(req: &mut PreprocessedRequest) {
    req.annotations
        .retain(|annotation| annotation != BYPASS_REMOTE_PREFILL_ANNOTATION);
}

/// Query-only requests are advisory worker lookups owned by the decode
/// `RoutingHost` in every activation state.
pub(super) fn is_query_only(req: &PreprocessedRequest) -> bool {
    req.get_annotation_value("query_instance_id").is_some()
}

/// Create a tracker for routing decisions when the upstream `DeltaGenerator`
/// did not provide one.
pub(super) fn ensure_tracker(req: &mut PreprocessedRequest) {
    if req.tracker.is_none() {
        req.tracker = Some(Arc::new(RequestTracker::new()));
    }
}

pub(super) fn session_affinity(context: &Context<()>) -> Result<Option<Arc<SessionAffinityId>>> {
    context
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| anyhow::anyhow!("invalid session affinity context: {message}"))
}

/// The prefill leg: a one-token copy of the request on its own cancellation
/// context, carrying the session affinity.
pub(super) fn prefill_request(
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

/// Direct routing mode serves only a prefill worker the request names, set
/// by an external router such as the EPP. Keyed on the prefill mode, not the
/// decode set's, and checked after the binding loads because that is where
/// the prefill mode lives.
pub(super) fn require_direct_prefill_worker(
    binding: &PrefillBinding,
    request: &PreprocessedRequest,
) -> Result<()> {
    let preselected_worker = request.routing.as_ref().and_then(|r| r.prefill_worker_id);
    if binding.prefill_router_mode.is_direct_routing() && preselected_worker.is_none() {
        anyhow::bail!(
            "Prefill worker ID required in Direct routing mode but none found in request. \
             Expected prefill_worker_id to be set via x-dynamo-prefill-instance-id header by external router (e.g., EPP)."
        );
    }
    Ok(())
}

pub(super) fn log_prefill_failure(error: &anyhow::Error) {
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
