// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::{
    engine::AsyncEngineContext,
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, ResponseStream, ServerStreamingEngine,
        SingleIn, propagate_first_response_guard,
    },
    protocols::annotated::Annotated,
};
use dynamo_scheduler::host::RoutingConstraints;
use futures::stream;
use tokio::sync::OwnedSemaphorePermit;

use super::{
    PrefillBinding, PrefillCompletion, PrefillDispatchErrorClass, PrefillOutcome, PrefillRouter,
    PreparedPrefill, build_decode_router_override, cancellation, classify_prefill_dispatch_error,
    extract_bootstrap_info, into_decode_request, merge_decode_topology_constraints,
    prefill_activation_pending_error, settle_prefill_terminal, strip_terminal_disaggregated_params,
};
use crate::{
    kv_router::{
        encoder_router::EncoderResultReleaseSupported,
        routing_host::{PrefillDispatch, ReplanLifecycle, cancelled_error},
    },
    local_model::runtime_config::DISAGG_PREFILL_CANCEL_ANYTIME_V1,
    protocols::{
        common::{
            extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
            llm_backend::{LLMEngineOutput, PreprocessedRequest},
            preprocessor::ENCODER_RESULT_RELEASE_SUPPORTED,
            timing::{RequestPhase, RequestTracker},
        },
        response_merge::{PREFILL_OUTPUT_ANNOTATION, PREFILL_OUTPUT_FIRST_ANNOTATION, PrefillNeed},
    },
};

/// Only outstanding exports need disposal or independent consumer execution.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EncoderHandoff {
    None,
    Releasable,
    MustConsume,
}

impl EncoderHandoff {
    pub(super) fn for_request(request: &SingleIn<PreprocessedRequest>) -> Self {
        if request.encoder_result.is_none() || request.staged_kv_cleanup {
            Self::None
        } else if request
            .get_optional::<EncoderResultReleaseSupported>(ENCODER_RESULT_RELEASE_SUPPORTED)
            .is_ok_and(|capability| capability.is_some_and(|capability| capability.0))
        {
            Self::Releasable
        } else {
            Self::MustConsume
        }
    }

    pub(super) fn check_cancellation(self, context: &dyn AsyncEngineContext) -> Result<()> {
        if self == Self::Releasable && context.is_stopped() {
            return Err(cancelled_error(context.id()));
        }
        Ok(())
    }
}

struct RemotePrefill {
    outcome: PrefillOutcome,
    topology_constraints: Option<RoutingConstraints>,
    completion: Option<super::PrefillTask>,
}

impl PrefillRouter {
    pub(super) async fn prefill_and_decode(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        handoff: EncoderHandoff,
        session_affinity: Option<Arc<SessionAffinityId>>,
        passthrough_phase: RequestPhase,
        replan_lifecycle: &mut Option<ReplanLifecycle>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (mut req, context) = request.into_parts();
        let engine_ctx = context.context();
        let original_max_tokens = req.stop_conditions.max_tokens;
        let tracker = req
            .tracker
            .get_or_insert_with(|| Arc::new(RequestTracker::new()))
            .clone();
        let prefill_phase_barrier = tracker.set_phase(RequestPhase::Prefill).await;
        handoff.check_cancellation(engine_ctx.as_ref())?;

        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);
        let mut prefill_context = independent_prefill_context(prefill_req, &context)?;
        if let Some(session_affinity) = session_affinity {
            prefill_context.insert(
                SESSION_AFFINITY_CONTEXT_KEY,
                session_affinity.as_ref().clone(),
            );
        }
        let prefill_engine_ctx = prefill_context.context();

        // A tracker barrier may have outlived the target or its serving generation.
        let Some(binding) = self.binding_for_request(&req)? else {
            drop(prefill_phase_barrier);
            {
                let _permit = tracker.set_phase(passthrough_phase).await;
            }
            return next.generate(context.map(|_| req)).await;
        };
        if binding.prefill_router_mode.is_direct_routing()
            && req
                .routing
                .as_ref()
                .and_then(|r| r.prefill_worker_id)
                .is_none()
        {
            anyhow::bail!(
                "Prefill worker ID required in Direct routing mode but none found in request. \
                 Expected prefill_worker_id to be set via x-dynamo-prefill-instance-id header by external router (e.g., EPP)."
            );
        }
        let RemotePrefill {
            outcome,
            topology_constraints,
            completion,
        } = async {
            let dispatched = self
                .dispatch_prefill(prefill_context, &binding, handoff, engine_ctx.clone())
                .await?;
            self.finish_prefill_dispatch(&req, dispatched, prefill_phase_barrier)
                .await
        }
        .await
        .inspect_err(log_prefill_failure)?;

        let outcome = match outcome {
            PrefillOutcome::Terminal { output } => {
                settle_prefill_terminal(&req, replan_lifecycle, &output);
                let output = strip_terminal_disaggregated_params(*output);
                return Ok(ResponseStream::new(
                    Box::pin(stream::once(async move { output })),
                    engine_ctx,
                ));
            }
            outcome => outcome,
        };

        // A stopped caller still needs Decode to receive and release staged KV.
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!(
                request_id = %engine_ctx.id(),
                "Context killed/stopped after prefill, allowing decode routing for KV transfer"
            );
        }
        tracing::debug!("Prefill completed, proceeding to decode");

        // Background Prefill releases this barrier after its routing attribution.
        {
            let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
        }
        let mut decode_req = into_decode_request(req, outcome);
        if let Some(topology_constraints) = topology_constraints {
            merge_decode_topology_constraints(&mut decode_req, topology_constraints);
        }
        decode_req.stop_conditions.max_tokens = original_max_tokens;
        let existing_override = decode_req.router_config_override.take();
        decode_req.router_config_override = Some(build_decode_router_override(
            existing_override,
            self.conditional_disagg_policy.is_enabled(),
        ));
        let annotated = |name| decode_req.annotations.iter().any(|a| a == name);
        let output = match (
            annotated(PREFILL_OUTPUT_ANNOTATION),
            annotated(PREFILL_OUTPUT_FIRST_ANNOTATION),
        ) {
            (false, _) => PrefillNeed::None,
            (true, false) => PrefillNeed::BeforeLast,
            (true, true) => PrefillNeed::BeforeFirst,
        };
        let decode = next.generate(context.map(|_| decode_req));
        match completion {
            Some(completion) => {
                completion
                    .forward_decode(decode, engine_ctx, output, prefill_engine_ctx)
                    .await
            }
            None => decode.await,
        }
    }

    async fn dispatch_prefill(
        &self,
        prefill_context: SingleIn<PreprocessedRequest>,
        binding: &Arc<PrefillBinding>,
        handoff: EncoderHandoff,
        client: Arc<dyn AsyncEngineContext>,
    ) -> Result<PrefillDispatch<PreparedPrefill>> {
        let prefill_ctx = prefill_context.context();
        binding
            .router
            .select_and_dispatch_prefill(prefill_context, |request, target| {
                if handoff != EncoderHandoff::None {
                    handoff.check_cancellation(client.as_ref())?;
                    if !matches!(self.activation_snapshot(), super::PrefillActivation::Active(current)
                        if Arc::ptr_eq(&current, binding))
                    {
                        return Err(prefill_activation_pending_error(
                            &self.model_name,
                            &self.namespace,
                        ));
                    }
                }
                let prepared =
                    self.prepare_prefill_dispatch(request, target, &binding.endpoint_id)?;
                let prefill_supports_cancellation =
                    self.model_manager.worker_supports_runtime_capability(
                        &binding.endpoint_id,
                        prepared.worker_id,
                        DISAGG_PREFILL_CANCEL_ANYTIME_V1,
                    );
                if handoff != EncoderHandoff::MustConsume
                    && let Some(guard) = cancellation::arm_for(
                        prefill_supports_cancellation,
                        self.decode_supports_prefill_cancellation,
                        client.clone(),
                        Arc::downgrade(&prefill_ctx),
                    )
                {
                    prefill_ctx.retain(guard);
                }
                Ok(prepared)
            })
            .await
    }

    async fn finish_prefill_dispatch(
        &self,
        request: &PreprocessedRequest,
        dispatch: PrefillDispatch<PreparedPrefill>,
        prefill_phase_barrier: OwnedSemaphorePermit,
    ) -> Result<RemotePrefill> {
        let PrefillDispatch {
            metadata: prepared,
            stream,
            booking,
        } = dispatch;
        if let (Some(state), Some(booking)) = (request.migration_state.as_ref(), booking) {
            state.store_prefill_release(booking);
        }
        let tracker = request.tracker.clone();
        let (outcome, completion) = if let Some(bootstrap_info) = prepared.bootstrap_info {
            let completion = self.spawn_prefill_task(stream, tracker, prefill_phase_barrier);
            (
                PrefillOutcome::Bootstrap {
                    bootstrap_info,
                    worker_id: prepared.worker_id,
                    prefill_dp_rank: prepared.prefill_dp_rank,
                },
                Some(completion),
            )
        } else {
            drop(prefill_phase_barrier);
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
            }
        };
        Ok(RemotePrefill {
            outcome,
            topology_constraints: prepared.topology_constraints,
            completion,
        })
    }
}

fn log_prefill_failure(error: &anyhow::Error) {
    match classify_prefill_dispatch_error(error.as_ref()) {
        PrefillDispatchErrorClass::Cancelled => {
            tracing::debug!(%error, "Remote prefill cancelled");
        }
        PrefillDispatchErrorClass::AtCapacity => {
            tracing::warn!(%error, "request rejected by prefill worker (at capacity)");
        }
        PrefillDispatchErrorClass::Failed => {
            tracing::error!(%error, "Remote prefill failed, failing request");
        }
    }
}

/// Preserve request metadata while letting Prefill outlive caller cancellation.
pub(super) fn independent_prefill_context(
    request: PreprocessedRequest,
    source: &Context<()>,
) -> Result<Context<PreprocessedRequest>> {
    let mut prefill =
        Context::with_id_and_metadata(request, source.id().to_string(), source.metadata().clone());
    propagate_first_response_guard(source, &mut prefill)?;
    Ok(prefill)
}
