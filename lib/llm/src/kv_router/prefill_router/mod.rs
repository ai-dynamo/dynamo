// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::PrefillLoadEstimator;
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, ManyOut, Operator, RouterMode, ServerStreamingEngine, SingleIn,
        async_trait,
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

use execution::link_child_context;
use inner::InnerPrefillRouter;
pub use types::PrefillError;
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
        let engine_ctx = context.context();

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If prefill router is not activated (no prefill workers discovered),
        // this is aggregated mode — route directly to decode.
        // With --enforce-disagg, fail instead of falling back.
        if self.prefill_router.get().is_none() {
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
                // We successfully used the peeked worker, so we must now advance the router state
                // to ensure the next request gets a different worker.
                if !self.router_mode.is_kv_routing()
                    && let Some(router) = self.prefill_router.get()
                {
                    router.select_next_worker();
                }

                let routing = prefill_req.routing_mut();
                routing.prefill_worker_id = Some(worker_id);
                routing.dp_rank = dp_rank;
                prefill_req.bootstrap_info = Some(bootstrap_info.clone());

                let prefill_context =
                    link_child_context(&engine_ctx, prefill_req, request_id.as_str());

                // Pass the phase barrier to the spawned task. It is released after routing
                // completes so `record_worker_full` finishes before phase changes to Decode.
                self.spawn_prefill_task(prefill_context, Some(worker_id), prefill_phase_barrier);

                Ok(PrefillOutcome::Bootstrap(bootstrap_info))
            }
            PrefillResolveDecision::Unavailable
            | PrefillResolveDecision::NotActivated
            | PrefillResolveDecision::NoBootstrapEndpoint => {
                // Original prefill path: wait for prefill to complete
                tracing::debug!("Using original prefill path");

                // Drop the phase barrier because we wait for prefill completion in this task,
                // so there is no race with set_phase(Decode) below.
                drop(prefill_phase_barrier);

                let prefill_context =
                    link_child_context(&engine_ctx, prefill_req, request_id.as_str());

                // In Direct mode, pass preselected_worker so execute_prefill uses
                // router.direct() instead of router.generate() (which bails in Direct mode).
                let (result, _worker_info) = Self::execute_prefill(
                    self.prefill_router.get().cloned(),
                    prefill_context,
                    preselected_worker,
                    None,
                )
                .await?;

                Ok(PrefillOutcome::Completed(result))
            }
        };

        // Abort if cancelled during prefill
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!("Abort entering decode after context is stopped or killed");
            return Err(anyhow::anyhow!(
                "Context id {} is stopped or killed",
                engine_ctx.id()
            ));
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
                    PrefillOutcome::Bootstrap(info) => {
                        decode_req.bootstrap_info = Some(info);
                    }
                    PrefillOutcome::Completed(result) => {
                        decode_req.prefill_result = Some(result);
                    }
                }

                // Restore original max_tokens for decode
                decode_req.stop_conditions.max_tokens = original_max_tokens;

                // Set router_config_override for decode:
                // - overlap_score_weight = 0 (no KV cache overlap scoring for decode)
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::RouterConfigOverride;

    #[test]
    fn decode_router_override_disables_overlap_and_prefill_tracking() {
        let override_config = build_decode_router_override(Some(RouterConfigOverride {
            router_temperature: Some(0.7),
            ..Default::default()
        }));

        assert_eq!(override_config.overlap_score_weight, Some(0.0));
        assert_eq!(override_config.assume_kv_reuse, Some(false));
        assert_eq!(override_config.track_prefill_tokens, Some(false));
        assert_eq!(override_config.router_temperature, Some(0.7));
    }
}
