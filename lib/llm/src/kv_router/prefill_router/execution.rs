// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_kv_router::protocols::{BlockExtraInfo, WorkerId};
use dynamo_runtime::{pipeline::SingleIn, protocols::maybe_error::MaybeError};

use super::{InnerPrefillRouter, PrefillError, PrefillResolveDecision, PrefillRouter};
use crate::protocols::common::{
    llm_backend::PreprocessedRequest,
    preprocessor::{BootstrapInfo, PrefillResult},
};

impl PrefillRouter {
    /// Select a prefill worker and resolve its bootstrap connection info.
    /// If preselected_worker is provided (GAIE Stage 2), use it directly.
    /// Otherwise, query for the best worker (KV mode) or select next worker (non-KV modes).
    pub(super) async fn resolve_prefill_worker(
        &self,
        req: &PreprocessedRequest,
        preselected_worker: Option<u64>,
    ) -> PrefillResolveDecision {
        let Some(endpoint_id) = self.endpoint_id.get() else {
            return PrefillResolveDecision::NotActivated;
        };
        if self.prefill_router.get().is_none() {
            return PrefillResolveDecision::NotActivated;
        }

        // Worker selection
        let (worker_id, dp_rank) = if let Some(id) = preselected_worker {
            let dp_rank = req
                .routing
                .as_ref()
                .and_then(|r| r.prefill_dp_rank.or(r.dp_rank));
            tracing::debug!(
                worker_id = id,
                dp_rank = ?dp_rank,
                "Using pre-selected prefill worker for bootstrap"
            );
            (id, dp_rank)
        } else {
            // Use shared worker selection logic (update_states=false for peek behavior)
            // Extract LORA name and priority jump from routing hints
            let lora_name = req.routing.as_ref().and_then(|r| r.lora_name.clone());
            let cache_salt = req.routing.as_ref().and_then(|r| r.cache_salt.clone());
            let priority_jump = req
                .routing
                .as_ref()
                .and_then(|r| r.priority_jump)
                .unwrap_or(0.0);
            let allowed_worker_ids = req
                .routing
                .as_ref()
                .and_then(|r| r.allowed_worker_ids.clone());
            let (routing_token_ids, block_mm_infos) = req.block_mm_routing_info();
            match self
                .query_prefill_worker(
                    routing_token_ids,
                    block_mm_infos,
                    false,
                    lora_name,
                    cache_salt,
                    priority_jump,
                    allowed_worker_ids,
                )
                .await
            {
                Ok((worker_id, dp_rank)) => (worker_id, dp_rank),
                Err(_) => return PrefillResolveDecision::Unavailable,
            }
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let Some(endpoint) = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)
        else {
            return PrefillResolveDecision::NoBootstrapEndpoint;
        };
        let Some(host) = endpoint.bootstrap_host else {
            return PrefillResolveDecision::NoBootstrapEndpoint;
        };
        let Some(port) = endpoint.bootstrap_port else {
            return PrefillResolveDecision::NoBootstrapEndpoint;
        };

        let bootstrap_room: u64 = rand::random_range(0..=i64::MAX.cast_unsigned());

        tracing::debug!(
            worker_id = worker_id,
            dp_rank = ?dp_rank,
            bootstrap_host = %host,
            bootstrap_port = port,
            bootstrap_room = bootstrap_room,
            router_mode = ?self.router_mode,
            "Built bootstrap_info upfront before prefill"
        );

        PrefillResolveDecision::Resolved {
            worker_id,
            dp_rank,
            bootstrap_info: BootstrapInfo {
                bootstrap_host: host,
                bootstrap_port: port,
                bootstrap_room,
            },
        }
    }

    /// Execute prefill with the given router and extract structured result.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// If `phase_transition_permit` is provided, it is dropped immediately after routing completes,
    /// allowing subsequent `set_phase` calls to proceed. This preserves the current synchronization:
    /// the prefill route must finish worker recording before the phase can change to Decode.
    ///
    /// Returns (PrefillResult, Option<(worker_id, dp_rank)>).
    pub(super) async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_transition_permit: Option<OwnedSemaphorePermit>,
    ) -> Result<(PrefillResult, Option<(u64, Option<u32>)>), PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        // Clone tracker before request is consumed by generate_to_worker.
        // Used to record prefill_complete_time for KV transfer latency metric.
        let tracker = request.tracker.clone();
        let mut prefill_response = router
            .generate_to_worker(request, target_worker)
            .await
            .map_err(|e| {
                PrefillError::PrefillError(
                    "failed to route to prefill worker".to_string(),
                    Some(e.into()),
                )
            })?;

        // Release the phase barrier now that routing completed and worker recording already ran.
        // Decode may proceed without waiting for prefill output streaming to finish.
        drop(phase_transition_permit);

        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        // Record when prefill result arrived at the router (for KV transfer latency metric).
        // This is after drop(phase_transition_permit) and after first_output is received.
        if let Some(ref tracker) = tracker {
            tracker.record_prefill_complete();
        }

        if let Some(err) = first_output.err() {
            return Err(PrefillError::PrefillError(
                "Prefill router returned error in output".to_string(),
                Some(Box::new(err)),
            ));
        }

        let mut prompt_tokens_details = first_output
            .data
            .as_ref()
            .and_then(|o| o.completion_usage.as_ref())
            .and_then(|u| u.prompt_tokens_details.clone());

        while let Some(next) = prefill_response.next().await {
            if let Some(o) = next.data.as_ref()
                && prompt_tokens_details.is_none()
            {
                prompt_tokens_details = o
                    .completion_usage
                    .as_ref()
                    .and_then(|u| u.prompt_tokens_details.clone());
            }
        }

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };

        let Some(disaggregated_params) = output.disaggregated_params.clone() else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };

        // Extract prefill worker ID and dp_rank from disaggregated_params
        let prefill_worker_info =
            disaggregated_params
                .get("worker_id")
                .and_then(|worker_id_json| {
                    let worker_id = worker_id_json
                        .get("prefill_worker_id")
                        .and_then(|v| v.as_u64())?;
                    let dp_rank = worker_id_json
                        .get("prefill_dp_rank")
                        .and_then(|v| v.as_u64())
                        .map(|r| r as u32);
                    Some((worker_id, dp_rank))
                });
        Ok((
            PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            prefill_worker_info,
        ))
    }

    /// Spawn prefill as a background task.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// The `phase_transition_permit` is passed to the spawned task and released after routing
    /// completes, allowing the main task's `set_phase(Decode)` to proceed.
    pub(super) fn spawn_prefill_task(
        &self,
        prefill_request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_transition_permit: OwnedSemaphorePermit,
    ) {
        let router = self.prefill_router.get().cloned();
        // Capture current span to propagate trace context to the spawned task
        let span = tracing::Span::current();

        tokio::spawn(
            async move {
                match Self::execute_prefill(
                    router,
                    prefill_request,
                    target_worker,
                    Some(phase_transition_permit),
                )
                .await
                {
                    Ok(_) => {
                        tracing::debug!("Prefill background task completed");
                    }
                    Err(e) => {
                        tracing::warn!("Prefill background task error: {e:?}");
                    }
                }
            }
            .instrument(span),
        );
    }

    /// Query the best prefill worker without executing a request.
    /// Returns (worker_id, dp_rank).
    ///
    /// This is the shared worker selection logic used by both `resolve_prefill_worker`
    /// and `query_route`.
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        update_states: bool,
        lora_name: Option<String>,
        cache_salt: Option<String>,
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<(u64, Option<u32>)> {
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(r) => {
                let (worker, _overlap) = r
                    .chooser
                    .find_best_match(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        lora_name,
                        cache_salt,
                        priority_jump,
                        None,
                        None,
                        allowed_worker_ids,
                    )
                    .await?;
                Ok((worker.worker_id, Some(worker.dp_rank)))
            }
            InnerPrefillRouter::SimpleRouter(r) => {
                let worker_id = if update_states {
                    r.select_next_worker()
                } else {
                    r.peek_next_worker()
                }
                .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok((worker_id, None))
            }
        }
    }

    /// Register externally-provided workers in the prefill router's slot tracker.
    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(InnerPrefillRouter::KvRouter(r)) = self.prefill_router.get() {
            r.chooser.register_workers(worker_ids);
        }
    }

    /// Check if disaggregated mode is currently active (prefill router activated).
    /// Uses the same `activated` flag as `can_serve_requests()` for consistency.
    pub fn is_activated(&self) -> bool {
        self.activated.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether disaggregated mode is strictly enforced (fail if no prefill workers).
    pub fn enforce_disagg(&self) -> bool {
        self.enforce_disagg
    }
}

// NVBugs 5969206: link_child_context removed — linking prefill as a child of
// engine_context caused kill propagation that tears down the RPC transport,
// interrupting NIXL KV cache transfers and leaking blocks permanently.
// Prefill context is now created without linking (Context::with_id only).
// Abort on the decode side is deferred via kv_transfer_complete_event in
// handler_base.py until the first generation result confirms KV receipt.
