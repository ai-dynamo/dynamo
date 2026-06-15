// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};
use dynamo_runtime::{
    error::{DynamoError, ErrorType, match_error_chain},
    pipeline::SingleIn,
    protocols::maybe_error::MaybeError,
};

use super::{
    InnerPrefillRouter, PrefillError, PrefillQueryOutcome, PrefillResolveDecision, PrefillRouter,
};
use crate::protocols::common::{
    llm_backend::PreprocessedRequest,
    preprocessor::{BootstrapInfo, PrefillResult, TraceLink},
    timing::{RequestTracker, RoutingData},
};

pub(super) struct PrefillCompletion {
    pub result: PrefillResult,
    /// `(worker_id, dp_rank)` for the worker that performed prefill, when the
    /// routing layer can identify it.
    pub worker_info: Option<(u64, Option<u32>)>,
    pub worker_link: Option<TraceLink>,
}

impl PrefillRouter {
    /// Select a prefill worker and resolve its bootstrap connection info.
    /// If preselected_worker is provided (GAIE Stage 2), use it directly.
    /// Otherwise, query for the best worker (KV mode) or select next worker (non-KV modes).
    pub(super) async fn resolve_prefill_worker(
        &self,
        context_id: &str,
        req: &PreprocessedRequest,
        preselected_worker: Option<u64>,
    ) -> PrefillResolveDecision {
        let Some(endpoint_id) = self.endpoint_id.get() else {
            return PrefillResolveDecision::NotActivated;
        };
        if self.prefill_router.get().is_none() {
            return PrefillResolveDecision::NotActivated;
        }

        // Treat a preselected prefill worker as a caller/external pin. Otherwise,
        // sticky affinity wins before this router writes generated bootstrap hints.
        let sticky_worker = if preselected_worker.is_none() {
            self.resolve_sticky_prefill_worker(context_id, req).await
        } else {
            None
        };

        // Worker selection. `permit` books occupancy for LL/P2C/DAW so the
        // later `direct()` dispatch is counted; held by the returned decision
        // for the request's lifetime. For an externally-fixed worker (sticky /
        // preselected) there is no selection race, so a plain track_dispatch
        // books it; the router-selected path books atomically inside
        // query_prefill_worker. `None` for KV/RoundRobin/Random.
        let book = |worker_id: u64| {
            self.prefill_router
                .get()
                .and_then(|r| r.track_dispatch(worker_id))
        };
        let (worker_id, dp_rank, permit) = if let Some(worker) = sticky_worker {
            tracing::debug!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                "Using sticky prefill worker for bootstrap"
            );
            (
                worker.worker_id,
                Some(worker.dp_rank),
                book(worker.worker_id),
            )
        } else if let Some(id) = preselected_worker {
            let dp_rank = req
                .routing
                .as_ref()
                .and_then(|r| r.prefill_dp_rank.or(r.dp_rank));
            tracing::debug!(
                worker_id = id,
                dp_rank = ?dp_rank,
                "Using pre-selected prefill worker for bootstrap"
            );
            (id, dp_rank, book(id))
        } else {
            // Use shared worker selection logic (update_states=false for peek behavior)
            // Extract queue and request metadata from routing hints.
            let lora_name = req.routing.as_ref().and_then(|r| r.lora_name.clone());
            let priority_jump = req
                .routing
                .as_ref()
                .and_then(|r| r.priority_jump)
                .unwrap_or(0.0);
            let strict_priority = req
                .routing
                .as_ref()
                .and_then(|r| r.strict_priority)
                .unwrap_or(0);
            let allowed_worker_ids = req
                .routing
                .as_ref()
                .and_then(|r| r.allowed_worker_ids.clone());
            let routing_constraints = req
                .routing
                .as_ref()
                .and_then(|r| r.routing_constraints.clone())
                .unwrap_or_default();
            let (routing_token_ids, block_mm_infos) = req.block_mm_routing_info();
            match self
                .query_prefill_worker(
                    routing_token_ids,
                    block_mm_infos,
                    false,
                    lora_name,
                    priority_jump,
                    strict_priority,
                    allowed_worker_ids,
                    routing_constraints,
                )
                .await
            {
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank,
                    permit,
                }) => (worker_id, dp_rank, permit),
                Ok(PrefillQueryOutcome::Backpressure {
                    reason,
                    queued_isl_tokens,
                    max_queued_isl_tokens,
                }) => {
                    return PrefillResolveDecision::Backpressure {
                        reason,
                        queued_isl_tokens,
                        max_queued_isl_tokens,
                    };
                }
                Err(_) => return PrefillResolveDecision::Unavailable,
            }
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let Some(endpoint) = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)
        else {
            return PrefillResolveDecision::NoBootstrapEndpoint {
                worker_id,
                dp_rank,
                permit,
            };
        };
        let Some(host) = endpoint.bootstrap_host else {
            return PrefillResolveDecision::NoBootstrapEndpoint {
                worker_id,
                dp_rank,
                permit,
            };
        };
        let Some(port) = endpoint.bootstrap_port else {
            return PrefillResolveDecision::NoBootstrapEndpoint {
                worker_id,
                dp_rank,
                permit,
            };
        };

        let dp_size: Option<u32> = self
            .model_manager
            .get_data_parallel_size(endpoint_id, worker_id);
        let r: u64 = rand::random_range(0..=i64::MAX.cast_unsigned());
        let bootstrap_room = compute_bootstrap_room(dp_rank, dp_size, r);

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
            permit,
        }
    }

    async fn resolve_sticky_prefill_worker(
        &self,
        context_id: &str,
        req: &PreprocessedRequest,
    ) -> Option<dynamo_kv_router::protocols::WorkerWithDpRank> {
        let router = self.prefill_router.get()?;
        let worker = router.sticky_worker_for_prefill(req)?;
        if router.unbind_ineligible_sticky_prefill_worker(context_id, req, worker) {
            return None;
        }

        match router
            .validate_sticky_prefill_worker(context_id, req, worker)
            .await
        {
            Ok(worker) => {
                router.refresh_sticky_prefill_worker(req);
                Some(worker)
            }
            Err(error) => {
                let unbound =
                    router.unbind_ineligible_sticky_prefill_worker(context_id, req, worker);
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    error = %error,
                    unbound_due_to_ineligibility = unbound,
                    "Sticky prefill worker routing failed; falling back to normal prefill routing"
                );
                None
            }
        }
    }

    /// Execute prefill with the given router and extract structured result.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// If `phase_transition_permit` is provided, it is dropped immediately after routing completes,
    /// allowing subsequent `set_phase` calls to proceed. This preserves the current synchronization:
    /// the prefill route must finish worker recording before the phase can change to Decode.
    pub(super) async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_transition_permit: Option<OwnedSemaphorePermit>,
    ) -> Result<PrefillCompletion, PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        // Clone tracker before request is consumed by generate_to_worker.
        // Used to record prefill_complete_time for KV transfer latency metric.
        let tracker = request.tracker.clone();
        // Only SimpleRouter honors target_worker directly. KvRouter reads the
        // pin from request routing and records the actual worker in RequestTracker.
        let simple_direct_worker_info = match &router {
            InnerPrefillRouter::SimpleRouter(_) => target_worker.map(|worker_id| (worker_id, None)),
            InnerPrefillRouter::KvRouter(_) => None,
        };
        let mut prefill_response = router
            .generate_to_worker(request, target_worker)
            .await
            .map_err(|e| {
                // A shed prefill worker returns ResourceExhausted. Carry it as the
                // source so the chain stays downcastable to 503; boxing the raw
                // anyhow error instead would hide that identity.
                if match_error_chain(e.as_ref(), &[ErrorType::ResourceExhausted], &[]) {
                    tracing::warn!(
                        worker_error = %e,
                        "Request rejected by prefill worker (at capacity) — returning HTTP 503"
                    );
                    return PrefillError::PrefillError(
                        "prefill worker overloaded".to_string(),
                        Some(Box::new(
                            DynamoError::builder()
                                .error_type(ErrorType::ResourceExhausted)
                                .message(e.to_string())
                                .build(),
                        )),
                    );
                }
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

        let worker_link = output.worker_trace_link.clone();

        let worker_info = prefill_worker_info(
            tracker.as_deref(),
            output.routing_data.as_ref(),
            simple_direct_worker_info,
        );
        Ok(PrefillCompletion {
            result: PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            worker_info,
            worker_link,
        })
    }

    /// Spawn prefill as a background task.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// The `phase_transition_permit` is passed to the spawned task and released after routing
    /// completes, allowing the main task's `set_phase(Decode)` to proceed.
    ///
    /// `load_permit` is held for the spawned task's lifetime; its drop emits the
    /// occupancy decrement for the bootstrap dispatch. `None` for modes without
    /// occupancy tracking.
    pub(super) fn spawn_prefill_task(
        &self,
        prefill_request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_transition_permit: OwnedSemaphorePermit,
        load_permit: Option<dynamo_runtime::pipeline::OccupancyPermit>,
    ) {
        let router = self.prefill_router.get().cloned();
        // Capture current span to propagate trace context to the spawned task
        let span = tracing::Span::current();

        tokio::spawn(
            async move {
                let _load_permit = load_permit; // drop emits decrement
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
    ///
    /// Returns `PrefillQueryOutcome::Routed` for the selected worker, or
    /// `PrefillQueryOutcome::Backpressure` when the prefill scheduler queue is
    /// saturated. This is the shared worker selection logic used by both
    /// `resolve_prefill_worker` and `query_route`.
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(r) => {
                let outcome = r
                    .chooser
                    .find_best_match_details(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        false,
                        lora_name,
                        priority_jump,
                        strict_priority,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        // KV tracks load via worker-pushed kv_metrics, not the
                        // occupancy counter, so no permit here.
                        Ok(PrefillQueryOutcome::Routed {
                            worker_id: worker.worker_id,
                            dp_rank: Some(worker.dp_rank),
                            permit: None,
                        })
                    }
                    crate::kv_router::FindBestMatchOutcome::Backpressure {
                        reason,
                        queued_isl_tokens,
                        max_queued_isl_tokens,
                    } => Ok(PrefillQueryOutcome::Backpressure {
                        reason,
                        queued_isl_tokens,
                        max_queued_isl_tokens,
                    }),
                }
            }
            InnerPrefillRouter::SimpleRouter(r) => {
                // Peek path (update_states=false) is the bootstrap resolve: for
                // LL/P2C/DAW, atomically select+book via select_and_reserve so a
                // concurrent resolve can't pick the same min; RoundRobin/Random
                // have no occupancy state, so fall back to a plain peek (no
                // permit). The select path (update_states=true) advances the
                // RoundRobin/Random cursor and never books occupancy.
                let (worker_id, permit) = if update_states {
                    let id = r
                        .select_next_worker()
                        .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                    (id, None)
                } else {
                    match r.select_and_reserve().await {
                        Some((id, permit)) => (id, Some(permit)),
                        None => {
                            let id = r.peek_next_worker().ok_or_else(|| {
                                anyhow::anyhow!("No workers available for prefill")
                            })?;
                            (id, None)
                        }
                    }
                };
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank: None,
                    permit,
                })
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

fn prefill_worker_info(
    tracker: Option<&RequestTracker>,
    routing_data: Option<&RoutingData>,
    simple_direct_worker_info: Option<(u64, Option<u32>)>,
) -> Option<(u64, Option<u32>)> {
    // Prefer router-owned attribution over forwarded payloads: KvRouter records
    // in the tracker, SimpleRouter direct routing uses the explicit target, and a
    // standalone router forwards it on `routing_data.worker_id`.
    tracker_prefill_worker_info(tracker)
        .or(simple_direct_worker_info)
        .or_else(|| routing_data_prefill_worker_info(routing_data))
}

fn tracker_prefill_worker_info(tracker: Option<&RequestTracker>) -> Option<(u64, Option<u32>)> {
    tracker
        .and_then(|tracker| tracker.get_worker_info())
        .and_then(|info| {
            info.prefill_worker_id
                .map(|worker_id| (worker_id, info.prefill_dp_rank))
        })
}

fn routing_data_prefill_worker_info(
    routing_data: Option<&RoutingData>,
) -> Option<(u64, Option<u32>)> {
    let info = routing_data?.worker_id.as_ref()?;
    let worker_id = info.prefill_worker_id?;
    Some((worker_id, info.prefill_dp_rank))
}

/// Derive a `bootstrap_room` from a pre-sampled `r` such that
/// `room % dp_size == dp_rank` and `room <= i64::MAX`. The 63-bit cap is the
/// existing room contract on the SGLang side. Falls back to `r` when
/// `dp_rank` or `dp_size` is unavailable. `r` must be in `[0, i64::MAX]`.
fn compute_bootstrap_room(dp_rank: Option<u32>, dp_size: Option<u32>, r: u64) -> u64 {
    let max_room = i64::MAX.cast_unsigned();
    debug_assert!(r <= max_room);
    match (dp_rank, dp_size) {
        (Some(rank), Some(size)) if size > 0 => {
            let size = size as u64;
            let rank = rank as u64;
            // Bound the quotient so `q * size + rank <= i64::MAX`.
            let max_q = (max_room - rank) / size;
            let q = r % (max_q + 1);
            q * size + rank
        }
        _ => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::timing::WORKER_TYPE_PREFILL;
    use crate::protocols::openai::nvext::WorkerIdInfo;

    const MAX_ROOM: u64 = i64::MAX as u64;

    /// `routing_data` carrying just the prefill attribution, as forwarded by a
    /// standalone router (see `inject_worker_id_from_tracker`).
    fn routing_data_with_prefill(
        prefill_worker_id: u64,
        prefill_dp_rank: Option<u32>,
    ) -> RoutingData {
        RoutingData {
            worker_id: Some(WorkerIdInfo {
                prefill_worker_id: Some(prefill_worker_id),
                prefill_dp_rank,
                decode_worker_id: None,
                decode_dp_rank: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn prefill_worker_info_prefers_tracker_over_routing_data_and_direct_target() {
        let tracker = RequestTracker::new();
        tracker.record_worker(10, Some(2), WORKER_TYPE_PREFILL);
        let routing_data = routing_data_with_prefill(20, Some(3));

        assert_eq!(
            prefill_worker_info(Some(&tracker), Some(&routing_data), Some((30, None))),
            Some((10, Some(2)))
        );
    }

    #[test]
    fn prefill_worker_info_prefers_direct_target_over_routing_data() {
        let routing_data = routing_data_with_prefill(20, Some(3));

        assert_eq!(
            prefill_worker_info(None, Some(&routing_data), Some((30, None))),
            Some((30, None))
        );
    }

    #[test]
    fn prefill_worker_info_falls_back_to_routing_data_worker_id() {
        let routing_data = routing_data_with_prefill(20, Some(3));

        assert_eq!(
            prefill_worker_info(None, Some(&routing_data), None),
            Some((20, Some(3)))
        );
    }

    #[test]
    fn prefill_worker_info_falls_back_to_direct_target() {
        assert_eq!(
            prefill_worker_info(None, None, Some((30, None))),
            Some((30, None))
        );
    }

    #[test]
    fn prefill_worker_info_returns_none_without_authoritative_source() {
        assert_eq!(prefill_worker_info(None, None, None), None);
    }

    #[test]
    fn bootstrap_room_falls_back_when_dp_unavailable() {
        // Missing rank, missing size, or both -> return r unchanged.
        assert_eq!(compute_bootstrap_room(None, None, 12345), 12345);
        assert_eq!(compute_bootstrap_room(Some(3), None, 12345), 12345);
        assert_eq!(compute_bootstrap_room(None, Some(8), 12345), 12345);
        // size=0 is a guard against divide-by-zero; treated as unavailable.
        assert_eq!(compute_bootstrap_room(Some(0), Some(0), 12345), 12345);
    }

    #[test]
    fn bootstrap_room_respects_63bit_cap_at_max_r() {
        // Sweep ranks for the sizes that overflowed in the buggy version:
        //   - size = 48 (i64::MAX % 48 = 31, so ranks 32..47 overflowed)
        //   - size = 49 (49 divides i64::MAX, so ranks 1..48 overflowed)
        //   - size = 7  (7 divides i64::MAX, so ranks 1..6 overflowed)
        for size in [3u32, 5, 6, 7, 9, 16, 32, 48, 49, 64, 128] {
            for rank in 0..size {
                let room = compute_bootstrap_room(Some(rank), Some(size), MAX_ROOM);
                assert!(
                    room <= MAX_ROOM,
                    "size={size} rank={rank} r=MAX produced {room} > i64::MAX",
                );
                assert_eq!(
                    room % size as u64,
                    rank as u64,
                    "size={size} rank={rank} broke modulo contract",
                );
            }
        }
    }

    #[test]
    fn bootstrap_room_modulo_contract_across_r() {
        // Across many `r` values, the modulo contract must hold and the
        // result must stay within the 63-bit cap.
        let r_samples = [
            0u64,
            1,
            47,
            48,
            49,
            1_000_000,
            1u64 << 32,
            (1u64 << 62) - 1,
            1u64 << 62,
            MAX_ROOM - 1,
            MAX_ROOM,
        ];
        for size in [3u32, 8, 48, 49] {
            for rank in [0u32, 1, size / 2, size - 1] {
                for &r in &r_samples {
                    let room = compute_bootstrap_room(Some(rank), Some(size), r);
                    assert!(
                        room <= MAX_ROOM,
                        "size={size} rank={rank} r={r} produced {room} > i64::MAX",
                    );
                    assert_eq!(
                        room % size as u64,
                        rank as u64,
                        "size={size} rank={rank} r={r} broke modulo contract",
                    );
                }
            }
        }
    }

    #[test]
    fn bootstrap_room_balances_dp_rank_assignments() {
        // For a non-power-of-two dp_size, sampling many rooms with the real
        // RNG must (a) put every room in its requested rank's modulo bucket
        // and (b) leave no rank starved -- each rank should claim roughly its
        // fair share when assignments cycle round-robin.
        let dp_size: u32 = 48;
        let trials_per_rank: usize = 2_000;

        let mut per_rank_counts = vec![0usize; dp_size as usize];
        let mut max_room_seen = 0u64;
        let mut min_room_seen = u64::MAX;

        for rank in 0..dp_size {
            for _ in 0..trials_per_rank {
                let r = rand::random_range(0..=MAX_ROOM);
                let room = compute_bootstrap_room(Some(rank), Some(dp_size), r);

                assert!(room <= MAX_ROOM, "room {room} exceeds i64::MAX");
                assert_eq!(
                    room % dp_size as u64,
                    rank as u64,
                    "room {room} did not land in rank {rank}'s bucket",
                );

                per_rank_counts[rank as usize] += 1;
                max_room_seen = max_room_seen.max(room);
                min_room_seen = min_room_seen.min(room);
            }
        }

        // Every rank received its requested share (nothing was silently dropped).
        for (rank, &count) in per_rank_counts.iter().enumerate() {
            assert_eq!(count, trials_per_rank, "rank {rank} count mismatch");
        }

        // Sanity check that the quotient sampler is not collapsing onto a
        // tiny region: with 96k samples in [0, i64::MAX], the spread should
        // cover most of the 63-bit range.
        let span = max_room_seen - min_room_seen;
        assert!(
            span > MAX_ROOM / 2,
            "rooms clustered in span={span}, expected wide spread across [0, i64::MAX]",
        );
    }

    #[test]
    fn bootstrap_room_is_deterministic_in_r() {
        // Same (rank, size, r) -> same room. Guards against accidental
        // re-introduction of an internal random call inside the helper.
        let room_a = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
        let room_b = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
        assert_eq!(room_a, room_b);
        assert_eq!(room_a % 48, 7);
    }
}

// NVBugs 5969206: link_child_context removed — linking prefill as a child of
// engine_context caused kill propagation that tears down the RPC transport,
// interrupting NIXL KV cache transfers and leaking blocks permanently.
// Prefill context is now created without linking (Context::with_id only).
// Abort on the decode side is deferred via kv_transfer_complete_event in
// handler_base.py until the first generation result confirms KV receipt.
