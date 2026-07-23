// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process worker selection for standalone mode.
//!
//! Wraps [`SelectionService`] and defines its worker registration and selection
//! types. Optionally synchronizes active load across EPP replicas through
//! [`crate::peer_discovery`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, anyhow};

use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest as CoreSelectAndReserveRequest, SelectionError,
    SelectionService, SelectionServiceBuilder, WorkerLifecycle, WorkerRequest as CoreWorkerRequest,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;

const DEFAULT_ROUTING_GROUP: &str = "default";

/// A worker the EPP registers into the selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub endpoint: String,
    pub block_size: u32,
    pub kv_events_endpoints: HashMap<u32, String>,
    pub replay_endpoint: Option<String>,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
    pub stable_routing_id: Option<String>,
}

/// A worker-selection request.
#[derive(Debug, Clone)]
pub struct SelectRequest {
    pub model_name: String,
    /// EPP-minted booking key (a fresh UUID per pick). Keyed by an id the EPP
    /// already knows so the booking is releasable even if this response is lost.
    pub reservation_id: String,
    pub token_ids: Vec<u32>,
    pub allowed_worker_ids: Option<HashSet<u64>>,
    pub priority_jump: Option<f64>,
    pub strict_priority: Option<u32>,
}

/// Observability overlap summary (matched token counts).
#[derive(Debug, Clone)]
pub struct OverlapSummary {
    pub longest_matched: u32,
    pub gpu: u32,
    pub cpu: u32,
    pub disk: u32,
}

/// The selector's choice for a prompt.
#[derive(Debug, Clone)]
pub struct SelectResponse {
    /// Booking key the selector recorded (echoes the request's `reservation_id`).
    pub reservation_id: String,
    pub worker_id: u64,
    pub endpoint: String,
    pub block_size: u32,
    pub overlap: OverlapSummary,
    pub effective_prefill_tokens: usize,
}

/// In-process runtime-free selector wrapping a [`SelectionService`].
pub struct Selector {
    service: Arc<SelectionService>,
    /// Cancels the peer-discovery watch on drop. The `SelectionService`'s own
    /// `Drop` tears down its core + replica-sync tasks.
    cancel: CancellationToken,
    /// Last catalog we pushed into the service, keyed by `worker_id`. Lets
    /// [`Selector::reconcile`] skip no-op upserts that would re-register
    /// KV-event listeners.
    current: Mutex<HashMap<u64, WorkerRegistration>>,
    /// Peer-discovery readiness in replicated mode: `None` when replication is
    /// disabled (single replica, always ready), or `Some(flag)` that latches
    /// `true` once the initial peer-set sync completes. ANDed into EPP health.
    peer_ready: Option<Arc<AtomicBool>>,
}

impl Selector {
    pub async fn new(cfg: &EppStandaloneConfig) -> Result<Self> {
        let kv_router_config = kv_router_config_from_dynamo_env();
        let cancel = CancellationToken::new();

        // If queueing is enabled, we need to validate that the max_num_batched_tokens is set.
        // Done once at startup to avoid validating on every reconcile.
        let queueing_enabled = kv_router_config
            .queueing_enabled(Some(&cfg.model_name))
            .map_err(|e| anyhow!("resolving router policy for model {}: {e}", cfg.model_name))?;
        if queueing_enabled && cfg.max_num_batched_tokens.unwrap_or(0) == 0 {
            anyhow::bail!(
                "DYN_EPP_MAX_NUM_BATCHED_TOKENS is required (and must be > 0) because the router \
                 scheduling policy enables queueing for model {}; set it to the engine's \
                 --max-num-batched-tokens",
                cfg.model_name
            );
        }

        let mut builder =
            SelectionServiceBuilder::new(kv_router_config).indexer_threads(cfg.selector_threads);

        let replication: Option<(String, u16)> = match &cfg.peer_service {
            Some(name) => Some((
                name.clone(),
                crate::peer_discovery::resolve_replica_sync_port(&cfg.namespace, name).await?,
            )),
            None => None,
        };

        if let Some((_, peer_sync_port)) = &replication {
            builder = builder.replica_sync(*peer_sync_port, Vec::new());
        }

        let service = Arc::new(
            builder
                .build()
                .await
                .map_err(|e| anyhow!("building embedded selection service: {e}"))?,
        );

        let peer_ready = if let Some((service_name, peer_sync_port)) = replication {
            // In replicated mode, we need to exclude ourselves from the peer set which requires the POD_IP
            let self_ip = std::env::var("POD_IP")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow!(
                        "DYN_EPP_PEER_SERVICE is set but POD_IP is unavailable; inject POD_IP \
                         via the downward API (fieldRef status.podIP) so this replica can \
                         exclude itself from its peer set"
                    )
                })?;
            Some(
                crate::peer_discovery::spawn(
                    service.clone(),
                    &cfg.namespace,
                    &service_name,
                    peer_sync_port,
                    self_ip,
                    cancel.clone(),
                )
                .await?,
            )
        } else {
            None
        };

        tracing::info!(
            indexer_threads = cfg.selector_threads,
            replicated = cfg.peer_service.is_some(),
            "Initialized in-process selection service"
        );

        Ok(Self {
            service,
            cancel,
            current: Mutex::new(HashMap::new()),
            peer_ready,
        })
    }

    pub fn peer_ready(&self) -> Option<Arc<AtomicBool>> {
        self.peer_ready.clone()
    }

    fn worker_request(reg: &WorkerRegistration) -> CoreWorkerRequest {
        CoreWorkerRequest {
            worker_id: reg.worker_id,
            model_name: reg.model_name.clone(),
            routing_group: DEFAULT_ROUTING_GROUP.to_string(),
            endpoint: Some(reg.endpoint.clone()),
            block_size: Some(reg.block_size),
            // Data parallel size is not yet implemented.
            // Default support for single rank DP.
            data_parallel_size: Some(1),
            kv_events_endpoints: reg.kv_events_endpoints.clone(),
            replay_endpoint: reg.replay_endpoint.clone(),
            total_kv_blocks: reg.total_kv_blocks,
            max_num_batched_tokens: reg.max_num_batched_tokens,
            stable_routing_id: reg.stable_routing_id.clone(),
            ..Default::default()
        }
    }

    pub async fn reconcile(&self, desired: &HashMap<u64, WorkerRegistration>) -> Result<()> {
        let mut current = self.current.lock().await;

        // Upsert new or changed workers.
        for (worker_id, reg) in desired {
            if current.get(worker_id) == Some(reg) {
                continue;
            }
            let record = self
                .service
                .upsert_worker(Self::worker_request(reg))
                .await
                .map_err(|e| anyhow!("upsert_worker failed: {e}"))?;
            // Only cache the registration once the core reports the worker Schedulable.
            if record.lifecycle == WorkerLifecycle::Schedulable {
                current.insert(*worker_id, reg.clone());
            } else {
                tracing::warn!(
                    worker_id = *worker_id,
                    lifecycle = ?record.lifecycle,
                    reasons = ?record.not_schedulable_reasons,
                    "Worker upserted but not schedulable; leaving uncached to retry on the next reconcile"
                );
            }
        }

        // Delete workers that are no longer desired.
        let stale: Vec<u64> = current
            .keys()
            .copied()
            .filter(|id| !desired.contains_key(id))
            .collect();
        for worker_id in stale {
            match self.service.delete_worker(worker_id).await {
                // A worker that was never registered is not an error (idempotent).
                Ok(_) | Err(SelectionError::NotFound(_)) => {}
                Err(e) => return Err(anyhow!("delete_worker failed: {e}")),
            }
            current.remove(&worker_id);
        }

        Ok(())
    }

    /// Select a worker for a prompt and book its load in one operation. Takes the
    /// request by value so per-request fields are moved into the core request
    /// rather than cloned on the hot path.
    pub async fn select_and_reserve(&self, req: SelectRequest) -> Result<SelectResponse> {
        let reservation_id = req.reservation_id;
        let core_req = CoreSelectAndReserveRequest {
            model_name: req.model_name,
            routing_group: DEFAULT_ROUTING_GROUP.to_string(),
            // The core keys both its selection cache and the scheduler booking off
            // this id; feed it the EPP-minted reservation id so the booking stays
            // EPP-known (releasable even if this response is lost).
            selection_id: Some(reservation_id.clone()),
            prompt: PromptRequest {
                token_ids: Some(req.token_ids),
                ..Default::default()
            },
            router_config_override: None,
            expected_output_tokens: None,
            session_id: None,
            priority_jump: req.priority_jump,
            strict_priority: req.strict_priority,
            pinned_worker: None,
            allowed_worker_ids: req.allowed_worker_ids,
            routing_constraints: RoutingConstraints::default(),
        };
        let resp = self
            .service
            .select_and_reserve(core_req)
            .await
            .map_err(|e| anyhow!("select_and_reserve failed: {e}"))?;
        Ok(SelectResponse {
            reservation_id,
            worker_id: resp.worker_id,
            endpoint: resp.endpoint,
            block_size: resp.block_size,
            overlap: OverlapSummary {
                longest_matched: resp.overlap.longest_matched,
                gpu: resp.overlap.gpu,
                cpu: resp.overlap.cpu,
                disk: resp.overlap.disk,
            },
            effective_prefill_tokens: resp.effective_prefill_tokens,
        })
    }

    /// Release a booking, removing the request from the selector's slot tracker /
    /// active-load accounting. Called when the gateway signals the response is
    /// complete. Idempotent: an unknown reservation (e.g. a body-less request
    /// that never booked) is treated as success.
    pub async fn free_reservation(&self, reservation_id: &str) -> Result<()> {
        match self.service.free_reservation(reservation_id).await {
            Ok(()) | Err(SelectionError::NotFound(_)) => Ok(()),
            Err(e) => Err(anyhow!("free_reservation failed: {e}")),
        }
    }

    /// Release a booking's prefill-token load at first token, keeping its decode
    /// load booked until `free_reservation`. Called in aggregated serving too, to
    /// keep the worker's load model accurate as decode continues. Idempotent: an
    /// unknown reservation is treated as success.
    pub async fn prefill_complete(&self, reservation_id: &str) -> Result<()> {
        match self.service.prefill_complete(reservation_id).await {
            Ok(()) | Err(SelectionError::NotFound(_)) => Ok(()),
            Err(e) => Err(anyhow!("prefill_complete failed: {e}")),
        }
    }

    /// Returns `true` once the selector can schedule at least one worker.
    pub async fn any_ready(&self) -> bool {
        self.service.ready().ready
    }
}

impl Drop for Selector {
    fn drop(&mut self) {
        // Stop the peer-discovery watch; the service's own Drop stops the core,
        // KV-event listeners, scheduling, and replica-sync tasks.
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal single-replica config (no peer service, so no cluster access).
    /// `max_num_batched_tokens` is set so `Selector::new` never fails its
    /// fast-fail check regardless of the ambient router policy.
    fn test_config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            selector_threads: 1,
            peer_service: None,
            inference_pool_name: "test-pool".to_string(),
            namespace: "test-ns".to_string(),
            model_name: "test-model".to_string(),
            tokenizer_service_url: "http://vllm-render:8000".to_string(),
            tokenizer_protocol: crate::epp_standalone_config::TokenizerProtocol::VllmRender,
            tokenizer_max_response_bytes: 16 * 1024 * 1024,
            tokenization_timeout_ms: 5_000,
            block_size: 16,
            kv_event_port: 5557,
            replay_port: None,
            total_kv_blocks: None,
            max_num_batched_tokens: Some(8192),
        }
    }

    /// A registration the core marks `Incomplete`: `block_size = 0` fails the
    /// schedulable-metadata check independent of router/kv-event config, so the
    /// upsert returns `Ok` with a non-`Schedulable` lifecycle.
    fn incomplete_registration(worker_id: u64) -> WorkerRegistration {
        WorkerRegistration {
            worker_id,
            model_name: "test-model".to_string(),
            endpoint: "http://10.0.0.1:8000".to_string(),
            block_size: 0,
            kv_events_endpoints: HashMap::new(),
            replay_endpoint: None,
            total_kv_blocks: None,
            max_num_batched_tokens: None,
            stable_routing_id: None,
        }
    }

    /// A registration the core marks `Schedulable` purely in-process. It carries
    /// every field the schedulable-metadata check needs: a non-empty endpoint, a
    /// non-zero `block_size`, and a well-formed KV-event endpoint per dp_rank.
    /// The KV-event endpoint is only a `tcp://` address string — the core's ZMQ
    /// SUB listener connects lazily and never needs a live publisher (see the
    /// `WorkerRegistry` register tests in `lib/kv-router`), so no network infra is
    /// required. Queueing is disabled under the default router policy, so
    /// `max_num_batched_tokens` is not required here.
    fn schedulable_registration(worker_id: u64) -> WorkerRegistration {
        WorkerRegistration {
            worker_id,
            model_name: "test-model".to_string(),
            endpoint: format!("http://10.0.0.{worker_id}:8000"),
            block_size: 16,
            kv_events_endpoints: HashMap::from([(
                0u32,
                format!("tcp://127.0.0.1:{}", 45_000 + worker_id),
            )]),
            replay_endpoint: None,
            total_kv_blocks: None,
            max_num_batched_tokens: None,
            stable_routing_id: Some(format!("vllm-{worker_id}")),
        }
    }

    /// A selection request keyed by `reservation_id` for the schedulable worker's
    /// model. The prompt is long enough to book non-trivial prefill load.
    fn select_request(reservation_id: &str) -> SelectRequest {
        SelectRequest {
            model_name: "test-model".to_string(),
            reservation_id: reservation_id.to_string(),
            token_ids: (1..=16).collect(),
            allowed_worker_ids: None,
            priority_jump: None,
            strict_priority: None,
        }
    }

    /// Reconcile a single schedulable worker into a fresh selector, asserting the
    /// core admitted it (so the reserve paths below actually book).
    async fn selector_with_schedulable_worker() -> Selector {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");
        selector
            .reconcile(&HashMap::from([(1u64, schedulable_registration(1))]))
            .await
            .expect("reconcile should succeed");
        assert!(
            selector.any_ready().await,
            "a complete worker must be schedulable in-process"
        );
        selector
    }

    /// Item 1: a successful reserve books load, and the final free releases it.
    /// `free_reservation` is idempotent: a second free of the same id is a no-op
    /// success (matching a duplicate completion signal from the gateway).
    #[tokio::test]
    async fn reserve_then_free_succeeds_and_is_idempotent() {
        let selector = selector_with_schedulable_worker().await;

        let resp = selector
            .select_and_reserve(select_request("res-1"))
            .await
            .expect("reserve should succeed against a schedulable worker");
        assert_eq!(
            resp.reservation_id, "res-1",
            "the booking key is echoed back"
        );
        assert_eq!(resp.worker_id, 1, "the only schedulable worker is chosen");
        assert_eq!(
            resp.effective_prefill_tokens, 16,
            "the full uncached prompt is booked as prefill load"
        );

        selector
            .free_reservation("res-1")
            .await
            .expect("freeing a live booking succeeds");
        selector
            .free_reservation("res-1")
            .await
            .expect("freeing an already-freed booking is an idempotent no-op");
    }

    /// Item 5: prefill completion releases prompt load exactly once and is
    /// idempotent — a repeated first-token signal must not error — and the final
    /// free still succeeds afterwards.
    #[tokio::test]
    async fn prefill_complete_then_free_is_idempotent() {
        let selector = selector_with_schedulable_worker().await;

        selector
            .select_and_reserve(select_request("res-p"))
            .await
            .expect("reserve should succeed");

        selector
            .prefill_complete("res-p")
            .await
            .expect("first prefill-complete succeeds");
        selector
            .prefill_complete("res-p")
            .await
            .expect("a repeated prefill-complete is an idempotent no-op");

        selector
            .free_reservation("res-p")
            .await
            .expect("free after prefill-complete succeeds");
        // Prefill-complete after free targets a gone booking → idempotent no-op.
        selector
            .prefill_complete("res-p")
            .await
            .expect("prefill-complete on a freed booking is a no-op");
    }

    /// Item 6 (core keying): bookings are tracked by their `reservation_id`, so
    /// freeing one reservation must not touch another. The booking id is the only
    /// key — freeing an unknown id is a no-op, freeing one live id leaves the
    /// other booked (a duplicate reserve of it still conflicts), and the freed id
    /// becomes reusable while the untouched one does not.
    #[tokio::test]
    async fn reservations_are_isolated_by_reservation_id() {
        let selector = selector_with_schedulable_worker().await;

        selector
            .select_and_reserve(select_request("res-a"))
            .await
            .expect("reserve res-a");
        selector
            .select_and_reserve(select_request("res-b"))
            .await
            .expect("reserve res-b");

        // A live booking id conflicts on re-reserve, proving the id is tracked.
        assert!(
            selector
                .select_and_reserve(select_request("res-b"))
                .await
                .is_err(),
            "re-reserving a live booking id must conflict"
        );

        // Freeing an unknown id must not disturb any live booking.
        selector
            .free_reservation("res-unknown")
            .await
            .expect("freeing an unknown id is a no-op");

        // Free only res-a. res-b must stay booked (still conflicts), while res-a
        // becomes reusable — so free() acted on exactly the id it was given.
        selector
            .free_reservation("res-a")
            .await
            .expect("free res-a");
        assert!(
            selector
                .select_and_reserve(select_request("res-b"))
                .await
                .is_err(),
            "freeing res-a must leave res-b booked"
        );
        selector
            .select_and_reserve(select_request("res-a"))
            .await
            .expect("res-a is reusable once freed");
    }

    /// Item 2 (selection-failure path): with no schedulable worker the core
    /// cannot admit the request, so `select_and_reserve` surfaces an error (which
    /// the picker maps to `PickError::RoutingFailed`). Uses `incomplete_registration`
    /// so a worker exists in the catalog but is never schedulable.
    #[tokio::test]
    async fn select_and_reserve_errors_without_a_schedulable_worker() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");
        selector
            .reconcile(&HashMap::from([(1u64, incomplete_registration(1))]))
            .await
            .expect("reconcile should succeed");
        assert!(
            !selector.any_ready().await,
            "an incomplete worker must not be schedulable"
        );

        assert!(
            selector
                .select_and_reserve(select_request("res-x"))
                .await
                .is_err(),
            "reserving with no schedulable worker must fail"
        );
    }

    /// Lifecycle callbacks are idempotent for a booking that never existed (e.g. a
    /// body-less request that never reserved): both `free_reservation` and
    /// `prefill_complete` treat an unknown id as success (NotFound → Ok).
    #[tokio::test]
    async fn free_and_prefill_of_unknown_reservation_are_noops() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");
        selector
            .free_reservation("never-booked")
            .await
            .expect("free of an unknown id is a no-op");
        selector
            .prefill_complete("never-booked")
            .await
            .expect("prefill-complete of an unknown id is a no-op");
    }

    #[tokio::test]
    async fn incomplete_worker_is_not_cached_as_reconciled() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");

        let desired = HashMap::from([(1u64, incomplete_registration(1))]);
        selector
            .reconcile(&desired)
            .await
            .expect("reconcile should succeed");

        // The worker came back Incomplete, so it must NOT be recorded as
        // reconciled — otherwise the identical next snapshot would skip the
        // re-upsert and the worker would stay silently unconverged.
        assert!(
            selector.current.lock().await.is_empty(),
            "Incomplete worker must not be cached as reconciled"
        );
        assert!(
            !selector.any_ready().await,
            "an Incomplete worker must not be schedulable"
        );

        // A second identical reconcile must re-attempt the upsert (cache miss),
        // not skip it, so the worker keeps getting a chance to converge.
        selector
            .reconcile(&desired)
            .await
            .expect("second reconcile should succeed");
        assert!(
            selector.current.lock().await.is_empty(),
            "Incomplete worker must still be uncached after a repeat reconcile"
        );
    }
}
