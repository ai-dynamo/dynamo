// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process, runtime-free worker selector.
//!
//! Standalone mode runs the runtime-free selection service **in-process**: the
//! EPP and a [`SelectionService`] are compiled into one binary and the EPP calls
//! the selector's Rust API directly — no HTTP client, no second Deployment.
//!
//! With `DYN_EPP_PEER_SERVICE` set, [`Selector`] enables the service's
//! replica-sync and starts a [`crate::peer_discovery`] watch of the EPP's own
//! Service, so replicated EPP pods discover their siblings and sync active load
//! over ZMQ (admission / prefill-complete / free). Without it, a single replica
//! runs fully local.
//!
//! This module also owns the small plain types the reflector → topology adapter →
//! router pipeline speaks ([`WorkerRegistration`], [`SelectRequest`],
//! [`SelectResponse`]); the selector maps them to the selection service's own
//! public request/response types (no JSON in the hot path).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};

use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest as CoreSelectAndReserveRequest, SelectionError,
    SelectionService, SelectionServiceBuilder, WorkerRequest as CoreWorkerRequest,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;

/// Tenant scope for standalone mode. Must match between worker registration and
/// selection; the selection service's own default is `"default"`.
const DEFAULT_TENANT: &str = "default";

/// A worker the EPP registers into the selector. Only the fields standalone
/// mode populates are included; the selector defaults the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub endpoint: String,
    pub block_size: u32,
    pub data_parallel_size: u32,
    pub kv_events_endpoints: HashMap<u32, String>,
    pub replay_endpoint: Option<String>,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
    pub stable_routing_id: Option<String>,
}

/// A worker-selection request. Raw `token_ids` let the selector compute
/// block/sequence hashes.
#[derive(Debug, Clone)]
pub struct SelectRequest {
    pub model_name: String,
    pub selection_id: Option<String>,
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
    pub selection_id: Option<String>,
    pub worker_id: u64,
    pub dp_rank: u32,
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
}

impl Selector {
    /// Build an in-process selector from the standalone config. `selector_threads`
    /// sizes the KV indexer pool; router scheduling comes from the standard
    /// `DYN_ROUTER_*` environment. When `peer_service` is set, replica-sync is
    /// enabled and a peer-discovery watch keeps the peer set in sync.
    pub async fn new(cfg: &EppStandaloneConfig) -> Result<Self> {
        let kv_router_config = kv_router_config_from_dynamo_env();
        let cancel = CancellationToken::new();

        let mut builder =
            SelectionServiceBuilder::new(kv_router_config).indexer_threads(cfg.selector_threads);

        // Config validation already guarantees peer_sync_port is Some whenever
        // peer_service is set; resolve the (name, port) pair once and surface the
        // invariant as an error rather than a panic.
        let replication: Option<(String, u16)> = match &cfg.peer_service {
            Some(name) => {
                let port = cfg.peer_sync_port.ok_or_else(|| {
                    anyhow!("peer_sync_port is required when peer_service is set")
                })?;
                Some((name.clone(), port))
            }
            None => None,
        };

        if let Some((_, peer_sync_port)) = &replication {
            // Bind this replica's ZMQ replica-sync port; peers are registered
            // dynamically by the EndpointSlice watch below.
            builder = builder.replica_sync(*peer_sync_port, Vec::new());
        }

        let service = Arc::new(
            builder
                .build()
                .await
                .map_err(|e| anyhow!("building embedded selection service: {e}"))?,
        );

        if let Some((service_name, peer_sync_port)) = replication {
            // POD_IP is required (not advisory) in replicated mode: without it we
            // can't exclude our own IP from the peer set, so this replica would
            // sync with itself and double-count its own load. Fail fast rather
            // than run in a knowingly-wrong state.
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
            // The EPP's own Service lives in the EPP's namespace (same as the
            // pool); reuse the single resolved namespace.
            crate::peer_discovery::spawn(
                service.clone(),
                &cfg.namespace,
                &service_name,
                peer_sync_port,
                self_ip,
                cancel.clone(),
            )
            .await?;
        }

        tracing::info!(
            indexer_threads = cfg.selector_threads,
            replicated = cfg.peer_service.is_some(),
            "Initialized in-process selection service"
        );

        Ok(Self {
            service,
            cancel,
            current: Mutex::new(HashMap::new()),
        })
    }

    fn worker_request(reg: &WorkerRegistration) -> CoreWorkerRequest {
        CoreWorkerRequest {
            worker_id: reg.worker_id,
            model_name: reg.model_name.clone(),
            tenant_id: DEFAULT_TENANT.to_string(),
            endpoint: Some(reg.endpoint.clone()),
            block_size: Some(reg.block_size),
            data_parallel_size: Some(reg.data_parallel_size),
            kv_events_endpoints: reg.kv_events_endpoints.clone(),
            replay_endpoint: reg.replay_endpoint.clone(),
            total_kv_blocks: reg.total_kv_blocks,
            max_num_batched_tokens: reg.max_num_batched_tokens,
            stable_routing_id: reg.stable_routing_id.clone(),
            ..Default::default()
        }
    }

    /// Drive the selector catalog toward `desired` (keyed by `worker_id`).
    /// Idempotent: safe to call repeatedly.
    pub async fn reconcile(&self, desired: &HashMap<u64, WorkerRegistration>) -> Result<()> {
        let mut current = self.current.lock().await;

        // Upsert new or changed workers.
        for (worker_id, reg) in desired {
            if current.get(worker_id) == Some(reg) {
                continue;
            }
            self.service
                .upsert_worker(Self::worker_request(reg))
                .await
                .map_err(|e| anyhow!("upsert_worker failed: {e}"))?;
            current.insert(*worker_id, reg.clone());
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
    /// request by value so per-request fields (`token_ids`, `model_name`, ...) are
    /// moved into the core request rather than cloned on the hot path.
    pub async fn select_and_reserve(&self, req: SelectRequest) -> Result<SelectResponse> {
        let core_req = CoreSelectAndReserveRequest {
            model_name: req.model_name,
            tenant_id: DEFAULT_TENANT.to_string(),
            selection_id: req.selection_id,
            reservation_id: None,
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
            selection_id: resp.selection_id,
            worker_id: resp.worker_id,
            dp_rank: resp.dp_rank,
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
