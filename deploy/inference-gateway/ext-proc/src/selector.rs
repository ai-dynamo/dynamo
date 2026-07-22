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

use dynamo_kv_router::config::{KvRouterConfig, kv_router_config_from_dynamo_env};
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
    reconcile_state: Mutex<ReconcileState>,
    /// Peer-discovery readiness in replicated mode: `None` when replication is
    /// disabled (single replica, always ready), or `Some(flag)` that latches
    /// `true` once the initial peer-set sync completes. ANDed into EPP health.
    peer_ready: Option<Arc<AtomicBool>>,
}

/// Local bookkeeping for desired-state reconciliation.
#[derive(Default)]
struct ReconcileState {
    /// Registrations confirmed schedulable by the service. Identical desired
    /// entries can skip upsert without preventing `Incomplete` retries.
    converged: HashMap<u64, WorkerRegistration>,
    /// Every worker ID this reconciler may have introduced into the service,
    /// including `Incomplete` and partially failed upserts. Stale detection must
    /// cover this set rather than only the schedulable convergence cache.
    tracked_worker_ids: HashSet<u64>,
}

impl Selector {
    pub async fn new(cfg: &EppStandaloneConfig) -> Result<Self> {
        Self::new_with_kv_router_config(cfg, kv_router_config_from_dynamo_env()).await
    }

    async fn new_with_kv_router_config(
        cfg: &EppStandaloneConfig,
        kv_router_config: KvRouterConfig,
    ) -> Result<Self> {
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
            reconcile_state: Mutex::new(ReconcileState::default()),
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

    pub async fn reconcile(&self, registrations: &[WorkerRegistration]) -> Result<()> {
        // Derive the bookkeeping key from the registration so callers cannot
        // provide a map key that disagrees with the ID sent to SelectionService.
        // Reject duplicates before mutating either local or service state.
        let mut desired = HashMap::with_capacity(registrations.len());
        for registration in registrations {
            if desired
                .insert(registration.worker_id, registration)
                .is_some()
            {
                anyhow::bail!("duplicate worker_id {}", registration.worker_id);
            }
        }

        let mut state = self.reconcile_state.lock().await;

        // Upsert new or changed workers.
        for (&worker_id, &reg) in &desired {
            // Track before calling the service so a partially applied upsert can
            // still be deleted if the worker later leaves the desired snapshot.
            state.tracked_worker_ids.insert(worker_id);
            if state.converged.get(&worker_id) == Some(reg) {
                continue;
            }
            let record = self
                .service
                .upsert_worker(Self::worker_request(reg))
                .await
                .map_err(|e| anyhow!("upsert_worker failed: {e}"))?;
            // Only cache the registration once the core reports the worker Schedulable.
            if record.lifecycle == WorkerLifecycle::Schedulable {
                state.converged.insert(worker_id, reg.clone());
            } else {
                state.converged.remove(&worker_id);
                tracing::warn!(
                    worker_id,
                    lifecycle = ?record.lifecycle,
                    reasons = ?record.not_schedulable_reasons,
                    "Worker upserted but not schedulable; leaving uncached to retry on the next reconcile"
                );
            }
        }

        // Delete every worker this reconciler may have introduced that is no
        // longer desired, including workers that never became schedulable.
        let stale: Vec<u64> = state
            .tracked_worker_ids
            .iter()
            .copied()
            .filter(|id| !desired.contains_key(id))
            .collect();
        for worker_id in stale {
            match self.service.delete_worker(worker_id).await {
                // A worker that was never registered is not an error (idempotent).
                Ok(_) | Err(SelectionError::NotFound(_)) => {}
                Err(e) => return Err(anyhow!("delete_worker failed: {e}")),
            }
            state.tracked_worker_ids.remove(&worker_id);
            state.converged.remove(&worker_id);
        }

        Ok(())
    }

    pub async fn select_and_reserve(&self, req: SelectRequest) -> Result<SelectResponse> {
        let core_req = CoreSelectAndReserveRequest {
            model_name: req.model_name,
            routing_group: DEFAULT_ROUTING_GROUP.to_string(),
            selection_id: req.selection_id,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model_policy_file() -> tempfile::NamedTempFile {
        let policy_file = tempfile::NamedTempFile::new().expect("create policy file");
        std::fs::write(
            policy_file.path(),
            r#"
models:
  queueing-model:
    default_policy_family: standard
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: queued
        policy_family: standard
        cache_bucket: all
        quantum: 1
        prefill_busy_threshold: 1
  threshold-free-model:
    default_policy_family: standard
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: direct
        policy_family: standard
        cache_bucket: all
        quantum: 1
"#,
        )
        .expect("write policy file");
        policy_file
    }

    fn router_config_with_policy(policy_file: &tempfile::NamedTempFile) -> KvRouterConfig {
        KvRouterConfig {
            router_policy_config: Some(policy_file.path().to_string_lossy().into_owned()),
            ..Default::default()
        }
    }

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

    #[tokio::test]
    async fn incomplete_worker_is_not_cached_as_reconciled() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");

        let desired = [incomplete_registration(1)];
        selector
            .reconcile(&desired)
            .await
            .expect("reconcile should succeed");

        // The worker came back Incomplete, so it must NOT be recorded as
        // reconciled — otherwise the identical next snapshot would skip the
        // re-upsert and the worker would stay silently unconverged.
        assert!(
            selector.reconcile_state.lock().await.converged.is_empty(),
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
            selector.reconcile_state.lock().await.converged.is_empty(),
            "Incomplete worker must still be uncached after a repeat reconcile"
        );
    }

    #[tokio::test]
    async fn stale_incomplete_worker_is_deleted() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");

        selector
            .reconcile(&[incomplete_registration(1)])
            .await
            .expect("incomplete worker should be tracked");
        selector
            .reconcile(&[])
            .await
            .expect("stale incomplete worker should be deleted");

        let worker = selector
            .service
            .list_workers(None, None)
            .into_iter()
            .find(|worker| worker.worker_id == 1)
            .expect("SelectionService retains the terminal catalog record");
        assert_eq!(worker.lifecycle, WorkerLifecycle::Unschedulable);
        let state = selector.reconcile_state.lock().await;
        assert!(state.tracked_worker_ids.is_empty());
        assert!(state.converged.is_empty());
    }

    #[tokio::test]
    async fn queueing_model_requires_max_num_batched_tokens_at_startup() {
        let policy_file = model_policy_file();
        let mut cfg = test_config();
        cfg.model_name = "queueing-model".to_string();
        cfg.max_num_batched_tokens = None;

        let error =
            Selector::new_with_kv_router_config(&cfg, router_config_with_policy(&policy_file))
                .await
                .err()
                .expect("queueing model must reject missing capacity");
        assert!(
            error
                .to_string()
                .contains("DYN_EPP_MAX_NUM_BATCHED_TOKENS is required"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn threshold_free_model_allows_missing_max_num_batched_tokens_at_startup() {
        let policy_file = model_policy_file();
        let mut cfg = test_config();
        cfg.model_name = "threshold-free-model".to_string();
        cfg.max_num_batched_tokens = None;

        Selector::new_with_kv_router_config(&cfg, router_config_with_policy(&policy_file))
            .await
            .expect("threshold-free model should allow missing capacity");
    }

    #[tokio::test]
    async fn duplicate_worker_ids_are_rejected_before_reconciliation() {
        let selector = Selector::new(&test_config())
            .await
            .expect("selector should build");
        let duplicate = incomplete_registration(1);

        let error = selector
            .reconcile(&[duplicate.clone(), duplicate])
            .await
            .expect_err("duplicate IDs must be rejected");
        assert!(error.to_string().contains("duplicate worker_id 1"));
        assert!(selector.service.list_workers(None, None).is_empty());
    }
}
