// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker registration, lease, and lifecycle management for direct Valkey events.

use super::*;

mod config;
pub(super) use config::{VALKEY_WORKER_UNREGISTER_TIMEOUT, valkey_gc_initial_delay_ms};
#[cfg(test)]
pub(super) use config::{
    parse_valkey_gc_inspection_budget, parse_valkey_gc_interval_ms,
    parse_valkey_required_replica_acks, parse_valkey_worker_events_enabled, valkey_index_namespace,
};
pub use config::{valkey_worker_config_from_env, valkey_worker_events_enabled};

const VALKEY_URLS_ENV: &str = "DYN_ROUTER_VALKEY_URLS";
const VALKEY_INDEX_SCOPE_ENV: &str = "DYN_ROUTER_VALKEY_INDEX_SCOPE";
const VALKEY_CONNECTION_POOL_SIZE_ENV: &str = "DYN_ROUTER_VALKEY_CONNECTION_POOL_SIZE";
const VALKEY_SENTINEL_URLS_ENV: &str = "DYN_ROUTER_VALKEY_SENTINEL_URLS";
const VALKEY_SENTINEL_MASTER_NAME_ENV: &str = "DYN_ROUTER_VALKEY_SENTINEL_MASTER_NAME";
const VALKEY_SENTINEL_QUORUM_ENV: &str = "DYN_ROUTER_VALKEY_SENTINEL_QUORUM";
const VALKEY_ALLOW_INSECURE_PLAINTEXT_ENV: &str = "DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT";
const VALKEY_ALLOW_DEGRADED_WRITES_ENV: &str = "DYN_ROUTER_VALKEY_ALLOW_DEGRADED_WRITES";
const VALKEY_WORKER_LEASE_MS_ENV: &str = "DYN_ROUTER_VALKEY_WORKER_LEASE_MS";
const VALKEY_GC_INTERVAL_MS_ENV: &str = "DYN_ROUTER_VALKEY_GC_INTERVAL_MS";
const VALKEY_GC_INSPECTION_BUDGET_ENV: &str = "DYN_ROUTER_VALKEY_GC_INSPECTION_BUDGET";
const DEFAULT_VALKEY_WORKER_LEASE_MS: u64 = 30_000;
const MIN_VALKEY_WORKER_LEASE_MS: u64 = 10_000;
const MAX_VALKEY_WORKER_LEASE_MS: u64 = 600_000;
const DEFAULT_VALKEY_GC_INTERVAL_MS: u64 = 60_000;
const MIN_VALKEY_GC_INTERVAL_MS: u64 = 1_000;
const MAX_VALKEY_GC_INTERVAL_MS: u64 = 86_400_000;
const DEFAULT_VALKEY_GC_INSPECTION_BUDGET: u32 = 256;
pub(super) const MAX_VALKEY_GC_INSPECTION_BUDGET: u32 = 1_048_576;

/// Owner-fenced capability shared by every direct KV-event publisher for one
/// worker process.
#[derive(Clone)]
pub struct ValkeyWorkerEventLease {
    owner_nonce: u64,
    pub(super) operation_cancel: CancellationToken,
    pub(super) integrity: ValkeyPublisherIntegrity,
}

impl ValkeyWorkerEventLease {
    pub fn owner_nonce(&self) -> u64 {
        self.owner_nonce
    }
}

/// Worker-owned Valkey registration lease. A heartbeat keeps all DP ranks
/// eligible while an independent, jittered task advances bounded lifecycle
/// GC. Crash expiry and conditional unregister reclaim state without frontend
/// races or sleep/wake tombstones.
pub struct ValkeyWorkerRegistration {
    event_lease: ValkeyWorkerEventLease,
    worker_id: WorkerId,
    lifecycle_cancel: CancellationToken,
    operation_cancel: CancellationToken,
    heartbeat_task: Option<JoinHandle<()>>,
    gc_task: Option<JoinHandle<()>>,
    active: bool,
}

/// Fully validated direct-worker Valkey configuration. Resolving this once at
/// worker startup keeps registration and publisher setup on one topology.
#[derive(Clone, Debug)]
pub struct ValkeyWorkerConfig {
    urls: String,
    index_scope: String,
    index_namespace: String,
    direct_event_pool_size: u32,
    required_replica_acks: Option<u32>,
    sentinel_urls: Option<String>,
    sentinel_master_name: Option<String>,
    sentinel_quorum: Option<usize>,
    allow_degraded_writes: bool,
    worker_lease_ms: u64,
    gc_interval_ms: Option<u64>,
    gc_inspection_budget: u32,
}

impl ValkeyWorkerConfig {
    async fn build_indexer(
        &self,
        component: &Component,
        kv_block_size: u32,
        cancellation_token: CancellationToken,
    ) -> Result<crate::kv_router::indexer::valkey::ValkeyIndexer> {
        if let (Some(urls), Some(master_name)) = (
            self.sentinel_urls.as_deref(),
            self.sentinel_master_name.as_deref(),
        ) {
            let sentinel = crate::valkey_transport::ValkeySentinelConfig::new(
                urls,
                master_name,
                self.sentinel_quorum,
            )?;
            crate::kv_router::indexer::valkey::ValkeyIndexer::new_worker_with_sentinel(
                &self.urls,
                self.direct_event_pool_size,
                self.required_replica_acks,
                &self.index_namespace,
                component.name(),
                Some(&self.index_scope),
                kv_block_size,
                cancellation_token,
                sentinel,
                self.allow_degraded_writes,
            )
            .await
        } else {
            crate::kv_router::indexer::valkey::ValkeyIndexer::new_worker(
                &self.urls,
                self.direct_event_pool_size,
                self.required_replica_acks,
                &self.index_namespace,
                component.name(),
                Some(&self.index_scope),
                kv_block_size,
                cancellation_token,
            )
        }
    }
}

impl ValkeyWorkerRegistration {
    /// Register every data-parallel rank owned by a worker in the shared Valkey
    /// admission index without creating a KV-event source or publishing
    /// synthetic cache ownership.
    ///
    /// Decode-only workers deliberately emit no KV cache events, but they are
    /// still valid authoritative-admission targets. Registration is awaited by
    /// worker startup so discovery cannot expose an ineligible rank while the
    /// module write (and, when configured, its replica acknowledgement) is
    /// still pending. Returns `None` when direct Valkey worker events are not
    /// configured.
    pub async fn register_from_env(
        component: &Component,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        dp_ranks: &[DpRank],
    ) -> Result<Option<Self>> {
        if kv_block_size == 0 {
            anyhow::bail!("Valkey worker registration requires a nonzero KV block size");
        }

        let ranks = dp_ranks.iter().copied().collect::<BTreeSet<_>>();
        if ranks.is_empty() {
            anyhow::bail!("Valkey worker registration requires at least one DP rank");
        }

        let Some(config) = valkey_worker_config_from_env(component)? else {
            return Ok(None);
        };
        Self::register_with_config(component, worker_id, kv_block_size, &ranks, config)
            .await
            .map(Some)
    }

    pub async fn register_with_config(
        component: &Component,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        dp_ranks: &BTreeSet<DpRank>,
        config: ValkeyWorkerConfig,
    ) -> Result<Self> {
        if kv_block_size == 0 {
            anyhow::bail!("Valkey worker registration requires a nonzero KV block size");
        }
        if dp_ranks.is_empty() {
            anyhow::bail!("Valkey worker registration requires at least one DP rank");
        }

        let _ = KvPublisherMetrics::from_component(component);
        let operation_cancel = CancellationToken::new();
        let indexer = config
            .build_indexer(component, kv_block_size, operation_cancel.clone())
            .await?;
        let worker_id = worker_id.unwrap_or_else(|| component.drt().connection_id());

        let ranks = dp_ranks.iter().copied().collect::<Vec<_>>();
        let owner_nonce = random_nonzero_owner_nonce();
        indexer
            .register_worker_lease(
                worker_id,
                owner_nonce,
                config.worker_lease_ms,
                config.gc_inspection_budget,
                &ranks,
            )
            .await
            .with_context(|| {
                format!(
                    "failed to acquire Valkey worker lease for worker {worker_id} DP ranks {ranks:?}"
                )
            })?;

        let integrity = ValkeyPublisherIntegrity::new(
            indexer.clone(),
            worker_id,
            owner_nonce,
            config.worker_lease_ms,
        );
        let event_lease = ValkeyWorkerEventLease {
            owner_nonce,
            operation_cancel: operation_cancel.clone(),
            integrity: integrity.clone(),
        };
        let lifecycle_cancel = CancellationToken::new();
        let heartbeat_task = spawn_worker_lease_heartbeat(
            integrity.clone(),
            config.worker_lease_ms,
            lifecycle_cancel.clone(),
        );
        let gc_task = config.gc_interval_ms.map(|interval_ms| {
            spawn_worker_lifecycle_gc(
                integrity,
                worker_id,
                owner_nonce,
                interval_ms,
                config.gc_inspection_budget,
                lifecycle_cancel.clone(),
            )
        });
        tracing::info!(
            worker_id,
            owner_nonce,
            lease_ms = config.worker_lease_ms,
            dp_ranks = ?dp_ranks,
            gc_interval_ms = ?config.gc_interval_ms,
            gc_inspection_budget = config.gc_inspection_budget,
            "Acquired worker-owned Valkey registration lease"
        );
        Ok(Self {
            event_lease,
            worker_id,
            lifecycle_cancel,
            operation_cancel,
            heartbeat_task: Some(heartbeat_task),
            gc_task,
            active: true,
        })
    }

    pub fn event_lease(&self) -> ValkeyWorkerEventLease {
        self.event_lease.clone()
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        self.lifecycle_cancel.cancel();
        let heartbeat_task = self.heartbeat_task.take();
        let gc_task = self.gc_task.take();
        let stop_heartbeat_and_unregister = async {
            if let Some(task) = heartbeat_task {
                task.abort();
                let _ = task.await;
            }
            if let Some(task) = gc_task {
                task.abort();
                let _ = task.await;
            }
            self.event_lease.integrity.unregister_for_shutdown().await
        };
        let result = match tokio::time::timeout(
            VALKEY_WORKER_UNREGISTER_TIMEOUT,
            stop_heartbeat_and_unregister,
        )
        .await
        {
            Ok(result) => result.with_context(|| {
                format!(
                    "failed to unregister Valkey worker lease for worker {} owner {}",
                    self.worker_id, self.event_lease.owner_nonce
                )
            }),
            Err(_) => Err(anyhow::anyhow!(
                "timed out after {}ms unregistering Valkey worker {} owner {}; server-side lease expiry remains the backstop",
                VALKEY_WORKER_UNREGISTER_TIMEOUT.as_millis(),
                self.worker_id,
                self.event_lease.owner_nonce
            )),
        };
        self.operation_cancel.cancel();
        result
    }
}

impl Drop for ValkeyWorkerRegistration {
    fn drop(&mut self) {
        self.lifecycle_cancel.cancel();
        self.operation_cancel.cancel();
        if let Some(task) = self.heartbeat_task.take() {
            task.abort();
        }
        if let Some(task) = self.gc_task.take() {
            task.abort();
        }
    }
}

fn random_nonzero_owner_nonce() -> u64 {
    loop {
        let bytes = Uuid::new_v4().into_bytes();
        let nonce = u64::from_be_bytes(bytes[..8].try_into().expect("UUID is 16 bytes"));
        if nonce != 0 {
            return nonce;
        }
    }
}

fn spawn_worker_lease_heartbeat(
    integrity: ValkeyPublisherIntegrity,
    lease_ms: u64,
    cancellation: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let interval = std::time::Duration::from_millis((lease_ms / 3).max(1));
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => return,
                _ = tokio::time::sleep(interval) => {}
            }
            if !integrity.is_healthy() {
                tokio::select! {
                    _ = cancellation.cancelled() => return,
                    _ = integrity.fence_once() => {}
                }
                continue;
            }
            if let Err(error) = integrity.renew_lease().await {
                tracing::error!(
                    error = %error,
                    "Valkey worker heartbeat entered the permanent integrity fence"
                );
                tokio::select! {
                    _ = cancellation.cancelled() => return,
                    _ = integrity.fence_once() => {}
                }
            }
        }
    })
}

pub(super) fn spawn_worker_lifecycle_gc(
    integrity: ValkeyPublisherIntegrity,
    worker_id: WorkerId,
    owner_nonce: u64,
    interval_ms: u64,
    inspection_budget: u32,
    cancellation: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let interval = Duration::from_millis(interval_ms);
        let mut next_delay =
            Duration::from_millis(valkey_gc_initial_delay_ms(interval_ms, owner_nonce));
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => return,
                _ = tokio::time::sleep(next_delay) => {}
            }

            let outcome = tokio::select! {
                _ = cancellation.cancelled() => return,
                outcome = integrity.gc_step_if_idle(inspection_budget) => outcome,
            };
            match outcome {
                Ok(GcStepOutcome::Completed(stats)) => {
                    if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                        metrics.increment_lifecycle_gc_step("completed");
                    }
                    tracing::debug!(
                        worker_id,
                        owner_nonce,
                        inspection_budget,
                        stats = ?stats,
                        "Completed bounded Valkey lifecycle GC tick"
                    );
                }
                Ok(GcStepOutcome::SkippedBusy) => {
                    if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                        metrics.increment_lifecycle_gc_step("skipped_busy");
                    }
                    tracing::debug!(
                        worker_id,
                        owner_nonce,
                        "Skipped Valkey lifecycle GC tick because worker mutations were active"
                    );
                }
                Ok(GcStepOutcome::SkippedFenced) => {
                    if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                        metrics.increment_lifecycle_gc_step("skipped_fenced");
                    }
                    tracing::debug!(
                        worker_id,
                        owner_nonce,
                        "Skipped Valkey lifecycle GC because worker integrity is fenced"
                    );
                }
                Err(error) => {
                    if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                        metrics.increment_lifecycle_gc_step("error");
                    }
                    tracing::warn!(
                        worker_id,
                        owner_nonce,
                        inspection_budget,
                        error = %error,
                        "Valkey lifecycle GC tick failed; serving and lease heartbeat continue"
                    );
                }
            }
            next_delay = interval;
        }
    })
}
