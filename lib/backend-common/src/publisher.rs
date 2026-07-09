// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side wiring for KV-aware-routing publishers.
//!
//! Two surfaces, both event-driven (no polling, no GIL on framework side):
//!
//! - [`KvEventPublisher`] (per dp_rank) — wired from the engine's
//!   [`KvEventSource`] declarations. Engine pushes stored/removed events
//!   to the publisher; framework relays to NATS.
//! - [`SnapshotPublisher`] — single per-worker handle for per-rank
//!   `ComponentSnapshot` writes. Engine pushes; publisher atomically
//!   updates the Rust `ComponentGauges` (for /metrics) AND the
//!   per-rank `WorkerMetricsPublisher` (for KV router NATS signal)
//!   inline.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use dynamo_llm::kv_router::publisher::{
    KvEventPublisher, KvEventPublisherShutdown, KvEventPublisherShutdownOutcome,
    KvEventSourceConfig, ValkeyWorkerConfig, ValkeyWorkerEventLease, ValkeyWorkerRegistration,
    WorkerMetricsPublisher, shutdown_publishers_and_wait,
};
use dynamo_runtime::component::Component;

use crate::engine::KvEventSource;
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::metrics::{ComponentGauges, EngineMetrics};
use crate::snapshot_publisher::SnapshotPublisher;

/// Live publisher handles owned by `Worker` for the lifetime of serving.
///
/// Publishers can also be held by engine callbacks, so explicit shutdown
/// handles are retained to stop and join their event tasks before the shared
/// Valkey owner lease is surrendered.
pub(crate) struct PublisherHandles {
    #[allow(dead_code)]
    kv_publishers: Vec<Arc<KvEventPublisher>>,
    kv_publisher_shutdowns: Vec<KvEventPublisherShutdown>,
    /// Stashed so the engine's `Arc<SnapshotPublisher>` reference stays
    /// valid for the worker's lifetime. Engines drop their copy when
    /// they shut down; we keep ours so the `WorkerMetricsPublisher`s
    /// inside don't drop their NATS endpoints prematurely.
    #[allow(dead_code)]
    snapshot_publisher: Option<Arc<SnapshotPublisher>>,
    /// Decode-only workers have no KV event source, but authoritative Valkey
    /// admission still requires every rank to be registered before discovery
    /// exposes the worker.
    #[allow(dead_code)]
    valkey_registration: Option<ValkeyWorkerRegistration>,
}

pub(crate) struct PublisherSetup {
    pub(crate) kv_sources: Vec<KvEventSource>,
    pub(crate) snapshot_dp_ranks: Vec<u32>,
    pub(crate) registration_dp_ranks: Vec<u32>,
    pub(crate) valkey_worker_config: Option<ValkeyWorkerConfig>,
    pub(crate) on_snapshot_ready: Option<crate::engine::OnSnapshotPublisherReady>,
    pub(crate) kv_cache_block_size: Option<u32>,
    pub(crate) enable_local_indexer: bool,
}

impl PublisherHandles {
    pub(crate) async fn shutdown(&mut self) {
        let publisher_outcome = shutdown_publishers_and_wait(&self.kv_publisher_shutdowns).await;
        self.kv_publisher_shutdowns.clear();
        self.kv_publishers.clear();
        if let Some(mut registration) = self.valkey_registration.take() {
            if publisher_outcome == KvEventPublisherShutdownOutcome::Drained {
                if let Err(error) = registration.shutdown().await {
                    tracing::warn!(
                        error = %error,
                        "Failed to unregister Valkey worker lease; server-side expiry remains the backstop"
                    );
                }
            } else {
                tracing::warn!(
                    ?publisher_outcome,
                    "Skipping Valkey worker unregister after forced publisher shutdown; server-side lease expiry remains the backstop"
                );
            }
        }
    }
}

// Sync — `KvEventPublisher::new_with_local_indexer` doesn't await. The
// snapshot router-publisher construction below is async because
// `create_endpoint` does.
fn setup_kv_publishers(
    component: &Component,
    sources: Vec<KvEventSource>,
    kv_cache_block_size: u32,
    enable_local_indexer: bool,
    valkey_event_lease: Option<&ValkeyWorkerEventLease>,
    publishers: &mut Vec<Arc<KvEventPublisher>>,
) -> Result<(), DynamoError> {
    publishers.reserve(sources.len());
    for source in sources {
        let dp_rank = source.dp_rank();
        let (source_config, on_ready) = match source {
            KvEventSource::Zmq {
                endpoint, topic, ..
            } => (
                Some(KvEventSourceConfig::Zmq {
                    endpoint,
                    topic,
                    image_token_id: None,
                }),
                None,
            ),
            KvEventSource::Push { on_ready, .. } => (None, Some(on_ready)),
        };
        let publisher = KvEventPublisher::new_with_local_indexer_and_worker_id_and_valkey_lease(
            component.clone(),
            None,
            kv_cache_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            None,
            valkey_event_lease.cloned(),
        )
        .map_err(|e| publisher_err(format!("kv publisher setup (dp_rank={dp_rank}): {e}")))?;
        let publisher = Arc::new(publisher);
        publishers.push(publisher.clone());
        if let Some(on_ready) = on_ready {
            // Partial-success: engines whose on_ready ran before this failure
            // have already started threads. The unwind path runs
            // `engine.cleanup` (see `Worker::cleanup_once`), which is the
            // sole hook for joining them.
            on_ready(publisher.clone()).map_err(|e| {
                publisher_err(format!("kv publisher on_ready (dp_rank={dp_rank}): {e}"))
            })?;
        }
    }
    Ok(())
}

/// Convert post-registration setup into a transaction. Any failure stops all
/// publishers created so far before surrendering the worker lease, ensuring
/// no accepted event can race with unregister.
async fn complete_or_rollback_setup<T, R, F, Fut>(
    result: Result<T, DynamoError>,
    kv_publishers: &mut Vec<Arc<KvEventPublisher>>,
    registration: &mut Option<R>,
    shutdown_registration: F,
) -> Result<T, DynamoError>
where
    F: FnOnce(R) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    let error = match result {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };

    let shutdowns = kv_publishers
        .iter()
        .map(|publisher| publisher.shutdown_handle())
        .collect::<Vec<_>>();
    let publisher_outcome = shutdown_publishers_and_wait(&shutdowns).await;
    kv_publishers.clear();
    rollback_registration_after_publishers(publisher_outcome, registration, shutdown_registration)
        .await;
    Err(error)
}

async fn rollback_registration_after_publishers<R, F, Fut>(
    publisher_outcome: KvEventPublisherShutdownOutcome,
    registration: &mut Option<R>,
    shutdown_registration: F,
) where
    F: FnOnce(R) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    let Some(registration) = registration.take() else {
        return;
    };
    if publisher_outcome != KvEventPublisherShutdownOutcome::Drained {
        // Dropping stops renewal without explicitly unregistering. The
        // server-side owner lease must remain until expiry because a forced
        // or timed-out publisher may still have an ambiguous write in flight.
        drop(registration);
        tracing::warn!(
            ?publisher_outcome,
            "Skipping Valkey worker unregister after unsafe setup rollback; server-side lease expiry remains the backstop"
        );
        return;
    }
    if let Err(unregister_error) = shutdown_registration(registration).await {
        tracing::warn!(
            error = %unregister_error,
            "Failed to unregister Valkey worker after publisher setup rollback; server-side expiry remains the backstop"
        );
    }
}

/// Build one `WorkerMetricsPublisher` per declared dp_rank. Each owns a
/// NATS endpoint advertising the rank's `kv_used_blocks` signal to the
/// KV router. Constructed eagerly so the `SnapshotPublisher` can route
/// per-rank writes inline.
async fn build_router_publishers(
    component: &Component,
    dp_ranks: &[u32],
) -> Result<HashMap<u32, Arc<WorkerMetricsPublisher>>, DynamoError> {
    let mut out = HashMap::with_capacity(dp_ranks.len());
    for &dp_rank in dp_ranks {
        let publisher = WorkerMetricsPublisher::new().map_err(|e| {
            publisher_err(format!("metrics publisher new (dp_rank={dp_rank}): {e}"))
        })?;
        publisher
            .create_endpoint(component.clone())
            .await
            .map_err(|e| {
                publisher_err(format!(
                    "metrics publisher endpoint (dp_rank={dp_rank}): {e}"
                ))
            })?;
        out.insert(dp_rank, Arc::new(publisher));
    }
    Ok(out)
}

fn publisher_err(message: String) -> DynamoError {
    // Publisher construction errors are almost always NATS-reach related.
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::CannotConnect))
        .message(message)
        .build()
}

pub(crate) async fn setup_publishers(
    component: &Component,
    engine_metrics: &EngineMetrics,
    setup: PublisherSetup,
) -> Result<PublisherHandles, DynamoError> {
    let PublisherSetup {
        kv_sources,
        snapshot_dp_ranks,
        registration_dp_ranks,
        valkey_worker_config,
        on_snapshot_ready,
        kv_cache_block_size,
        enable_local_indexer,
    } = setup;
    let mut valkey_registration = match (
        kv_cache_block_size,
        registration_dp_ranks.is_empty(),
        valkey_worker_config,
    ) {
        (Some(block_size), false, Some(config)) => ValkeyWorkerRegistration::register_with_config(
            component,
            None,
            block_size,
            &registration_dp_ranks.into_iter().collect(),
            config,
        )
        .await
        .map_err(|error| {
            publisher_err(format!(
                "Valkey worker-rank registration before discovery: {error}"
            ))
        })?
        .into(),
        _ => None,
    };

    // KV event publishers require the engine's block size; without it, the
    // router can't translate token IDs into cache blocks. Snapshot publisher
    // is independent — load reporting works regardless of cache structure.
    let valkey_event_lease = valkey_registration
        .as_ref()
        .map(ValkeyWorkerRegistration::event_lease);
    let mut kv_publishers = Vec::new();
    let post_registration_setup = async {
        if let Some(block_size) = kv_cache_block_size {
            setup_kv_publishers(
                component,
                kv_sources,
                block_size,
                enable_local_indexer,
                valkey_event_lease.as_ref(),
                &mut kv_publishers,
            )?;
        } else if !kv_sources.is_empty() {
            tracing::warn!(
                "engine declared {} kv_event_sources but kv_cache_block_size is None; skipping KV event publishers",
                kv_sources.len()
            );
        }

        let snapshot_publisher = if snapshot_dp_ranks.is_empty() {
            None
        } else {
            let router_publishers = build_router_publishers(component, &snapshot_dp_ranks).await?;
            let gauges = Arc::new(ComponentGauges::new(engine_metrics, &snapshot_dp_ranks)?);
            let publisher = Arc::new(SnapshotPublisher::new(gauges, router_publishers));
            if let Some(on_ready) = on_snapshot_ready {
                on_ready(publisher.clone())
                    .map_err(|e| publisher_err(format!("snapshot publisher on_ready: {e}")))?;
            }
            Some(publisher)
        };
        Ok(snapshot_publisher)
    }
    .await;

    let snapshot_publisher = complete_or_rollback_setup(
        post_registration_setup,
        &mut kv_publishers,
        &mut valkey_registration,
        |mut registration| async move { registration.shutdown().await },
    )
    .await?;
    let kv_publisher_shutdowns = kv_publishers
        .iter()
        .map(|publisher| publisher.shutdown_handle())
        .collect();

    Ok(PublisherHandles {
        kv_publishers,
        kv_publisher_shutdowns,
        snapshot_publisher,
        valkey_registration,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use super::*;

    #[tokio::test]
    async fn post_registration_failure_runs_immediate_rollback() {
        struct MockRegistration;

        let shutdown_called = Arc::new(AtomicBool::new(false));
        let shutdown_observer = shutdown_called.clone();
        let mut registration = Some(MockRegistration);
        let mut publishers = Vec::new();
        let injected_error = publisher_err("injected post-registration failure".to_string());

        let result = complete_or_rollback_setup(
            Err::<(), _>(injected_error),
            &mut publishers,
            &mut registration,
            move |_registration| async move {
                shutdown_observer.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;

        assert!(result.is_err());
        assert!(registration.is_none());
        assert!(shutdown_called.load(Ordering::SeqCst));
    }

    async fn assert_unsafe_rollback_leaves_lease_for_expiry(
        outcome: KvEventPublisherShutdownOutcome,
    ) {
        struct MockRegistration;

        let shutdown_called = Arc::new(AtomicBool::new(false));
        let shutdown_observer = shutdown_called.clone();
        let mut registration = Some(MockRegistration);

        rollback_registration_after_publishers(
            outcome,
            &mut registration,
            move |_registration| async move {
                shutdown_observer.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;

        assert!(registration.is_none());
        assert!(!shutdown_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn forced_setup_rollback_leaves_registration_for_lease_expiry() {
        assert_unsafe_rollback_leaves_lease_for_expiry(KvEventPublisherShutdownOutcome::Forced)
            .await;
    }

    #[tokio::test]
    async fn timed_out_setup_rollback_leaves_registration_for_lease_expiry() {
        assert_unsafe_rollback_leaves_lease_for_expiry(KvEventPublisherShutdownOutcome::TimedOut)
            .await;
    }
}
