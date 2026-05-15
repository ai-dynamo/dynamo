// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side wiring for KV-aware-routing publishers.

use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::kv_router::publisher::{
    KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher,
};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::engine::{KvEventSource, Metrics, MetricsSource};
use crate::error::{BackendError, DynamoError, ErrorType};

const METRICS_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Live publisher handles owned by `Worker` for the lifetime of serving.
/// `kv_publishers` and `metrics_publishers` are kept alive solely so their
/// `Drop` impls run on shutdown.
pub(crate) struct PublisherHandles {
    #[allow(dead_code)]
    kv_publishers: Vec<Arc<KvEventPublisher>>,
    #[allow(dead_code)]
    metrics_publishers: Vec<Arc<WorkerMetricsPublisher>>,
    metrics_task: Option<JoinHandle<()>>,
    cancel: CancellationToken,
}

impl PublisherHandles {
    pub(crate) async fn shutdown(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.metrics_task.take()
            && let Err(e) = task.await
        {
            tracing::warn!(error = ?e, "metrics task panicked or was aborted");
        }
    }
}

// Sync — `KvEventPublisher::new_with_local_indexer` doesn't await. The
// metrics counterpart below is async because `create_endpoint` does.
fn setup_kv_publishers(
    component: &Component,
    sources: Vec<KvEventSource>,
    kv_cache_block_size: u32,
    enable_local_indexer: bool,
) -> Result<Vec<Arc<KvEventPublisher>>, DynamoError> {
    let mut publishers = Vec::with_capacity(sources.len());
    for source in sources {
        let dp_rank = source.dp_rank();
        let (source_config, on_ready) = match source {
            KvEventSource::Zmq {
                endpoint, topic, ..
            } => (Some(KvEventSourceConfig::Zmq { endpoint, topic }), None),
            KvEventSource::Push { on_ready, .. } => (None, Some(on_ready)),
        };
        let publisher = KvEventPublisher::new_with_local_indexer(
            component.clone(),
            kv_cache_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            None,
        )
        .map_err(|e| publisher_err(format!("kv publisher setup (dp_rank={dp_rank}): {e}")))?;
        let publisher = Arc::new(publisher);
        if let Some(on_ready) = on_ready {
            // Partial-success: engines whose on_ready ran before this failure
            // have already started threads. The unwind path runs
            // `engine.cleanup` (see `Worker::cleanup_once`), which is the
            // sole hook for joining them.
            on_ready(publisher.clone()).map_err(|e| {
                publisher_err(format!("kv publisher on_ready (dp_rank={dp_rank}): {e}"))
            })?;
        }
        publishers.push(publisher);
    }
    Ok(publishers)
}

async fn setup_metrics_publishers(
    component: &Component,
    snapshots: Vec<MetricsSource>,
    cancel: CancellationToken,
) -> Result<(Vec<Arc<WorkerMetricsPublisher>>, Option<JoinHandle<()>>), DynamoError> {
    if snapshots.is_empty() {
        return Ok((Vec::new(), None));
    }
    let mut pairs = Vec::with_capacity(snapshots.len());
    let mut publishers = Vec::with_capacity(snapshots.len());
    for snap in snapshots {
        let dp_rank = snap.dp_rank;
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
        let publisher = Arc::new(publisher);
        pairs.push((publisher.clone(), snap));
        publishers.push(publisher);
    }
    // `secondary()` so the fixed-interval poll loop doesn't share the
    // primary runtime with request-handling tasks.
    let task = component
        .drt()
        .runtime()
        .secondary()
        .spawn(run_metrics_loop(pairs, cancel));
    Ok((publishers, Some(task)))
}

/// One task per worker drives every rank's snapshot per tick. For Python
/// engines this serializes GIL acquisition through one OS thread instead of
/// fanning out N contending tokio tasks at the snapshot cadence.
async fn run_metrics_loop(
    pairs: Vec<(Arc<WorkerMetricsPublisher>, MetricsSource)>,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(METRICS_POLL_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut warned_ranks: Vec<u32> = Vec::new();
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = ticker.tick() => publish_tick(&pairs, &mut warned_ranks),
        }
    }
}

fn publish_tick(
    pairs: &[(Arc<WorkerMetricsPublisher>, MetricsSource)],
    warned_ranks: &mut Vec<u32>,
) {
    for (publisher, snap) in pairs {
        let Some(Metrics {
            kv_used_blocks: Some(kv_used_blocks),
        }) = (snap.snapshot)()
        else {
            continue;
        };
        if let Err(e) = publisher.publish(Some(snap.dp_rank), None, Some(kv_used_blocks)) {
            if !warned_ranks.contains(&snap.dp_rank) {
                warned_ranks.push(snap.dp_rank);
                tracing::warn!(dp_rank = snap.dp_rank, error = %e,
                    "metrics publish failed; suppressing further warnings");
            } else {
                tracing::debug!(dp_rank = snap.dp_rank, error = %e, "metrics publish failed");
            }
        }
    }
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
    kv_sources: Vec<KvEventSource>,
    metrics_snapshots: Vec<MetricsSource>,
    kv_cache_block_size: Option<u32>,
    enable_local_indexer: bool,
) -> Result<PublisherHandles, DynamoError> {
    // KV event publishers require the engine's block size; without it, the
    // router can't translate token IDs into cache blocks. Metrics publishers
    // are independent — load reporting works regardless of cache structure.
    let kv_publishers = if let Some(block_size) = kv_cache_block_size {
        setup_kv_publishers(component, kv_sources, block_size, enable_local_indexer)?
    } else {
        if !kv_sources.is_empty() {
            tracing::warn!(
                "engine declared {} kv_event_sources but kv_cache_block_size is None; skipping KV event publishers",
                kv_sources.len()
            );
        }
        Vec::new()
    };
    let cancel = CancellationToken::new();
    let (metrics_publishers, metrics_task) =
        setup_metrics_publishers(component, metrics_snapshots, cancel.clone()).await?;
    Ok(PublisherHandles {
        kv_publishers,
        metrics_publishers,
        metrics_task,
        cancel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// `PublisherHandles::shutdown` cancels the shared token and awaits the
    /// metrics task. Drives one synthetic task that loops on the cancel
    /// token; verifying it exits proves the cancellation contract end-to-end.
    #[tokio::test]
    async fn publisher_handles_shutdown_cancels_metric_task() {
        let cancel = CancellationToken::new();
        let observed_cancel = Arc::new(AtomicU64::new(0));
        let counter = observed_cancel.clone();
        let token = cancel.clone();
        let task = tokio::spawn(async move {
            tokio::select! {
                _ = token.cancelled() => {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
                _ = tokio::time::sleep(Duration::from_secs(5)) => {}
            }
        });
        let mut handles = PublisherHandles {
            kv_publishers: Vec::new(),
            metrics_publishers: Vec::new(),
            metrics_task: Some(task),
            cancel,
        };
        handles.shutdown().await;
        assert_eq!(
            observed_cancel.load(Ordering::SeqCst),
            1,
            "metric task must observe cancellation during shutdown"
        );
    }

    /// First `shutdown` takes the task and joins it; second is a no-op.
    #[tokio::test]
    async fn publisher_handles_shutdown_is_idempotent() {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let task = tokio::spawn(async move { token.cancelled().await });
        let mut handles = PublisherHandles {
            kv_publishers: Vec::new(),
            metrics_publishers: Vec::new(),
            metrics_task: Some(task),
            cancel,
        };
        handles.shutdown().await;
        assert!(
            handles.metrics_task.is_none(),
            "first shutdown takes the task"
        );
        assert!(handles.cancel.is_cancelled());
        handles.shutdown().await;
    }
}
