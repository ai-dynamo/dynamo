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

use crate::engine::{ComponentMetricsPublisher, ComponentMetricsSource, KvEventSource};
use crate::error::{BackendError, DynamoError, ErrorType};

const METRICS_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Live publisher handles owned by `Worker` for the lifetime of serving.
/// `kv_publishers` and `metrics_publishers` are kept alive solely so their
/// `Drop` impls run on shutdown. `component_publisher` is reached from
/// `Worker::cleanup_once` / `Worker::drain` to record lifecycle gauges.
pub(crate) struct PublisherHandles {
    #[allow(dead_code)]
    kv_publishers: Vec<Arc<KvEventPublisher>>,
    #[allow(dead_code)]
    metrics_publishers: Vec<Arc<WorkerMetricsPublisher>>,
    pub(crate) component_publisher: Option<Arc<dyn ComponentMetricsPublisher>>,
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
    component_publisher: Option<Arc<dyn ComponentMetricsPublisher>>,
    cancel: CancellationToken,
) -> Result<(Vec<Arc<WorkerMetricsPublisher>>, Option<JoinHandle<()>>), DynamoError> {
    let Some(component_publisher) = component_publisher else {
        return Ok((Vec::new(), None));
    };
    let sources = component_publisher.sources();
    if sources.is_empty() {
        return Ok((Vec::new(), None));
    }
    let mut pairs = Vec::with_capacity(sources.len());
    let mut publishers = Vec::with_capacity(sources.len());
    for source in sources {
        let dp_rank = source.dp_rank;
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
        pairs.push((publisher.clone(), source));
        publishers.push(publisher);
    }
    // `secondary()` so the fixed-interval poll loop doesn't share the
    // primary runtime with request-handling tasks.
    let task = component
        .drt()
        .runtime()
        .secondary()
        .spawn(run_metrics_loop(pairs, component_publisher, cancel));
    Ok((publishers, Some(task)))
}

/// One task per worker drives every rank's snapshot per tick. For Python
/// engines this serializes GIL acquisition through one OS thread instead of
/// fanning out N contending tokio tasks at the snapshot cadence.
async fn run_metrics_loop(
    pairs: Vec<(Arc<WorkerMetricsPublisher>, ComponentMetricsSource)>,
    component_publisher: Arc<dyn ComponentMetricsPublisher>,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(METRICS_POLL_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut warned_ranks: Vec<u32> = Vec::new();
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = ticker.tick() => publish_tick(&pairs, &component_publisher, &mut warned_ranks),
        }
    }
}

fn publish_tick(
    pairs: &[(Arc<WorkerMetricsPublisher>, ComponentMetricsSource)],
    component_publisher: &Arc<dyn ComponentMetricsPublisher>,
    warned_ranks: &mut Vec<u32>,
) {
    for (router_publisher, source) in pairs {
        let Some(snapshot) = (source.snapshot)() else {
            continue;
        };
        // Single source of truth feeds both consumers: router-input signal
        // for the KV router, and `dynamo_component_*` gauges for /metrics.
        component_publisher.update(&snapshot);
        if let Err(e) = router_publisher.publish(
            Some(source.dp_rank),
            None,
            Some(snapshot.kv_used_blocks),
        ) {
            if !warned_ranks.contains(&source.dp_rank) {
                warned_ranks.push(source.dp_rank);
                tracing::warn!(dp_rank = source.dp_rank, error = %e,
                    "metrics publish failed; suppressing further warnings");
            } else {
                tracing::debug!(dp_rank = source.dp_rank, error = %e, "metrics publish failed");
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
    component_publisher: Option<Arc<dyn ComponentMetricsPublisher>>,
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
    let component_publisher_for_handles = component_publisher.clone();
    let (metrics_publishers, metrics_task) =
        setup_metrics_publishers(component, component_publisher, cancel.clone()).await?;
    Ok(PublisherHandles {
        kv_publishers,
        metrics_publishers,
        component_publisher: component_publisher_for_handles,
        metrics_task,
        cancel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::engine::ComponentSnapshot;

    /// Recording publisher: snapshot fn pulls from `state`, `update` records.
    struct RecordingPublisher {
        sources: Vec<ComponentMetricsSource>,
        updates: Arc<Mutex<Vec<ComponentSnapshot>>>,
        cleanup_time: Arc<Mutex<Option<f64>>>,
        drain_time: Arc<Mutex<Option<f64>>>,
    }

    impl ComponentMetricsPublisher for RecordingPublisher {
        fn sources(&self) -> Vec<ComponentMetricsSource> {
            self.sources.clone()
        }
        fn update(&self, snapshot: &ComponentSnapshot) {
            self.updates.lock().unwrap().push(*snapshot);
        }
        fn set_cleanup_time(&self, seconds: f64) {
            *self.cleanup_time.lock().unwrap() = Some(seconds);
        }
        fn set_drain_time(&self, seconds: f64) {
            *self.drain_time.lock().unwrap() = Some(seconds);
        }
    }

    fn snapshot_source(
        dp_rank: u32,
        state: Arc<Mutex<Option<ComponentSnapshot>>>,
    ) -> ComponentMetricsSource {
        ComponentMetricsSource {
            snapshot: Arc::new(move || *state.lock().unwrap()),
            dp_rank,
        }
    }

    /// `publish_tick` drives both consumers from the same snapshot: the
    /// router-input `WorkerMetricsPublisher` AND the component publisher's
    /// `update`. Verifies the "single source of truth" guarantee.
    #[test]
    fn publish_tick_feeds_both_consumers() {
        let state = Arc::new(Mutex::new(Some(ComponentSnapshot {
            kv_used_blocks: 42,
            kv_total_blocks: 100,
            gpu_cache_usage: 0.42,
            kv_cache_hit_rate: Some(0.85),
            dp_rank: 3,
        })));
        let source = snapshot_source(3, state);
        let router_pub = Arc::new(WorkerMetricsPublisher::new().expect("wmp::new"));
        let updates = Arc::new(Mutex::new(Vec::<ComponentSnapshot>::new()));
        let component_pub: Arc<dyn ComponentMetricsPublisher> = Arc::new(RecordingPublisher {
            sources: vec![source.clone()],
            updates: updates.clone(),
            cleanup_time: Arc::new(Mutex::new(None)),
            drain_time: Arc::new(Mutex::new(None)),
        });

        let pairs = vec![(router_pub.clone(), source)];
        let mut warned: Vec<u32> = Vec::new();
        publish_tick(&pairs, &component_pub, &mut warned);

        // Component publisher saw the snapshot exactly once.
        let recorded = updates.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].kv_used_blocks, 42);
        assert_eq!(recorded[0].kv_total_blocks, 100);
        assert_eq!(recorded[0].dp_rank, 3);
        // No warnings — router publish succeeded.
        assert!(warned.is_empty());
    }

    /// Snapshot returning `None` is the engine signalling "no data yet."
    /// `publish_tick` must skip both sides — no spurious gauge update, no
    /// router publish with stale data.
    #[test]
    fn publish_tick_skips_none_snapshot() {
        let state: Arc<Mutex<Option<ComponentSnapshot>>> = Arc::new(Mutex::new(None));
        let source = snapshot_source(0, state);
        let router_pub = Arc::new(WorkerMetricsPublisher::new().expect("wmp::new"));
        let updates = Arc::new(Mutex::new(Vec::<ComponentSnapshot>::new()));
        let component_pub: Arc<dyn ComponentMetricsPublisher> = Arc::new(RecordingPublisher {
            sources: vec![source.clone()],
            updates: updates.clone(),
            cleanup_time: Arc::new(Mutex::new(None)),
            drain_time: Arc::new(Mutex::new(None)),
        });

        let pairs = vec![(router_pub, source)];
        let mut warned: Vec<u32> = Vec::new();
        publish_tick(&pairs, &component_pub, &mut warned);

        assert!(updates.lock().unwrap().is_empty(), "no update on None");
        assert!(warned.is_empty());
    }

    /// Default trait impls are no-op; `RecordingPublisher` overrides them so
    /// `Worker::cleanup_once` / drain can hand the elapsed time to whoever
    /// owns the gauges. Both setters are independent: setting one doesn't
    /// touch the other.
    #[test]
    fn lifecycle_setters_record_independently() {
        let publisher = RecordingPublisher {
            sources: vec![],
            updates: Arc::new(Mutex::new(Vec::new())),
            cleanup_time: Arc::new(Mutex::new(None)),
            drain_time: Arc::new(Mutex::new(None)),
        };
        publisher.set_drain_time(0.25);
        assert_eq!(*publisher.drain_time.lock().unwrap(), Some(0.25));
        assert_eq!(*publisher.cleanup_time.lock().unwrap(), None);
        publisher.set_cleanup_time(1.5);
        assert_eq!(*publisher.cleanup_time.lock().unwrap(), Some(1.5));
    }

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
            component_publisher: None,
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
            component_publisher: None,
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
