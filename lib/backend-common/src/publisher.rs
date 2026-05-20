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
use std::sync::Arc;

use dynamo_llm::kv_router::publisher::{
    KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher,
};
use dynamo_runtime::component::Component;

use crate::engine::KvEventSource;
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::metrics::{ComponentGauges, EngineMetrics};
use crate::snapshot_publisher::SnapshotPublisher;

/// Live publisher handles owned by `Worker` for the lifetime of serving.
/// All variants are kept alive solely so their `Drop` impls run on
/// shutdown — there is no background task to join.
pub(crate) struct PublisherHandles {
    #[allow(dead_code)]
    kv_publishers: Vec<Arc<KvEventPublisher>>,
    /// Stashed so the engine's `Arc<SnapshotPublisher>` reference stays
    /// valid for the worker's lifetime. Engines drop their copy when
    /// they shut down; we keep ours so the `WorkerMetricsPublisher`s
    /// inside don't drop their NATS endpoints prematurely.
    #[allow(dead_code)]
    snapshot_publisher: Option<Arc<SnapshotPublisher>>,
}

// Sync — `KvEventPublisher::new_with_local_indexer` doesn't await. The
// snapshot router-publisher construction below is async because
// `create_endpoint` does.
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
    kv_sources: Vec<KvEventSource>,
    dp_ranks: Vec<u32>,
    on_publisher_ready: Option<crate::engine::OnSnapshotPublisherReady>,
    kv_cache_block_size: Option<u32>,
    enable_local_indexer: bool,
) -> Result<PublisherHandles, DynamoError> {
    // KV event publishers require the engine's block size; without it, the
    // router can't translate token IDs into cache blocks. Snapshot publisher
    // is independent — load reporting works regardless of cache structure.
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

    let snapshot_publisher = if dp_ranks.is_empty() {
        None
    } else {
        let router_publishers = build_router_publishers(component, &dp_ranks).await?;
        let gauges = Arc::new(ComponentGauges::new(engine_metrics)?);
        let publisher = Arc::new(SnapshotPublisher::new(gauges, router_publishers));
        if let Some(on_ready) = on_publisher_ready {
            on_ready(publisher.clone())
                .map_err(|e| publisher_err(format!("snapshot publisher on_ready: {e}")))?;
        }
        Some(publisher)
    };

    Ok(PublisherHandles {
        kv_publishers,
        snapshot_publisher,
    })
}
