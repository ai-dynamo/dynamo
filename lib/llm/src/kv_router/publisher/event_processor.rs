// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::*;
use dynamo_runtime::transports::nats::NatsQueue;

#[cfg(test)]
use crate::kv_router::metrics::KV_PUBLISHER_EVENT_SOURCE_EVENT_PLANE;
use crate::kv_router::metrics::{
    KV_PUBLISHER_EVENT_STAGE_ACCEPTED, KV_PUBLISHER_EVENT_STAGE_RECEIVED,
    kv_cache_event_type_label, kv_publisher_metrics,
};

use super::DEFAULT_MAX_BATCH_BLOCKS;
use super::batching::BatchingState;
use super::dedup::EventDedupFilter;
use super::sinks::{JetStreamPublisher, emit};

fn increment_event_metric(
    metric_source: Option<&'static str>,
    stage: &'static str,
    event_type: &'static str,
) {
    if let Some(source) = metric_source
        && let Some(metrics) = kv_publisher_metrics()
    {
        metrics.increment_event(stage, event_type, source);
    }
}

async fn run_event_processor_loop_with_metric_source<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    metric_source: Option<&'static str>,
    max_batch_blocks: usize,
) {
    let mut batching_state = BatchingState::new();
    let mut dedup = EventDedupFilter::new();
    let mut last_raw_input_id: Option<u64> = None;

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                break;
            }
            event = rx.recv() => {
                let Some(placement_event) = event else {
                    tracing::debug!("Event processor channel closed.");
                    batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                    break;
                };

                let raw_event_id = placement_event.event.event_id;
                if let Some(last_id) = last_raw_input_id
                    && raw_event_id > last_id + 1
                {
                    let gap = raw_event_id - last_id - 1;
                    tracing::warn!(
                        worker_id,
                        last_raw_input_id = last_id,
                        raw_event_id,
                        gap,
                        "Input event gap detected: raw events dropped before batching"
                    );
                    if let Some(metrics) = kv_publisher_metrics() {
                        metrics.increment_engines_dropped_events(gap);
                    } else {
                        tracing::warn!(
                            worker_id,
                            gap,
                            "Failed to record dropped events metric: metrics not initialized"
                        );
                    }
                }
                last_raw_input_id = Some(raw_event_id);

                let storage_tier = placement_event.placement.tier;
                let event = placement_event.event;
                let event_type = kv_cache_event_type_label(&event.data);
                increment_event_metric(
                    metric_source,
                    KV_PUBLISHER_EVENT_STAGE_RECEIVED,
                    event_type,
                );
                tracing::trace!(
                    "Event processor for worker_id {} processing event: {:?}",
                    worker_id,
                    event.data
                );

                let dp_rank_changed =
                    batching_state.has_pending() && event.dp_rank != batching_state.last_dp_rank;
                let storage_tier_changed =
                    batching_state.has_pending() && storage_tier != batching_state.last_storage_tier;

                match event.data {
                    KvCacheEventData::Removed(data) => {
                        if batching_state.pending_stored.is_some()
                            || dp_rank_changed
                            || storage_tier_changed
                        {
                            batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        }
                        match &mut batching_state.pending_removed {
                            Some(pending) => pending.block_hashes.extend(data.block_hashes),
                            None => {
                                batching_state.pending_removed = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Stored(data) => {
                        let should_flush = dp_rank_changed
                            || storage_tier_changed
                            || batching_state.pending_removed.is_some()
                            || batching_state.pending_stored.as_ref().is_some_and(|p| {
                                data.parent_hash != p.blocks.last().map(|b| b.block_hash)
                            });
                        if should_flush {
                            batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        }
                        match &mut batching_state.pending_stored {
                            Some(pending) => pending.blocks.extend(data.blocks),
                            None => {
                                batching_state.pending_stored = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Cleared => {
                        batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                        dedup.clear();
                        emit(
                            &publisher,
                            &local_indexer,
                            worker_id,
                            storage_tier,
                            KvCacheEvent {
                                event_id: batching_state.next_publish_id,
                                data: KvCacheEventData::Cleared,
                                dp_rank: event.dp_rank,
                            },
                        )
                        .await;
                        batching_state.next_publish_id += 1;
                    }
                }

                increment_event_metric(
                    metric_source,
                    KV_PUBLISHER_EVENT_STAGE_ACCEPTED,
                    event_type,
                );
                batching_state.last_dp_rank = event.dp_rank;
                batching_state.last_storage_tier = storage_tier;

                if batching_state.has_pending()
                    && (timeout_ms.is_none_or(|ms| batching_state.is_timeout_elapsed(ms))
                        || batching_state.pending_block_count() > max_batch_blocks)
                {
                    batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                }
            }
            _ = tokio::time::sleep(
                timeout_ms
                    .map(|ms| batching_state.remaining_timeout(ms))
                    .unwrap_or(Duration::from_secs(3600))
            ), if timeout_ms.is_some() && batching_state.has_pending() => {
                batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
            }
        }
    }
}

#[cfg(test)]
pub(super) async fn run_event_processor_loop<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    max_batch_blocks: usize,
) {
    run_event_processor_loop_with_metric_source(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        timeout_ms,
        Some(KV_PUBLISHER_EVENT_SOURCE_EVENT_PLANE),
        max_batch_blocks,
    )
    .await
}

#[cfg(test)]
pub(super) async fn start_event_processor<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    start_event_processor_with_metric_source(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        Some(KV_PUBLISHER_EVENT_SOURCE_EVENT_PLANE),
    )
    .await
}

pub(super) async fn start_event_processor_with_metric_source<
    P: RouterEventSink + Send + Sync + 'static,
>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
    metric_source: Option<&'static str>,
) {
    run_event_processor_loop_with_metric_source(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        metric_source,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

pub(super) async fn start_event_processor_jetstream_with_metric_source(
    publisher: NatsQueue,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<PlacementEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
    metric_source: Option<&'static str>,
) {
    run_event_processor_loop_with_metric_source(
        JetStreamPublisher(publisher),
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        metric_source,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}
