// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::*;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::metrics::kv_publisher_metrics;

use super::DEFAULT_MAX_BATCH_BLOCKS;
use super::batching::BatchingState;
use super::dedup::EventDedupFilter;
use super::sinks::{IntegrityState, JetStreamPublisher, ValkeyEventPublisher, emit};
use super::{PlacementEventReceiver, PublisherInput};

async fn wait_for_integrity_signal(recovery: Option<&ValkeyEventPublisher>) {
    match recovery {
        Some(recovery) => recovery.integrity().wait_for_state_change().await,
        None => std::future::pending().await,
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
    run_event_processor_loop_inner(
        publisher,
        None,
        worker_id,
        cancellation_token,
        PlacementEventReceiver::Unbounded(rx),
        local_indexer,
        timeout_ms,
        max_batch_blocks,
    )
    .await;
}

#[expect(clippy::too_many_arguments)]
async fn run_event_processor_loop_inner<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    recovery: Option<ValkeyEventPublisher>,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: PlacementEventReceiver,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    max_batch_blocks: usize,
) {
    let mut batching_state = BatchingState::new();
    let mut dedup = EventDedupFilter::new();
    let mut last_raw_input_id: Option<u64> = None;
    let mut draining = false;

    loop {
        if let Some(recovery) = &recovery
            && recovery.integrity().state() != IntegrityState::Healthy
        {
            // Nothing accepted before an integrity fault may be published.
            // The process stays fenced, so discard all bounded old ingress.
            batching_state = BatchingState::new();
            dedup.clear();
            last_raw_input_id = None;
            while rx.try_recv().is_ok() {
                if let Some(metrics) = kv_publisher_metrics() {
                    metrics.increment_input_dropped_event("integrity_fenced_queue");
                }
            }

            if draining || cancellation_token.is_cancelled() {
                rx.close();
                tracing::warn!(
                    worker_id,
                    "Stopping a faulted direct-Valkey publisher during bounded shutdown; worker unregister/lease expiry remains the final fence"
                );
                break;
            }

            match recovery.integrity().state() {
                IntegrityState::Faulted => {
                    tokio::select! {
                        _ = cancellation_token.cancelled() => continue,
                        _ = recovery.integrity().fence_once() => {}
                    }
                }
                IntegrityState::Fencing | IntegrityState::Fenced => {
                    // Another DP-rank owns the best-effort unregister, or it
                    // has finished. Remain fenced until process shutdown.
                    tokio::select! {
                        _ = cancellation_token.cancelled() => {}
                        _ = recovery.integrity().wait_for_state_change() => {}
                        input = rx.recv() => {
                            if input.is_some() {
                                if let Some(metrics) = kv_publisher_metrics() {
                                    metrics.increment_input_dropped_event("integrity_fenced_queue");
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
                IntegrityState::Healthy => {}
            }
            continue;
        }

        tokio::select! {
            _ = wait_for_integrity_signal(recovery.as_ref()), if recovery.is_some() => {
                continue;
            }
            _ = cancellation_token.cancelled(), if !draining => {
                tracing::info!("KV Event source received cancellation signal; draining queued events");
                // Reject new events while preserving every event accepted before
                // shutdown. The owner lease remains live until this loop exits.
                rx.close();
                draining = true;
            }
            input = rx.recv() => {
                let Some(PublisherInput { event: placement_event }) = input else {
                    tracing::debug!("Event processor channel closed.");
                    batching_state.flush(&publisher, &local_indexer, worker_id, &mut dedup).await;
                    break;
                };

                if let Some(recovery) = &recovery
                    && recovery.integrity().state() != IntegrityState::Healthy
                {
                    if let Some(metrics) = kv_publisher_metrics() {
                        metrics.increment_input_dropped_event("integrity_fenced_queue");
                    }
                    continue;
                }

                let raw_event_id = placement_event.event.event_id;
                if let Some(last_id) = last_raw_input_id
                    && raw_event_id > last_id.saturating_add(1)
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
                    if let Some(recovery) = &recovery {
                        recovery.integrity().mark_fault("raw_event_gap");
                        if let Some(metrics) = kv_publisher_metrics() {
                            metrics.increment_input_dropped_event("raw_event_gap");
                        }
                        continue;
                    }
                }
                if let Some(last_id) = last_raw_input_id
                    && raw_event_id <= last_id
                    && let Some(recovery) = &recovery
                {
                    tracing::error!(
                        worker_id,
                        last_raw_input_id = last_id,
                        raw_event_id,
                        "Input event id regressed or repeated; fencing direct-Valkey metadata"
                    );
                    recovery.integrity().mark_fault("raw_event_regression");
                    if let Some(metrics) = kv_publisher_metrics() {
                        metrics.increment_input_dropped_event("raw_event_regression");
                    }
                    continue;
                }
                last_raw_input_id = Some(raw_event_id);

                let storage_tier = placement_event.placement.tier;
                let event = placement_event.event;
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

    // Direct Valkey publication returns after bounded queue acceptance so
    // rank processors can fill worker-sized pipelines. Do not let graceful
    // shutdown unregister the owner lease until every accepted normalized
    // event has committed or the shared integrity domain has fenced it.
    if let Some(recovery) = &recovery {
        recovery.integrity().wait_for_idle().await;
    }
}

pub(super) async fn start_event_processor<P: RouterEventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: impl Into<PlacementEventReceiver>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop_inner(
        publisher,
        None,
        worker_id,
        cancellation_token,
        rx.into(),
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

pub(super) async fn start_direct_valkey_event_processor<
    P: RouterEventSink + Send + Sync + 'static,
>(
    publisher: P,
    recovery: ValkeyEventPublisher,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: PlacementEventReceiver,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop_inner(
        publisher,
        Some(recovery),
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

pub(super) async fn start_event_processor_jetstream(
    publisher: NatsQueue,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: impl Into<PlacementEventReceiver>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop_inner(
        JetStreamPublisher(publisher),
        None,
        worker_id,
        cancellation_token,
        rx.into(),
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}
