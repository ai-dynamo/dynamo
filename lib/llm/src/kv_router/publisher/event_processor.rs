// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::*;

use crate::kv_router::metrics::kv_publisher_metrics;

use super::DEFAULT_MAX_BATCH_BLOCKS;
use super::batching::BatchingState;
use super::dedup::EventDedupFilter;
use super::sinks::{RouterEventBatchSink, emit};

pub(super) async fn run_event_processor_loop<P: RouterEventBatchSink + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<Vec<PlacementEvent>>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    max_batch_blocks: usize,
) {
    let mut batching_state = BatchingState::new();
    let mut dedup = EventDedupFilter::new();
    let mut last_raw_input_id: Option<u64> = None;

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                let mut output = Vec::new();
                batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                publish_output(&publisher, worker_id, &output).await;
                break;
            }
            event_batch = rx.recv() => {
                let Some(event_batch) = event_batch else {
                    tracing::debug!("Event processor channel closed.");
                    let mut output = Vec::new();
                    batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                    publish_output(&publisher, worker_id, &output).await;
                    break;
                };
                let mut output = Vec::new();

                // Process the complete source list before returning to `select!` so
                // another channel item, the timeout, or cancellation cannot split it.
                for placement_event in event_batch {
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
                    tracing::trace!(
                        "Event processor for worker_id {} processing event: {:?}",
                        worker_id,
                        event.data
                    );

                    let dp_rank_changed =
                        batching_state.has_pending() && event.dp_rank != batching_state.last_dp_rank;
                    let storage_tier_changed = batching_state.has_pending()
                        && storage_tier != batching_state.last_storage_tier;

                    match event.data {
                        KvCacheEventData::Removed(data) => {
                            if batching_state.pending_stored.is_some()
                                || dp_rank_changed
                                || storage_tier_changed
                            {
                                batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
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
                                batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                            }
                            match &mut batching_state.pending_stored {
                                Some(pending) => pending.blocks.extend(data.blocks),
                                None => {
                                    batching_state.pending_stored = Some(data);
                                }
                            }
                        }
                        KvCacheEventData::Cleared => {
                            batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                            dedup.clear();
                            emit(
                                &local_indexer,
                                worker_id,
                                storage_tier,
                                KvCacheEvent {
                                    event_id: batching_state.next_publish_id,
                                    data: KvCacheEventData::Cleared,
                                    dp_rank: event.dp_rank,
                                },
                                &mut output,
                            )
                            .await;
                            batching_state.next_publish_id += 1;
                        }
                    }

                    batching_state.last_dp_rank = event.dp_rank;
                    batching_state.last_storage_tier = storage_tier;

                    // Bound coalesced output without splitting an individual source
                    // event or returning to `select!` midway through the native list.
                    if batching_state.has_pending()
                        && batching_state.pending_block_count() >= max_batch_blocks
                    {
                        batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                    }
                }

                // Without a timeout, flush the compatible tail at the native-list
                // boundary. With a timeout, retain it for possible cross-list batching.
                if batching_state.has_pending()
                    && match timeout_ms {
                        None => true,
                        Some(ms) => batching_state.is_timeout_elapsed(ms),
                    }
                {
                    batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                }
                publish_output(&publisher, worker_id, &output).await;
            }
            _ = tokio::time::sleep(
                timeout_ms
                    .map(|ms| batching_state.remaining_timeout(ms))
                    .unwrap_or(Duration::from_secs(3600))
            ), if timeout_ms.is_some() && batching_state.has_pending() => {
                let mut output = Vec::new();
                batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                publish_output(&publisher, worker_id, &output).await;
            }
        }
    }
}

async fn publish_output<P: RouterEventBatchSink>(
    publisher: &P,
    worker_id: u64,
    output: &[RouterEvent],
) {
    if output.is_empty() {
        return;
    }
    if let Err(e) = publisher.publish_events(output).await {
        tracing::error!(
            worker_id,
            attempted_event_count = output.len(),
            error = %e,
            "One or more KV event publishes failed"
        );
    }
}

pub(super) async fn start_event_processor<P: RouterEventBatchSink + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<Vec<PlacementEvent>>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}
