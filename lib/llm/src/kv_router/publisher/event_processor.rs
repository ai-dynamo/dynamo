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
    let mut batching_state = BatchingState::new(max_batch_blocks);
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
                'event_batch: for placement_event in event_batch {
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
                    let residency_domain = placement_event.placement.residency_domain;
                    tracing::trace!(
                        "Event processor for worker_id {} processing event: {:?}",
                        worker_id,
                        placement_event.event.data
                    );

                    match &placement_event.event.data {
                        KvCacheEventData::Removed(_) | KvCacheEventData::Stored(_) => {
                            batching_state
                                .push(
                                    placement_event,
                                    &local_indexer,
                                    worker_id,
                                    &mut dedup,
                                    &mut output,
                                )
                                .await;
                        }
                        KvCacheEventData::Cleared => {
                            batching_state.flush(&local_indexer, worker_id, &mut dedup, &mut output).await;
                            let event = placement_event.event;
                            dedup.clear_rank_domain(event.dp_rank, residency_domain);
                            let applied = emit(
                                &local_indexer,
                                worker_id,
                                storage_tier,
                                residency_domain,
                                KvCacheEvent {
                                    event_id: batching_state.next_publish_id,
                                    data: KvCacheEventData::Cleared,
                                    dp_rank: event.dp_rank,
                                },
                                &mut output,
                            )
                            .await;
                            if !applied {
                                output.pop();
                                tracing::error!(
                                    worker_id,
                                    dp_rank = event.dp_rank,
                                    ?residency_domain,
                                    "Fencing KV event publisher after local reset failure"
                                );
                                // NOTE: This token owns the publisher's discovery and recovery
                                // advertisement too. Once the local reset barrier fails, withdrawing
                                // the complete source is the only safe way to avoid advertising a
                                // cursor whose local snapshot has diverged.
                                cancellation_token.cancel();
                                break 'event_batch;
                            }
                            batching_state.next_publish_id = batching_state
                                .next_publish_id
                                .checked_add(1)
                                .expect("KV event publisher outbound cursor exhausted");
                        }
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
