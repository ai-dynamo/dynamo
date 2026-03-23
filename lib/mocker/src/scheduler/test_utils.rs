// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::anyhow;
use dynamo_kv_router::indexer::{
    KvIndexerInterface, KvIndexerMetrics, LocalKvIndexer, METRIC_EVENT_REMOVED,
    METRIC_EVENT_STORED, METRIC_STATUS_BLOCK_NOT_FOUND, METRIC_STATUS_INVALID_BLOCK,
    METRIC_STATUS_OK, METRIC_STATUS_PARENT_NOT_FOUND,
};
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, LocalBlockHash, RouterEvent, WorkerId, WorkerWithDpRank,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

use super::{DirectRequest, OutputSignal, SchedulerHandle};
use crate::common::protocols::KvCacheEventSink;

pub(crate) struct RouterIndexerHarness {
    indexer: Arc<LocalKvIndexer>,
    metrics: Arc<KvIndexerMetrics>,
    worker: WorkerWithDpRank,
}

impl RouterIndexerHarness {
    pub(crate) fn new(block_size: u32, worker_id: WorkerId) -> Self {
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            block_size,
            metrics.clone(),
            4096,
        ));

        Self {
            indexer,
            metrics,
            worker: WorkerWithDpRank::new(worker_id, 0),
        }
    }

    pub(crate) async fn apply_events<I>(&self, events: I)
    where
        I: IntoIterator<Item = RouterEvent>,
    {
        for event in events {
            self.indexer.apply_event_with_buffer(event).await.unwrap();
        }
        let _ = self.indexer.flush().await;
        self.assert_no_event_errors();
    }

    pub(crate) async fn overlap_for_hashes(&self, local_hashes: Vec<LocalBlockHash>) -> u32 {
        self.indexer
            .find_matches(local_hashes)
            .await
            .unwrap()
            .scores
            .get(&self.worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn ok_count(&self, event_type: &'static str) -> u64 {
        metric_value(&self.metrics, event_type, METRIC_STATUS_OK)
    }

    pub(crate) fn status_count(&self, event_type: &'static str, status: &'static str) -> u64 {
        metric_value(&self.metrics, event_type, status)
    }

    pub(crate) fn invalid_counts(&self) -> Vec<(&'static str, &'static str, u64)> {
        [METRIC_EVENT_STORED, METRIC_EVENT_REMOVED]
            .into_iter()
            .flat_map(|event_type| {
                [
                    METRIC_STATUS_PARENT_NOT_FOUND,
                    METRIC_STATUS_BLOCK_NOT_FOUND,
                    METRIC_STATUS_INVALID_BLOCK,
                ]
                .into_iter()
                .map(move |status| (event_type, status, self.status_count(event_type, status)))
            })
            .collect()
    }

    pub(crate) fn invalid_event_count(&self) -> u64 {
        self.invalid_counts()
            .into_iter()
            .map(|(_, _, count)| count)
            .sum()
    }

    pub(crate) fn spawn_forwarder(&self) -> (Arc<TestKvEventSink>, JoinHandle<()>) {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<RouterEvent>();
        let sink = Arc::new(TestKvEventSink {
            worker_id: self.worker.worker_id,
            event_tx,
        });
        let indexer = self.indexer.clone();
        let forwarder = tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                indexer.apply_event_with_buffer(event).await.unwrap();
            }
            let _ = indexer.flush().await;
        });
        (sink, forwarder)
    }

    pub(crate) async fn flush(&self) {
        let _ = self.indexer.flush().await;
    }

    pub(crate) fn assert_no_event_errors(&self) {
        let breakdown = self
            .invalid_counts()
            .into_iter()
            .filter(|(_, _, count)| *count > 0)
            .map(|(event_type, status, count)| format!("{event_type}/{status}={count}"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(
            self.invalid_event_count(),
            0,
            "router indexer reported invalid KV events{}",
            if breakdown.is_empty() {
                String::new()
            } else {
                format!(": {breakdown}")
            }
        );
    }

    pub(crate) fn shutdown(&self) {
        self.indexer.shutdown();
    }
}

#[derive(Clone)]
pub(crate) struct TestKvEventSink {
    worker_id: WorkerId,
    event_tx: mpsc::UnboundedSender<RouterEvent>,
}

impl KvCacheEventSink for TestKvEventSink {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::new(self.worker_id, event))
            .map_err(|_| anyhow!("router test event channel closed"))
    }
}

pub(crate) fn metric_value(
    metrics: &KvIndexerMetrics,
    event_type: &'static str,
    status: &'static str,
) -> u64 {
    metrics
        .kv_cache_events_applied
        .get_metric_with_label_values(&[event_type, status])
        .unwrap()
        .get()
}

pub(crate) fn stored_hashes(events: &[RouterEvent]) -> Vec<LocalBlockHash> {
    events
        .iter()
        .filter_map(|event| match &event.event.data {
            KvCacheEventData::Stored(store) => Some(
                store
                    .blocks
                    .iter()
                    .map(|block| block.tokens_hash)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

pub(crate) fn nth_stored_hashes(events: &[RouterEvent], nth: usize) -> Vec<LocalBlockHash> {
    events
        .iter()
        .filter_map(|event| match &event.event.data {
            KvCacheEventData::Stored(store) => Some(
                store
                    .blocks
                    .iter()
                    .map(|block| block.tokens_hash)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .nth(nth)
        .unwrap_or_default()
}

pub(crate) fn removed_event_count(events: &[RouterEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event.event.data, KvCacheEventData::Removed(_)))
        .count()
}

/// Send `num_requests` to a scheduler, collect all output signals, and assert
/// that the scheduler produces exactly `num_requests * max_output_tokens` signals
/// and returns to idle (0 active decode blocks).
///
/// When `use_shared_tokens` is true, the first half of each request shares a
/// common prefix to exercise prefix caching / radix tree reuse.
pub(crate) async fn assert_scheduler_completes_all(
    scheduler: &dyn SchedulerHandle,
    output_rx: &mut mpsc::UnboundedReceiver<OutputSignal>,
    num_requests: usize,
    input_len: usize,
    max_output_tokens: usize,
    use_shared_tokens: bool,
) {
    let shared_tokens = if use_shared_tokens {
        Some(
            (0..input_len / 2)
                .map(|_| rand::random::<u32>() % 50000)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    for _ in 0..num_requests {
        let input_tokens = if let Some(ref shared) = shared_tokens {
            let mut tokens = shared.clone();
            tokens.extend((0..input_len / 2).map(|_| rand::random::<u32>() % 50000));
            tokens
        } else {
            (0..input_len)
                .map(|_| rand::random::<u32>() % 50000)
                .collect::<Vec<_>>()
        };

        scheduler.receive(DirectRequest {
            tokens: input_tokens,
            max_output_tokens,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms: None,
        });
    }

    let expected_tokens = num_requests * max_output_tokens;
    let mut received_tokens = 0;

    let timeout = tokio::time::sleep(Duration::from_secs(2));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            biased;
            Some(_) = output_rx.recv() => {
                received_tokens += 1;
                if received_tokens >= expected_tokens {
                    break;
                }
                timeout.set(tokio::time::sleep(Duration::from_secs(2)));
            }
            _ = &mut timeout => break,
        }
    }

    assert_eq!(
        received_tokens, expected_tokens,
        "Expected {expected_tokens} output signals, got {received_tokens}"
    );

    tokio::time::sleep(Duration::from_millis(100)).await;
    let metrics = scheduler.metrics_receiver().borrow().clone();
    assert_eq!(
        metrics.active_decode_blocks, 0,
        "Scheduler should be idle after all requests complete, got {} active blocks",
        metrics.active_decode_blocks
    );
    assert_eq!(
        metrics.gpu_cache_usage_perc, 0.0,
        "Scheduler should report zero cache usage after draining, got {}",
        metrics.gpu_cache_usage_perc
    );
    assert!(
        metrics.total_blocks > 0,
        "Scheduler should populate total_blocks, got {}",
        metrics.total_blocks
    );
}
