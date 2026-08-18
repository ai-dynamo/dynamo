// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

const METRICS_REPLAY_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<BTreeMap<DpRank, WorkerMetrics>>,
    rx: tokio::sync::watch::Receiver<BTreeMap<DpRank, WorkerMetrics>>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(BTreeMap::new());
        Ok(Self { tx, rx })
    }

    pub fn publish(
        &self,
        dp_rank: Option<DpRank>,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
    ) -> Result<()> {
        if active_decode_blocks.is_none() && kv_used_blocks.is_none() {
            anyhow::bail!("worker metrics publish requires at least one load metric");
        }

        let metrics = WorkerMetrics {
            dp_rank: dp_rank.unwrap_or(0),
            active_decode_blocks,
            kv_used_blocks,
        };
        tracing::trace!(
            "Publish metrics: dp_rank={}, active_decode_blocks={:?}, kv_used_blocks={:?}",
            metrics.dp_rank,
            metrics.active_decode_blocks,
            metrics.kv_used_blocks
        );
        self.tx.send_modify(|current| {
            current.insert(metrics.dp_rank, metrics);
        });
        Ok(())
    }

    pub async fn create_endpoint(&self, endpoint: Endpoint) -> Result<()> {
        let worker_id = endpoint.drt().connection_id();
        let event_publisher = EventPublisher::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await?;
        self.start_metrics_publishing(event_publisher, worker_id);
        Ok(())
    }

    pub(super) fn start_metrics_publishing(&self, event_publisher: EventPublisher, worker_id: u64) {
        let metrics_rx = self.rx.clone();

        tokio::spawn(async move {
            let mut rx = metrics_rx;
            let mut current_metrics = rx.borrow_and_update().clone();
            let mut pending_publish = current_metrics.clone();
            let publish_timer = tokio::time::sleep(tokio::time::Duration::ZERO);
            tokio::pin!(publish_timer);
            let mut replay = tokio::time::interval_at(
                tokio::time::Instant::now() + METRICS_REPLAY_INTERVAL,
                METRICS_REPLAY_INTERVAL,
            );
            replay.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping event-plane background task"
                            );
                            break;
                        }

                        let latest = rx.borrow_and_update().clone();
                        let start_publish_timer = pending_publish.is_empty();
                        for (dp_rank, metrics) in latest {
                            if current_metrics.get(&dp_rank) == Some(&metrics) {
                                continue;
                            }
                            current_metrics.insert(dp_rank, metrics.clone());
                            pending_publish.insert(dp_rank, metrics);
                        }
                        if start_publish_timer && !pending_publish.is_empty() {
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now()
                                    + tokio::time::Duration::from_millis(1)
                            );
                        }
                    }
                    _ = &mut publish_timer, if !pending_publish.is_empty() => {
                        for metrics in std::mem::take(&mut pending_publish).into_values() {
                            publish_metrics(&event_publisher, worker_id, &metrics).await;
                        }
                    }
                    _ = replay.tick(), if !current_metrics.is_empty() => {
                        for metrics in current_metrics.values() {
                            publish_metrics(&event_publisher, worker_id, metrics).await;
                        }
                    }
                }
            }
        });
    }
}

async fn publish_metrics(
    event_publisher: &EventPublisher,
    worker_id: u64,
    metrics: &WorkerMetrics,
) {
    let active_load = ActiveLoad {
        worker_id,
        dp_rank: metrics.dp_rank,
        active_decode_blocks: metrics.active_decode_blocks,
        active_prefill_tokens: None,
        kv_used_blocks: metrics.kv_used_blocks,
    };

    if let Err(error) = event_publisher.publish(&active_load).await {
        tracing::warn!(%error, "failed to publish worker metrics");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_retains_the_latest_snapshot_for_every_dp_rank() {
        let publisher = WorkerMetricsPublisher::new().unwrap();
        publisher.publish(Some(0), None, Some(10)).unwrap();
        publisher.publish(Some(1), None, Some(20)).unwrap();
        publisher.publish(Some(0), None, Some(30)).unwrap();

        let current = publisher.rx.borrow();
        assert_eq!(current.len(), 2);
        assert_eq!(current[&0].kv_used_blocks, Some(30));
        assert_eq!(current[&1].kv_used_blocks, Some(20));
    }
}
