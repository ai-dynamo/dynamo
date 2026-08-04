// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<HashMap<DpRank, WorkerMetrics>>,
    rx: tokio::sync::watch::Receiver<HashMap<DpRank, WorkerMetrics>>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        Ok(Self { tx, rx })
    }

    pub fn publish(
        &self,
        dp_rank: Option<DpRank>,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
        num_waiting_reqs: Option<u64>,
    ) -> Result<()> {
        if active_decode_blocks.is_none() && kv_used_blocks.is_none() && num_waiting_reqs.is_none()
        {
            anyhow::bail!("worker metrics publish requires at least one load metric");
        }

        let metrics = WorkerMetrics {
            dp_rank: dp_rank.unwrap_or(0),
            active_decode_blocks,
            kv_used_blocks,
            num_waiting_reqs,
        };
        tracing::trace!(
            dp_rank = metrics.dp_rank,
            active_decode_blocks = ?metrics.active_decode_blocks,
            kv_used_blocks = ?metrics.kv_used_blocks,
            num_waiting_reqs = ?metrics.num_waiting_reqs,
            "Publishing worker metrics"
        );
        self.tx.send_modify(|metrics_by_rank| {
            metrics_by_rank.insert(metrics.dp_rank, metrics);
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
            let mut last_metrics: HashMap<DpRank, WorkerMetrics> = HashMap::new();
            let mut pending_publish: HashMap<DpRank, WorkerMetrics> = HashMap::new();
            let publish_timer = tokio::time::sleep(tokio::time::Duration::ZERO);
            tokio::pin!(publish_timer);

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping event-plane background task"
                            );
                            break;
                        }

                        let metrics_by_rank = rx.borrow_and_update().clone();
                        for (dp_rank, metrics) in metrics_by_rank {
                            if last_metrics.get(&dp_rank) == Some(&metrics) {
                                continue;
                            }

                            if pending_publish.is_empty() {
                                publish_timer.as_mut().reset(
                                    tokio::time::Instant::now()
                                        + tokio::time::Duration::from_millis(1),
                                );
                            }
                            pending_publish.insert(dp_rank, metrics.clone());
                            last_metrics.insert(dp_rank, metrics);
                        }
                    }
                    _ = &mut publish_timer, if !pending_publish.is_empty() => {
                        let mut metrics_to_publish = std::mem::take(&mut pending_publish)
                            .into_iter()
                            .collect::<Vec<_>>();
                        metrics_to_publish.sort_unstable_by_key(|(dp_rank, _)| *dp_rank);
                        for (_, metrics) in metrics_to_publish {
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: metrics.active_decode_blocks,
                                active_prefill_tokens: None,
                                kv_used_blocks: metrics.kv_used_blocks,
                                num_waiting_reqs: metrics.num_waiting_reqs,
                            };

                            if let Err(e) = event_publisher.publish(&active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_the_latest_metrics_for_every_dp_rank() {
        let publisher = WorkerMetricsPublisher::new().unwrap();
        publisher.publish(Some(0), None, Some(10), Some(1)).unwrap();
        publisher.publish(Some(1), None, Some(20), Some(2)).unwrap();
        publisher.publish(Some(0), None, Some(11), Some(3)).unwrap();

        let metrics = publisher.rx.borrow();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[&0].kv_used_blocks, Some(11));
        assert_eq!(metrics[&0].num_waiting_reqs, Some(3));
        assert_eq!(metrics[&1].kv_used_blocks, Some(20));
        assert_eq!(metrics[&1].num_waiting_reqs, Some(2));
    }
}
