// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

const PUBLISH_DEBOUNCE: Duration = Duration::from_millis(1);

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
}

struct PendingMetrics {
    metrics: WorkerMetrics,
    deadline: tokio::time::Instant,
}

struct WorkerMetricsDebouncer {
    debounce: Duration,
    last_metrics: HashMap<DpRank, WorkerMetrics>,
    pending: HashMap<DpRank, PendingMetrics>,
}

impl WorkerMetricsDebouncer {
    fn new(debounce: Duration) -> Self {
        Self {
            debounce,
            last_metrics: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    fn observe(
        &mut self,
        metrics_by_rank: &HashMap<DpRank, WorkerMetrics>,
        now: tokio::time::Instant,
    ) {
        for (&dp_rank, metrics) in metrics_by_rank {
            if self.last_metrics.get(&dp_rank) == Some(metrics) {
                continue;
            }

            self.last_metrics.insert(dp_rank, metrics.clone());
            self.pending.insert(
                dp_rank,
                PendingMetrics {
                    metrics: metrics.clone(),
                    deadline: now + self.debounce,
                },
            );
        }
    }

    fn next_deadline(&self) -> Option<tokio::time::Instant> {
        self.pending.values().map(|pending| pending.deadline).min()
    }

    fn take_due(&mut self, now: tokio::time::Instant) -> Vec<WorkerMetrics> {
        let due_ranks = self
            .pending
            .iter()
            .filter_map(|(&dp_rank, pending)| (pending.deadline <= now).then_some(dp_rank))
            .collect::<Vec<_>>();

        due_ranks
            .into_iter()
            .filter_map(|dp_rank| self.pending.remove(&dp_rank))
            .map(|pending| pending.metrics)
            .collect()
    }
}

#[async_trait::async_trait]
pub(super) trait WorkerMetricsSink: Send + 'static {
    async fn publish(&self, active_load: ActiveLoad) -> Result<()>;
}

#[async_trait::async_trait]
impl WorkerMetricsSink for EventPublisher {
    async fn publish(&self, active_load: ActiveLoad) -> Result<()> {
        EventPublisher::publish(self, &active_load).await
    }
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
        self.tx.send_if_modified(|metrics_by_rank| {
            if metrics_by_rank.get(&metrics.dp_rank) == Some(&metrics) {
                return false;
            }

            metrics_by_rank.insert(metrics.dp_rank, metrics);
            true
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
        self.start_metrics_publishing_with(event_publisher, worker_id);
    }

    pub(super) fn start_metrics_publishing_with<S>(&self, sink: S, worker_id: u64)
    where
        S: WorkerMetricsSink,
    {
        let metrics_rx = self.rx.clone();

        tokio::spawn(async move {
            let mut rx = metrics_rx;
            let mut debouncer = WorkerMetricsDebouncer::new(PUBLISH_DEBOUNCE);
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

                        let now = tokio::time::Instant::now();
                        debouncer.observe(&rx.borrow_and_update(), now);
                        if let Some(deadline) = debouncer.next_deadline() {
                            publish_timer.as_mut().reset(deadline);
                        }
                    }
                    _ = &mut publish_timer, if debouncer.next_deadline().is_some() => {
                        for metrics in debouncer.take_due(tokio::time::Instant::now()) {
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: metrics.active_decode_blocks,
                                active_prefill_tokens: None,
                                kv_used_blocks: metrics.kv_used_blocks,
                            };

                            if let Err(e) = sink.publish(active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }

                        if let Some(deadline) = debouncer.next_deadline() {
                            publish_timer.as_mut().reset(deadline);
                        }
                    }
                }
            }
        });
    }
}
