// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::{Component, Namespace};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<WorkerMetrics>,
    rx: tokio::sync::watch::Receiver<WorkerMetrics>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(WorkerMetrics::default());
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
        self.tx
            .send(metrics)
            .map_err(|_| anyhow::anyhow!("metrics channel closed"))
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let worker_id = component.drt().connection_id();
        self.start_nats_metrics_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    pub(super) fn start_nats_metrics_publishing(&self, namespace: Namespace, worker_id: u64) {
        let nats_rx = self.rx.clone();

        tokio::spawn(async move {
            let event_publisher =
                match EventPublisher::for_namespace(&namespace, KV_METRICS_SUBJECT).await {
                    Ok(publisher) => publisher,
                    Err(e) => {
                        tracing::error!("Failed to create metrics publisher: {}", e);
                        return;
                    }
                };

            let mut rx = nats_rx;
            let mut last_metrics: Option<WorkerMetrics> = None;
            let mut pending_publish: Option<WorkerMetrics> = None;
            let mut publish_timer =
                Box::pin(tokio::time::sleep(tokio::time::Duration::from_secs(0)));
            publish_timer.as_mut().reset(tokio::time::Instant::now());

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping NATS background task"
                            );
                            break;
                        }

                        let metrics = rx.borrow_and_update().clone();
                        let has_changed = last_metrics.as_ref() != Some(&metrics);

                        if has_changed {
                            pending_publish = Some(metrics.clone());
                            last_metrics = Some(metrics);
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now()
                                    + tokio::time::Duration::from_millis(1)
                            );
                        }
                    }
                    _ = &mut publish_timer => {
                        if let Some(metrics) = pending_publish.take() {
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: metrics.active_decode_blocks,
                                active_prefill_tokens: None,
                                kv_used_blocks: metrics.kv_used_blocks,
                            };

                            if let Err(e) = event_publisher.publish(&active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }

                        publish_timer.as_mut().reset(
                            tokio::time::Instant::now()
                                + tokio::time::Duration::from_secs(3600)
                        );
                    }
                }
            }
        });
    }
}
