// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-specific glue for [`ActiveSequencesMultiWorker`].
//!
//! This module provides the concrete [`SequencePublisher`] and [`SequenceSubscriber`]
//! implementations that wire the runtime-agnostic business logic (in `dynamo_kv_router`)
//! to NATS event transport and Prometheus metrics.

use dynamo_kv_router::ActiveSequenceStride;
pub use dynamo_kv_router::multi_worker_sequence::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    SequenceSubscriber, SequenceTrackerOptions,
};
use dynamo_kv_router::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceStrideError, WorkerWithDpRank,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio_util::sync::CancellationToken;

use super::metrics::{RouterWorkerStatusMetrics, WORKER_LOAD_METRICS};
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, KV_METRICS_SUBJECT};
use crate::local_model::runtime_config::ModelRuntimeConfig;
#[cfg(test)]
use dynamo_kv_router::protocols::PrefillLoadHint;

/// Concrete [`SequencePublisher`] backed by NATS [`EventPublisher`] and Prometheus gauges.
pub struct RuntimeSequencePublisher {
    event_publisher: EventPublisher,
    metrics_publisher: Arc<EventPublisher>,
    worker_status_metrics: Arc<RouterWorkerStatusMetrics>,
}

impl SequencePublisher for RuntimeSequencePublisher {
    async fn publish_event(&self, event: &ActiveSequenceEvent) -> anyhow::Result<()> {
        self.event_publisher.publish(event).await
    }

    fn publish_load(&self, load: ActiveLoad) {
        let publisher = self.metrics_publisher.clone();
        tokio::spawn(async move {
            if let Err(e) = publisher.publish(&load).await {
                tracing::trace!(
                    "Failed to publish ActiveLoad to NATS for worker (id={}, dp_rank={}): {e:?}",
                    load.worker_id,
                    load.dp_rank
                );
            }
        });
    }

    fn publish_load_batch(&self, loads: Vec<ActiveLoad>) {
        let publisher = self.metrics_publisher.clone();
        tokio::spawn(async move {
            for load in loads {
                if let Err(e) = publisher.publish(&load).await {
                    tracing::trace!(
                        "Failed to publish ActiveLoad to NATS for worker (id={}, dp_rank={}): {e:?}",
                        load.worker_id,
                        load.dp_rank
                    );
                }
            }
        });
    }

    fn observe_load(
        &self,
        worker: &WorkerWithDpRank,
        worker_type: &str,
        blocks: usize,
        tokens: usize,
    ) {
        WORKER_LOAD_METRICS.observe(
            worker.worker_id,
            worker.dp_rank,
            worker_type,
            blocks,
            tokens,
        );
    }

    fn observe_worker_registered(&self, worker: &WorkerWithDpRank, worker_type: &str) {
        self.worker_status_metrics
            .set_registered(worker.worker_id, worker.dp_rank, worker_type);
    }

    fn observe_worker_removed(&self, worker: &WorkerWithDpRank, worker_type: &str) {
        self.worker_status_metrics
            .remove_worker(worker.worker_id, worker.dp_rank, worker_type);
    }

    fn observe_replica_stride_mismatch(
        &self,
        expected: ActiveSequenceStride,
        received: Result<ActiveSequenceStride, ActiveSequenceStrideError>,
        worker_type: &str,
    ) {
        self.worker_status_metrics
            .increment_active_sequence_stride_mismatch(expected, received, worker_type);
    }
}

/// Concrete [`SequenceSubscriber`] backed by NATS typed event stream.
pub struct RuntimeSequenceSubscriber {
    inner: dynamo_runtime::transports::event_plane::TypedEventSubscriber<ActiveSequenceEvent>,
}

impl SequenceSubscriber for RuntimeSequenceSubscriber {
    async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
        match self.inner.next().await? {
            Ok((_envelope, event)) => Some(Ok(event)),
            Err(e) => Some(Err(e)),
        }
    }

    fn poll_next_event(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
        match self.inner.poll_next(cx) {
            Poll::Ready(Some(Ok((_envelope, event)))) => Poll::Ready(Some(Ok(event))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Type alias for the runtime-wired multi-worker sequence tracker.
pub type ActiveSequencesMulti = ActiveSequencesMultiWorker<RuntimeSequencePublisher>;

/// Convenience async constructor that creates the NATS publishers/subscribers
/// and returns an `Arc<ActiveSequencesMulti>` with replica sync already running.
pub async fn create_multi_worker_sequences(
    component: Component,
    block_size: usize,
    workers_with_configs: HashMap<u64, ModelRuntimeConfig>,
    router_id: u64,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
    options: SequenceTrackerOptions,
) -> Result<Arc<ActiveSequencesMulti>> {
    let event_publisher =
        EventPublisher::for_component(&component, ACTIVE_SEQUENCES_SUBJECT).await?;
    let metrics_publisher =
        Arc::new(EventPublisher::for_namespace(component.namespace(), KV_METRICS_SUBJECT).await?);
    let worker_status_metrics = RouterWorkerStatusMetrics::from_component(&component);

    let publisher = RuntimeSequencePublisher {
        event_publisher,
        metrics_publisher,
        worker_status_metrics,
    };

    let dp_range: HashMap<u64, (u32, u32)> = workers_with_configs
        .into_iter()
        .map(|(id, config)| {
            (
                id,
                (config.data_parallel_start_rank, config.data_parallel_size),
            )
        })
        .collect();

    let multi_worker = ActiveSequencesMultiWorker::new_with_options(
        publisher,
        block_size,
        dp_range,
        router_id,
        worker_type,
        options,
    );

    let arc = Arc::new(multi_worker);

    if options.replica_sync {
        let subscriber = EventSubscriber::for_component(&component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();
        let subscriber = RuntimeSequenceSubscriber { inner: subscriber };
        arc.start_replica_sync(subscriber, cancellation_token.child_token());
    }

    arc.start_periodic_force_expiry_across_all_workers(cancellation_token.child_token());

    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::time::Instant;

    fn tracking_hint(tokens: usize) -> Option<PrefillLoadHint> {
        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        })
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_cross_instance_sync() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_size = 4;

        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        let namespace = distributed.namespace("test_cross_instance_sync")?;
        let component = namespace.component("sequences")?;
        let active_sequence_stride = ActiveSequenceStride::new(3).unwrap();

        let mut workers_with_configs = HashMap::new();

        let mut config_worker_0 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        config_worker_0.data_parallel_size = 2;
        workers_with_configs.insert(0, config_worker_0);

        let config_worker_1 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        workers_with_configs.insert(1, config_worker_1);

        let seq_manager_1 = create_multi_worker_sequences(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
            SequenceTrackerOptions {
                replica_sync: true,
                active_sequence_stride,
                ..Default::default()
            },
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            component,
            block_size,
            workers_with_configs,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
            SequenceTrackerOptions {
                replica_sync: true,
                active_sequence_stride,
                ..Default::default()
            },
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let decay_now = Instant::now();

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: Some(active_sequence_stride.sample_dense((0..9).collect())),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(12),
                worker: WorkerWithDpRank::new(0, 0),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: Some(active_sequence_stride.sample_dense((9..15).collect())),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(8),
                worker: WorkerWithDpRank::new(0, 1),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_2.add_request(
            SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: Some(active_sequence_stride.sample_dense((0..12).collect())),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(16),
                worker: WorkerWithDpRank::new(1, 0),
                lora_name: None,
            },
            decay_now,
        )?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase1 = seq_manager_1.active_blocks();
        let tokens_phase1 = seq_manager_1.active_tokens(Instant::now());

        let worker_0_dp0 = WorkerWithDpRank::new(0, 0);
        let worker_0_dp1 = WorkerWithDpRank::new(0, 1);
        let worker_1_dp0 = WorkerWithDpRank::new(1, 0);

        assert_eq!(
            blocks_phase1[&worker_0_dp0], 9,
            "Worker 0 dp_rank 0 should estimate 9 active blocks (from request_0)"
        );
        assert_eq!(
            blocks_phase1[&worker_0_dp1], 6,
            "Worker 0 dp_rank 1 should estimate 6 active blocks (from request_1)"
        );
        assert_eq!(
            blocks_phase1[&worker_1_dp0], 12,
            "Worker 1 dp_rank 0 should estimate 12 active blocks (from request_2 added by seq_manager_2)"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp0], 12,
            "Worker 0 dp_rank 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp1], 8,
            "Worker 0 dp_rank 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1_dp0], 16,
            "Worker 1 dp_rank 0 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        seq_manager_1.mark_prefill_completed(&"request_0".to_string(), Instant::now())?;
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        assert_eq!(
            seq_manager_1.active_tokens(Instant::now())[&worker_0_dp0],
            0
        );
        assert_eq!(
            seq_manager_2.active_tokens(Instant::now())[&worker_0_dp0],
            0
        );

        seq_manager_1.free(&"request_2".to_string(), Instant::now())?;

        seq_manager_2.free(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_1".to_string(), Instant::now())?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase2 = seq_manager_2.active_blocks();
        let tokens_phase2 = seq_manager_2.active_tokens(Instant::now());

        let all_workers = vec![
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
        ];

        for worker in all_workers {
            assert_eq!(
                blocks_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active blocks after all requests freed",
                worker.worker_id, worker.dp_rank
            );
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active tokens after all requests freed",
                worker.worker_id, worker.dp_rank
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_no_token_sequence_sync() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_size = 4;

        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        let namespace = distributed.namespace("test_no_token_seq_sync")?;
        let component = namespace.component("sequences")?;

        let mut workers_with_configs = HashMap::new();
        workers_with_configs.insert(
            0,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            1,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            2,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );

        let seq_manager_1 = create_multi_worker_sequences(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
            SequenceTrackerOptions {
                replica_sync: true,
                ..Default::default()
            },
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            component,
            block_size,
            workers_with_configs,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
            SequenceTrackerOptions {
                replica_sync: true,
                ..Default::default()
            },
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let decay_now = Instant::now();

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(12),
                worker: WorkerWithDpRank::from_worker_id(0),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(8),
                worker: WorkerWithDpRank::from_worker_id(1),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_2.add_request(
            SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(16),
                worker: WorkerWithDpRank::from_worker_id(2),
                lora_name: None,
            },
            decay_now,
        )?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase1 = seq_manager_1.active_tokens(Instant::now());

        let worker_0 = WorkerWithDpRank::from_worker_id(0);
        let worker_1 = WorkerWithDpRank::from_worker_id(1);
        let worker_2 = WorkerWithDpRank::from_worker_id(2);

        assert_eq!(
            tokens_phase1[&worker_0], 12,
            "Worker 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1], 8,
            "Worker 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_2], 16,
            "Worker 2 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        seq_manager_1.mark_prefill_completed(&"request_2".to_string(), Instant::now())?;
        seq_manager_1.free(&"request_2".to_string(), Instant::now())?;

        seq_manager_2.mark_prefill_completed(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.mark_prefill_completed(&"request_1".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_1".to_string(), Instant::now())?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase2 = seq_manager_2.active_tokens(Instant::now());

        for worker_id in 0..=2 {
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker {} should have 0 active tokens after all requests freed",
                worker_id
            );
        }

        Ok(())
    }
}
