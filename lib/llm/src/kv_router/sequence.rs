// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-specific glue for [`ActiveSequencesMultiWorker`].
//!
//! This module provides the concrete [`SequencePublisher`] and [`SequenceSubscriber`]
//! implementations that wire the runtime-agnostic business logic (in `dynamo_kv_router`)
//! to NATS event transport and Prometheus metrics.

pub use dynamo_kv_router::multi_worker_sequence::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    SequenceSubscriber,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use std::collections::HashMap;
use std::sync::Arc;

use super::metrics::WORKER_LOAD_METRICS;
use super::protocols::{ActiveLoad, ActiveSequenceEvent, WorkerWithDpRank};
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, KV_METRICS_SUBJECT};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Concrete [`SequencePublisher`] backed by NATS [`EventPublisher`] and Prometheus gauges.
pub struct RuntimeSequencePublisher {
    event_publisher: EventPublisher,
    metrics_publisher: Arc<EventPublisher>,
}

#[async_trait]
impl SequencePublisher for RuntimeSequencePublisher {
    async fn publish_event(&self, event: &ActiveSequenceEvent) -> anyhow::Result<()> {
        self.event_publisher.publish(event).await
    }

    fn publish_load(&self, load: ActiveLoad) {
        let publisher = self.metrics_publisher.clone();
        tokio::spawn(async move {
            if let Err(e) = publisher.publish(&load).await {
                tracing::trace!("Failed to publish ActiveLoad to NATS: {e:?}");
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
}

/// Concrete [`SequenceSubscriber`] backed by NATS typed event stream.
pub struct RuntimeSequenceSubscriber {
    inner: dynamo_runtime::transports::event_plane::TypedEventSubscriber<ActiveSequenceEvent>,
}

#[async_trait]
impl SequenceSubscriber for RuntimeSequenceSubscriber {
    async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
        match self.inner.next().await? {
            Ok((_envelope, event)) => Some(Ok(event)),
            Err(e) => Some(Err(e)),
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
    replica_sync: bool,
    router_id: u64,
    worker_type: &'static str,
) -> Result<Arc<ActiveSequencesMulti>> {
    let event_publisher =
        EventPublisher::for_component(&component, ACTIVE_SEQUENCES_SUBJECT).await?;
    let metrics_publisher = Arc::new(
        EventPublisher::for_namespace(component.namespace(), KV_METRICS_SUBJECT).await?,
    );

    let publisher = RuntimeSequencePublisher {
        event_publisher,
        metrics_publisher,
    };

    let dp_sizes: HashMap<u64, u32> = workers_with_configs
        .into_iter()
        .map(|(id, config)| (id, config.data_parallel_size))
        .collect();

    let multi_worker = ActiveSequencesMultiWorker::new(
        publisher,
        block_size,
        dp_sizes,
        replica_sync,
        router_id,
        worker_type,
    );

    let arc = Arc::new(multi_worker);

    if replica_sync {
        let subscriber = EventSubscriber::for_component(&component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();
        let subscriber = RuntimeSequenceSubscriber { inner: subscriber };
        let cancel_token = component.drt().runtime().child_token();
        arc.start_replica_sync(subscriber, cancel_token);
    }

    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};

    #[test]
    fn test_active_sequences_shared_blocks() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("request_1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.add_request("request_2".to_string(), Some(vec![4]), 4, 0, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.add_request("request_3".to_string(), Some(vec![1, 2, 3, 4]), 16, 4, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.free(&"request_2".to_string());
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_3".to_string());
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
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
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            component,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: Some(vec![0, 1, 2]),
                isl: 12,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(0, 0),
                lora_name: None,
            })
            .await?;

        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: Some(vec![3, 4]),
                isl: 8,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(0, 1),
                lora_name: None,
            })
            .await?;

        seq_manager_2
            .add_request(SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: Some(vec![0, 1, 2, 3]),
                isl: 16,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(1, 0),
                lora_name: None,
            })
            .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase1 = seq_manager_1.active_blocks();
        let tokens_phase1 = seq_manager_1.active_tokens();

        let worker_0_dp0 = WorkerWithDpRank::new(0, 0);
        let worker_0_dp1 = WorkerWithDpRank::new(0, 1);
        let worker_1_dp0 = WorkerWithDpRank::new(1, 0);

        assert_eq!(
            blocks_phase1[&worker_0_dp0], 3,
            "Worker 0 dp_rank 0 should have 3 active blocks (from request_0)"
        );
        assert_eq!(
            blocks_phase1[&worker_0_dp1], 2,
            "Worker 0 dp_rank 1 should have 2 active blocks (from request_1)"
        );
        assert_eq!(
            blocks_phase1[&worker_1_dp0], 4,
            "Worker 1 dp_rank 0 should have 4 active blocks (from request_2 added by seq_manager_2)"
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

        seq_manager_1.free(&"request_2".to_string()).await?;

        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase2 = seq_manager_2.active_blocks();
        let tokens_phase2 = seq_manager_2.active_tokens();

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
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            component,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: None,
                isl: 12,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(0),
                lora_name: None,
            })
            .await?;

        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: None,
                isl: 8,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(1),
                lora_name: None,
            })
            .await?;

        seq_manager_2
            .add_request(SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: None,
                isl: 16,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(2),
                lora_name: None,
            })
            .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase1 = seq_manager_1.active_tokens();

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

        seq_manager_1
            .mark_prefill_completed(&"request_2".to_string())
            .await?;
        seq_manager_1.free(&"request_2".to_string()).await?;

        seq_manager_2
            .mark_prefill_completed(&"request_0".to_string())
            .await?;
        seq_manager_2
            .mark_prefill_completed(&"request_1".to_string())
            .await?;
        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase2 = seq_manager_2.active_tokens();

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
