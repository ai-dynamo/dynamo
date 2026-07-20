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
use dynamo_kv_router::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventBatch, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use bytes::Bytes;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber, MsgpackCodec};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::metrics::{RouterWorkerStatusMetrics, WORKER_LOAD_METRICS};
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, KV_METRICS_SUBJECT};
use crate::local_model::runtime_config::ModelRuntimeConfig;
#[cfg(test)]
use dynamo_kv_router::protocols::PrefillLoadHint;

const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
const MAX_REPLICA_BATCH_PAYLOAD_BYTES: usize = 1_000_000;

#[derive(Serialize)]
struct ActiveSequenceEventBatchRef<'a> {
    events: &'a [ActiveSequenceEvent],
}

struct EncodedReplicaBatch {
    event_range: Range<usize>,
    payload: Bytes,
    oversized: bool,
}

/// Concrete [`SequencePublisher`] backed by NATS [`EventPublisher`] and Prometheus gauges.
pub struct RuntimeSequencePublisher {
    event_tx: Option<mpsc::Sender<ActiveSequenceEvent>>,
    metrics_publisher: Arc<EventPublisher>,
    worker_status_metrics: Arc<RouterWorkerStatusMetrics>,
}

impl SequencePublisher for RuntimeSequencePublisher {
    async fn publish_event(&self, event: &ActiveSequenceEvent) -> anyhow::Result<()> {
        let Some(event_tx) = &self.event_tx else {
            return Ok(());
        };
        event_tx
            .send(event.clone())
            .await
            .map_err(|_| anyhow::anyhow!("active-sequence replica publisher is closed"))
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
}

fn encode_replica_batch(events: &[ActiveSequenceEvent]) -> Result<Bytes> {
    MsgpackCodec.encode_payload(&ActiveSequenceEventBatchRef { events })
}

fn encode_replica_batch_partitions(
    events: &[ActiveSequenceEvent],
) -> Result<Vec<EncodedReplicaBatch>> {
    let mut batches = Vec::new();
    let mut batch_start = 0;

    while batch_start < events.len() {
        let remaining = &events[batch_start..];
        let remaining_payload = encode_replica_batch(remaining)?;
        if remaining_payload.len() <= MAX_REPLICA_BATCH_PAYLOAD_BYTES {
            batches.push(EncodedReplicaBatch {
                event_range: batch_start..events.len(),
                payload: remaining_payload,
                oversized: false,
            });
            break;
        }

        let first_payload = encode_replica_batch(&remaining[..1])?;
        if first_payload.len() > MAX_REPLICA_BATCH_PAYLOAD_BYTES {
            batches.push(EncodedReplicaBatch {
                event_range: batch_start..batch_start + 1,
                payload: first_payload,
                oversized: true,
            });
            batch_start += 1;
            continue;
        }

        let mut best_len = 1;
        let mut best_payload = first_payload;
        let mut low = 2;
        let mut high = remaining.len() - 1;
        while low <= high {
            let candidate_len = low + (high - low) / 2;
            let candidate_payload = encode_replica_batch(&remaining[..candidate_len])?;
            if candidate_payload.len() <= MAX_REPLICA_BATCH_PAYLOAD_BYTES {
                best_len = candidate_len;
                best_payload = candidate_payload;
                low = candidate_len + 1;
            } else {
                high = candidate_len - 1;
            }
        }

        batches.push(EncodedReplicaBatch {
            event_range: batch_start..batch_start + best_len,
            payload: best_payload,
            oversized: false,
        });
        batch_start += best_len;
    }

    Ok(batches)
}

async fn publish_replica_batch(publisher: &EventPublisher, events: &[ActiveSequenceEvent]) {
    let batches = match encode_replica_batch_partitions(events) {
        Ok(batches) => batches,
        Err(error) => {
            tracing::error!(
                event_count = events.len(),
                error = %error,
                "Failed to encode active-sequence replica batch"
            );
            return;
        }
    };

    for batch in batches {
        let batch_events = &events[batch.event_range.clone()];
        let first_request_id = &batch_events
            .first()
            .expect("replica batch must contain an event")
            .request_id;
        let last_request_id = &batch_events
            .last()
            .expect("replica batch must contain an event")
            .request_id;

        if batch.oversized {
            tracing::warn!(
                request_id = %first_request_id,
                payload_bytes = batch.payload.len(),
                max_payload_bytes = MAX_REPLICA_BATCH_PAYLOAD_BYTES,
                "Active-sequence replica event exceeds the batch payload limit; publishing intact"
            );
        }

        if let Err(error) = publisher.publish_bytes_ref(batch.payload.as_ref()).await {
            tracing::error!(
                transport = ?publisher.transport_kind(),
                event_count = batch_events.len(),
                payload_bytes = batch.payload.len(),
                first_request_id = %first_request_id,
                last_request_id = %last_request_id,
                error = %error,
                "Failed to publish active-sequence replica batch"
            );
        }
    }
}

async fn collect_replica_batch(
    first_event: ActiveSequenceEvent,
    event_rx: &mut mpsc::Receiver<ActiveSequenceEvent>,
    cancellation_token: &CancellationToken,
) -> (Vec<ActiveSequenceEvent>, bool) {
    let mut events = Vec::with_capacity(MAX_REPLICA_BATCH_EVENTS);
    events.push(first_event);
    let deadline = Instant::now() + MAX_REPLICA_BATCH_DURATION;
    let flush_timer = tokio::time::sleep_until(deadline);
    tokio::pin!(flush_timer);

    while events.len() < MAX_REPLICA_BATCH_EVENTS {
        tokio::select! {
            _ = cancellation_token.cancelled() => return (events, true),
            _ = &mut flush_timer => break,
            event = event_rx.recv() => match event {
                Some(event) => events.push(event),
                None => return (events, true),
            },
        }
    }

    (events, false)
}

async fn run_replica_batch_publisher(
    publisher: EventPublisher,
    mut event_rx: mpsc::Receiver<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
) {
    loop {
        let first_event = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            event = event_rx.recv() => match event {
                Some(event) => event,
                None => break,
            },
        };
        let (events, stop_after_flush) =
            collect_replica_batch(first_event, &mut event_rx, &cancellation_token).await;
        publish_replica_batch(&publisher, &events).await;
        if stop_after_flush {
            break;
        }
    }
}

/// Concrete [`SequenceSubscriber`] backed by NATS typed event stream.
pub struct RuntimeSequenceSubscriber {
    inner: dynamo_runtime::transports::event_plane::TypedEventSubscriber<ActiveSequenceEventBatch>,
    pending: VecDeque<ActiveSequenceEvent>,
}

impl SequenceSubscriber for RuntimeSequenceSubscriber {
    async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match self.inner.next().await? {
                Ok((_envelope, batch)) => self.pending.extend(batch.events),
                Err(error) => return Some(Err(error)),
            }
        }
    }

    fn poll_next_event(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            match self.inner.poll_next(cx) {
                Poll::Ready(Some(Ok((_envelope, batch)))) => self.pending.extend(batch.events),
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Type alias for the runtime-wired multi-worker sequence tracker.
pub type ActiveSequencesMulti = ActiveSequencesMultiWorker<RuntimeSequencePublisher>;

/// Convenience async constructor that creates the NATS publishers/subscribers
/// and returns an `Arc<ActiveSequencesMulti>` with replica sync already running.
pub async fn create_multi_worker_sequences(
    endpoint: Endpoint,
    block_size: usize,
    workers_with_configs: HashMap<u64, ModelRuntimeConfig>,
    replica_sync: bool,
    router_id: u64,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<Arc<ActiveSequencesMulti>> {
    let event_tx = if replica_sync {
        let event_publisher =
            EventPublisher::for_endpoint(&endpoint, ACTIVE_SEQUENCES_SUBJECT).await?;
        let (event_tx, event_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
        tokio::spawn(run_replica_batch_publisher(
            event_publisher,
            event_rx,
            cancellation_token.child_token(),
        ));
        Some(event_tx)
    } else {
        None
    };
    let metrics_publisher =
        Arc::new(EventPublisher::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await?);
    let worker_status_metrics = RouterWorkerStatusMetrics::from_component(endpoint.component());

    let publisher = RuntimeSequencePublisher {
        event_tx,
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

    let multi_worker = ActiveSequencesMultiWorker::new(
        publisher,
        block_size,
        dp_range,
        replica_sync,
        router_id,
        worker_type,
    );

    let arc = Arc::new(multi_worker);

    if replica_sync {
        let subscriber = EventSubscriber::for_endpoint(&endpoint, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEventBatch>();
        let subscriber = RuntimeSequenceSubscriber {
            inner: subscriber,
            pending: VecDeque::new(),
        };
        arc.start_replica_sync(subscriber, cancellation_token.child_token());
    }

    arc.start_periodic_force_expiry_across_all_workers(cancellation_token.child_token());

    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::ActiveSequenceEventData;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::time::Instant;

    fn tracking_hint(tokens: usize) -> Option<PrefillLoadHint> {
        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        })
    }

    fn free_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::Free,
            router_id: 7,
            lora_name: None,
        }
    }

    fn large_add_event(request_id: impl Into<String>, hashes: usize) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: Some(vec![u64::MAX; hashes]),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: None,
            },
            router_id: 7,
            lora_name: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn active_sequence_batch_collection_uses_time_and_count_caps() {
        let (event_tx, mut event_rx) = mpsc::channel(MAX_REPLICA_BATCH_EVENTS + 1);
        for request_id in 0..100 {
            event_tx
                .send(free_event(format!("free-{request_id}")))
                .await
                .unwrap();
        }

        let first = event_rx.recv().await.unwrap();
        let start = Instant::now();
        let (events, stop) =
            collect_replica_batch(first, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(events.len(), 100);
        assert_eq!(Instant::now() - start, MAX_REPLICA_BATCH_DURATION);
        let batches = encode_replica_batch_partitions(&events).unwrap();
        assert_eq!(batches.len(), 1);
        let decoded: ActiveSequenceEventBatch =
            MsgpackCodec.decode_payload(&batches[0].payload).unwrap();
        assert_eq!(decoded.events.len(), 100);
        for (request_id, event) in decoded.events.iter().enumerate() {
            assert_eq!(event.request_id, format!("free-{request_id}"));
        }

        for request_id in 0..=MAX_REPLICA_BATCH_EVENTS {
            event_tx
                .send(free_event(format!("count-{request_id}")))
                .await
                .unwrap();
        }
        let first = event_rx.recv().await.unwrap();
        let start = Instant::now();
        let (events, stop) =
            collect_replica_batch(first, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(events.len(), MAX_REPLICA_BATCH_EVENTS);
        assert_eq!(Instant::now(), start);
        assert_eq!(event_rx.len(), 1);

        let last = event_rx.recv().await.unwrap();
        let (remaining, stop) =
            collect_replica_batch(last, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn active_sequence_batches_split_at_payload_limit_and_roundtrip() {
        const NATS_DEFAULT_MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

        let max_payload = vec![0; MAX_REPLICA_BATCH_PAYLOAD_BYTES];
        let max_sized_envelope = MsgpackCodec
            .encode_envelope_parts(
                u64::MAX,
                u64::MAX,
                u64::MAX,
                ACTIVE_SEQUENCES_SUBJECT,
                &max_payload,
            )
            .unwrap();
        assert!(max_sized_envelope.len() < NATS_DEFAULT_MAX_PAYLOAD_BYTES);

        let events = vec![
            large_add_event("large-0", 80_000),
            large_add_event("large-1", 80_000),
        ];
        assert!(encode_replica_batch(&events).unwrap().len() > MAX_REPLICA_BATCH_PAYLOAD_BYTES);

        let batches = encode_replica_batch_partitions(&events).unwrap();
        assert_eq!(batches.len(), 2);
        let mut decoded_request_ids = Vec::new();
        for batch in &batches {
            assert!(!batch.oversized);
            assert!(batch.payload.len() <= MAX_REPLICA_BATCH_PAYLOAD_BYTES);
            let envelope = MsgpackCodec
                .encode_envelope_parts(
                    u64::MAX,
                    u64::MAX,
                    u64::MAX,
                    ACTIVE_SEQUENCES_SUBJECT,
                    batch.payload.as_ref(),
                )
                .unwrap();
            assert!(envelope.len() < NATS_DEFAULT_MAX_PAYLOAD_BYTES);

            let decoded: ActiveSequenceEventBatch =
                MsgpackCodec.decode_payload(&batch.payload).unwrap();
            decoded_request_ids.extend(decoded.events.into_iter().map(|event| event.request_id));
        }
        assert_eq!(decoded_request_ids, ["large-0", "large-1"]);
    }

    #[test]
    fn active_sequence_batch_preserves_single_oversized_event() {
        let events = vec![large_add_event("oversized", 120_000)];
        let batches = encode_replica_batch_partitions(&events).unwrap();

        assert_eq!(batches.len(), 1);
        assert!(batches[0].oversized);
        assert!(batches[0].payload.len() > MAX_REPLICA_BATCH_PAYLOAD_BYTES);
        assert_eq!(batches[0].event_range, 0..1);
    }

    #[tokio::test]
    async fn active_sequence_replica_sync_isolated_by_endpoint() -> Result<()> {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let namespace = distributed.namespace(format!(
            "active-sequence-endpoint-isolation-{}",
            uuid::Uuid::new_v4()
        ))?;
        let component = namespace.component("workers")?;
        let endpoint_a = component.endpoint("generate-a");
        let endpoint_b = component.endpoint("generate-b");
        let workers = HashMap::from([(0, ModelRuntimeConfig::new())]);

        let cancel = CancellationToken::new();
        let sequences_a = create_multi_worker_sequences(
            endpoint_a.clone(),
            4,
            workers.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;
        let sequences_a_peer = create_multi_worker_sequences(
            endpoint_a,
            4,
            workers.clone(),
            true,
            3,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;
        let sequences_b = create_multi_worker_sequences(
            endpoint_b,
            4,
            workers,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;

        let worker = WorkerWithDpRank::new(0, 0);
        tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
            for request_index in 0..100 {
                if sequences_a_peer.active_blocks()[&worker] > 0 {
                    break;
                }

                sequences_a.add_request(
                    SequenceRequest {
                        request_id: format!("endpoint-a-request-{request_index}"),
                        token_sequence: Some(vec![1, 2, 3, 4]),
                        track_prefill_tokens: true,
                        expected_output_tokens: None,
                        prefill_load_hint: tracking_hint(4),
                        worker,
                        lora_name: None,
                    },
                    Instant::now(),
                )?;
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }

            anyhow::ensure!(sequences_a_peer.active_blocks()[&worker] > 0);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        assert!(sequences_a.active_blocks()[&worker] > 0);
        assert!(sequences_a_peer.active_blocks()[&worker] > 0);
        let leaked_to_b = tokio::time::timeout(tokio::time::Duration::from_millis(250), async {
            loop {
                if sequences_b.active_blocks()[&worker] > 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert!(
            leaked_to_b.is_err(),
            "endpoint B received endpoint A sequence state"
        );
        assert_eq!(sequences_b.active_blocks()[&worker], 0);
        cancel.cancel();
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_cross_instance_sync() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_size = 4;

        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        let namespace = distributed.namespace("test_cross_instance_sync")?;
        let endpoint = namespace.component("sequences")?.endpoint("generate");

        let mut workers_with_configs = HashMap::new();

        let mut config_worker_0 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        config_worker_0.data_parallel_size = 2;
        workers_with_configs.insert(0, config_worker_0);

        let config_worker_1 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        workers_with_configs.insert(1, config_worker_1);

        let seq_manager_1 = create_multi_worker_sequences(
            endpoint.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            endpoint,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let decay_now = Instant::now();

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: Some(vec![0, 1, 2]),
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
                token_sequence: Some(vec![3, 4]),
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
                token_sequence: Some(vec![0, 1, 2, 3]),
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
        let endpoint = namespace.component("sequences")?.endpoint("generate");

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
            endpoint.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            endpoint,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
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
