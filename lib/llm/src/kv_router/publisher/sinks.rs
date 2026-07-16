// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, StorageTier};
use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

pub(super) trait RouterEventBatchSink: Send + Sync {
    fn publish_events(&self, events: &[RouterEvent]) -> impl Future<Output = Result<()>> + Send;
}

impl<P: RouterEventSink + Send + Sync> RouterEventBatchSink for P {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut first_error = None;
        for event in events {
            if let Err(error) = self.publish_event(event).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl RouterEventBatchSink for EventPlanePublisher {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        match self.0.transport_kind() {
            // NATS Core retains its existing singleton RouterEvent payload.
            EventTransportKind::Nats => {
                let mut first_error = None;
                for event in events {
                    if let Err(error) = self.0.publish(event).await
                        && first_error.is_none()
                    {
                        first_error = Some(error);
                    }
                }
                first_error.map_or(Ok(()), Err)
            }
            // ZMQ peers must run the same version: its payload is Vec<RouterEvent>.
            EventTransportKind::Zmq => self.0.publish_bytes(encode_zmq_event_batch(events)?).await,
        }
    }
}

/// Encode the complete ordered event list as one ZMQ payload.
pub(super) fn encode_zmq_event_batch(events: &[RouterEvent]) -> Result<Vec<u8>> {
    Ok(rmp_serde::to_vec_named(events)?)
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

pub(super) async fn emit(
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
    output: &mut Vec<RouterEvent>,
) {
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    output.push(router_event);
}
