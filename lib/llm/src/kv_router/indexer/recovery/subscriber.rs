// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::worker_query::WorkerQueryClient;
use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::Indexer;
use anyhow::Result;
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::{
    component::Endpoint,
    discovery::EventTransportKind,
    prelude::*,
    transports::event_plane::{EventSubscriber, TypedEventSubscriber},
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Start a background task for event consumption using the event plane.
///
/// This function:
/// - Uses the configured event plane (NATS Core or ZMQ)
/// - On worker Added: dumps worker's local indexer into router
/// - On worker Removed: removes worker from router indexer
async fn start_kv_router_background_event_plane(
    endpoint: Endpoint,
    indexer: Indexer,
    transport_kind: EventTransportKind,
    workers_with_configs: RuntimeConfigWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<()> {
    // Subscribe to KV events BEFORE spawning the discovery/recovery loop.
    // This ensures no events are lost between the initial dump fetch and the
    // subscription becoming active — the tree state at fetch time is guaranteed
    // to be a subset of what the subscription will deliver.
    let subscriber =
        EventSubscriber::for_endpoint_with_transport(&endpoint, KV_EVENT_SUBJECT, transport_kind)
            .await?
            .typed::<Vec<RouterEvent>>();

    // Brief delay to let the subscription fully establish with the NATS server
    // before recovery fetches the initial dump from workers.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // WorkerQueryClient handles its own discovery loop for lifecycle + initial recovery.
    // No blocking wait — recovery happens asynchronously as endpoints are discovered.
    let worker_query_client = WorkerQueryClient::spawn(
        endpoint.clone(),
        indexer,
        workers_with_configs,
        model,
        worker_type,
        cancellation_token.child_token(),
    )
    .await?;
    let endpoint_id = endpoint.id();

    match transport_kind {
        EventTransportKind::Nats => {
            tracing::info!(
                endpoint = %endpoint_id,
                "KV Router using NATS Core subscription (local_indexer mode)"
            );
        }
        EventTransportKind::Zmq => {
            tracing::info!(
                endpoint = %endpoint_id,
                "KV Router using ZMQ event plane subscription (local_indexer mode)"
            );
        }
    }

    tokio::spawn(async move {
        consume_events(subscriber, worker_query_client, cancellation_token).await;
    });

    Ok(())
}

async fn consume_events(
    mut subscriber: TypedEventSubscriber<Vec<RouterEvent>>,
    worker_query_client: Arc<WorkerQueryClient>,
    cancellation_token: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("KV Router event plane background task received cancellation signal");
                break;
            }

            result = subscriber.next() => {
                let Some(result) = result else {
                    tracing::warn!("KV Router event-plane stream closed");
                    break;
                };
                let (envelope, events) = match result {
                    Ok((envelope, events)) => (envelope, events),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to receive RouterEvent batch from event plane; publisher/subscriber versions or payload formats may not match: {e:?}"
                        );
                        continue;
                    }
                };

                tracing::trace!(
                    event_count = events.len(),
                    "Received event payload from publisher {} (seq {})",
                    envelope.publisher_id,
                    envelope.sequence
                );
                for event in events {
                    if cancellation_token.is_cancelled() {
                        break;
                    }
                    forward_live_event(&worker_query_client, event).await;
                }
            }
        }
    }

    tracing::debug!("KV Router event plane background task exiting");
}

async fn forward_live_event(worker_query_client: &Arc<WorkerQueryClient>, event: RouterEvent) {
    tracing::trace!(
        "Forwarding live event to recovery coordinator for worker {} dp_rank {} event_id {}",
        event.worker_id,
        event.event.dp_rank,
        event.event.event_id
    );
    worker_query_client.handle_live_event(event).await;
}

pub async fn start_subscriber(
    endpoint: Endpoint,
    indexer: Indexer,
    workers_with_configs: RuntimeConfigWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let component = endpoint.component();
    let transport_kind = component.drt().default_event_transport_kind();

    if transport_kind == EventTransportKind::Zmq {
        tracing::info!("Using ZMQ event plane subscription (local_indexer mode)");
    } else {
        tracing::info!("Using NATS Core subscription (local_indexer mode)");
    }

    start_kv_router_background_event_plane(
        endpoint,
        indexer,
        transport_kind,
        workers_with_configs,
        model,
        worker_type,
        cancellation_token,
    )
    .await
}
