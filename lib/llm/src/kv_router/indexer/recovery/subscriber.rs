// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::worker_query::WorkerQueryClient;
use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::Indexer;
use anyhow::Result;
use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{KV_EVENT_SUBJECT, RouterEvent},
};
use dynamo_runtime::{
    component::Component,
    discovery::EventTransportKind,
    prelude::*,
    transports::event_plane::{EventSubscriber, TypedEventSubscriber},
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Start a simplified background task for event consumption using the event plane.
///
/// This is used when local indexer mode is enabled. Unlike `start_kv_router_background`,
/// this function:
/// - Uses the event plane (NATS Core or ZMQ) instead of JetStream
/// - Does not support snapshots, purging, or durable consumers
/// - On worker Added: dumps worker's local indexer into router
/// - On worker Removed: removes worker from router indexer
///
/// This is appropriate when workers have local indexers enabled.
async fn start_kv_router_background_event_plane(
    component: Component,
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
        EventSubscriber::for_component_with_transport(&component, KV_EVENT_SUBJECT, transport_kind)
            .await?
            .typed::<Vec<RouterEvent>>();

    // Brief delay to let the subscription fully establish with the NATS server
    // before recovery fetches the initial dump from workers.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // WorkerQueryClient handles its own discovery loop for lifecycle + initial recovery.
    // No blocking wait — recovery happens asynchronously as endpoints are discovered.
    let worker_query_client = WorkerQueryClient::spawn(
        component.clone(),
        indexer,
        workers_with_configs,
        model,
        worker_type,
        cancellation_token.child_token(),
    )
    .await?;
    let kv_event_subject = format!(
        "namespace.{}.component.{}.{}",
        component.namespace().name(),
        component.name(),
        KV_EVENT_SUBJECT
    );

    match transport_kind {
        EventTransportKind::Nats => {
            tracing::info!(
                subject = %kv_event_subject,
                "KV Router using NATS Core subscription (local_indexer mode)"
            );
        }
        EventTransportKind::Zmq => {
            tracing::info!(
                subject = %kv_event_subject,
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

            Some(result) = subscriber.next() => {
                let (envelope, events) = match result {
                    Ok((envelope, events)) => (envelope, events),
                    Err(e) => {
                        tracing::warn!("Failed to receive RouterEvent payload from event plane: {e:?}");
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

/// Helper to decide which subscriber (JetStream or Event Plane) to start based on config
pub async fn start_subscriber(
    component: Component,
    kv_router_config: &KvRouterConfig,
    indexer: Indexer,
    workers_with_configs: RuntimeConfigWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let transport_kind = component.drt().default_event_transport_kind();

    // Start subscriber - durable_kv_events flag determines the mode:
    // - durable_kv_events=false (default): Use NATS Core / generic event plane (requires workers to have local_indexer enabled)
    // - durable_kv_events=true: Use JetStream for durability and multi-replica consistency
    if kv_router_config.durable_kv_events {
        tracing::warn!(
            "--durable-kv-events is deprecated and will be removed in a future release. \
             The event-plane subscriber (local_indexer mode) is now the recommended path."
        );
        if transport_kind != EventTransportKind::Nats {
            anyhow::bail!(
                "--durable-kv-events requires NATS event plane, but runtime is configured for {transport_kind:?}"
            );
        }
        tracing::info!("Using JetStream subscription (--durable-kv-events enabled)");

        let consumer_id = component.drt().discovery().instance_id().to_string();
        super::jetstream::start_kv_router_background(
            component,
            consumer_id,
            indexer,
            kv_router_config,
            cancellation_token,
        )
        .await
    } else {
        if transport_kind == EventTransportKind::Zmq {
            if kv_router_config.router_snapshot_threshold.is_some()
                || kv_router_config.router_reset_states
            {
                tracing::warn!(
                    "ZMQ event plane does not support KV snapshots or state reset; ignoring snapshot/reset settings"
                );
            }
            tracing::info!("Using ZMQ event plane subscription (local_indexer mode)");
        } else {
            tracing::info!("Using NATS Core subscription (local_indexer mode)");
        }

        start_kv_router_background_event_plane(
            component,
            indexer,
            transport_kind,
            workers_with_configs,
            model,
            worker_type,
            cancellation_token,
        )
        .await
    }
}
