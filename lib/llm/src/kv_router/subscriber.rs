// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::{Indexer, worker_query::WorkerQueryClient};
use anyhow::Result;
use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{KV_EVENT_SUBJECT, RouterEvent},
};
use dynamo_runtime::{
    component::Component, discovery::EventTransportKind, prelude::*,
    transports::event_plane::EventSubscriber,
};

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
) -> Result<()> {
    let cancellation_token = component.drt().primary_token();

    // Subscribe to KV events BEFORE spawning the discovery/recovery loop.
    // This ensures no events are lost between the initial dump fetch and the
    // subscription becoming active — the tree state at fetch time is guaranteed
    // to be a subset of what the subscription will deliver.
    let mut subscriber =
        EventSubscriber::for_component_with_transport(&component, KV_EVENT_SUBJECT, transport_kind)
            .await?
            .typed::<RouterEvent>();

    // Brief delay to let the subscription fully establish with the NATS server
    // before recovery fetches the initial dump from workers.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // WorkerQueryClient handles its own discovery loop for lifecycle + initial recovery.
    // No blocking wait — recovery happens asynchronously as endpoints are discovered.
    let worker_query_client = WorkerQueryClient::spawn(component.clone(), indexer).await?;
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
        loop {
            tokio::select! {
                biased;

                _ = cancellation_token.cancelled() => {
                    tracing::debug!("KV Router event plane background task received cancellation signal");
                    break;
                }

                // Handle event consumption from event plane subscription
                Some(result) = subscriber.next() => {
                    let (envelope, event) = match result {
                        Ok((envelope, event)) => (envelope, event),
                        Err(e) => {
                            tracing::warn!("Failed to receive RouterEvent from event plane: {e:?}");
                            continue;
                        }
                    };

                    tracing::trace!(
                        "Received event from publisher {} (seq {})",
                        envelope.publisher_id,
                        envelope.sequence
                    );

                    tracing::trace!(
                        "Forwarding live event to recovery coordinator for worker {} dp_rank {} event_id {}",
                        event.worker_id,
                        event.event.dp_rank,
                        event.event.event_id
                    );
                    worker_query_client.handle_live_event(event).await;
                }
            }
        }

        tracing::debug!("KV Router event plane background task exiting");
    });

    Ok(())
}

/// Helper to decide which subscriber (JetStream or Event Plane) to start based on config
pub async fn start_subscriber(
    component: Component,
    kv_router_config: &KvRouterConfig,
    indexer: Indexer,
) -> Result<()> {
    let transport_kind = EventTransportKind::from_env_or_default();

    // Start subscriber - durable_kv_events flag determines the mode:
    // - durable_kv_events=false (default): Use NATS Core / generic event plane (requires workers to have local_indexer enabled)
    // - durable_kv_events=true: Use JetStream for durability and multi-replica consistency
    if kv_router_config.durable_kv_events {
        tracing::warn!(
            "--durable-kv-events is deprecated and will be removed in a future release. \
             The event-plane subscriber (local_indexer mode) is now the recommended path."
        );
        if transport_kind == EventTransportKind::Zmq {
            tracing::warn!(
                "--durable-kv-events requires NATS, but ZMQ event plane is configured; falling back to JetStream anyway"
            );
        }
        tracing::info!("Using JetStream subscription (--durable-kv-events enabled)");

        let consumer_id = component.drt().discovery().instance_id().to_string();
        super::jetstream::start_kv_router_background(
            component,
            consumer_id,
            indexer,
            kv_router_config,
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

        start_kv_router_background_event_plane(component, indexer, transport_kind).await
    }
}
