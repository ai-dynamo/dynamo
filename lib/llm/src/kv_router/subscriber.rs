// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::kv_router::{
    Indexer, KV_EVENT_SUBJECT, KvRouterConfig,
    protocols::{DpRank, RouterEvent, WorkerId},
    worker_query::WorkerQueryClient,
};
use anyhow::Result;
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
/// This function first recovers state from all currently registered workers before
/// spawning the background task, ensuring the router is ready before returning.
///
/// This is appropriate when workers have local indexers enabled.
async fn start_kv_router_background_event_plane(
    component: Component,
    indexer: Indexer,
    transport_kind: EventTransportKind,
) -> Result<()> {
    let cancellation_token = component.drt().primary_token();
    // WorkerQueryClient handles its own discovery loop for lifecycle + initial recovery.
    // No blocking wait — recovery happens asynchronously as endpoints are discovered.
    let worker_query_client = WorkerQueryClient::spawn(component.clone(), indexer.clone()).await?;

    // Subscribe to KV events using the selected event plane transport
    let mut subscriber =
        EventSubscriber::for_component_with_transport(&component, KV_EVENT_SUBJECT, transport_kind)
            .await?
            .typed::<RouterEvent>();
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
        // Track last received event ID per (worker, dp_rank) for gap detection
        // Each dp_rank has its own monotonic event ID sequence
        let mut last_event_ids: HashMap<(WorkerId, DpRank), u64> = HashMap::new();

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

                    let worker_id = event.worker_id;
                    let dp_rank = event.event.dp_rank;
                    let event_id = event.event.event_id;
                    let event_key = (worker_id, dp_rank);

                    tracing::trace!(
                        "Received event from publisher {} (seq {})",
                        envelope.publisher_id,
                        envelope.sequence
                    );

                    // Gap detection: check if event ID is monotonically increasing per (worker, dp_rank)
                    // Note: event_id <= last_id is duplicate/out-of-order, apply anyway (idempotent)
                    if let Some(&last_id) = last_event_ids.get(&event_key)
                        && event_id > last_id + 1
                    {
                        let gap_start = last_id + 1;
                        let gap_end = event_id - 1;
                        let gap_size = gap_end - gap_start + 1;
                        tracing::warn!(
                            "Event ID gap detected for worker {worker_id} dp_rank {dp_rank}, recovering events [{gap_start}, {gap_end}], gap_size: {gap_size}"
                        );

                        if let Err(e) = worker_query_client
                            .recover_from_worker(worker_id, dp_rank, Some(gap_start), Some(gap_end))
                            .await
                        {
                            tracing::error!(
                                "Failed to recover gap events for worker {worker_id} dp_rank {dp_rank} (gap_start: {gap_start}, gap_end: {gap_end}); proceeding with current event anyway: {e}"
                            );
                        }
                    }

                    // Update last seen event ID (use max to handle out-of-order)
                    last_event_ids
                        .entry(event_key)
                        .and_modify(|id| *id = (*id).max(event_id))
                        .or_insert(event_id);

                    // Forward the RouterEvent to the indexer
                    indexer.apply_event(event).await;
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
