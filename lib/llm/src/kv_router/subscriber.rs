// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background processes for the KV Router including event consumption and snapshot uploads.

use std::{collections::HashSet, time::Duration};

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    prelude::*,
    traits::events::EventPublisher,
    transports::{
        etcd::{Client as EtcdClient, WatchEvent},
        nats::{NatsQueue, Slug},
    },
};
use rand::Rng;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::KV_ROUTERS_ROOT_PATH,
    kv_router::{
        KV_EVENT_SUBJECT,
        indexer::{DumpRequest, GetWorkersRequest, RouterEvent},
        protocols::WorkerId,
    },
};

const CHECK_INTERVAL_BASE: Duration = Duration::from_secs(1);
const CHECK_INTERVAL_JITTER_MS: i64 = 100;

/// Download snapshot from peer router via Dynamo endpoint.
async fn download_snapshot_from_peer(
    component: &Component,
    kv_events_tx: &mpsc::Sender<RouterEvent>,
) -> Result<()> {
    tracing::info!("Attempting to fetch snapshot from peer routers via Dynamo endpoint...");

    // List instances of the same component (peer routers)
    let instances = match component.list_instances().await {
        Ok(instances) => instances,
        Err(e) => {
            tracing::debug!("Failed to list router instances: {:?}", e);
            return Err(e);
        }
    };

    if instances.is_empty() {
        tracing::debug!("No peer router instances found");
        return Err(anyhow::anyhow!("No peer routers available"));
    }

    tracing::info!(
        "Found {} peer router instance(s), will attempt to fetch snapshot",
        instances.len()
    );

    // Try each peer instance
    for instance in instances {
        let instance_id = instance.id();
        tracing::debug!("Trying to fetch snapshot from instance {}", instance_id);

        // Create endpoint client for snapshot endpoint
        let snapshot_endpoint = component.endpoint("snapshot");
        let client = match snapshot_endpoint.client().await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "Failed to create client for instance {}: {:?}",
                    instance_id,
                    e
                );
                continue;
            }
        };

        // Use PushRouter pattern (follows existing patterns in codebase)
        use dynamo_runtime::pipeline::PushRouter;
        use dynamo_runtime::protocols::annotated::Annotated;
        use dynamo_runtime::protocols::maybe_error::MaybeError;
        use dynamo_runtime::stream::StreamExt;

        let router = match PushRouter::<(), Annotated<serde_json::Value>>::from_client(
            client,
            Default::default(),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    "Failed to create router for instance {}: {:?}",
                    instance_id,
                    e
                );
                continue;
            }
        };

        // Call snapshot endpoint (transport-agnostic - uses Dynamo's request plane)
        let mut stream = match router.direct(().into(), instance_id).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "Failed to call snapshot endpoint on instance {}: {:?}",
                    instance_id,
                    e
                );
                continue;
            }
        };

        // Get response
        if let Some(annotated_response) = stream.next().await {
            // Check for errors in the annotated response
            if let Some(err) = annotated_response.err() {
                tracing::warn!(
                    "Snapshot endpoint returned error from instance {}: {:?}",
                    instance_id,
                    err
                );
                continue;
            }

            // Extract the data from the annotated response
            let response = match annotated_response.data {
                Some(data) => data,
                None => {
                    tracing::warn!(
                        "Snapshot response from instance {} has no data",
                        instance_id
                    );
                    continue;
                }
            };

            // Parse snapshot response
            let events_value = &response["events"];
            if events_value.is_null() {
                tracing::warn!(
                    "Snapshot response from instance {} has no events",
                    instance_id
                );
                continue;
            }

            let events: Vec<RouterEvent> = match serde_json::from_value(events_value.clone()) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        "Failed to deserialize events from instance {}: {:?}",
                        instance_id,
                        e
                    );
                    continue;
                }
            };

            tracing::info!(
                "Successfully fetched snapshot with {} events from peer instance {} via Dynamo endpoint",
                events.len(),
                instance_id
            );

            // Send events to indexer to reconstruct tree
            for event in events {
                if let Err(e) = kv_events_tx.send(event).await {
                    tracing::error!("Failed to send event to indexer: {:?}", e);
                    return Err(anyhow::anyhow!("Failed to send events to indexer"));
                }
            }

            tracing::info!("Snapshot loaded from peer router (sequence-based recovery complete)");
            return Ok(());
        }
    }

    tracing::debug!("Could not fetch snapshot from any peer router");
    Err(anyhow::anyhow!("No peer snapshot available"))
}

/// Start a unified background task for event consumption
#[allow(clippy::too_many_arguments)]
pub async fn start_kv_router_background(
    component: Component,
    consumer_uuid: String,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    remove_worker_tx: mpsc::Sender<WorkerId>,
    _maybe_get_workers_tx: Option<mpsc::Sender<GetWorkersRequest>>,
    _maybe_snapshot_tx: Option<mpsc::Sender<DumpRequest>>,
    cancellation_token: CancellationToken,
    _router_snapshot_threshold: Option<u32>,
    router_reset_states: bool,
) -> Result<()> {
    // Set up NATS connections
    let stream_name = Slug::slugify(&format!("{}.{}", component.subject(), KV_EVENT_SUBJECT))
        .to_string()
        .replace("_", "-");
    let nats_server =
        std::env::var("NATS_SERVER").unwrap_or_else(|_| "nats://localhost:4222".to_string());

    // Create NatsQueue for event consumption
    let mut nats_queue = NatsQueue::new_with_consumer(
        stream_name.clone(),
        nats_server.clone(),
        std::time::Duration::from_secs(60), // 1 minute timeout
        consumer_uuid.clone(),
    );
    nats_queue.connect_with_reset(router_reset_states).await?;

    // Get etcd client (needed for router watching)
    let etcd_client = component
        .drt()
        .etcd_client()
        .ok_or_else(|| anyhow::anyhow!("etcd client not available"))?;

    // Handle initial state based on router_reset_states flag
    if !router_reset_states {
        // Try peer snapshot first (transport-agnostic, no NATS object store)
        match download_snapshot_from_peer(&component, &kv_events_tx).await {
            Ok(_) => {
                tracing::info!("Router state loaded from peer via Dynamo endpoint");
            }
            Err(e) => {
                tracing::info!(
                    "Peer snapshot not available ({}), starting with empty KV tree. \
                    Tree will rebuild naturally as workers send KV events.",
                    e
                );
            }
        }
    } else {
        tracing::info!("Router reset_states flag enabled, starting with empty KV tree");
    }

    // Cleanup orphaned consumers on startup
    cleanup_orphaned_consumers(&mut nats_queue, &etcd_client, &component, &consumer_uuid).await;

    // Watch for router deletions to clean up orphaned consumers
    let (_prefix_str, mut router_replicas_rx) = etcd_client
        .kv_get_and_watch_prefix(&format!("{}/", KV_ROUTERS_ROOT_PATH))
        .await?
        .dissolve();

    // Get the generate endpoint and watch for instance deletions
    let generate_endpoint = component.endpoint("generate");
    let (_instance_prefix, mut instance_event_rx) = etcd_client
        .kv_get_and_watch_prefix(generate_endpoint.etcd_root())
        .await?
        .dissolve();

    // Verify we have dynamic instance source for KV routing
    let client = generate_endpoint.client().await?;
    match client.instance_source.as_ref() {
        dynamo_runtime::component::InstanceSource::Dynamic(_) => {}
        dynamo_runtime::component::InstanceSource::Static => {
            anyhow::bail!("Expected dynamic instance source for KV routing");
        }
    };

    tokio::spawn(async move {
        // Create interval with jitter
        let jitter_ms =
            rand::rng().random_range(-CHECK_INTERVAL_JITTER_MS..=CHECK_INTERVAL_JITTER_MS);
        let interval_duration = Duration::from_millis(
            (CHECK_INTERVAL_BASE.as_millis() as i64 + jitter_ms).max(1) as u64,
        );
        let mut check_interval = tokio::time::interval(interval_duration);
        check_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;

                _ = cancellation_token.cancelled() => {
                    tracing::debug!("KV Router background task received cancellation signal");
                    // Clean up the queue and remove the durable consumer
                    // TODO: durable consumer cannot cleanup if ungraceful shutdown (crash)
                    if let Err(e) = nats_queue.shutdown(None).await {
                        tracing::warn!("Failed to shutdown NatsQueue: {e}");
                    }
                    break;
                }

                // Handle generate endpoint instance deletion events
                Some(event) = instance_event_rx.recv() => {
                    let WatchEvent::Delete(kv) = event else {
                        continue;
                    };

                    let key = String::from_utf8_lossy(kv.key());

                    let Some(worker_id_str) = key.split(&['/', ':'][..]).next_back() else {
                        tracing::warn!("Could not extract worker ID from instance key: {key}");
                        continue;
                    };

                    // Parse as hexadecimal (base 16)
                    let Ok(worker_id) = u64::from_str_radix(worker_id_str, 16) else {
                        tracing::warn!("Could not parse worker ID from instance key: {key}");
                        continue;
                    };

                    tracing::info!("Generate endpoint instance deleted, removing worker {worker_id}");
                    if let Err(e) = remove_worker_tx.send(worker_id).await {
                        tracing::warn!("Failed to send worker removal for worker {worker_id}: {e}");
                    }
                }

                // Handle event consumption
                result = nats_queue.dequeue_task(None) => {
                    match result {
                        Ok(Some(bytes)) => {
                            let event: RouterEvent = match serde_json::from_slice(&bytes) {
                                Ok(event) => event,
                                Err(e) => {
                                    tracing::warn!("Failed to deserialize RouterEvent: {e:?}");
                                    continue;
                                }
                            };

                            // Forward the RouterEvent to the indexer
                            if let Err(e) = kv_events_tx.send(event).await {
                                tracing::warn!(
                                    "failed to send kv event to indexer; shutting down: {e:?}"
                                );
                                break;
                            }
                        },
                        Ok(None) => {
                            tracing::trace!("Dequeue timeout, continuing");
                        },
                        Err(e) => {
                            tracing::error!("Failed to dequeue task: {e:?}");
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }

                // Handle router deletion events
                Some(event) = router_replicas_rx.recv() => {
                    let WatchEvent::Delete(kv) = event else {
                        // We only care about deletions for cleaning up consumers
                        continue;
                    };

                    let key = String::from_utf8_lossy(kv.key());
                    tracing::info!("Detected router replica deletion: {key}");

                    // Only process deletions for routers on the same component
                    if !key.contains(component.path().as_str()) {
                        tracing::trace!(
                            "Skipping router deletion from different component (key: {key}, subscriber component: {})",
                            component.path()
                        );
                        continue;
                    }

                    // Extract the router UUID from the key
                    let Some(router_uuid) = key.split('/').next_back() else {
                        tracing::warn!("Could not extract UUID from router key: {key}");
                        continue;
                    };

                    // The consumer UUID is the router UUID
                    let consumer_to_delete = router_uuid.to_string();

                    tracing::info!("Attempting to delete orphaned consumer: {consumer_to_delete}");

                    // Delete the consumer (allow race condition if multiple routers try to delete)
                    if let Err(e) = nats_queue.shutdown(Some(consumer_to_delete.clone())).await {
                        tracing::warn!("Failed to delete consumer {consumer_to_delete}: {e}");
                    } else {
                        tracing::info!("Successfully deleted orphaned consumer: {consumer_to_delete}");
                    }
                }
            }
        }

        // Clean up the queue and remove the durable consumer
        if let Err(e) = nats_queue.shutdown(None).await {
            tracing::warn!("Failed to shutdown NatsQueue: {e}");
        }
    });

    Ok(())
}

/// Cleanup orphaned NATS consumers that no longer have corresponding etcd router entries
async fn cleanup_orphaned_consumers(
    nats_queue: &mut NatsQueue,
    etcd_client: &EtcdClient,
    component: &Component,
    consumer_uuid: &str,
) {
    let Ok(consumers) = nats_queue.list_consumers().await else {
        return;
    };

    let router_prefix = format!("{}/{}/", KV_ROUTERS_ROOT_PATH, component.path());
    let Ok(router_entries) = etcd_client.kv_get_prefix(&router_prefix).await else {
        return;
    };

    let active_uuids: HashSet<String> = router_entries
        .iter()
        .filter_map(|kv| {
            String::from_utf8_lossy(kv.key())
                .split('/')
                .next_back()
                .map(str::to_string)
        })
        .collect();

    for consumer in consumers {
        if consumer == consumer_uuid {
            // Never delete myself (extra/redundant safeguard)
            continue;
        }
        if !active_uuids.contains(&consumer) {
            tracing::info!("Cleaning up orphaned consumer: {consumer}");
            let _ = nats_queue.shutdown(Some(consumer)).await;
        }
    }
}
