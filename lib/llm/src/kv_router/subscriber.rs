// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background processes for the KV Router including event consumption and snapshot uploads.

use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::{
    component::Component, prelude::*, traits::events::EventPublisher, transports::nats::NatsQueue,
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kv_router::{
    KV_EVENT_SUBJECT, RADIX_STATE_BUCKET, RADIX_STATE_FILE, ROUTER_SNAPSHOT_LOCK,
    indexer::{DumpRequest, RouterEvent},
};

/// Start a unified background task for event consumption and optional snapshot management
pub async fn start_kv_router_background(
    component: Component,
    consumer_uuid: String,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    snapshot_tx: Option<mpsc::Sender<DumpRequest>>,
    cancellation_token: CancellationToken,
    snapshot_threshold: Option<u32>,
    reset_states: bool,
) -> Result<()> {
    // Set up NATS connections
    let stream_name =
        format!("{}.{}", component.subject(), KV_EVENT_SUBJECT).replace(['/', '\\', '.', '_'], "-");
    let nats_server =
        std::env::var("NATS_SERVER").unwrap_or_else(|_| "nats://localhost:4222".to_string());

    // Create NatsQueue for event consumption
    let mut nats_queue = NatsQueue::new_with_consumer(
        stream_name.clone(),
        nats_server.clone(),
        std::time::Duration::from_secs(300), // Very long timeout (5 minutes)
        consumer_uuid,
    );
    nats_queue.connect_with_reset(reset_states).await?;

    // Always create NATS client (needed for both reset and snapshots)
    let client_options = dynamo_runtime::transports::nats::Client::builder()
        .server(&nats_server)
        .build()?;
    let nats_client = client_options.connect().await?;

    // Create bucket name for snapshots/state
    let bucket_name =
        format!("{}-{RADIX_STATE_BUCKET}", component.name()).replace(['/', '\\', '.', '_'], "-");

    // Handle initial state based on reset_states flag
    if reset_states {
        // Delete the bucket to reset state
        tracing::info!("Resetting router state, deleting bucket: {bucket_name}");
        if let Err(e) = nats_client.object_store_delete_bucket(&bucket_name).await {
            tracing::warn!("Failed to delete bucket (may not exist): {e:?}");
        }
    } else {
        // Try to download initial state from object store
        let url = url::Url::parse(&format!(
            "nats://{}/{bucket_name}/{RADIX_STATE_FILE}",
            nats_client.addr()
        ))?;

        match nats_client
            .object_store_download_data::<Vec<RouterEvent>>(url)
            .await
        {
            Ok(events) => {
                tracing::info!(
                    "Successfully downloaded {} events from object store",
                    events.len()
                );
                // Send all events to the indexer
                for event in events {
                    if let Err(e) = kv_events_tx.send(event).await {
                        tracing::warn!("Failed to send initial event to indexer: {e:?}");
                    }
                }
                tracing::info!("Successfully sent all initial events to indexer");
            }
            Err(e) => {
                tracing::info!(
                    "Did not initialize radix state from NATs object store (likely no snapshots yet): {e:?}"
                );
            }
        }
    }

    // Only set up snapshot-related resources if snapshot_tx is provided and threshold is set
    let snapshot_resources = if snapshot_tx.is_some() && snapshot_threshold.is_some() {
        // Get etcd client for distributed locking
        let etcd_client = component
            .drt()
            .etcd_client()
            .ok_or_else(|| anyhow::anyhow!("etcd client not available for distributed locking"))?;

        // Lock key for snapshot uploads
        let lock_key = format!("/{}/{}", ROUTER_SNAPSHOT_LOCK, component.name());

        Some((
            nats_client.clone(),
            bucket_name.clone(),
            etcd_client,
            lock_key,
        ))
    } else {
        None
    };

    tokio::spawn(async move {
        let mut check_interval = tokio::time::interval(Duration::from_secs(1));
        check_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("KV Router background task received cancellation signal");
                    // Clean up the queue and remove the durable consumer
                    if let Err(e) = nats_queue.shutdown().await {
                        tracing::warn!("Failed to shutdown NatsQueue: {e}");
                    }
                    break;
                }

                // Handle event consumption
                result = nats_queue.dequeue_task(Some(std::time::Duration::from_secs(0))) => {
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

                // Handle periodic stream checking and purging (only if snapshot_tx is provided)
                _ = check_interval.tick() => {
                    let Some((snapshot_tx, resources)) = snapshot_tx.as_ref().zip(snapshot_resources.as_ref()) else {
                        continue;
                    };

                    let (nats_client, bucket_name, etcd_client, lock_key) = resources;

                    // Check total messages in the stream
                    let Ok(message_count) = nats_queue.get_stream_messages().await else {
                        tracing::warn!("Failed to get stream message count");
                        continue;
                    };

                    // Guard clause: skip if message count is too low
                    let threshold = snapshot_threshold.unwrap_or(u32::MAX) as u64;
                    if message_count <= threshold {
                        continue;
                    }

                    tracing::info!("Stream has {message_count} messages, attempting to acquire lock for purge and snapshot");

                    // Try to acquire distributed lock
                    let lock_acquired = match etcd_client.kv_create(
                        lock_key,
                        b"locked".to_vec(),
                        Some(etcd_client.lease_id())
                    ).await {
                        Ok(_) => {
                            tracing::debug!("Successfully acquired snapshot lock");
                            true
                        }
                        Err(_) => {
                            tracing::debug!("Another instance already holds the snapshot lock");
                            false
                        }
                    };

                    // Guard clause: skip if lock not acquired
                    if !lock_acquired {
                        continue;
                    }

                    // Perform purge and snapshot upload
                    match perform_purge_and_snapshot(
                        &mut nats_queue,
                        nats_client,
                        snapshot_tx,
                        bucket_name
                    ).await {
                        Ok(_) => tracing::info!("Successfully performed purge and snapshot"),
                        Err(e) => tracing::error!("Failed to perform purge and snapshot: {e:?}"),
                    }

                    // Release the lock
                    if let Err(e) = etcd_client.kv_delete(lock_key.clone(), None).await {
                        tracing::warn!("Failed to release snapshot lock: {e:?}");
                    }
                }
            }
        }

        // Clean up the queue and remove the durable consumer
        if let Err(e) = nats_queue.shutdown().await {
            tracing::warn!("Failed to shutdown NatsQueue: {e}");
        }
    });

    Ok(())
}

/// Perform purge and snapshot upload operations
async fn perform_purge_and_snapshot(
    nats_queue: &mut NatsQueue,
    nats_client: &dynamo_runtime::transports::nats::Client,
    snapshot_tx: &mpsc::Sender<DumpRequest>,
    bucket_name: &str,
) -> anyhow::Result<()> {
    // TODO: Radix tree snapshot may not match ack floor unless this replica has min ack'ed seq.
    // Could add sleep to increase likelihood. However, radix tree can only be ahead of purge point
    // (not behind), and KV events are idempotent, so replaying already-applied events is safe.

    // First, perform the purge of acknowledged messages
    nats_queue.purge_acknowledged().await?;

    // Request a snapshot from the indexer
    let (resp_tx, resp_rx) = oneshot::channel();
    let dump_req = DumpRequest { resp: resp_tx };

    snapshot_tx
        .send(dump_req)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to send dump request: {e:?}"))?;

    // Wait for the dump response
    let events = resp_rx
        .await
        .map_err(|e| anyhow::anyhow!("Failed to receive dump response: {e:?}"))?;

    // Upload the snapshot to NATS object store
    let url = url::Url::parse(&format!(
        "nats://{}/{bucket_name}/{RADIX_STATE_FILE}",
        nats_client.addr()
    ))?;

    nats_client
        .object_store_upload_data(&events, url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to upload snapshot: {e:?}"))?;

    tracing::info!(
        "Successfully uploaded radix tree snapshot with {} events to bucket {bucket_name}",
        events.len()
    );

    Ok(())
}
