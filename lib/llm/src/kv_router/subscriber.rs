// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background processes for the KV Router including event consumption and snapshot uploads.

use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    prelude::*,
    traits::events::EventPublisher,
    transports::{
        etcd::WatchEvent,
        nats::{NatsQueue, Slug},
    },
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::KV_ROUTERS_ROOT_PATH,
    kv_router::{
        KV_EVENT_SUBJECT, RADIX_STATE_BUCKET, RADIX_STATE_FILE, ROUTER_CLEANUP_LOCK,
        ROUTER_SNAPSHOT_LOCK,
        indexer::{DumpRequest, RouterEvent},
    },
};

/// Resources required for snapshot operations
#[derive(Clone)]
struct SnapshotResources {
    nats_client: dynamo_runtime::transports::nats::Client,
    bucket_name: String,
    etcd_client: dynamo_runtime::transports::etcd::Client,
    lock_name: String,
}

impl SnapshotResources {
    /// Try to acquire distributed lock for snapshot operations
    /// Returns Some(lock_response) if lock acquired, None if another instance holds it
    async fn lock(&self) -> Option<etcd_client::LockResponse> {
        match self
            .etcd_client
            .lock(self.lock_name.clone(), Some(self.etcd_client.lease_id()))
            .await
        {
            Ok(response) => {
                tracing::debug!(
                    "Successfully acquired snapshot lock with key: {:?}",
                    response.key()
                );
                Some(response)
            }
            Err(e) => {
                tracing::debug!("Another instance already holds the snapshot lock: {e:?}");
                None
            }
        }
    }

    /// Release the distributed lock
    async fn unlock(&self, lock_response: etcd_client::LockResponse) {
        if let Err(e) = self.etcd_client.unlock(lock_response.key()).await {
            tracing::warn!("Failed to release snapshot lock: {e:?}");
        }
    }
}

/// Start a unified background task for event consumption and optional snapshot management
#[allow(clippy::too_many_arguments)]
pub async fn start_kv_router_background(
    component: Component,
    consumer_uuid: String,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    remove_worker_tx: mpsc::Sender<crate::kv_router::indexer::WorkerId>,
    snapshot_tx: Option<mpsc::Sender<DumpRequest>>,
    cancellation_token: CancellationToken,
    router_snapshot_threshold: Option<u32>,
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
        consumer_uuid,
    );
    nats_queue.connect_with_reset(router_reset_states).await?;

    // Always create NATS client (needed for both reset and snapshots)
    let client_options = dynamo_runtime::transports::nats::Client::builder()
        .server(&nats_server)
        .build()?;
    let nats_client = client_options.connect().await?;

    // Create bucket name for snapshots/state
    let bucket_name = Slug::slugify(&format!("{}-{RADIX_STATE_BUCKET}", component.subject()))
        .to_string()
        .replace("_", "-");

    // Handle initial state based on router_reset_states flag
    if router_reset_states {
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
            .object_store_download_data::<Vec<RouterEvent>>(&url)
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

    // Get etcd client (needed for both snapshots and router watching)
    let etcd_client = component
        .drt()
        .etcd_client()
        .ok_or_else(|| anyhow::anyhow!("etcd client not available"))?;

    // Watch for router deletions to clean up orphaned consumers
    let (_prefix_str, _watcher, mut router_replicas_rx) = etcd_client
        .kv_get_and_watch_prefix(&format!("{}/", KV_ROUTERS_ROOT_PATH))
        .await?
        .dissolve();
    let cleanup_lock_name = format!("{}/{}", ROUTER_CLEANUP_LOCK, component.subject());

    // Get the generate endpoint and watch for instance deletions
    let generate_endpoint = component.endpoint("generate");
    let (_instance_prefix, _instance_watcher, mut instance_event_rx) = etcd_client
        .kv_get_and_watch_prefix(generate_endpoint.etcd_root())
        .await?
        .dissolve();

    // Only set up snapshot-related resources if snapshot_tx is provided and threshold is set
    let snapshot_resources = if snapshot_tx.is_some() && router_snapshot_threshold.is_some() {
        let lock_name = format!("{}/{}", ROUTER_SNAPSHOT_LOCK, component.subject());

        Some(SnapshotResources {
            nats_client,
            bucket_name,
            etcd_client: etcd_client.clone(),
            lock_name,
        })
    } else {
        None
    };

    tokio::spawn(async move {
        let mut check_interval = tokio::time::interval(Duration::from_secs(1));
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

                    // Extract the hex worker ID after the colon (e.g., "generate:694d99badb9f7c07" -> "694d99badb9f7c07")
                    let Some(worker_id_str) = key.split(':').next_back() else {
                        tracing::warn!("Could not extract worker ID from instance key: {}", key);
                        continue;
                    };

                    // Parse as hexadecimal (base 16)
                    let Ok(worker_id) = i64::from_str_radix(worker_id_str, 16) else {
                        tracing::warn!("Could not parse worker ID from instance key: {}", key);
                        continue;
                    };

                    tracing::info!("Generate endpoint instance deleted, removing worker {}", worker_id);
                    if let Err(e) = remove_worker_tx.send(worker_id).await {
                        tracing::warn!("Failed to send worker removal for worker {}: {}", worker_id, e);
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

                // Handle periodic stream checking and purging (only if snapshot_tx is provided)
                _ = check_interval.tick() => {
                    let Some((snapshot_tx, resources)) = snapshot_tx.as_ref().zip(snapshot_resources.as_ref()) else {
                        continue;
                    };

                    // Check total messages in the stream
                    let Ok(message_count) = nats_queue.get_stream_messages().await else {
                        tracing::warn!("Failed to get stream message count");
                        continue;
                    };

                    // Guard clause: skip if message count is too low
                    let threshold = router_snapshot_threshold.unwrap_or(u32::MAX) as u64;
                    if message_count <= threshold {
                        continue;
                    }

                    tracing::info!("Stream has {message_count} messages, attempting to acquire lock for purge and snapshot");

                    // Try to acquire distributed lock
                    let Some(lock_response) = resources.lock().await else {
                        continue;
                    };

                    // Perform snapshot upload and purge
                    match purge_then_snapshot(
                        &mut nats_queue,
                        snapshot_tx,
                        resources
                    ).await {
                        Ok(_) => tracing::info!("Successfully performed purge and snapshot"),
                        Err(e) => tracing::error!("Failed to perform purge and snapshot: {e:?}"),
                    }

                    // Release the lock
                    resources.unlock(lock_response).await;
                }

                // Handle router deletion events
                Some(event) = router_replicas_rx.recv() => {
                    let WatchEvent::Delete(kv) = event else {
                        // We only care about deletions for cleaning up consumers
                        continue;
                    };

                    let key = String::from_utf8_lossy(kv.key());
                    tracing::info!("Router deleted: {}", key);

                    // Extract the router UUID from the key (format: kv_routers/<model>/<uuid>)
                    let Some(router_uuid) = key.split('/').next_back() else {
                        tracing::warn!("Could not extract UUID from router key: {}", key);
                        continue;
                    };

                    // The consumer UUID is the router UUID
                    let consumer_to_delete = router_uuid.to_string();

                    tracing::info!("Attempting to delete orphaned consumer: {}", consumer_to_delete);

                    // Try to acquire cleanup lock before deleting consumer
                    match etcd_client
                        .lock(cleanup_lock_name.clone(), Some(etcd_client.lease_id()))
                        .await
                    {
                        Ok(lock_response) => {
                            tracing::debug!(
                                "Acquired cleanup lock for deleting consumer: {}",
                                consumer_to_delete
                            );

                            // Delete the consumer
                            if let Err(e) = nats_queue.shutdown(Some(consumer_to_delete.clone())).await {
                                tracing::warn!("Failed to delete consumer {}: {}", consumer_to_delete, e);
                            } else {
                                tracing::info!("Successfully deleted orphaned consumer: {}", consumer_to_delete);
                            }

                            // Release the lock
                            if let Err(e) = etcd_client.unlock(lock_response.key()).await {
                                tracing::warn!("Failed to release cleanup lock: {e:?}");
                            }
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Could not acquire cleanup lock for consumer {}: {e:?}",
                                consumer_to_delete
                            );
                        }
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

/// Perform snapshot upload and purge operations
async fn purge_then_snapshot(
    nats_queue: &mut NatsQueue,
    snapshot_tx: &mpsc::Sender<DumpRequest>,
    resources: &SnapshotResources,
) -> anyhow::Result<()> {
    // Purge before snapshot ensures new/warm-restarted routers won't replay already-acknowledged messages.
    // Since KV events are idempotent, this ordering reduces unnecessary reprocessing while maintaining
    // at-least-once delivery guarantees. The snapshot will capture the clean state after purge.

    // First, purge acknowledged messages from the stream
    nats_queue.purge_acknowledged().await?;

    // Now request a snapshot from the indexer (which reflects the post-purge state)
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
        "nats://{}/{}/{RADIX_STATE_FILE}",
        resources.nats_client.addr(),
        resources.bucket_name
    ))?;

    resources
        .nats_client
        .object_store_upload_data(&events, &url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to upload snapshot: {e:?}"))?;

    tracing::info!(
        "Successfully uploaded radix tree snapshot with {} events to bucket {}",
        events.len(),
        resources.bucket_name
    );

    Ok(())
}
