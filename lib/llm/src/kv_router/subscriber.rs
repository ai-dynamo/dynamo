// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, collections::HashSet, time::Duration};

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    config::environment_names::nats as env_nats,
    discovery::{DiscoveryEvent, DiscoveryQuery},
    prelude::*,
    traits::events::EventPublisher,
    transports::nats::{NatsQueue, Slug},
};
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kv_router::{
    KV_EVENT_SUBJECT, RADIX_STATE_BUCKET, RADIX_STATE_FILE,
    indexer::{DumpRequest, GetWorkersRequest, RouterEvent},
    protocols::WorkerId,
    router_discovery_query,
    worker_query::WorkerQueryClient,
};

/// Delay between snapshot reads to verify stability
const SNAPSHOT_STABILITY_DELAY: Duration = Duration::from_millis(100);
const MAX_SNAPSHOT_STABILITY_ATTEMPTS: usize = 10;

const CHECK_INTERVAL_BASE: Duration = Duration::from_secs(1);
const CHECK_INTERVAL_JITTER_MS: i64 = 100;

// ============================================================================
// Local KvIndexer-based Recovery
// ============================================================================

/// Recover missed events from all workers with local indexers.
///
/// This function should be called on router startup to catch up on any events
/// that were missed while the router was offline.
///
/// # Arguments
///
/// * `worker_query_client` - Client for querying worker local indexers
/// * `last_received_event_ids` - Map of worker ID to last received event ID
/// * `worker_ids` - List of worker IDs to recover from
/// * `event_tx` - Channel to send recovered events to the indexer
///
/// # Returns
///
/// Total number of events recovered across all workers
pub async fn recover_from_all_workers(
    worker_query_client: &WorkerQueryClient,
    last_received_event_ids: &HashMap<WorkerId, u64>,
    worker_ids: &Vec<WorkerId>,
    event_tx: &mpsc::Sender<RouterEvent>,
) -> usize {
    let mut total_recovered = 0;
    let mut successful_workers = 0;
    let mut failed_workers = 0;

    for &worker_id in worker_ids {
        // Skip workers without local indexer
        if !worker_query_client.has_local_indexer(worker_id) {
            tracing::debug!(
                worker_id,
                "Skipping recovery - worker does not have local indexer enabled"
            );
            continue;
        }

        // If we haven't seen any events from this worker, start from beginning (None)
        // If we've seen events, start from last_known_id + 1
        let start_event_id = last_received_event_ids
            .get(&worker_id)
            .map(|&last_id| last_id + 1);

        match recover_from_worker(
            worker_query_client,
            worker_id,
            start_event_id,
            None, // Get all events after start_event_id
            event_tx,
        )
        .await
        {
            Ok(count) => {
                total_recovered += count;
                if count > 0 {
                    successful_workers += 1;
                }
            }
            Err(_) => {
                failed_workers += 1;
            }
        }
    }

    // Log summary
    if total_recovered > 0 || failed_workers > 0 {
        tracing::info!(
            total_recovered,
            successful_workers,
            failed_workers,
            "Startup recovery completed"
        );
    }

    total_recovered
}

/// Recover missed KV events from a specific worker.
///
/// # Arguments
///
/// * `worker_query_client` - Client for querying worker local indexers
/// * `worker_id` - The worker to recover from
/// * `start_event_id` - First event ID to fetch (inclusive), or None to start from beginning
/// * `end_event_id` - Last event ID to fetch (inclusive), or None for all
/// * `event_tx` - Channel to send recovered events to the indexer
///
/// # Returns
///
/// Number of events recovered, or error if recovery failed
pub async fn recover_from_worker(
    worker_query_client: &WorkerQueryClient,
    worker_id: WorkerId,
    start_event_id: Option<u64>,
    end_event_id: Option<u64>,
    event_tx: &mpsc::Sender<RouterEvent>,
) -> Result<usize> {
    if worker_query_client.has_local_indexer(worker_id) {
        tracing::debug!(
            worker_id,
            start_event_id = ?start_event_id,
            end_event_id = ?end_event_id,
            "Attempting recovery from worker"
        );
    } else {
        tracing::warn!(
            "Worker {} does not have local indexer enabled, skipping recovery",
            worker_id
        );
        return Ok(0);
    }

    // Query worker for events in range
    let response = worker_query_client
        .query_worker(worker_id, start_event_id, end_event_id)
        .await?;

    let events_count = response.events.len();

    if events_count == 0 {
        tracing::debug!(
            worker_id,
            start_event_id = ?start_event_id,
            "No missed events to recover from worker"
        );
        return Ok(0);
    }

    tracing::info!(
        worker_id,
        start_event_id = ?start_event_id,
        events_count,
        "Recovered {} missed events from worker",
        events_count
    );

    // Apply recovered events to the indexer
    for event in response.events {
        if let Err(e) = event_tx.send(event).await {
            tracing::error!(
                worker_id,
                error = %e,
                "Failed to send recovered event to indexer"
            );
            anyhow::bail!("Failed to send recovered event: {}", e);
        }
    }

    Ok(events_count)
}

// ============================================================================
// Snapshot Management
// ============================================================================

/// Download a stable snapshot from object store and send events to the indexer.
/// Retries until two consecutive reads match or max attempts is reached.
async fn download_stable_snapshot(
    nats_client: &dynamo_runtime::transports::nats::Client,
    bucket_name: &str,
    kv_events_tx: &mpsc::Sender<RouterEvent>,
) -> Result<()> {
    let url = url::Url::parse(&format!(
        "nats://{}/{bucket_name}/{RADIX_STATE_FILE}",
        nats_client.addr()
    ))?;

    // Try to get initial snapshot
    let Ok(mut prev_events) = nats_client
        .object_store_download_data::<Vec<RouterEvent>>(&url)
        .await
    else {
        tracing::debug!(
            "Failed to download snapshots. This is normal for freshly started Router replicas."
        );
        return Ok(());
    };

    // Keep trying until we get two consecutive stable reads
    for attempt in 1..=MAX_SNAPSHOT_STABILITY_ATTEMPTS {
        tokio::time::sleep(SNAPSHOT_STABILITY_DELAY).await;

        let curr_events = match nats_client
            .object_store_download_data::<Vec<RouterEvent>>(&url)
            .await
        {
            Ok(events) => events,
            Err(e) => {
                tracing::warn!(
                    "Snapshot read failed on attempt {attempt}, using previous snapshot with {} events: {e:?}",
                    prev_events.len()
                );
                break;
            }
        };

        // Check if snapshot is stable (two consecutive reads match)
        if prev_events == curr_events {
            tracing::info!(
                "Successfully downloaded stable snapshot with {} events from object store (stable after {attempt} attempts)",
                curr_events.len()
            );
            prev_events = curr_events;
            break;
        }

        tracing::debug!(
            "Snapshot changed between reads on attempt {attempt} ({} -> {} events), retrying",
            prev_events.len(),
            curr_events.len()
        );
        prev_events = curr_events;

        if attempt == MAX_SNAPSHOT_STABILITY_ATTEMPTS {
            tracing::warn!(
                "Max stability attempts reached, using latest snapshot with {} events",
                prev_events.len()
            );
        }
    }

    // Send all events to the indexer
    for event in prev_events {
        if let Err(e) = kv_events_tx.send(event).await {
            tracing::warn!("Failed to send initial event to indexer: {e:?}");
        }
    }
    tracing::info!("Successfully sent all initial events to indexer");

    Ok(())
}

/// Resources required for snapshot operations
#[derive(Clone)]
struct SnapshotResources {
    nats_client: dynamo_runtime::transports::nats::Client,
    bucket_name: String,
    instances_rx: tokio::sync::watch::Receiver<Vec<dynamo_runtime::component::Instance>>,
    get_workers_tx: mpsc::Sender<GetWorkersRequest>,
    snapshot_tx: mpsc::Sender<DumpRequest>,
}

impl SnapshotResources {
    /// Perform snapshot upload and purge operations
    async fn purge_then_snapshot(
        &self,
        nats_queue: &mut NatsQueue,
        remove_worker_tx: &mpsc::Sender<WorkerId>,
    ) -> anyhow::Result<()> {
        // Purge before snapshot ensures new/warm-restarted routers won't replay already-acknowledged messages.
        // Since KV events are idempotent, this ordering reduces unnecessary reprocessing while maintaining
        // at-least-once delivery guarantees. The snapshot will capture the clean state after purge.
        tracing::info!("Purging acknowledged messages and performing snapshot of radix tree");
        let start_time = std::time::Instant::now();

        // Clean up stale workers before snapshot
        // Get current worker IDs from instances_rx
        let current_instances = self.instances_rx.borrow().clone();
        let current_worker_ids: std::collections::HashSet<u64> = current_instances
            .iter()
            .map(|instance| instance.instance_id)
            .collect();

        // Get worker IDs from the indexer
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let get_workers_req = GetWorkersRequest { resp: resp_tx };

        if let Err(e) = self.get_workers_tx.send(get_workers_req).await {
            tracing::warn!("Failed to send get_workers request during snapshot: {e:?}");
        } else {
            match resp_rx.await {
                Ok(indexer_worker_ids) => {
                    // Find workers in indexer but not in current instances
                    for worker_id in indexer_worker_ids {
                        if !current_worker_ids.contains(&worker_id) {
                            tracing::info!(
                                "Removing stale worker {worker_id} from indexer during snapshot"
                            );
                            if let Err(e) = remove_worker_tx.send(worker_id).await {
                                tracing::warn!(
                                    "Failed to send remove_worker for stale worker {worker_id}: {e:?}"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to receive worker IDs from indexer: {e:?}");
                }
            }
        }

        // First, purge acknowledged messages from the stream
        nats_queue.purge_acknowledged().await?;

        // Now request a snapshot from the indexer (which reflects the post-purge state)
        let (resp_tx, resp_rx) = oneshot::channel();
        let dump_req = DumpRequest { resp: resp_tx };

        self.snapshot_tx
            .send(dump_req)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send dump request: {e:?}"))?;

        // Wait for the dump response
        let events = resp_rx
            .await
            .map_err(|e| anyhow::anyhow!("Failed to receive dump response: {e:?}"))?;

        // Upload the snapshot to NATS object store in background (non-blocking)
        let nats_client = self.nats_client.clone();
        let bucket_name = self.bucket_name.clone();
        let event_count = events.len();
        tokio::spawn(async move {
            let Ok(url) = url::Url::parse(&format!(
                "nats://{}/{bucket_name}/{RADIX_STATE_FILE}",
                nats_client.addr(),
            )) else {
                tracing::warn!("Failed to parse snapshot URL");
                return;
            };

            if let Err(e) = nats_client.object_store_upload_data(&events, &url).await {
                tracing::warn!("Failed to upload snapshot: {e:?}");
                return;
            }

            tracing::info!(
                "Successfully uploaded snapshot with {event_count} events to bucket {bucket_name} in {}ms",
                start_time.elapsed().as_millis()
            );
        });

        Ok(())
    }
}

/// Start a unified background task for event consumption and optional snapshot management
#[allow(clippy::too_many_arguments)]
pub async fn start_kv_router_background(
    component: Component,
    consumer_id: String,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    remove_worker_tx: mpsc::Sender<WorkerId>,
    maybe_get_workers_tx: Option<mpsc::Sender<GetWorkersRequest>>,
    maybe_snapshot_tx: Option<mpsc::Sender<DumpRequest>>,
    cancellation_token: CancellationToken,
    router_snapshot_threshold: Option<u32>,
    router_reset_states: bool,
) -> Result<()> {
    // Set up NATS connections
    let stream_name = Slug::slugify(&format!("{}.{}", component.subject(), KV_EVENT_SUBJECT))
        .to_string()
        .replace("_", "-");
    let nats_server = std::env::var(env_nats::NATS_SERVER)
        .unwrap_or_else(|_| "nats://localhost:4222".to_string());

    // Create NatsQueue for event consumption
    let mut nats_queue = NatsQueue::new_with_consumer(
        stream_name.clone(),
        nats_server.clone(),
        std::time::Duration::from_secs(60), // 1 minute timeout
        consumer_id.clone(),
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
    if !router_reset_states {
        // Try to download initial state from object store with stability check
        download_stable_snapshot(&nats_client, &bucket_name, &kv_events_tx).await?;
    } else {
        // Delete the bucket to reset state
        tracing::info!("Resetting router state, deleting bucket: {bucket_name}");
        if let Err(e) = nats_client.object_store_delete_bucket(&bucket_name).await {
            tracing::warn!("Failed to delete bucket (may not exist): {e:?}");
        }
    }

    // Cleanup orphaned consumers on startup
    cleanup_orphaned_consumers(&mut nats_queue, &component, &consumer_id).await;

    // Get the generate endpoint and watch for instance deletions
    let generate_endpoint = component.endpoint("generate");
    let discovery_client = component.drt().discovery();
    let generate_discovery_key = DiscoveryQuery::Endpoint {
        namespace: component.namespace().name().to_string(),
        component: component.name().to_string(),
        endpoint: "generate".to_string(),
    };
    let mut instance_event_stream = discovery_client
        .list_and_watch(generate_discovery_key, Some(cancellation_token.clone()))
        .await?;

    // Watch for router deletions to clean up orphaned consumers via discovery
    let router_discovery_key = router_discovery_query(component.namespace().name());
    let mut router_event_stream = discovery_client
        .list_and_watch(router_discovery_key, Some(cancellation_token.clone()))
        .await?;

    // Get instances_rx for tracking current workers
    let client = generate_endpoint.client().await?;
    let instances_rx = client.instance_source.as_ref().clone();

    // Only set up snapshot-related resources if snapshot_tx, get_workers_tx, and threshold are provided
    let snapshot_resources = if let (Some(get_workers_tx), Some(snapshot_tx), Some(_)) = (
        maybe_get_workers_tx,
        maybe_snapshot_tx,
        router_snapshot_threshold,
    ) {
        Some(SnapshotResources {
            nats_client,
            bucket_name,
            instances_rx,
            get_workers_tx,
            snapshot_tx,
        })
    } else {
        None
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
                Some(discovery_event_result) = instance_event_stream.next() => {
                    let Ok(discovery_event) = discovery_event_result else {
                        continue;
                    };

                    let DiscoveryEvent::Removed(worker_id) = discovery_event else {
                        continue;
                    };

                    tracing::warn!(
                        worker_id = worker_id,
                        "DISCOVERY: Generate endpoint instance removed, removing worker"
                    );

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

                // Handle periodic stream checking and purging (only if snapshot_resources is provided)
                _ = check_interval.tick() => {
                    let Some(resources) = snapshot_resources.as_ref() else {
                        continue;
                    };

                    // Check total messages in the stream
                    let Ok(message_count) = nats_queue.get_stream_messages().await else {
                        tracing::warn!("Failed to get stream message count");
                        continue;
                    };

                    let threshold = router_snapshot_threshold.unwrap_or(u32::MAX) as u64;

                    if message_count <= threshold {
                        continue;
                    }

                    tracing::info!("Stream has {message_count} messages (threshold: {threshold}), performing purge and snapshot");

                    match resources.purge_then_snapshot(
                        &mut nats_queue,
                        &remove_worker_tx,
                    ).await {
                        Ok(_) => tracing::info!("Successfully performed purge and snapshot"),
                        Err(e) => tracing::debug!("Could not perform purge and snapshot: {e:?}"),
                    }
                }

                // Handle router deletion events via discovery
                Some(router_event_result) = router_event_stream.next() => {
                    let Ok(router_event) = router_event_result else {
                        continue;
                    };

                    let DiscoveryEvent::Removed(router_instance_id) = router_event else {
                        // We only care about removals for cleaning up consumers
                        continue;
                    };

                    // The consumer UUID is the instance_id in hex format
                    let consumer_to_delete = router_instance_id.to_string();

                    tracing::info!(
                        router_instance_id = router_instance_id,
                        "DISCOVERY: Router instance removed, attempting to delete orphaned consumer: {consumer_to_delete}"
                    );

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

/// Cleanup orphaned NATS consumers that no longer have corresponding router entries
async fn cleanup_orphaned_consumers(
    nats_queue: &mut NatsQueue,
    component: &Component,
    consumer_id: &str,
) {
    let Ok(consumers) = nats_queue.list_consumers().await else {
        return;
    };

    // Get active routers from discovery
    let discovery = component.drt().discovery();
    let Ok(router_instances) = discovery
        .list(router_discovery_query(component.namespace().name()))
        .await
    else {
        tracing::debug!("Failed to list router instances from discovery, skipping cleanup");
        return;
    };

    // Build set of active router instance IDs
    let active_instance_ids: HashSet<String> = router_instances
        .iter()
        .map(|instance| instance.instance_id().to_string())
        .collect();

    for consumer in consumers {
        if consumer == consumer_id {
            // Never delete myself (extra/redundant safeguard)
            continue;
        }
        if !active_instance_ids.contains(&consumer) {
            tracing::info!("Cleaning up orphaned consumer: {consumer}");
            let _ = nats_queue.shutdown(Some(consumer)).await;
        }
    }
}
