// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, collections::HashSet, time::Duration};

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    config::environment_names::nats as env_nats,
    discovery::{DiscoveryEvent, DiscoveryQuery, EventTransportKind},
    prelude::*,
    transports::event_plane::EventSubscriber,
    transports::nats::{NatsQueue, Slug},
};
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kv_router::{
    KV_EVENT_SUBJECT, RADIX_STATE_BUCKET, RADIX_STATE_FILE,
    indexer::{DumpRequest, GetWorkersRequest, RouterEvent, WorkerKvQueryResponse},
    protocols::WorkerId,
    router_discovery_query,
    worker_query::WorkerQueryClient,
};

/// Helper function to create a KV stream name from a component and subject.
///
/// Generates a slugified stream name in the format:
/// `namespace-{namespace}-component-{component}-{subject}`
fn create_kv_stream_name(component: &Component, subject: &str) -> String {
    Slug::slugify(&format!(
        "namespace.{}.component.{}.{}",
        component.namespace().name(),
        component.name(),
        subject
    ))
    .to_string()
    .replace("_", "-")
}

/// Delay between snapshot reads to verify stability
const SNAPSHOT_STABILITY_DELAY: Duration = Duration::from_millis(100);
const MAX_SNAPSHOT_STABILITY_ATTEMPTS: usize = 10;

const CHECK_INTERVAL_BASE: Duration = Duration::from_secs(1);
const CHECK_INTERVAL_JITTER_MS: i64 = 100;

// Worker query retry configuration
const WORKER_QUERY_MAX_RETRIES: u32 = 8;
const WORKER_QUERY_INITIAL_BACKOFF_MS: u64 = 200;

// ============================================================================
// Discovery Helpers
// ============================================================================

/// Wait for at least one worker instance to be discovered.
/// Returns a peekable stream of discovery events for the generate endpoint.
async fn wait_for_worker_instance(
    component: &Component,
    cancellation_token: &CancellationToken,
) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<DiscoveryEvent>> + Send>>> {
    let discovery_client = component.drt().discovery();
    let generate_discovery_key = DiscoveryQuery::Endpoint {
        namespace: component.namespace().name().to_string(),
        component: component.name().to_string(),
        endpoint: "generate".to_string(),
    };

    let mut stream = discovery_client
        .list_and_watch(generate_discovery_key, Some(cancellation_token.clone()))
        .await?
        .peekable();

    tracing::info!("KV subscriber waiting for at least one worker instance...");
    std::pin::Pin::new(&mut stream).peek().await;

    Ok(Box::pin(stream))
}

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
                "Skipping recovery - worker {worker_id} does not have local indexer enabled"
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
            "Startup recovery completed: {total_recovered} events recovered from {successful_workers} workers, {failed_workers} workers failed"
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
            "Attempting recovery from worker {worker_id}, start_event_id: {start_event_id:?}, end_event_id: {end_event_id:?}"
        );
    } else {
        tracing::warn!("Worker {worker_id} does not have local indexer enabled, skipping recovery");
        return Ok(0);
    }

    // Query worker for events in range, with retry logic for transient failures
    // (e.g., worker's query service not yet re-subscribed after NATS restart)
    let mut response = None;
    let mut last_error = None;

    for attempt in 0..WORKER_QUERY_MAX_RETRIES {
        match worker_query_client
            .query_worker(worker_id, start_event_id, end_event_id)
            .await
        {
            Ok(resp) => {
                if attempt > 0 {
                    tracing::info!("Worker {worker_id} query succeeded after retry {attempt}");
                }
                response = Some(resp);
                break;
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < WORKER_QUERY_MAX_RETRIES - 1 {
                    let backoff_ms = WORKER_QUERY_INITIAL_BACKOFF_MS * 2_u64.pow(attempt);
                    tracing::warn!(
                        "Worker {worker_id} query failed on attempt {attempt}, retrying after {backoff_ms}ms"
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }

    let response = match response {
        Some(r) => r,
        None => return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No response"))),
    };

    // Handle response variants
    let events = match response {
        WorkerKvQueryResponse::Events(events) => {
            tracing::debug!(
                "Got {count} buffered events from worker {worker_id}",
                count = events.len()
            );
            events
        }
        WorkerKvQueryResponse::TreeDump(events) => {
            tracing::info!(
                "Got tree dump from worker {worker_id} (range too old or unspecified), count: {count}",
                count = events.len()
            );
            events
        }
        WorkerKvQueryResponse::TooNew {
            requested_start,
            requested_end,
            newest_available,
        } => {
            tracing::warn!(
                "Worker {worker_id} requested range is newer than available data: requested_start: {requested_start:?}, requested_end: {requested_end:?}, newest_available: {newest_available}"
            );
            return Ok(0);
        }
        WorkerKvQueryResponse::InvalidRange { start_id, end_id } => {
            anyhow::bail!("Invalid range: end_id ({end_id}) < start_id ({start_id})");
        }
        WorkerKvQueryResponse::Error(message) => {
            anyhow::bail!("Worker {worker_id} query failed: {message}");
        }
    };

    let events_count = events.len();

    if events_count == 0 {
        tracing::debug!(
            "No events to recover from worker {worker_id}, start_event_id: {start_event_id:?}"
        );
        return Ok(0);
    }

    tracing::info!(
        "Recovered {events_count} events from worker {worker_id}, start_event_id: {start_event_id:?}"
    );

    // Apply recovered events to the indexer
    for event in events {
        if let Err(e) = event_tx.send(event).await {
            tracing::error!(
                "Failed to send recovered event to indexer for worker {worker_id}: {e}"
            );
            anyhow::bail!("Failed to send recovered event: {e}");
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
    let stream_name = create_kv_stream_name(&component, KV_EVENT_SUBJECT);
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
    let event_plane_subject = format!(
        "namespace.{}.component.{}",
        component.namespace().name(),
        component.name()
    );
    let bucket_name = Slug::slugify(&format!("{}-{RADIX_STATE_BUCKET}", event_plane_subject))
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

    // Wait for at least one worker instance before proceeding
    let mut instance_event_stream =
        wait_for_worker_instance(&component, &cancellation_token).await?;

    // Watch for router deletions to clean up orphaned consumers via discovery
    let generate_endpoint = component.endpoint("generate");
    let discovery_client = component.drt().discovery();
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

                    let DiscoveryEvent::Removed(id) = discovery_event else {
                        continue;
                    };

                    let worker_id = id.instance_id();

                    tracing::warn!(
                        "DISCOVERY: Generate endpoint instance removed, removing worker {worker_id}"
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

                    let DiscoveryEvent::Removed(id) = router_event else {
                        // We only care about removals for cleaning up consumers
                        continue;
                    };

                    let router_instance_id = id.instance_id();

                    // The consumer ID is the instance_id as a string
                    let consumer_to_delete = router_instance_id.to_string();

                    tracing::info!(
                        "DISCOVERY: Router instance {router_instance_id} removed, attempting to delete orphaned consumer: {consumer_to_delete}"
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

/// Handle a worker discovery event (added or removed).
async fn handle_worker_discovery(
    event: DiscoveryEvent,
    worker_query_client: &WorkerQueryClient,
    kv_events_tx: &mpsc::Sender<RouterEvent>,
    remove_worker_tx: &mpsc::Sender<WorkerId>,
) {
    match event {
        DiscoveryEvent::Added(instance) => {
            let worker_id = instance.instance_id();
            tracing::info!(
                "DISCOVERY: Worker {worker_id} added, dumping local indexer into router"
            );

            match recover_from_worker(
                worker_query_client,
                worker_id,
                None, // Start from beginning
                None, // Get all events
                kv_events_tx,
            )
            .await
            {
                Ok(count) => {
                    tracing::info!(
                        "Successfully dumped worker {worker_id}'s local indexer, recovered {count} events"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to dump worker {worker_id}'s local indexer (may not have local indexer enabled): {e}"
                    );
                }
            }
        }
        DiscoveryEvent::Removed(id) => {
            let worker_id = id.instance_id();
            tracing::warn!("DISCOVERY: Worker {worker_id} removed, removing from router indexer");

            if let Err(e) = remove_worker_tx.send(worker_id).await {
                tracing::warn!("Failed to send worker removal for worker {worker_id}: {e}");
            }
        }
    }
}

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
pub async fn start_kv_router_background_event_plane(
    component: Component,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    remove_worker_tx: mpsc::Sender<WorkerId>,
    cancellation_token: CancellationToken,
    worker_query_client: WorkerQueryClient,
    transport_kind: EventTransportKind,
) -> Result<()> {
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

    // Wait for at least one worker instance before proceeding
    let mut instance_event_stream =
        wait_for_worker_instance(&component, &cancellation_token).await?;

    // Drain and process all existing workers before spawning the background loop.
    // list_and_watch returns existing instances first, so we poll with a short timeout
    // to process all initial workers synchronously before the router becomes "ready".
    loop {
        // Use a short timeout to detect when initial discovery events are exhausted
        let poll_result =
            tokio::time::timeout(Duration::from_millis(100), instance_event_stream.next()).await;

        match poll_result {
            Ok(Some(Ok(event))) => {
                handle_worker_discovery(
                    event,
                    &worker_query_client,
                    &kv_events_tx,
                    &remove_worker_tx,
                )
                .await;
            }
            Ok(Some(Err(e))) => {
                tracing::warn!("Error receiving discovery event during initial sync: {e}");
            }
            Ok(None) => {
                // Stream ended
                tracing::warn!("Discovery stream ended during initial sync");
                break;
            }
            Err(_) => {
                // Timeout - no more initial events
                tracing::debug!("Initial worker discovery sync complete");
                break;
            }
        }
    }

    tokio::spawn(async move {
        // Track last received event ID per worker for gap detection
        let mut last_event_ids: HashMap<WorkerId, u64> = HashMap::new();

        loop {
            tokio::select! {
                biased;

                _ = cancellation_token.cancelled() => {
                    tracing::debug!("KV Router event plane background task received cancellation signal");
                    break;
                }

                // Handle generate endpoint instance add/remove events
                Some(discovery_event_result) = instance_event_stream.next() => {
                    let Ok(event) = discovery_event_result else {
                        continue;
                    };

                    handle_worker_discovery(
                        event,
                        &worker_query_client,
                        &kv_events_tx,
                        &remove_worker_tx,
                    )
                    .await;
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
                    let event_id = event.event.event_id;

                    // Use envelope metadata for additional debugging
                    tracing::trace!(
                        "Received event from publisher {} (seq {})",
                        envelope.publisher_id,
                        envelope.sequence
                    );

                    // Gap detection: check if event ID is monotonically increasing per worker
                    // Note: event_id <= last_id is duplicate/out-of-order, apply anyway (idempotent)
                    if let Some(&last_id) = last_event_ids.get(&worker_id)
                        && event_id > last_id + 1
                    {
                        // Gap detected - recover missing events before processing current
                        let gap_start = last_id + 1;
                        let gap_end = event_id - 1;
                        let gap_size = gap_end - gap_start + 1;
                        tracing::warn!(
                            "Event ID gap detected for worker {worker_id}, recovering events [{gap_start}, {gap_end}], gap_size: {gap_size}"
                        );

                        // Note: While recovering, new events may queue in the subscriber's
                        // internal buffer. We don't explicitly buffer them here for simplicity.
                        // The subscriber will process them in order after recovery completes.
                        if let Err(e) = recover_from_worker(
                            &worker_query_client,
                            worker_id,
                            Some(gap_start),
                            Some(gap_end),
                            &kv_events_tx,
                        ).await {
                            tracing::error!(
                                "Failed to recover gap events for worker {worker_id} (gap_start: {gap_start}, gap_end: {gap_end}); proceeding with current event anyway: {e}"
                            );
                            // Note: If recovery fails, we still apply the current event.
                            // The tree will have a gap, but it's better than dropping the event.
                        }
                    }
                    // First event from this worker is always valid - we accept whatever ID it has.
                    // This handles initial startup and worker restarts without requiring event 0.

                    // Update last seen event ID (use max to handle out-of-order)
                    last_event_ids
                        .entry(worker_id)
                        .and_modify(|id| *id = (*id).max(event_id))
                        .or_insert(event_id);

                    // Forward the RouterEvent to the indexer
                    if let Err(e) = kv_events_tx.send(event).await {
                        tracing::warn!(
                            "failed to send kv event to indexer; shutting down: {e:?}"
                        );
                        break;
                    }
                }
            }
        }

        tracing::debug!("KV Router event plane background task exiting");
    });

    Ok(())
}

/// Backwards-compatible wrapper for NATS Core local-indexer mode.
pub async fn start_kv_router_background_nats_core(
    component: Component,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    remove_worker_tx: mpsc::Sender<WorkerId>,
    cancellation_token: CancellationToken,
    worker_query_client: WorkerQueryClient,
) -> Result<()> {
    start_kv_router_background_event_plane(
        component,
        kv_events_tx,
        remove_worker_tx,
        cancellation_token,
        worker_query_client,
        EventTransportKind::Nats,
    )
    .await
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
