// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background processes for the KV Router including event consumption and snapshot uploads.

use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::{
    component::Component, traits::events::EventPublisher, transports::nats::NatsQueue,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::kv_router::{
    KV_EVENT_SUBJECT,
    indexer::{DumpRequest, RadixUploader, RouterEvent},
};

/// Start a background task to consume events from NatsQueue and forward them to the indexer
pub async fn start_event_consumer(
    component: Component,
    consumer_uuid: String,
    kv_events_tx: mpsc::Sender<RouterEvent>,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let stream_name =
        format!("{}.{}", component.subject(), KV_EVENT_SUBJECT).replace(['/', '\\', '.', '_'], "-");
    let nats_server =
        std::env::var("NATS_SERVER").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let mut nats_queue = NatsQueue::new_with_consumer(
        stream_name,
        nats_server,
        std::time::Duration::from_secs(300), // Very long timeout (5 minutes)
        consumer_uuid,
    );

    nats_queue.connect().await?;

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::info!("Event consumer received cancellation signal");
                    break;
                }
                result = nats_queue.dequeue_task(Some(std::time::Duration::from_secs(300))) => {
                    match result {
                        Ok(Some(bytes)) => {
                            let event: RouterEvent = match serde_json::from_slice(&bytes) {
                                Ok(event) => event,
                                Err(e) => {
                                    tracing::warn!("Failed to deserialize RouterEvent: {:?}", e);
                                    continue;
                                }
                            };

                            // Forward the RouterEvent to the indexer
                            if let Err(e) = kv_events_tx.send(event).await {
                                tracing::warn!(
                                    "failed to send kv event to indexer; shutting down: {:?}",
                                    e
                                );
                                break;
                            }
                        },
                        Ok(None) => {
                            tracing::trace!("Dequeue timeout, continuing");
                        },
                        Err(e) => {
                            tracing::error!("Failed to dequeue task: {:?}", e);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
            }
        }

        // Clean up the queue and remove the durable consumer
        if let Err(e) = nats_queue.shutdown().await {
            tracing::warn!("Failed to shutdown NatsQueue: {}", e);
        }
    });

    Ok(())
}

/// Start a RadixUploader for periodic snapshot uploads to NATS object store
pub async fn start_radix_uploader(
    component: Component,
    snapshot_tx: mpsc::Sender<DumpRequest>,
    upload_interval: Duration,
    cancellation_token: CancellationToken,
) -> Result<RadixUploader> {
    // Create NATS client
    let nats_server =
        std::env::var("NATS_SERVER").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let client_options = dynamo_runtime::transports::nats::Client::builder()
        .server(&nats_server)
        .build()?;
    let nats_client = client_options.connect().await?;

    // Create bucket name from component name
    let bucket_name =
        format!("{}-router-snapshot", component.name()).replace(['/', '\\', '.', '_'], "-");

    let uploader = RadixUploader::new(
        nats_client,
        snapshot_tx,
        upload_interval,
        bucket_name.clone(),
        cancellation_token,
    );

    tracing::info!(
        "RadixUploader initialized with bucket: {}, interval: {:?}",
        bucket_name,
        upload_interval
    );
    Ok(uploader)
}
