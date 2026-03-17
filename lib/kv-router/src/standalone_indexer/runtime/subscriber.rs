// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    DistributedRuntime, discovery::EventTransportKind, transports::event_plane::EventSubscriber,
};

use crate::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use crate::standalone_indexer::registry::WorkerRegistry;

pub async fn spawn_event_subscriber(
    drt: &DistributedRuntime,
    namespace: &str,
    worker_component_name: &str,
    registry: Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> Result<()> {
    let transport_kind = EventTransportKind::from_env_or_default();
    let worker_component = drt.namespace(namespace)?.component(worker_component_name)?;
    let mut subscriber = EventSubscriber::for_component_with_transport(
        &worker_component,
        KV_EVENT_SUBJECT,
        transport_kind,
    )
    .await?
    .typed::<RouterEvent>();

    let kv_event_subject = format!(
        "namespace.{}.component.{}.{}",
        namespace, worker_component_name, KV_EVENT_SUBJECT
    );

    match transport_kind {
        EventTransportKind::Nats => {
            tracing::info!(
                subject = %kv_event_subject,
                "KV Indexer subscribing to NATS Core events"
            );
        }
        EventTransportKind::Zmq => {
            tracing::info!(
                subject = %kv_event_subject,
                "KV Indexer subscribing to ZMQ event plane"
            );
        }
    }

    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;

                _ = cancel_token.cancelled() => {
                    tracing::debug!("Event subscriber received cancellation signal");
                    break;
                }

                Some(result) = subscriber.next() => {
                    let (_envelope, event) = match result {
                        Ok((envelope, event)) => (envelope, event),
                        Err(err) => {
                            tracing::warn!("Failed to receive RouterEvent from event plane: {err:?}");
                            continue;
                        }
                    };

                    let worker_id = event.worker_id;
                    if let Some(indexer) = registry.get_indexer_for_worker(worker_id) {
                        indexer.apply_event(event).await;
                    } else {
                        tracing::trace!(
                            worker_id,
                            "Received event for unknown worker (not yet discovered?)"
                        );
                    }
                }
            }
        }

        tracing::info!("Event subscriber exiting");
    });

    Ok(())
}
