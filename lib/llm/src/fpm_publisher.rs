// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward Pass Metrics (FPM = ForwardPassMetrics) relay.
//!
//! Subscribes to the raw ZMQ PUB from `InstrumentedScheduler` (running in
//! a vLLM EngineCore child process) and re-publishes the payloads to the
//! Dynamo event plane with automatic discovery registration.
//!
//! This follows the same two-layer architecture as
//! [`crate::kv_router::publisher::KvEventPublisher`], but is much simpler:
//! no event transformation, no batching, no local indexer — just raw byte relay.

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::utils::zmq::{connect_sub_socket, multipart_message};

const FPM_TOPIC: &str = "forward-pass-metrics";

/// A relay that bridges ForwardPassMetrics from a local raw ZMQ PUB socket
/// to the Dynamo event plane.
pub struct FpmEventRelay {
    cancel: CancellationToken,
}

impl FpmEventRelay {
    /// Create and start a new relay.
    ///
    /// - `component`: Dynamo component (provides runtime + discovery scope).
    /// - `zmq_endpoint`: Local ZMQ PUB address to subscribe to
    ///   (e.g., `tcp://127.0.0.1:20380`).
    pub fn new(component: Component, zmq_endpoint: String) -> Result<Self> {
        let rt = component.drt().runtime().secondary();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let publisher =
            rt.block_on(async { EventPublisher::for_component(&component, FPM_TOPIC).await })?;

        rt.spawn(async move {
            Self::relay_loop(zmq_endpoint, publisher, cancel_clone).await;
        });

        Ok(Self { cancel })
    }

    /// Shut down the relay task.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn relay_loop(
        zmq_endpoint: String,
        publisher: EventPublisher,
        cancel: CancellationToken,
    ) {
        let socket = match connect_sub_socket(&zmq_endpoint, None).await {
            Ok(socket) => socket,
            Err(error) => {
                tracing::error!(endpoint = %zmq_endpoint, error = %error, "FPM relay: failed to connect");
                return;
            }
        };
        let mut socket = socket;
        tracing::info!("FPM relay: connected to {zmq_endpoint}");

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::info!("FPM relay: shutting down");
                    break;
                }
                result = socket.next() => {
                    match result {
                        Some(Ok(frames)) => {
                            let mut frames = multipart_message(frames);
                            // ZMQ multipart: [topic, seq, payload]
                            if frames.len() == 3 {
                                let payload = frames.swap_remove(2);
                                if let Err(e) = publisher.publish_bytes(payload).await {
                                    tracing::warn!("FPM relay: event plane publish failed: {e}");
                                }
                            } else {
                                tracing::warn!(
                                    "FPM relay: unexpected ZMQ frame count: expected 3, got {}",
                                    frames.len()
                            );
                            }
                        }
                        Some(Err(e)) => {
                            tracing::error!("FPM relay: ZMQ recv failed: {e}");
                            break;
                        }
                        None => {
                            tracing::error!("FPM relay: ZMQ stream ended");
                            break;
                        }
                    }
                }
            }
        }
    }
}

impl Drop for FpmEventRelay {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
