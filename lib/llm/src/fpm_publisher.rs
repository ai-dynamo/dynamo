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

use std::time::Duration;

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_runtime::{
    component::Component, traits::DistributedRuntimeProvider,
    transports::event_plane::EventPublisher,
};

const FPM_TOPIC: &str = "forward-pass-metrics";
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const INITIAL_SETUP_BACKOFF_MS: u64 = 10;
const MAX_SETUP_BACKOFF_MS: u64 = 5000;
const MAX_SETUP_BACKOFF_EXPONENT: u32 = 8;

fn calculate_setup_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_SETUP_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_SETUP_BACKOFF_EXPONENT)),
        MAX_SETUP_BACKOFF_MS,
    )
}

async fn connect_sub_socket_with_retry(
    endpoint: &str,
    cancellation_token: &CancellationToken,
    log_prefix: &str,
) -> Option<SubSocket> {
    let mut consecutive_errors = 0u32;

    loop {
        if cancellation_token.is_cancelled() {
            tracing::debug!("{log_prefix}: cancelled before connecting to {endpoint}");
            return None;
        }

        let mut socket = SubSocket::new();

        match socket.subscribe("").await {
            Ok(()) => {}
            Err(error) => {
                consecutive_errors += 1;
                let backoff_ms = calculate_setup_backoff_ms(consecutive_errors);
                tracing::warn!(
                    error = %error,
                    consecutive_errors = consecutive_errors,
                    backoff_ms = backoff_ms,
                    "{log_prefix}: failed to subscribe on ZMQ socket during setup, retrying"
                );
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => return None,
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
                continue;
            }
        }

        match socket.connect(endpoint).await {
            Ok(()) => return Some(socket),
            Err(error) => {
                consecutive_errors += 1;
                let backoff_ms = calculate_setup_backoff_ms(consecutive_errors);
                tracing::warn!(
                    error = %error,
                    consecutive_errors = consecutive_errors,
                    backoff_ms = backoff_ms,
                    "{log_prefix}: failed to connect ZMQ SUB during setup, retrying"
                );
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => return None,
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
            }
        }
    }
}

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
        let Some(mut socket) =
            connect_sub_socket_with_retry(&zmq_endpoint, &cancel, "FPM relay").await
        else {
            return;
        };
        tracing::info!("FPM relay: connected to {zmq_endpoint}");

        let mut consecutive_errors: u32 = 0;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::info!("FPM relay: shutting down");
                    break;
                }
                result = socket.recv() => {
                    match result {
                        Ok(msg) => {
                            consecutive_errors = 0;
                            let mut frames: Vec<Vec<u8>> =
                                msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();
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
                        Err(error) => {
                            consecutive_errors += 1;
                            tracing::warn!(
                                "FPM relay: ZMQ recv error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {error}"
                            );
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                tracing::error!("FPM relay: too many consecutive errors, exiting");
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(100)).await;
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
