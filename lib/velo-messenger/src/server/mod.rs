// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active message server.

pub(crate) mod dispatcher;
pub(crate) mod system_handlers;

pub(crate) use system_handlers::register_system_handlers;

use crate::common::{
    events::{EventType, Outcome, decode_event_header},
    messages::decode_active_message,
    responses::{ResponseManager, decode_response_header},
};

use std::sync::Arc;

use bytes::Bytes;
use tokio_util::task::TaskTracker;
use velo_backend::{DataStreams, VeloBackend};

pub(crate) use dispatcher::{ControlMessage, DispatcherHub, HandlerContext};

/// Handler for event frames received on the shared ack/event channel.
/// Higher-level crates (e.g., one that wraps velo-events) implement this.
pub trait EventFrameHandler: Send + Sync {
    fn on_event(&self, raw_handle: u128, is_error: bool, payload: Bytes);
}

pub(crate) struct ActiveMessageServer {
    _tracker: TaskTracker,
    control_tx: flume::Sender<ControlMessage>,
    hub: Arc<DispatcherHub>,
}

impl ActiveMessageServer {
    pub async fn new(
        response_manager: ResponseManager,
        event_handler: Option<Arc<dyn EventFrameHandler>>,
        data_streams: DataStreams,
        backend: Arc<VeloBackend>,
        tracker: TaskTracker,
    ) -> Self {
        let (message_rx, response_rx, event_rx) = data_streams.into_parts();

        // Create control channel for dispatcher hub (bounded for backpressure)
        let (control_tx, control_rx) = flume::bounded(1000);

        // Create dispatcher hub (shareable)
        let hub = Arc::new(DispatcherHub::new(backend.clone(), control_rx));

        // Spawn dispatcher hub control task
        let hub_clone = hub.clone();
        tracker.spawn(async move {
            while hub_clone.process_control().await {
                // Continue processing control messages
            }
            tracing::debug!(target: "velo_messenger::server", "Dispatcher hub shutting down");
            Ok::<(), anyhow::Error>(())
        });

        // Spawn message handler with direct dispatch (hot path)
        tracker.spawn(create_message_handler(message_rx, hub.clone()));

        tracker.spawn(create_response_handler(
            response_manager.clone(),
            response_rx,
        ));
        tracker.spawn(create_ack_and_event_handler(
            response_manager.clone(),
            event_handler,
            event_rx,
        ));
        Self {
            _tracker: tracker,
            control_tx,
            hub,
        }
    }

    /// Get a reference to the dispatcher hub
    pub(crate) fn hub(&self) -> &Arc<DispatcherHub> {
        &self.hub
    }

    /// Get a clone of the control channel sender
    pub(crate) fn control_tx(&self) -> flume::Sender<ControlMessage> {
        self.control_tx.clone()
    }
}

/// Message handler task - receives messages from backend and dispatches to handlers
/// This is the HOT PATH - optimized for low latency with direct dispatch
async fn create_message_handler(
    message_rx: flume::Receiver<(Bytes, Bytes)>,
    hub: Arc<DispatcherHub>,
) -> anyhow::Result<()> {
    while let Ok((header, payload)) = message_rx.recv_async().await {
        match decode_active_message(header, payload) {
            Ok(message) => {
                tracing::debug!(
                    target: "velo_messenger::server",
                    handler = %message.metadata.handler_name,
                    "Received active message"
                );

                let ctx = HandlerContext {
                    message_id: message.metadata.response_id,
                    payload: message.payload.clone(),
                    response_type: message.metadata.response_type,
                    headers: message.metadata.headers.clone(),
                    system: hub.system().clone(),
                };

                // Direct dispatch - inline, no channel hop!
                hub.dispatch_message(&message.metadata.handler_name, ctx);
            }
            Err(e) => {
                tracing::error!(target: "velo_messenger::server", "Failed to decode active message: {}", e);
            }
        }
    }
    Ok(())
}

/// Creates a task that handles responses from the response channel.
async fn create_response_handler(
    response_manager: ResponseManager,
    response_rx: flume::Receiver<(Bytes, Bytes)>,
) -> anyhow::Result<()> {
    while let Ok((header, payload)) = response_rx.recv_async().await {
        match decode_response_header(header) {
            Ok((response_id, outcome, _headers)) => match outcome {
                Outcome::Ok => {
                    response_manager.complete_outcome(response_id, Ok(Some(payload)));
                }
                Outcome::Error => {
                    let error_message =
                        String::from_utf8(payload.to_vec()).unwrap_or("unknown error".to_string());
                    response_manager.complete_outcome(response_id, Err(error_message));
                }
            },
            Err(e) => {
                tracing::error!(target: "velo_messenger::server", "Failed to decode response header: {}", e);
            }
        }
    }
    Ok(())
}

/// Creates a task that handles events and acks from the event channel.
async fn create_ack_and_event_handler(
    response_manager: ResponseManager,
    event_handler: Option<Arc<dyn EventFrameHandler>>,
    event_rx: flume::Receiver<(Bytes, Bytes)>,
) -> anyhow::Result<()> {
    while let Ok((header, payload)) = event_rx.recv_async().await {
        let event_type = decode_event_header(header);
        match event_type {
            Some(EventType::Ack(response_id, Outcome::Ok)) => {
                response_manager.complete_outcome(response_id, Ok(Some(payload)));
            }
            Some(EventType::Ack(response_id, Outcome::Error)) => {
                let error_message =
                    String::from_utf8(payload.to_vec()).unwrap_or("unknown error".to_string());
                response_manager.complete_outcome(response_id, Err(error_message));
            }
            Some(EventType::Event(raw_handle, Outcome::Ok)) => {
                if let Some(ref handler) = event_handler {
                    handler.on_event(raw_handle, false, payload);
                } else {
                    tracing::warn!(
                        target: "velo_messenger::server",
                        raw_handle = raw_handle,
                        "Received event frame but no EventFrameHandler configured"
                    );
                }
            }
            Some(EventType::Event(raw_handle, Outcome::Error)) => {
                if let Some(ref handler) = event_handler {
                    handler.on_event(raw_handle, true, payload);
                } else {
                    tracing::warn!(
                        target: "velo_messenger::server",
                        raw_handle = raw_handle,
                        "Received error event frame but no EventFrameHandler configured"
                    );
                }
            }
            None => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::events::{EventType, Outcome, encode_event_header};
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::time::{Duration, timeout};

    struct TestEventHandler {
        called: AtomicBool,
    }

    impl EventFrameHandler for TestEventHandler {
        fn on_event(&self, _raw_handle: u128, _is_error: bool, _payload: Bytes) {
            self.called.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn ack_ok_completes_response() -> anyhow::Result<()> {
        let worker_id = 7;
        let response_manager = ResponseManager::new(worker_id);
        let (tx, rx) = flume::bounded(1);

        let handler = tokio::spawn(create_ack_and_event_handler(
            response_manager.clone(),
            None,
            rx,
        ));

        let mut awaiter = response_manager.register_outcome()?;
        let response_id = awaiter.response_id();

        let header = encode_event_header(EventType::Ack(response_id, Outcome::Ok));
        tx.send((header, Bytes::from("ok"))).expect("send frame");
        drop(tx);

        let result = timeout(Duration::from_millis(200), awaiter.recv()).await;
        assert!(result.is_ok());

        handler.await??;
        Ok(())
    }

    #[tokio::test]
    async fn event_frame_dispatched_to_handler() -> anyhow::Result<()> {
        let worker_id = 7;
        let response_manager = ResponseManager::new(worker_id);
        let event_handler = Arc::new(TestEventHandler {
            called: AtomicBool::new(false),
        });
        let (tx, rx) = flume::bounded(1);

        let eh = event_handler.clone();
        let handler = tokio::spawn(create_ack_and_event_handler(response_manager, Some(eh), rx));

        let raw_handle: u128 = 42;
        let header = encode_event_header(EventType::Event(raw_handle, Outcome::Ok));
        tx.send((header, Bytes::new())).expect("send frame");
        drop(tx);

        handler.await??;
        assert!(event_handler.called.load(Ordering::SeqCst));
        Ok(())
    }
}
