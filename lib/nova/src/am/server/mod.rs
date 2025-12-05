// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Server

pub(crate) mod dispatcher;
pub(crate) mod system_handlers;

pub(crate) use system_handlers::register_system_handlers;

use crate::am::common::{
    events::{EventType, Outcome, decode_event_header},
    messages::decode_active_message,
    responses::{ResponseManager, decode_response_header},
};
use crate::events::EventManager;

use std::sync::Arc;

use bytes::Bytes;
use dynamo_nova_backend::{DataStreams, NovaBackend};
use tokio_util::task::TaskTracker;

pub(crate) use dispatcher::{ControlMessage, DispatcherHub, HandlerContext};

pub(crate) struct ActiveMessageServer {
    _tracker: TaskTracker,
    control_tx: flume::Sender<ControlMessage>,
    hub: Arc<DispatcherHub>,
}

impl ActiveMessageServer {
    pub async fn new(
        response_manager: ResponseManager,
        event_manager: EventManager,
        data_streams: DataStreams,
        backend: Arc<NovaBackend>,
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
            tracing::debug!(target: "dynamo_nova::server", "Dispatcher hub shutting down");
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
            event_manager,
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
                    target: "dynamo_nova::server",
                    handler = %message.metadata.handler_name,
                    "Received active message"
                );

                let ctx = HandlerContext {
                    message_id: message.metadata.response_id,
                    payload: message.payload.clone(),
                    response_type: message.metadata.response_type,
                    headers: message.metadata.headers.clone(),
                    system: hub.system().clone(), // Inject system from OnceLock
                };

                // Direct dispatch - inline, no channel hop!
                hub.dispatch_message(&message.metadata.handler_name, ctx);
            }
            Err(e) => {
                tracing::error!(target: "dynamo_nova::server", "Failed to decode active message: {}", e);
            }
        }
    }
    Ok(())
}

/// Creates a task that handles responses from the response channel.
/// All unary responses are handled here.
async fn create_response_handler(
    response_manager: ResponseManager,
    response_rx: flume::Receiver<(Bytes, Bytes)>,
) -> anyhow::Result<()> {
    while let Ok((header, payload)) = response_rx.recv_async().await {
        match decode_response_header(header) {
            Ok((response_id, outcome, _headers)) => {
                // Note: We ignore headers here as they're already included in the response payload
                // The headers are echoed back from the request
                match outcome {
                    Outcome::Ok => {
                        response_manager.complete_outcome(response_id, Ok(Some(payload)));
                    }
                    Outcome::Error => {
                        let error_message = String::from_utf8(payload.to_vec())
                            .unwrap_or("unknown error".to_string());
                        response_manager.complete_outcome(response_id, Err(error_message));
                    }
                }
            }
            Err(e) => {
                tracing::error!(target: "dynamo_nova::server", "Failed to decode response header: {}", e);
            }
        }
    }
    Ok(())
}

/// Creates a task that handles events and acks from the event channel.
/// All events and am_sync responses are handled here.
async fn create_ack_and_event_handler(
    response_manager: ResponseManager,
    event_manager: EventManager,
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
            Some(EventType::Event(event_handle, Outcome::Ok)) => {
                if let Err(e) = event_manager.trigger(event_handle) {
                    tracing::warn!("Failed to trigger event {}: {}", event_handle, e);
                }
            }
            Some(EventType::Event(event_handle, Outcome::Error)) => {
                let error_message =
                    String::from_utf8(payload.to_vec()).unwrap_or("unknown error".to_string());
                if let Err(e) = event_manager.poison(event_handle, error_message) {
                    tracing::warn!("Failed to poison event {}: {}", event_handle, e);
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
    use crate::am::common::events::{EventType, Outcome, encode_event_header};
    use crate::events::{EventPoison, LocalEventSystem};
    use tokio::time::{Duration, timeout};

    #[tokio::test]
    async fn event_error_frame_poisones_event_with_id() -> anyhow::Result<()> {
        let worker_id = 7;
        let response_manager = ResponseManager::new(worker_id);
        let event_manager = LocalEventSystem::new(worker_id);
        let (tx, rx) = flume::bounded(1);

        let handler = tokio::spawn(create_ack_and_event_handler(
            response_manager,
            event_manager.clone(),
            rx,
        ));

        let event = event_manager.new_event()?;
        let handle = event.handle();

        let header = encode_event_header(EventType::Event(handle, Outcome::Error));
        let reason = "boom";
        tx.send((header, Bytes::from(reason.as_bytes().to_vec())))
            .expect("send frame");
        drop(tx); // close channel so handler can exit

        let err = timeout(Duration::from_millis(200), event_manager.awaiter(handle)?)
            .await
            .expect("waiter timed out")
            .expect_err("expected poison");

        let poison = err.downcast::<EventPoison>().expect("poison error");
        assert_eq!(poison.handle(), handle);
        assert!(poison.reason().contains(reason));

        handler.await??;
        Ok(())
    }
}
