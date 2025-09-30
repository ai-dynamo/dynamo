// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info};

use crate::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, ActiveMessageHandler, HandlerEvent, HandlerId},
    response::ResponseContext,
    responses::{
        HealthCheckResponse, ListHandlersResponse, RegisterServiceResponse, WaitForHandlerResponse,
    },
};

use super::manager::ManagerState;

#[derive(Debug)]
pub struct RegisterServiceHandler {
    client: Arc<super::client::ZmqActiveMessageClient>,
}

impl RegisterServiceHandler {
    pub fn new(client: Arc<super::client::ZmqActiveMessageClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RegisterServicePayload {
    instance_id: String,
    endpoint: String,
}

#[async_trait]
impl ActiveMessageHandler for RegisterServiceHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let payload: RegisterServicePayload = message.deserialize()?;

        let instance_id = uuid::Uuid::parse_str(&payload.instance_id)?;
        let peer = PeerInfo::new(instance_id, payload.endpoint.clone());

        let registration_result = self.client.connect_to_peer(peer).await;
        let registered = registration_result.is_ok();

        if registered {
            info!(
                "Registered service {} at {}",
                payload.instance_id, payload.endpoint
            );
        } else {
            debug!(
                "Failed to register service {} at {}: {:?}",
                payload.instance_id, payload.endpoint, registration_result
            );
        }

        if let ResponseContext::Single(sender) = response {
            let response = RegisterServiceResponse {
                registered,
                instance_id: payload.instance_id,
                endpoint: payload.endpoint,
            };
            sender.send(response).await?;
        }

        registration_result
    }

    fn name(&self) -> &str {
        "_register_service"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "instance_id": { "type": "string" },
                        "endpoint": { "type": "string" }
                    },
                    "required": ["instance_id", "endpoint"]
                })
            });
        Some(&SCHEMA)
    }
}

#[derive(Debug)]
pub struct ListHandlersHandler {
    state: Arc<RwLock<ManagerState>>,
}

impl ListHandlersHandler {
    pub fn new(state: Arc<RwLock<ManagerState>>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl ActiveMessageHandler for ListHandlersHandler {
    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let state = self.state.read().await;
        let handlers: Vec<HandlerId> = state.handlers.keys().cloned().collect();
        drop(state);

        debug!("Available handlers: {:?}", handlers);

        if let ResponseContext::Single(sender) = response {
            let response = ListHandlersResponse { handlers };
            sender.send(response).await?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "_list_handlers"
    }
}

#[derive(Debug)]
pub struct WaitForHandlerHandler {
    state: Arc<RwLock<ManagerState>>,
    handler_events_tx: broadcast::Sender<HandlerEvent>,
}

impl WaitForHandlerHandler {
    pub fn new(
        state: Arc<RwLock<ManagerState>>,
        handler_events_tx: broadcast::Sender<HandlerEvent>,
    ) -> Self {
        Self {
            state,
            handler_events_tx,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WaitForHandlerPayload {
    handler_name: String,
    timeout_ms: Option<u64>,
}

#[async_trait]
impl ActiveMessageHandler for WaitForHandlerHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let payload: WaitForHandlerPayload = message.deserialize()?;

        let state = self.state.read().await;
        if state.handlers.contains_key(&payload.handler_name) {
            debug!("Handler '{}' already registered", payload.handler_name);

            if let ResponseContext::Single(sender) = response {
                let response = WaitForHandlerResponse {
                    handler_name: payload.handler_name,
                    available: true,
                };
                sender.send(response).await?;
            }

            return Ok(());
        }
        drop(state);

        let mut rx = self.handler_events_tx.subscribe();

        let timeout = payload
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(30));

        let deadline = tokio::time::Instant::now() + timeout;
        let mut handler_found = false;

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Ok(HandlerEvent::Registered { name, .. }) => {
                            if name == payload.handler_name {
                                debug!("Handler '{}' is now registered", payload.handler_name);
                                handler_found = true;
                                break;
                            }
                        }
                        Ok(HandlerEvent::Deregistered { .. }) => {}
                        Err(_) => {
                            anyhow::bail!("Handler event channel closed");
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    debug!(
                        "Timeout waiting for handler '{}'",
                        payload.handler_name
                    );
                    break;
                }
            }
        }

        if let ResponseContext::Single(sender) = response {
            let response = WaitForHandlerResponse {
                handler_name: payload.handler_name.clone(),
                available: handler_found,
            };
            sender.send(response).await?;
        }

        if handler_found {
            Ok(())
        } else {
            anyhow::bail!("Timeout waiting for handler '{}'", payload.handler_name)
        }
    }

    fn name(&self) -> &str {
        "_wait_for_handler"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "handler_name": { "type": "string" },
                        "timeout_ms": { "type": "number" }
                    },
                    "required": ["handler_name"]
                })
            });
        Some(&SCHEMA)
    }
}

#[derive(Debug)]
pub struct HealthCheckHandler;

#[async_trait]
impl ActiveMessageHandler for HealthCheckHandler {
    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        debug!("Health check received");

        if let ResponseContext::Single(sender) = response {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let response = HealthCheckResponse {
                status: "ok".to_string(),
                timestamp,
            };
            sender.send(response).await?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "_health_check"
    }
}

#[derive(Debug)]
pub struct AckHandler {
    client: Arc<super::client::ZmqActiveMessageClient>,
}

impl AckHandler {
    pub fn new(client: Arc<super::client::ZmqActiveMessageClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AckPayload {
    ack_id: String,
    status: AckStatus,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum AckStatus {
    Ack,
    Nack { error: String },
}

#[async_trait]
impl ActiveMessageHandler for AckHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let payload: AckPayload = message.deserialize()?;
        let ack_id = uuid::Uuid::parse_str(&payload.ack_id)?;

        match payload.status {
            AckStatus::Ack => {
                self.client
                    .complete_ack(ack_id, message.sender_instance)
                    .await?;
                debug!("Processed ACK for {}", ack_id);
            }
            AckStatus::Nack { error } => {
                self.client
                    .complete_nack(ack_id, message.sender_instance, error.clone())
                    .await?;
                debug!("Processed NACK for {}: {}", ack_id, error);
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "_ack"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "ack_id": { "type": "string" },
                        "status": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": { "const": "Ack" }
                                    },
                                    "required": ["type"]
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": { "const": "Nack" },
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "error": { "type": "string" }
                                            },
                                            "required": ["error"]
                                        }
                                    },
                                    "required": ["type", "data"]
                                }
                            ]
                        }
                    },
                    "required": ["ack_id", "status"]
                })
            });
        Some(&SCHEMA)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_message::{
        client::ActiveMessageClient,
        handler::ActiveMessage,
        response::{ResponseContext, SingleResponseSender},
    };
    use std::sync::Arc;
    use tokio::sync::{RwLock, oneshot};
    use uuid::Uuid;

    #[derive(Debug, Clone)]
    struct MockClient {
        instance_id: Uuid,
        endpoint: String,
    }

    #[async_trait]
    impl ActiveMessageClient for MockClient {
        fn instance_id(&self) -> uuid::Uuid {
            self.instance_id
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        async fn send_message(
            &self,
            _target: uuid::Uuid,
            _handler: &str,
            _payload: bytes::Bytes,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn broadcast_message(
            &self,
            _handler: &str,
            _payload: bytes::Bytes,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn list_peers(&self) -> anyhow::Result<Vec<crate::active_message::client::PeerInfo>> {
            Ok(vec![])
        }

        async fn connect_to_peer(
            &self,
            _peer: crate::active_message::client::PeerInfo,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn disconnect_from_peer(&self, _instance_id: uuid::Uuid) -> anyhow::Result<()> {
            Ok(())
        }

        async fn await_handler(
            &self,
            _instance_id: uuid::Uuid,
            _handler: &str,
            _timeout: Option<std::time::Duration>,
        ) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn list_handlers(&self, _instance_id: uuid::Uuid) -> anyhow::Result<Vec<String>> {
            Ok(vec![])
        }

        async fn send_raw_message(
            &self,
            _target: uuid::Uuid,
            _message: ActiveMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_acceptance(
            &self,
            _message_id: Uuid,
            _sender: oneshot::Sender<()>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_response(
            &self,
            _message_id: Uuid,
            _sender: oneshot::Sender<bytes::Bytes>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_ack(
            &self,
            _ack_id: Uuid,
            _timeout: std::time::Duration,
        ) -> anyhow::Result<oneshot::Receiver<Result<(), String>>> {
            let (_tx, rx) = oneshot::channel();
            Ok(rx)
        }
    }

    #[tokio::test]
    async fn test_health_check_handler_response() {
        let handler = HealthCheckHandler;
        let mock_client = MockClient {
            instance_id: Uuid::new_v4(),
            endpoint: "test://localhost".to_string(),
        };

        let response_sender = SingleResponseSender::new(
            Arc::new(mock_client.clone()),
            Uuid::new_v4(),
            Uuid::new_v4(),
            "_health_check".to_string(),
        );
        let response_context = ResponseContext::Single(response_sender);

        let message = ActiveMessage {
            message_id: Uuid::new_v4(),
            handler_name: "_health_check".to_string(),
            sender_instance: Uuid::new_v4(),
            payload: bytes::Bytes::new(),
            metadata: serde_json::Value::Null,
        };

        // This should succeed without sending a response since we can't easily intercept the send
        let result = handler
            .handle(message, &mock_client, response_context)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_handlers_response_structure() {
        let response = ListHandlersResponse {
            handlers: vec!["handler1".to_string(), "handler2".to_string()],
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: ListHandlersResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.handlers.len(), 2);
        assert!(deserialized.handlers.contains(&"handler1".to_string()));
        assert!(deserialized.handlers.contains(&"handler2".to_string()));
    }

    #[tokio::test]
    async fn test_health_check_response_structure() {
        let response = HealthCheckResponse {
            status: "ok".to_string(),
            timestamp: 1234567890,
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: HealthCheckResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.status, "ok");
        assert_eq!(deserialized.timestamp, 1234567890);
    }

    #[tokio::test]
    async fn test_register_service_response_structure() {
        let response = RegisterServiceResponse {
            registered: true,
            instance_id: "test-id".to_string(),
            endpoint: "test://endpoint".to_string(),
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: RegisterServiceResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(deserialized.registered);
        assert_eq!(deserialized.instance_id, "test-id");
        assert_eq!(deserialized.endpoint, "test://endpoint");
    }

    #[tokio::test]
    async fn test_wait_for_handler_response_structure() {
        let response = WaitForHandlerResponse {
            handler_name: "test_handler".to_string(),
            available: true,
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: WaitForHandlerResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.handler_name, "test_handler");
        assert!(deserialized.available);
    }
}
