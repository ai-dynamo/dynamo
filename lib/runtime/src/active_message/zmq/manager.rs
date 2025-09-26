// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{Mutex, RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::active_message::{
    client::ActiveMessageClient,
    handler::{ActiveMessage, ActiveMessageHandler, HandlerEvent, HandlerId, InstanceId},
    manager::{ActiveMessageManager, HandlerConfig},
    response::{ResponseContext, SingleResponseSender},
};

use super::{
    builtin_handlers::{
        AckHandler, HealthCheckHandler, ListHandlersHandler, RegisterServiceHandler,
        WaitForHandlerHandler,
    },
    client::ZmqActiveMessageClient,
    transport::ZmqTransport,
};

use crate::utils::tasks::tracker::{LogOnlyPolicy, TaskTracker, UnlimitedScheduler};

pub(crate) struct HandlerEntry {
    pub handler: Arc<dyn ActiveMessageHandler>,
    pub task_tracker: TaskTracker,
}

pub struct ManagerState {
    pub instance_id: InstanceId,
    pub endpoint: String,
    pub(crate) handlers: HashMap<HandlerId, HandlerEntry>,
    pub client: Arc<ZmqActiveMessageClient>,
    pub handler_events_tx: broadcast::Sender<HandlerEvent>,
}

impl std::fmt::Debug for ManagerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagerState")
            .field("instance_id", &self.instance_id)
            .field("endpoint", &self.endpoint)
            .field("handlers", &self.handlers.keys())
            .field("client", &self.client)
            .finish()
    }
}

pub struct ZmqActiveMessageManager {
    state: Arc<RwLock<ManagerState>>,
    client: Arc<ZmqActiveMessageClient>,
    handler_events_tx: broadcast::Sender<HandlerEvent>,
    cancel_token: CancellationToken,
    receiver_task: Arc<Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
    ack_cleanup_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl ZmqActiveMessageManager {
    pub fn zmq_client(&self) -> Arc<ZmqActiveMessageClient> {
        self.client.clone()
    }

    pub fn manager_state(&self) -> Arc<RwLock<ManagerState>> {
        self.state.clone()
    }

    pub async fn new(endpoint: String, cancel_token: CancellationToken) -> Result<Self> {
        let instance_id = InstanceId::new_v4();

        let context = tmq::Context::new();
        let sub_transport = ZmqTransport::new_subscriber_bound(&context, &endpoint)?;
        let bound_endpoint = sub_transport
            .local_endpoint()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bound endpoint"))?
            .clone();

        info!(
            "ZmqActiveMessageManager bound to {} (instance: {})",
            bound_endpoint, instance_id
        );

        let client = Arc::new(ZmqActiveMessageClient::new(
            instance_id,
            bound_endpoint.clone(),
        ));

        let (handler_events_tx, _) = broadcast::channel(1024);

        let state = Arc::new(RwLock::new(ManagerState {
            instance_id,
            endpoint: bound_endpoint,
            handlers: HashMap::new(),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
        }));

        let manager = Self {
            state: state.clone(),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
            cancel_token: cancel_token.clone(),
            receiver_task: Arc::new(Mutex::new(None)),
            ack_cleanup_task: Arc::new(Mutex::new(None)),
        };

        manager.register_builtin_handlers().await?;

        let receiver_task = tokio::spawn(Self::receive_loop(
            state.clone(),
            sub_transport,
            cancel_token.clone(),
        ));

        let ack_cleanup_task =
            tokio::spawn(Self::ack_cleanup_loop(client.clone(), cancel_token.clone()));

        // Store the task handles
        *manager.receiver_task.lock().await = Some(receiver_task);
        *manager.ack_cleanup_task.lock().await = Some(ack_cleanup_task);

        Ok(manager)
    }

    async fn ack_cleanup_loop(
        client: Arc<ZmqActiveMessageClient>,
        cancel_token: CancellationToken,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    client.cleanup_expired_acks().await;
                }
                _ = cancel_token.cancelled() => {
                    debug!("ACK cleanup loop cancelled");
                    break;
                }
            }
        }
    }

    async fn register_builtin_handlers(&self) -> Result<()> {
        let state = self.state.read().await;
        let client = state.client.clone();
        let handler_events_tx = state.handler_events_tx.clone();
        drop(state);

        let ack_handler = Arc::new(AckHandler::new(client.clone()));
        self.register_handler(ack_handler, None).await?;

        let register_service = Arc::new(RegisterServiceHandler::new(client.clone()));
        self.register_handler(register_service, None).await?;

        let list_handlers = Arc::new(ListHandlersHandler::new(self.state.clone()));
        self.register_handler(list_handlers, None).await?;

        let wait_for_handler = Arc::new(WaitForHandlerHandler::new(
            self.state.clone(),
            handler_events_tx,
        ));
        self.register_handler(wait_for_handler, None).await?;

        let health_check = Arc::new(HealthCheckHandler);
        self.register_handler(health_check, None).await?;

        Ok(())
    }

    async fn receive_loop(
        state: Arc<RwLock<ManagerState>>,
        mut transport: ZmqTransport,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    info!("Receive loop cancelled");
                    break;
                }

                result = transport.receive() => {
                    match result {
                        Ok(message) => {
                            if let Err(e) = Self::handle_message(state.clone(), message).await {
                                error!("Failed to handle message: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to receive message: {}", e);
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_message(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        // Handle internal response routing first
        if message.handler_name == "_accept" {
            return Self::handle_acceptance(state, message).await;
        }

        if message.handler_name == "_response" {
            return Self::handle_response(state, message).await;
        }

        let state_read = state.read().await;
        let handler_name = message.handler_name.clone();

        let handler_entry = match state_read.handlers.get(&handler_name) {
            Some(entry) => entry,
            None => {
                warn!("No handler found for: {}", handler_name);
                return Ok(());
            }
        };

        let handler = handler_entry.handler.clone();
        let client = state_read.client.clone();
        let task_tracker = handler_entry.task_tracker.clone();

        drop(state_read);

        debug!("Dispatching message to handler: {}", handler_name);

        // Extract acceptance and response configuration
        let acceptance_mode = message
            .metadata
            .get("_mode")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let accept_id = if matches!(acceptance_mode.as_deref(), Some("confirmed")) {
            message
                .metadata
                .get("_accept_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
        } else {
            None
        };
        let with_response_accept_id = if matches!(acceptance_mode.as_deref(), Some("with_response"))
        {
            message
                .metadata
                .get("_accept_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
        } else {
            None
        };

        // Set up response context based on message metadata
        let response_context = if matches!(acceptance_mode.as_deref(), Some("with_response")) {
            if let Some(response_id_str) = message
                .metadata
                .get("_response_id")
                .and_then(|v| v.as_str())
            {
                if let Ok(response_id) = Uuid::parse_str(response_id_str) {
                    ResponseContext::Single(SingleResponseSender::new(
                        client.clone(),
                        message.sender_instance,
                        response_id,
                        message.handler_name.clone(),
                    ))
                } else {
                    ResponseContext::None
                }
            } else {
                ResponseContext::None
            }
        } else {
            ResponseContext::None
        };

        task_tracker.spawn(async move {
            // Validate payload first if ACK is expected
            if let Some(accept_id) = accept_id {
                match handler.validate_schema(&message.payload) {
                    Ok(()) => {
                        // Payload validation succeeded - send ACK
                        if let Err(e) = client.send_ack(message.sender_instance, accept_id).await {
                            error!("Failed to send ACK for {}: {}", accept_id, e);
                        } else {
                            debug!("Sent ACK for message {}", accept_id);
                        }
                    }
                    Err(validation_error) => {
                        // Payload validation failed - send NACK
                        let error_msg = format!("Payload validation failed: {}", validation_error);
                        if let Err(e) = client
                            .send_nack(message.sender_instance, accept_id, error_msg.clone())
                            .await
                        {
                            error!("Failed to send NACK for {}: {}", accept_id, e);
                        } else {
                            debug!("Sent NACK for message {}: {}", accept_id, error_msg);
                        }
                        // Don't execute handler if validation failed
                        return Ok(());
                    }
                }
            }

            // Handle acceptance for with_response mode
            if let Some(accept_id) = with_response_accept_id {
                // Send acceptance message
                let accept_message = ActiveMessage {
                    message_id: Uuid::new_v4(),
                    handler_name: "_accept".to_string(),
                    sender_instance: client.instance_id(),
                    payload: Bytes::new(),
                    metadata: serde_json::json!({
                        "_accept_for": accept_id.to_string()
                    }),
                };

                if let Err(e) = client
                    .send_raw_message(message.sender_instance, accept_message)
                    .await
                {
                    error!("Failed to send acceptance for {}: {}", accept_id, e);
                } else {
                    debug!("Sent acceptance for message {}", accept_id);
                }
            }

            // Execute handler if validation passed (or no validation required)
            if let Err(e) = handler.handle(message, &*client, response_context).await {
                error!("Handler '{}' failed: {}", handler_name, e);
            }
            Ok(())
        });

        Ok(())
    }

    async fn handle_acceptance(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        if let Some(accept_for_str) = message.metadata.get("_accept_for").and_then(|v| v.as_str())
            && let Ok(accept_id) = Uuid::parse_str(accept_for_str)
        {
            let state_read = state.read().await;
            return state_read
                .client
                .complete_acceptance(accept_id, message.sender_instance)
                .await;
        }
        error!("Invalid acceptance message: {:?}", message);
        Ok(())
    }

    async fn handle_response(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        if let Some(response_to_str) = message
            .metadata
            .get("_response_to")
            .and_then(|v| v.as_str())
            && let Ok(response_id) = Uuid::parse_str(response_to_str)
        {
            let state_read = state.read().await;
            return state_read
                .client
                .complete_response(response_id, message.sender_instance, message.payload)
                .await;
        }
        error!("Invalid response message: {:?}", message);
        Ok(())
    }
}

#[async_trait]
impl ActiveMessageManager for ZmqActiveMessageManager {
    fn client(&self) -> Arc<dyn ActiveMessageClient> {
        self.client.clone()
    }

    async fn register_handler(
        &self,
        handler: Arc<dyn ActiveMessageHandler>,
        config: Option<HandlerConfig>,
    ) -> Result<()> {
        let handler_name = handler.name().to_string();

        let config = config.unwrap_or_default();
        let task_tracker = config.task_tracker.unwrap_or_else(|| {
            TaskTracker::builder()
                .scheduler(UnlimitedScheduler::new())
                .error_policy(LogOnlyPolicy::new())
                .build()
                .expect("Failed to create default task tracker")
        });

        let mut state = self.state.write().await;

        if state.handlers.contains_key(&handler_name) {
            anyhow::bail!("Handler '{}' already registered", handler_name);
        }

        let entry = HandlerEntry {
            handler,
            task_tracker,
        };

        state.handlers.insert(handler_name.clone(), entry);

        let event = HandlerEvent::Registered {
            name: handler_name.clone(),
            instance: state.instance_id,
        };

        let _ = state.handler_events_tx.send(event);

        debug!("Registered handler: {}", handler_name);

        Ok(())
    }

    async fn deregister_handler(&self, name: &str) -> Result<()> {
        let mut state = self.state.write().await;

        if state.handlers.remove(name).is_none() {
            anyhow::bail!("Handler '{}' not found", name);
        }

        let event = HandlerEvent::Deregistered {
            name: name.to_string(),
            instance: state.instance_id,
        };

        let _ = state.handler_events_tx.send(event);

        debug!("Deregistered handler: {}", name);

        Ok(())
    }

    async fn list_handlers(&self) -> Vec<HandlerId> {
        let state = self.state.read().await;
        state.handlers.keys().cloned().collect()
    }

    fn handler_events(&self) -> broadcast::Receiver<HandlerEvent> {
        self.handler_events_tx.subscribe()
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ZmqActiveMessageManager");
        self.cancel_token.cancel();

        // Join background tasks first
        if let Some(receiver_task) = self.receiver_task.lock().await.take() {
            debug!("Joining receiver task");
            if let Err(e) = receiver_task.await {
                warn!("Receiver task join error: {:?}", e);
            }
        }

        if let Some(ack_cleanup_task) = self.ack_cleanup_task.lock().await.take() {
            debug!("Joining ACK cleanup task");
            if let Err(e) = ack_cleanup_task.await {
                warn!("ACK cleanup task join error: {:?}", e);
            }
        }

        // Join handler task trackers
        let state = self.state.read().await;
        for (name, entry) in &state.handlers {
            debug!("Joining task tracker for handler: {}", name);
            entry.task_tracker.join().await;
        }

        Ok(())
    }
}
