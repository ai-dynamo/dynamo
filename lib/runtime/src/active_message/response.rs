// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response context types for handling different response patterns in ActiveMessage system.

use anyhow::Result;
use bytes::Bytes;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::client::ActiveMessageClient;
use super::handler::InstanceId;

/// Response context provided to handlers indicating what kind of response is expected
#[derive(Debug)]
pub enum ResponseContext {
    /// No response expected - fire and forget
    None,
    /// Single typed response expected
    Single(SingleResponseSender),
}

/// Sender for single response back to the original requester
#[derive(Debug)]
pub struct SingleResponseSender {
    client: Arc<dyn ActiveMessageClient>,
    target: InstanceId,
    message_id: Uuid,
    handler_name: String,
}

impl SingleResponseSender {
    /// Create a new single response sender
    pub(crate) fn new(
        client: Arc<dyn ActiveMessageClient>,
        target: InstanceId,
        message_id: Uuid,
        handler_name: String,
    ) -> Self {
        Self {
            client,
            target,
            message_id,
            handler_name,
        }
    }

    /// Send a typed response back to the requester
    pub async fn send<T: Serialize>(&self, response: T) -> Result<()> {
        let payload = Bytes::from(serde_json::to_vec(&response)?);

        // Send response as internal message
        let response_message = super::handler::ActiveMessage {
            message_id: Uuid::new_v4(),
            handler_name: "_response".to_string(),
            sender_instance: self.client.instance_id(),
            payload,
            metadata: serde_json::json!({
                "_response_to": self.message_id.to_string(),
                "_original_handler": self.handler_name,
            }),
        };

        self.client.send_raw_message(self.target, response_message).await
    }

    /// Send raw bytes as response
    pub async fn send_raw(&self, payload: Bytes) -> Result<()> {
        let response_message = super::handler::ActiveMessage {
            message_id: Uuid::new_v4(),
            handler_name: "_response".to_string(),
            sender_instance: self.client.instance_id(),
            payload,
            metadata: serde_json::json!({
                "_response_to": self.message_id.to_string(),
                "_original_handler": self.handler_name,
            }),
        };

        self.client.send_raw_message(self.target, response_message).await
    }
}