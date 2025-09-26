// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use super::client::ActiveMessageClient;
use super::response::ResponseContext;

pub type HandlerId = String;
pub type InstanceId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMessage {
    pub message_id: Uuid,
    pub handler_name: HandlerId,
    pub sender_instance: InstanceId,
    pub payload: Bytes,
    pub metadata: serde_json::Value,
}

impl ActiveMessage {
    pub fn new(handler_name: impl Into<String>, payload: Bytes) -> Self {
        Self {
            message_id: Uuid::new_v4(),
            handler_name: handler_name.into(),
            sender_instance: Uuid::new_v4(),
            payload,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_sender(mut self, sender: InstanceId) -> Self {
        self.sender_instance = sender;
        self
    }

    /// Deserialize payload to typed request
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize payload: {}", e))
    }

    /// Get payload as JSON value
    pub fn as_json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse payload as JSON: {}", e))
    }
}

#[async_trait]
pub trait ActiveMessageHandler: Send + Sync + std::fmt::Debug {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()>;

    fn name(&self) -> &str;

    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        if let Some(schema) = self.schema() {
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            let instance = jsonschema::JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandlerEvent {
    Registered {
        name: HandlerId,
        instance: InstanceId,
    },
    Deregistered {
        name: HandlerId,
        instance: InstanceId,
    },
}

pub const SYSTEM_HANDLER_PREFIX: &str = "_";

pub fn is_system_handler(name: &str) -> bool {
    name.starts_with(SYSTEM_HANDLER_PREFIX)
}

pub fn validate_handler_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Handler name cannot be empty");
    }

    if is_system_handler(name) {
        anyhow::bail!(
            "User-space handlers cannot start with '{}'. This prefix is reserved for system handlers.",
            SYSTEM_HANDLER_PREFIX
        );
    }

    Ok(())
}
