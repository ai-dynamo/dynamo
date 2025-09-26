// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    handler::{ActiveMessage, ActiveMessageHandler},
    response::ResponseContext,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeRequest {
    pub x: i32,
    pub y: i32,
    pub operation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeResponse {
    pub result: i32,
}

#[derive(Debug)]
pub struct ComputeHandler;

#[async_trait]
impl ActiveMessageHandler for ComputeHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let request: ComputeRequest = message.deserialize()?;

        let result = match request.operation.as_str() {
            "add" => request.x + request.y,
            "multiply" => request.x * request.y,
            _ => {
                info!("Unknown operation: {}", request.operation);
                return Ok(());
            }
        };

        info!(
            "Computed {} {} {} = {}",
            request.x, request.operation, request.y, result
        );

        // Handle response if one was expected
        match response {
            ResponseContext::Single(sender) => {
                let response = ComputeResponse { result };
                sender.send(response).await?;
                info!("Sent response: {}", result);
            }
            ResponseContext::None => {
                info!("No response expected");
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "compute"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" },
                        "y": { "type": "number" },
                        "operation": { "type": "string" }
                    },
                    "required": ["x", "y", "operation"]
                })
            });
        Some(&SCHEMA)
    }
}
