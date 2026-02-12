// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode},
    protocols::annotated::Annotated,
    protocols::maybe_error::MaybeError,
};
use serde::{Deserialize, Serialize};

/// Response from the worker's cache_control service mesh endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControlResponse {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpinned_count: Option<u32>,
}

impl MaybeError for CacheControlResponse {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        CacheControlResponse {
            status: "error".to_string(),
            message: Some(err.to_string()),
            pinned_count: None,
            unpinned_count: None,
        }
    }

    fn err(&self) -> Option<anyhow::Error> {
        if self.status == "error" {
            Some(anyhow::anyhow!(
                "cache_control error: {}",
                self.message.as_deref().unwrap_or("unknown")
            ))
        } else {
            None
        }
    }
}

/// A PushRouter client typed for cache_control requests/responses.
pub type CacheControlClient = PushRouter<serde_json::Value, Annotated<CacheControlResponse>>;

/// Create a cache_control client from a component.
///
/// Connects to the "cache_control" endpoint on the given component and returns
/// a typed PushRouter client for sending cache control operations (pin_prefix,
/// unpin_prefix) to workers.
pub async fn create_cache_control_client(component: &Component) -> Result<CacheControlClient> {
    let client = component.endpoint("cache_control").client().await?;
    Ok(CacheControlClient::from_client(client, RouterMode::KV).await?)
}
