// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode},
    protocols::annotated::Annotated,
};

/// A PushRouter client typed for cache_control requests/responses.
///
/// Both request and response are untyped JSON. The worker's cache_control
/// endpoint returns {"status": "ok"/"error", ...} but the router treats
/// PIN as fire-and-forget and only logs the response at debug level.
pub type CacheControlClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Create a cache_control client from a component.
///
/// Connects to the "cache_control" endpoint on the given component and returns
/// a PushRouter client for sending cache control operations (pin_prefix,
/// unpin_prefix) to workers.
pub async fn create_cache_control_client(component: &Component) -> Result<CacheControlClient> {
    let client = component.endpoint("cache_control").client().await?;
    Ok(CacheControlClient::from_client(client, RouterMode::KV).await?)
}
