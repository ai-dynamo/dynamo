// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the `EndpointPicker` trait and its associated types (`Endpoint`,
//! `RequestInfo`, `PickResult`, `PickError`). This mirrors the Go LW-EPP's
//! `EndpointPicker` interface from GAIE #2834. The ext_proc server is generic
//! over this trait — it handles the Envoy protocol, the picker handles the
//! routing decision.

use std::collections::HashMap;

/// Endpoint represents a model server pod endpoint available for serving requests.
/// Mirrors Go `epplight.Endpoint` in pkg/lwepp/datastore/datastore.go
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Pod name
    pub pod_name: String,
    /// Pod IP address
    pub address: String,
    /// Target port
    pub port: String,
    /// Pod labels
    pub labels: HashMap<String, String>,
}

impl Endpoint {
    /// Returns the endpoint in "ip:port" format.
    pub fn address_port(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

/// RequestInfo contains metadata about the incoming HTTP request.
/// Mirrors Go `epplight.RequestInfo`.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// HTTP request headers, preserved as ordered pairs so that repeated
    /// header keys (valid in HTTP) are not silently collapsed.
    pub headers: Vec<(String, String)>,
    /// Raw request body (empty for GET)
    pub body: Vec<u8>,
    /// Model name extracted from the request body
    pub model: String,
    /// From x-gateway-destination-endpoint-subset metadata
    pub candidate_subset: Vec<String>,
}

/// PickResult contains the endpoint selection result.
/// Mirrors Go `epplight.PickResult`, extended with Dynamo-specific
/// routing headers that the backend workers need.
#[derive(Debug, Clone, Default)]
pub struct PickResult {
    /// Primary endpoint in "ip:port" format
    pub endpoint: String,
    /// Optional fallback endpoints in "ip:port" format
    pub fallbacks: Vec<String>,
    /// Extra headers to inject into the forwarded request.
    /// Used by Dynamo for routing metadata (worker IDs, DP ranks, routing mode).
    pub headers: Vec<(String, String)>,
    /// Pre-computed token IDs from the picker's tokenization.
    /// Injected into the request body as `nvext.token_data` so the backend
    /// skips redundant tokenization. Mirrors Go EPP's `setTokenizedPrompt`.
    pub token_ids: Option<Vec<u32>>,
}

/// EndpointPicker is the central abstraction for endpoint selection.
/// Mirrors Go `epplight.EndpointPicker` interface.
///
/// Implementations receive request metadata and a list of available endpoints,
/// and return the chosen endpoint(s). The ext_proc server handles all Envoy
/// protocol details, subset filtering, and pod discovery.
#[tonic::async_trait]
pub trait EndpointPicker: Send + Sync + 'static {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError>;
}

/// Error from an endpoint picker.
#[derive(Debug, thiserror::Error)]
pub enum PickError {
    #[error("no endpoints available")]
    NoEndpoints,
    #[error("routing failed: {0}")]
    RoutingFailed(String),
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
}
