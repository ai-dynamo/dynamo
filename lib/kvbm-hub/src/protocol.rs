// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP wire protocol shared between [`HubClient`](crate::HubClient) and
//! [`HubServer`](crate::HubServer).
//!
//! The hub exposes two axum listeners:
//! - **Port 1337** — peer discovery HTTP endpoints (see [`paths`]).
//! - **Port 8337** — control-plane endpoints (registration, heartbeat, health).
//!
//! All request/response bodies are JSON.

use serde::{Deserialize, Serialize};
use velo_common::{InstanceId, PeerInfo, WorkerId};

/// Default HTTP port for peer-discovery lookups (the `PeerDiscovery` surface).
pub const DEFAULT_DISCOVERY_PORT: u16 = 1337;

/// Default HTTP port for the control plane (registration, heartbeat).
pub const DEFAULT_CONTROL_PORT: u16 = 8337;

/// URL path fragments for the HTTP API.
pub mod paths {
    /// Peer-discovery lookup by `InstanceId`.
    ///
    /// `GET /v1/peers/instance/{instance_id}` → [`super::PeerLookupResponse`]
    pub const PEERS_BY_INSTANCE: &str = "/v1/peers/instance/{instance_id}";

    /// Peer-discovery lookup by `WorkerId`.
    ///
    /// `GET /v1/peers/worker/{worker_id}` → [`super::PeerLookupResponse`]
    pub const PEERS_BY_WORKER: &str = "/v1/peers/worker/{worker_id}";

    /// Register an instance.
    ///
    /// `POST /v1/instances` with body [`super::RegisterRequest`]
    /// → [`super::RegisterResponse`]
    pub const INSTANCES: &str = "/v1/instances";

    /// Unregister an instance.
    ///
    /// `DELETE /v1/instances/{instance_id}`
    pub const INSTANCE_BY_ID: &str = "/v1/instances/{instance_id}";

    /// Liveness heartbeat.
    ///
    /// `POST /v1/instances/{instance_id}/heartbeat` → [`super::HeartbeatResponse`]
    pub const INSTANCE_HEARTBEAT: &str = "/v1/instances/{instance_id}/heartbeat";

    /// Hub health probe.
    ///
    /// `GET /health` → `200 OK`
    pub const HEALTH: &str = "/health";
}

/// Request body for `POST /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    /// Full peer information (instance id + opaque worker address).
    pub peer_info: PeerInfo,
}

/// Response body for `POST /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    /// The registered instance id, echoed back.
    pub instance_id: InstanceId,
}

/// Response body for peer-discovery lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerLookupResponse {
    /// Full peer information for the requested id.
    pub peer_info: PeerInfo,
}

/// Response body for `POST /v1/instances/{instance_id}/heartbeat`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeartbeatResponse {
    /// Whether the hub considers this instance registered.
    pub acknowledged: bool,
}

/// Typed error body returned by the hub on non-2xx responses.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct ErrorBody {
    /// Stable machine-readable error code.
    pub code: ErrorCode,
    /// Human-readable description.
    pub message: String,
}

/// Stable error codes returned by the hub API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Requested instance or worker id is not registered.
    NotFound,
    /// Registration conflicts with an existing registration.
    Conflict,
    /// Request payload was malformed.
    BadRequest,
    /// Unexpected server-side failure.
    Internal,
}

/// Path helper: format [`paths::PEERS_BY_INSTANCE`] with a concrete id.
pub fn peers_by_instance(id: InstanceId) -> String {
    format!("/v1/peers/instance/{id}")
}

/// Path helper: format [`paths::PEERS_BY_WORKER`] with a concrete id.
pub fn peers_by_worker(id: WorkerId) -> String {
    format!("/v1/peers/worker/{}", id.as_u64())
}

/// Path helper: format [`paths::INSTANCE_BY_ID`] with a concrete id.
pub fn instance_by_id(id: InstanceId) -> String {
    format!("/v1/instances/{id}")
}

/// Path helper: format [`paths::INSTANCE_HEARTBEAT`] with a concrete id.
pub fn instance_heartbeat(id: InstanceId) -> String {
    format!("/v1/instances/{id}/heartbeat")
}
