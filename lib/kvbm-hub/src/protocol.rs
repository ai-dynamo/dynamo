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

    /// Hub-initiated velo probe.
    ///
    /// `POST /v1/instances/{instance_id}/probe` → [`super::ProbeResponse`]
    pub const INSTANCE_PROBE: &str = "/v1/instances/{instance_id}/probe";

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
    /// The hub's own velo `InstanceId`, when the hub runs with a velo
    /// participant. Clients can resolve the hub's `PeerInfo` via
    /// `GET /v1/peers/instance/{id}` and wire it into their own Velo for
    /// bidirectional active messaging. `None` when the hub has no velo.
    #[serde(default)]
    pub hub_instance_id: Option<InstanceId>,
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

/// Response body for `POST /v1/instances/{instance_id}/probe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResponse {
    /// Echoed sequence number from the velo heartbeat.
    pub seq: u64,
    /// Whether the instance reported itself healthy.
    pub ok: bool,
}

/// Response body for `GET /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListInstancesResponse {
    /// All currently registered instances.
    pub instances: Vec<PeerInfo>,
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

/// Path helper: format [`paths::INSTANCE_PROBE`] with a concrete id.
pub fn instance_probe(id: InstanceId) -> String {
    format!("/v1/instances/{id}/probe")
}

#[cfg(test)]
mod tests {
    use super::*;
    use velo_common::WorkerAddress;

    fn make_peer_info() -> PeerInfo {
        let id = InstanceId::new_v4();
        PeerInfo::new(id, WorkerAddress::from_encoded(b"test".to_vec()))
    }

    #[test]
    fn heartbeat_response_serde_round_trip() {
        for acknowledged in [true, false] {
            let orig = HeartbeatResponse { acknowledged };
            let json = serde_json::to_string(&orig).unwrap();
            let back: HeartbeatResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(back.acknowledged, acknowledged);
        }
    }

    #[test]
    fn heartbeat_response_default_is_not_acknowledged() {
        assert!(!HeartbeatResponse::default().acknowledged);
    }

    #[test]
    fn register_request_serde_round_trip() {
        let peer_info = make_peer_info();
        let orig = RegisterRequest {
            peer_info: peer_info.clone(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: RegisterRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.peer_info.instance_id(), peer_info.instance_id());
    }

    #[test]
    fn register_response_serde_round_trip() {
        let instance_id = InstanceId::new_v4();
        let hub_instance_id = Some(InstanceId::new_v4());
        let orig = RegisterResponse {
            instance_id,
            hub_instance_id,
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: RegisterResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.instance_id, instance_id);
        assert_eq!(back.hub_instance_id, hub_instance_id);
    }

    #[test]
    fn register_response_accepts_legacy_payload_without_hub_instance_id() {
        let instance_id = InstanceId::new_v4();
        let legacy_json = format!("{{\"instance_id\":\"{instance_id}\"}}");
        let back: RegisterResponse = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(back.instance_id, instance_id);
        assert!(back.hub_instance_id.is_none());
    }

    #[test]
    fn peer_lookup_response_serde_round_trip() {
        let peer_info = make_peer_info();
        let orig = PeerLookupResponse {
            peer_info: peer_info.clone(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: PeerLookupResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.peer_info.instance_id(), peer_info.instance_id());
    }

    #[test]
    fn error_code_serde_all_variants() {
        for code in [
            ErrorCode::NotFound,
            ErrorCode::Conflict,
            ErrorCode::BadRequest,
            ErrorCode::Internal,
        ] {
            let json = serde_json::to_string(&code).unwrap();
            let back: ErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, code);
        }
    }

    #[test]
    fn error_body_serde_round_trip() {
        let orig = ErrorBody {
            code: ErrorCode::NotFound,
            message: "instance abc not found".to_string(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: ErrorBody = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code, ErrorCode::NotFound);
        assert_eq!(back.message, "instance abc not found");
        assert_eq!(back.to_string(), "instance abc not found");
    }

    #[test]
    fn path_helpers_contain_id() {
        let id = InstanceId::new_v4();
        let worker_id = id.worker_id();
        let id_str = id.to_string();

        assert!(peers_by_instance(id).contains(&id_str));
        assert!(peers_by_worker(worker_id).contains(&worker_id.as_u64().to_string()));
        assert!(instance_by_id(id).contains(&id_str));
        assert!(instance_heartbeat(id).contains(&id_str));
    }

    #[test]
    fn path_helpers_have_expected_prefixes() {
        let id = InstanceId::new_v4();
        let worker_id = id.worker_id();

        assert!(peers_by_instance(id).starts_with("/v1/peers/instance/"));
        assert!(peers_by_worker(worker_id).starts_with("/v1/peers/worker/"));
        assert!(instance_by_id(id).starts_with("/v1/instances/"));
        assert!(instance_heartbeat(id).starts_with("/v1/instances/"));
        assert!(instance_heartbeat(id).ends_with("/heartbeat"));
    }
}
