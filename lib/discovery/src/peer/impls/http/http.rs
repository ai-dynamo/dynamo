// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared HTTP types for discovery service and client.
//!
//! This module contains DTOs and types used for communication between
//! HTTP clients and the discovery service.

use serde::{Deserialize, Serialize};

use crate::{InstanceId, PeerInfo, WorkerAddress, WorkerId};

/// Request to register a peer in the discovery service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    /// Instance ID of the peer
    pub instance_id: InstanceId,
    /// Address where the worker can be reached
    pub worker_address: WorkerAddress,
}

/// Response from registering a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    /// Whether the registration was successful
    pub success: bool,
    /// Optional error message if registration failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response containing peer information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfoResponse {
    /// The peer information
    pub peer_info: PeerInfo,
}

/// Response containing a list of peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerListResponse {
    /// List of discovered peers
    pub peers: Vec<PeerInfo>,
}

/// Bootstrap peer information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapPeer {
    /// Peer ID in the P2P network
    pub peer_id: String,
    /// Multiaddresses where the peer can be reached
    pub addresses: Vec<String>,
}

/// Response containing bootstrap peers for a cluster_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapPeersResponse {
    /// The cluster_id this bootstrap list is for
    pub cluster_id: String,
    /// List of bootstrap peers
    pub peers: Vec<BootstrapPeer>,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status ("ok" or error message)
    pub status: String,
    /// Whether the meta P2P swarm is operational
    pub meta_swarm_ready: bool,
}

/// Error response from the discovery service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Optional error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    /// Create a new error response.
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            details: None,
        }
    }

    /// Create a new error response with details.
    pub fn with_details(error: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            details: Some(details.into()),
        }
    }
}

/// HTTP endpoint paths.
pub mod endpoints {
    /// Register a peer by instance ID and address.
    /// POST /register
    /// Body: RegisterRequest
    pub const REGISTER: &str = "/register";

    /// Heartbeat to extend registration TTL.
    /// PUT /heartbeat/{instance_id}
    pub const HEARTBEAT: &str = "/heartbeat";

    /// Unregister a peer by instance ID.
    /// DELETE /unregister/{instance_id}
    pub const UNREGISTER: &str = "/unregister";

    /// Discover a peer by worker ID.
    /// GET /discover/worker/{worker_id}
    pub const DISCOVER_WORKER: &str = "/discover/worker";

    /// Discover a peer by instance ID.
    /// GET /discover/instance/{instance_id}
    pub const DISCOVER_INSTANCE: &str = "/discover/instance";

    /// Discover all peers.
    /// GET /discover/all
    pub const DISCOVER_ALL: &str = "/discover/all";

    /// Get bootstrap peers for a cluster_id.
    /// GET /bootstrap-peers/{cluster_id}
    pub const BOOTSTRAP_PEERS: &str = "/bootstrap-peers";

    /// Health check endpoint.
    /// GET /health
    pub const HEALTH: &str = "/health";

    // Monitoring API endpoints
    /// Get overall service statistics.
    /// GET /api/status
    pub const API_STATUS: &str = "/api/status";

    /// Get list of all clusters.
    /// GET /api/clusters
    pub const API_CLUSTERS: &str = "/api/clusters";

    /// Get instances for a specific cluster.
    /// GET /api/clusters/{cluster_id}
    pub const API_CLUSTER_INSTANCES: &str = "/api/clusters";

    /// Get all instances with detailed information.
    /// GET /api/instances
    pub const API_INSTANCES: &str = "/api/instances";

    /// Server-Sent Events stream for real-time updates.
    /// GET /api/events
    pub const API_EVENTS: &str = "/api/events";
}

// ============================================================================
// Monitoring API Types
// ============================================================================

/// Health status of an instance based on TTL remaining.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// TTL > 30 seconds remaining
    Healthy,
    /// TTL 10-30 seconds remaining
    Warning,
    /// TTL < 10 seconds remaining
    Critical,
}

/// Detailed instance information for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceDetails {
    /// Instance ID
    pub instance_id: InstanceId,
    /// Worker ID derived from instance ID
    pub worker_id: WorkerId,
    /// Worker network address
    pub worker_address: WorkerAddress,
    /// TTL remaining in seconds
    pub ttl_remaining_secs: u64,
    /// Health status based on TTL
    pub health_status: HealthStatus,
    /// Unix timestamp of last heartbeat (seconds since epoch)
    pub last_heartbeat_unix: u64,
}

/// Overall service statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Total number of registered instances
    pub total_instances: usize,
    /// Number of healthy instances (TTL > 30s)
    pub healthy_instances: usize,
    /// Number of warning instances (TTL 10-30s)
    pub warning_instances: usize,
    /// Number of critical instances (TTL < 10s)
    pub critical_instances: usize,
    /// Number of clusters/prefixes being served
    pub total_clusters: usize,
    /// Service uptime in seconds
    pub uptime_secs: u64,
}

/// Information about a cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster ID
    pub cluster_id: String,
    /// Number of instances in this cluster
    pub instance_count: usize,
    /// Number of healthy instances
    pub healthy_count: usize,
}

/// Response containing list of clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClustersResponse {
    /// List of clusters
    pub clusters: Vec<ClusterInfo>,
}

/// Response containing detailed instance information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstancesResponse {
    /// List of instances with details
    pub instances: Vec<InstanceDetails>,
}

/// Server-Sent Event types for real-time monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MonitoringEvent {
    /// New instance registered
    Registered {
        instance_id: InstanceId,
        worker_id: WorkerId,
        ttl_secs: u64,
    },
    /// Instance sent heartbeat
    Heartbeat {
        instance_id: InstanceId,
        ttl_remaining_secs: u64,
    },
    /// Instance expired
    Expired { instance_id: InstanceId },
    /// Instance explicitly unregistered
    Unregistered { instance_id: InstanceId },
    /// Periodic stats update
    StatsUpdate {
        total_instances: usize,
        healthy: usize,
        warning: usize,
        critical: usize,
    },
}
