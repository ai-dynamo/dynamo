// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Request Plane Server Interface
//!
//! This module defines a transport-agnostic interface for request plane servers.
//! All transport implementations (HTTP, TCP, NATS) implement this trait to provide
//! a consistent interface for endpoint registration and management.

use super::*;
use crate::SystemHealth;
use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;

/// Unified interface for request plane servers
///
/// This trait abstracts over different transport mechanisms (HTTP/2, TCP, NATS)
/// providing a consistent interface for registering endpoints and managing server lifecycle.
///
/// # Design Principles
///
/// 1. **Transport Agnostic**: Implementations can be swapped without changing business logic
/// 2. **Multiplexed**: All servers handle multiple endpoints on a single port/connection
/// 3. **Async by Default**: All operations are async to support high concurrency
/// 4. **Health Monitoring**: Servers provide health status for monitoring
///
/// # Example
///
/// ```ignore
/// use dynamo_runtime::pipeline::network::ingress::RequestPlaneServer;
///
/// async fn register(server: &dyn RequestPlaneServer) -> Result<()> {
///     server.register_endpoint(
///         "generate".to_string(),
///         handler,
///         instance_id,
///         "dynamo".to_string(),
///         "backend".to_string(),
///         system_health,
///     ).await?;
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    /// Register an endpoint handler with the server
    ///
    /// # Arguments
    ///
    /// * `endpoint_name` - Name/path for routing (e.g., "generate", "health")
    /// * `service_handler` - Handler that processes incoming requests
    /// * `instance_id` - Unique instance identifier for this endpoint
    /// * `namespace` - Service namespace (e.g., "dynamo")
    /// * `component_name` - Component name (e.g., "backend", "frontend")
    /// * `system_health` - Health tracking for this endpoint
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if registration succeeds, or an error if:
    /// - Endpoint name is already registered
    /// - Server is not running or has been stopped
    /// - Transport-specific errors occur
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()>;

    /// Unregister an endpoint from the server
    ///
    /// # Arguments
    ///
    /// * `endpoint_name` - Name of the endpoint to unregister
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if unregistration succeeds or endpoint doesn't exist.
    /// Errors are only returned for transport-specific failures.
    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()>;

    /// Get server bind address or identifier
    ///
    /// Returns a transport-specific address string:
    /// - HTTP: `"http://0.0.0.0:8888"`
    /// - TCP: `"tcp://0.0.0.0:9999"`
    /// - NATS: `"nats://localhost:4222"`
    ///
    /// Used for logging, debugging, and service discovery.
    fn address(&self) -> String;

    /// Get the transport name
    ///
    /// Returns a static string identifier for the transport type.
    /// Used for logging and debugging.
    ///
    /// # Examples
    ///
    /// - `"http"` - HTTP/2 transport
    /// - `"tcp"` - Raw TCP transport
    /// - `"nats"` - NATS messaging
    fn transport_name(&self) -> &'static str;

    /// Check if server is healthy and ready to accept requests
    ///
    /// Returns `true` if the server is operational and can handle requests.
    /// This is a lightweight check that doesn't perform actual network I/O.
    ///
    /// Implementations should return `false` if:
    /// - Server has been explicitly stopped
    /// - Underlying transport is disconnected
    /// - Server encountered a fatal error
    fn is_healthy(&self) -> bool;
}
