// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service Discovery Interface
//!
//! This module defines the ServiceDiscovery interface that can be satisfied by different backends
//! (etcd, kubernetes, etc). This interface de-couples Dynamo from specific discovery mechanisms.

use crate::{Result, CancellationToken};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tokio::sync::mpsc;

/// Status of a service instance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceStatus {
    /// Instance is ready to receive traffic
    Ready,
    /// Instance is not ready to receive traffic
    NotReady,
}

/// Events that can occur for service instances
#[derive(Debug, Clone)]
pub enum InstanceEvent {
    /// A new instance has been added and is ready for traffic
    Added(Instance),
    /// An instance has been removed or is no longer ready for traffic
    Removed(Instance),
}

/// Stream of instance events
pub type InstanceEventStream = Pin<Box<dyn Stream<Item = InstanceEvent> + Send>>;

/// Represents a discovered service instance
#[derive(Debug, Clone)]
pub struct Instance {
    /// Unique identifier for the instance
    pub id: String,
    /// Network address of the instance (e.g., "10.1.2.3:8080")
    pub address: String,
    /// Namespace the instance belongs to
    pub namespace: String,
    /// Component type of the instance
    pub component: String,
}

impl Instance {
    /// Create a new Instance
    pub fn new(id: String, address: String, namespace: String, component: String) -> Self {
        Self {
            id,
            address,
            namespace,
            component,
        }
    }

    /// Get metadata associated with this instance
    /// Note: Implementations should make an HTTP request to /metadata endpoint
    /// This is a placeholder that returns empty metadata for now
    pub async fn metadata(&self) -> Result<HashMap<String, serde_json::Value>> {
        // TODO: Implement HTTP client to fetch metadata from /metadata endpoint
        // This should be implemented by the specific backend (ETCD, Kubernetes, etc.)
        // For now, return empty metadata
        tracing::warn!(
            "metadata() called for instance {} but not implemented - returning empty metadata",
            self.id
        );
        Ok(HashMap::new())
    }
}

/// Handle for managing a registered service instance
#[async_trait]
pub trait InstanceHandle: Send + Sync {
    /// Get the unique identifier for this instance
    fn instance_id(&self) -> &str;

    /// Set metadata associated with this instance
    async fn set_metadata(&self, metadata: HashMap<String, serde_json::Value>) -> Result<()>;

    /// Mark the instance as ready or not ready for traffic
    async fn set_ready(&self, status: InstanceStatus) -> Result<()>;
}

/// Service Discovery trait for client-side discovery operations
#[async_trait]
pub trait ServiceDiscovery: Send + Sync {
    /// List all instances that match the given namespace and component
    /// Returns a list of Instance objects that are currently ready for traffic
    async fn list_instances(&self, namespace: &str, component: &str) -> Result<Vec<Instance>>;

    /// Watch for events for instances that match (namespace, component)
    /// Returns a stream of events (InstanceAddedEvent, InstanceRemovedEvent)
    async fn watch(&self, namespace: &str, component: &str) -> Result<InstanceEventStream>;
}

/// Service Registry trait for server-side instance registration
#[async_trait]
pub trait ServiceRegistry: Send + Sync {
    /// Register a new instance of the given namespace and component
    /// Returns an InstanceHandle that can be used to manage the instance
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Arc<dyn InstanceHandle>>;
}

/// Combined trait for both service discovery and registry operations
pub trait ServiceDiscoveryRegistry: ServiceDiscovery + ServiceRegistry {}

/// Blanket implementation for types that implement both ServiceDiscovery and ServiceRegistry
impl<T> ServiceDiscoveryRegistry for T where T: ServiceDiscovery + ServiceRegistry {}

// Re-export commonly used types
pub use InstanceEvent::{Added as InstanceAddedEvent, Removed as InstanceRemovedEvent};

// Implementations
pub mod etcd;
pub mod kubernetes;
