// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes implementation of the ServiceDiscovery interface using EndpointSlices
//!
//! This implementation follows the EndpointSlice discovery mechanism as described in the design doc:
//! 1. Pods have namespace and component labels that match the method args
//! 2. A Kubernetes Service selects pods based on namespace and component labels
//! 3. EndpointSlices are automatically managed by Kubernetes and track ready pod endpoints
//! 4. We watch EndpointSlices to get notifications when instances become ready/unavailable

use super::{
    Instance, InstanceEvent, InstanceEventStream, InstanceHandle, InstanceStatus, ServiceDiscovery,
    ServiceRegistry,
};
use crate::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tokio::sync::{RwLock, mpsc};

// TODO: Add kube crate dependency to Cargo.toml
// use kube::{Api, Client, ResourceExt, api::{ListParams, WatchEvent, WatchParams}};
// use k8s_openapi::api::discovery::v1::EndpointSlice;
// use k8s_openapi::api::core::v1::{Pod, Service};

/// Kubernetes-specific metadata stored in pods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesInstanceMetadata {
    /// Instance metadata as key-value pairs
    pub metadata: HashMap<String, serde_json::Value>,
    /// Instance readiness status
    pub status: InstanceStatus,
}

/// Kubernetes implementation of InstanceHandle
/// This represents a pod that has been registered for service discovery
pub struct KubernetesInstanceHandle {
    /// Pod name (used as instance ID)
    pod_name: String,
    /// Namespace the pod is in
    namespace: String,
    /// Component label value
    component: String,
    /// Current metadata stored in memory
    metadata: Arc<RwLock<KubernetesInstanceMetadata>>,
    /// HTTP server for exposing /metadata endpoint
    metadata_server: Option<tokio::task::JoinHandle<()>>,
    /// Current readiness status
    readiness_status: Arc<RwLock<InstanceStatus>>,
}

impl KubernetesInstanceHandle {
    pub fn new(pod_name: String, namespace: String, component: String) -> Self {
        let metadata = KubernetesInstanceMetadata {
            metadata: HashMap::new(),
            status: InstanceStatus::NotReady,
        };

        Self {
            pod_name,
            namespace,
            component,
            metadata: Arc::new(RwLock::new(metadata)),
            metadata_server: None,
            readiness_status: Arc::new(RwLock::new(InstanceStatus::NotReady)),
        }
    }

    /// Start HTTP server to expose metadata at /metadata endpoint
    async fn start_metadata_server(&mut self, port: u16) -> Result<()> {
        let metadata = self.metadata.clone();
        let readiness_status = self.readiness_status.clone();

        let app = axum::Router::new()
            .route(
                "/metadata",
                axum::routing::get(move || async move {
                    let metadata = metadata.read().await;
                    axum::Json(metadata.metadata.clone())
                }),
            )
            .route(
                "/health",
                axum::routing::get({
                    let readiness_status = readiness_status.clone();
                    move || async move {
                        let status = readiness_status.read().await;
                        match *status {
                            InstanceStatus::Ready => axum::http::StatusCode::OK,
                            InstanceStatus::NotReady => axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        }
                    }
                }),
            );

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
            .await
            .map_err(|e| crate::error!("Failed to bind to port {}: {}", port, e))?;

        let server_handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("Metadata server error: {}", e);
            }
        });

        self.metadata_server = Some(server_handle);
        Ok(())
    }
}

// HTTP handlers are now inline in the route definitions

#[async_trait]
impl InstanceHandle for KubernetesInstanceHandle {
    fn instance_id(&self) -> &str {
        &self.pod_name
    }

    async fn set_metadata(&self, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        let mut current_metadata = self.metadata.write().await;
        current_metadata.metadata = metadata;
        Ok(())
    }

    async fn set_ready(&self, status: InstanceStatus) -> Result<()> {
        {
            let mut current_metadata = self.metadata.write().await;
            current_metadata.status = status.clone();
        }
        {
            let mut readiness_status = self.readiness_status.write().await;
            *readiness_status = status;
        }
        Ok(())
    }
}

/// Kubernetes implementation of ServiceDiscovery and ServiceRegistry using EndpointSlices
pub struct KubernetesServiceDiscovery {
    /// Kubernetes client (placeholder - requires kube crate)
    // client: Client,
    /// Namespace to operate in
    namespace: String,
    /// Label prefix for dynamo components
    label_prefix: String,
}

impl KubernetesServiceDiscovery {
    /// Create a new Kubernetes service discovery client
    pub async fn new(namespace: String) -> Result<Self> {
        // TODO: Initialize Kubernetes client when kube crate is available
        // let client = Client::try_default().await
        //     .map_err(|e| crate::error!("Failed to create Kubernetes client: {}", e))?;

        Ok(Self {
            // client,
            namespace,
            label_prefix: "nvidia.com/dynamo".to_string(),
        })
    }

    /// Create a new Kubernetes service discovery client with custom label prefix
    pub async fn with_label_prefix(namespace: String, label_prefix: String) -> Result<Self> {
        // TODO: Initialize Kubernetes client when kube crate is available
        // let client = Client::try_default().await
        //     .map_err(|e| crate::error!("Failed to create Kubernetes client: {}", e))?;

        Ok(Self {
            // client,
            namespace,
            label_prefix,
        })
    }

    /// Get the service name for a given namespace and component
    fn get_service_name(&self, namespace: &str, component: &str) -> String {
        format!("{}-{}-service", namespace, component)
    }

    /// Get label selectors for namespace and component
    fn get_label_selectors(&self, namespace: &str, component: &str) -> HashMap<String, String> {
        let mut selectors = HashMap::new();
        selectors.insert(
            format!("{}-namespace", self.label_prefix),
            namespace.to_string(),
        );
        selectors.insert(
            format!("{}-component", self.label_prefix),
            component.to_string(),
        );
        selectors
    }

    /// Parse instance from EndpointSlice endpoint
    fn parse_instance_from_endpoint(
        &self,
        _endpoint: &serde_json::Value, // Placeholder for EndpointSlice endpoint
        _namespace: &str,
        _component: &str,
    ) -> Result<Option<Instance>> {
        // TODO: Implement when kube crate is available
        // This should:
        // 1. Check if endpoint.conditions.ready is true
        // 2. Extract pod name from endpoint.targetRef.name
        // 3. Get address from endpoint.addresses[0]
        // 4. Return Instance if ready, None otherwise

        tracing::warn!("parse_instance_from_endpoint not implemented - requires kube crate");
        Ok(None)
    }

    /// Get current pod name from environment or pod metadata
    async fn get_current_pod_name(&self) -> Result<String> {
        // Try to get pod name from environment variable (set by Kubernetes)
        if let Ok(pod_name) = std::env::var("HOSTNAME") {
            return Ok(pod_name);
        }

        // Fallback: try to read from pod metadata file
        if let Ok(pod_name) = tokio::fs::read_to_string("/etc/podinfo/name").await {
            return Ok(pod_name.trim().to_string());
        }

        // Last resort: generate a unique name
        Ok(format!("pod-{}", &uuid::Uuid::new_v4().to_string()[..8]))
    }

    /// Verify that the current pod has the required labels
    async fn verify_pod_labels(&self, _namespace: &str, _component: &str) -> Result<()> {
        // TODO: Implement when kube crate is available
        // This should:
        // 1. Get current pod using pod name
        // 2. Verify it has the required namespace and component labels
        // 3. Return error if labels don't match

        tracing::warn!(
            "verify_pod_labels not implemented - assuming pod has correct labels for {}/{}",
            _namespace,
            _component
        );
        Ok(())
    }

    /// Ensure service exists for the given namespace and component
    async fn ensure_service_exists(&self, _namespace: &str, _component: &str) -> Result<()> {
        // TODO: Implement when kube crate is available
        // This should:
        // 1. Check if service exists
        // 2. Create service if it doesn't exist with proper label selectors
        // 3. Service should select pods with namespace and component labels

        tracing::warn!(
            "ensure_service_exists not implemented - assuming service exists for {}/{}",
            _namespace,
            _component
        );
        Ok(())
    }
}

#[async_trait]
impl ServiceDiscovery for KubernetesServiceDiscovery {
    async fn list_instances(&self, _namespace: &str, _component: &str) -> Result<Vec<Instance>> {
        // TODO: Implement when kube crate is available
        // This should:
        // 1. Get service name for namespace/component
        // 2. Query EndpointSlices for the service using label selector
        // 3. Parse endpoints that are ready
        // 4. Return list of Instance objects

        tracing::warn!(
            "list_instances not implemented for Kubernetes - returning empty list for {}/{}",
            _namespace,
            _component
        );
        Ok(Vec::new())
    }

    async fn watch(&self, _namespace: &str, _component: &str) -> Result<InstanceEventStream> {
        // TODO: Implement when kube crate is available
        // This should:
        // 1. Set up kubectl watch for EndpointSlices with service label selector
        // 2. Parse watch events and convert to InstanceEvent
        // 3. Return stream of events

        tracing::warn!(
            "watch not implemented for Kubernetes - returning empty stream for {}/{}",
            _namespace,
            _component
        );

        let stream = stream! {
            // Empty stream for now - yield nothing and then end
            // In a real implementation, this would watch EndpointSlices
            if false {
                yield InstanceEvent::Added(Instance::new("".to_string(), "".to_string(), "".to_string(), "".to_string()));
            }
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl ServiceRegistry for KubernetesServiceDiscovery {
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Arc<dyn InstanceHandle>> {
        // Verify pod has correct labels
        self.verify_pod_labels(namespace, component).await?;

        // Ensure service exists
        self.ensure_service_exists(namespace, component).await?;

        // Get current pod name
        let pod_name = self.get_current_pod_name().await?;

        // Create instance handle
        let mut handle =
            KubernetesInstanceHandle::new(pod_name, self.namespace.clone(), component.to_string());

        // Start metadata server on port 8080 (should match readiness probe)
        handle.start_metadata_server(8080).await?;

        Ok(Arc::new(handle))
    }
}

/// Helper function to create label selector string from map
fn create_label_selector(labels: &HashMap<String, String>) -> String {
    labels
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kubernetes_service_discovery_creation() -> Result<()> {
        let discovery = KubernetesServiceDiscovery::new("test-namespace".to_string()).await?;
        assert_eq!(discovery.namespace, "test-namespace");
        assert_eq!(discovery.label_prefix, "nvidia.com/dynamo");
        Ok(())
    }

    #[tokio::test]
    async fn test_service_name_generation() -> Result<()> {
        let discovery = KubernetesServiceDiscovery::new("test-ns".to_string()).await?;
        let service_name = discovery.get_service_name("dynamo", "decode");
        assert_eq!(service_name, "dynamo-decode-service");
        Ok(())
    }

    #[tokio::test]
    async fn test_label_selectors() -> Result<()> {
        let discovery = KubernetesServiceDiscovery::new("test-ns".to_string()).await?;
        let selectors = discovery.get_label_selectors("dynamo", "decode");

        assert_eq!(
            selectors.get("nvidia.com/dynamo-namespace"),
            Some(&"dynamo".to_string())
        );
        assert_eq!(
            selectors.get("nvidia.com/dynamo-component"),
            Some(&"decode".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_instance_handle_creation() -> Result<()> {
        let handle = KubernetesInstanceHandle::new(
            "test-pod-123".to_string(),
            "test-namespace".to_string(),
            "test-component".to_string(),
        );

        assert_eq!(handle.instance_id(), "test-pod-123");

        // Test setting metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "key1".to_string(),
            serde_json::Value::String("value1".to_string()),
        );
        handle.set_metadata(metadata.clone()).await?;

        // Test setting ready status
        handle.set_ready(InstanceStatus::Ready).await?;

        Ok(())
    }
}
