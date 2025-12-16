// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Annotation key for localhost test mode
/// When this annotation is set to "true" on an EndpointSlice, the discovery daemon
/// will fetch metadata from localhost instead of the pod IP. This enables local
/// development against a real Kubernetes cluster.
///
/// Usage in EndpointSlice:
/// ```yaml
/// metadata:
///   annotations:
///     nvidia.com/dynamo-discovery-localhost: "true"
/// ```
pub const LOCALHOST_TEST_ANNOTATION: &str = "nvidia.com/dynamo-discovery-localhost";

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish()
}

/// Check if an EndpointSlice has localhost test mode enabled
fn is_localhost_test_mode(slice: &EndpointSlice) -> bool {
    slice
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(LOCALHOST_TEST_ANNOTATION))
        .map(|value| value == "true")
        .unwrap_or(false)
}

/// Extract the system port from an EndpointSlice's ports
/// Looks for a port with name "system", returns None if not found
fn extract_system_port(slice: &EndpointSlice) -> Option<u16> {
    slice.ports.as_ref().and_then(|ports| {
        ports
            .iter()
            .find(|p| p.name.as_deref() == Some("system"))
            .and_then(|p| p.port.map(|port| port as u16))
    })
}

/// Extract endpoint information from an EndpointSlice
/// Returns (instance_id, pod_name, pod_ip, system_port) tuples for ready endpoints
///
/// When the EndpointSlice has the localhost test annotation (`nvidia.com/dynamo-discovery-localhost=true`),
/// the returned IP will be `127.0.0.1` instead of the actual endpoint address. This enables
/// local development where the developer runs workers locally while using a real Kubernetes
/// cluster for discovery coordination.
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String, String, u16)> {
    // Check if localhost test mode is enabled for this slice
    let localhost_mode = is_localhost_test_mode(slice);
    let slice_name = slice.metadata.name.as_deref().unwrap_or("unknown");

    if localhost_mode {
        tracing::debug!(
            "EndpointSlice '{}' has localhost test mode enabled - will use 127.0.0.1 for metadata fetch",
            slice_name
        );
    }

    // Extract system port from EndpointSlice ports
    let system_port = match extract_system_port(slice) {
        Some(port) => port,
        None => {
            tracing::warn!(
                "EndpointSlice '{}' did not have a system port defined",
                slice_name
            );
            return Vec::new();
        }
    };

    let mut result = Vec::new();
    let endpoints = &slice.endpoints;

    for endpoint in endpoints {
        // Check if endpoint is ready
        let is_ready = endpoint
            .conditions
            .as_ref()
            .and_then(|c| c.ready)
            .unwrap_or(false);

        if !is_ready {
            continue;
        }

        // Get pod name from targetRef
        let pod_name = match endpoint.target_ref.as_ref() {
            Some(target_ref) => target_ref.name.as_deref().unwrap_or(""),
            None => continue,
        };

        if pod_name.is_empty() {
            continue;
        }

        let instance_id = hash_pod_name(pod_name);

        // In localhost test mode, use 127.0.0.1 instead of the actual pod IP
        // This allows local workers to be discovered via a real K8s cluster
        let ip = if localhost_mode {
            "127.0.0.1".to_string()
        } else {
            match endpoint.addresses.first() {
                Some(addr) => addr.clone(),
                None => continue,
            }
        };

        result.push((instance_id, pod_name.to_string(), ip, system_port));
    }

    result
}

/// Pod information extracted from environment
#[derive(Debug, Clone)]
pub(super) struct PodInfo {
    pub pod_name: String,
    pub pod_namespace: String,
    pub pod_uid: String,
    pub system_port: u16,
}

impl PodInfo {
    /// Discover pod information from environment variables
    ///
    /// Required environment variables:
    /// - `POD_NAME`: Name of the pod (required)
    /// - `POD_UID`: UID of the pod (required for CR owner reference)
    /// - `POD_NAMESPACE`: Namespace of the pod (defaults to "default")
    pub fn from_env() -> Result<Self> {
        let pod_name = std::env::var("POD_NAME")
            .map_err(|_| anyhow::anyhow!("POD_NAME environment variable not set"))?;

        let pod_uid = std::env::var("POD_UID")
            .map_err(|_| anyhow::anyhow!("POD_UID environment variable not set"))?;

        let pod_namespace = std::env::var("POD_NAMESPACE").unwrap_or_else(|_| {
            tracing::warn!("POD_NAMESPACE not set, defaulting to 'default'");
            "default".to_string()
        });

        // Read system server port from config
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        let system_port = config.system_port as u16;

        Ok(Self {
            pod_name,
            pod_namespace,
            pod_uid,
            system_port,
        })
    }
}
