// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish()
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
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String, String, u16)> {
    // Extract system port from EndpointSlice ports
    let system_port = match extract_system_port(slice) {
        Some(port) => port,
        None => {
            let slice_name = slice.metadata.name.as_deref().unwrap_or("unknown");
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

        // Get first IP only (avoid duplicate instance IDs)
        if let Some(ip) = endpoint.addresses.first() {
            result.push((instance_id, pod_name.to_string(), ip.clone(), system_port));
        }
    }

    result
}

/// Pod information extracted from environment
#[derive(Debug, Clone)]
pub(super) struct PodInfo {
    pub pod_name: String,
    pub pod_namespace: String,
    pub system_port: u16,
}

impl PodInfo {
    /// Discover pod information from environment variables
    pub fn from_env() -> Result<Self> {
        let pod_name = std::env::var("POD_NAME")
            .map_err(|_| anyhow::anyhow!("POD_NAME environment variable not set"))?;

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
            system_port,
        })
    }
}
