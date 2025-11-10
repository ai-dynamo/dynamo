// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::hash::{Hash, Hasher};

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish()
}

/// Parse port number from pod name (format: pod-name-<port>)
/// Returns Some(port) if successfully parsed, None otherwise
pub(super) fn parse_port_from_pod_name(pod_name: &str) -> Option<u16> {
    pod_name.rsplit('-')
        .next()
        .and_then(|last| last.parse::<u16>().ok())
}

/// Extract endpoint information from an EndpointSlice
/// Returns (instance_id, pod_name, pod_ip) tuples for ready endpoints
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String, String)> {
    let mut result = Vec::new();
    
    let endpoints = &slice.endpoints;
    
    for endpoint in endpoints {
        // Check if endpoint is ready
        let is_ready = endpoint.conditions.as_ref()
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
            result.push((instance_id, pod_name.to_string(), ip.clone()));
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
            .map_err(|_| crate::error!("POD_NAME environment variable not set"))?;
        
        let pod_namespace = std::env::var("POD_NAMESPACE")
            .unwrap_or_else(|_| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let pod_name = "test-pod-123";
        let hash1 = hash_pod_name(pod_name);
        let hash2 = hash_pod_name(pod_name);
        assert_eq!(hash1, hash2, "Hash should be consistent");
    }

    #[test]
    fn test_hash_uniqueness() {
        let hash1 = hash_pod_name("pod-1");
        let hash2 = hash_pod_name("pod-2");
        assert_ne!(hash1, hash2, "Different pods should have different hashes");
    }

    #[test]
    fn test_parse_port_from_pod_name() {
        // Valid port numbers
        assert_eq!(
            parse_port_from_pod_name("dynamo-test-worker-8080"),
            Some(8080)
        );
        assert_eq!(
            parse_port_from_pod_name("my-service-9000"),
            Some(9000)
        );
        assert_eq!(
            parse_port_from_pod_name("test-3000"),
            Some(3000)
        );
        assert_eq!(
            parse_port_from_pod_name("a-b-c-80"),
            Some(80)
        );
        
        // Invalid - no port number at end
        assert_eq!(
            parse_port_from_pod_name("dynamo-test-worker"),
            None
        );
        assert_eq!(
            parse_port_from_pod_name("8080-worker"),
            None
        );
        assert_eq!(
            parse_port_from_pod_name("worker-abc"),
            None
        );
        assert_eq!(
            parse_port_from_pod_name(""),
            None
        );
    }
}

