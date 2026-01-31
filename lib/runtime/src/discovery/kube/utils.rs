// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    // Clear top 11 bits to ensure it can be safely rounded to IEEE-754 f64
    const INSTANCE_ID_MASK: u64 = 0x001F_FFFF_FFFF_FFFFu64;
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish() & INSTANCE_ID_MASK
}

/// Extract endpoint information from an EndpointSlice
/// Returns (instance_id, pod_name) tuples for ready endpoints
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String)> {
    let mut result = Vec::new();

    for endpoint in &slice.endpoints {
        let is_ready = endpoint
            .conditions
            .as_ref()
            .and_then(|c| c.ready)
            .unwrap_or(false);

        if !is_ready {
            continue;
        }

        let pod_name = match endpoint.target_ref.as_ref() {
            Some(target_ref) => target_ref.name.as_deref().unwrap_or(""),
            None => continue,
        };

        if pod_name.is_empty() {
            continue;
        }

        let instance_id = hash_pod_name(pod_name);

        result.push((instance_id, pod_name.to_string()));
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

/// Default path for Kubernetes Downward API volume mount
const DEFAULT_PODINFO_PATH: &str = "/etc/podinfo";

impl PodInfo {
    /// Read a value from a Downward API file, falling back to environment variable
    fn read_from_file_or_env(file_path: &Path, env_var: &str) -> Option<String> {
        // First try reading from file (Downward API volume mount)
        // This is preferred after CRIU restore since env vars contain stale values
        if let Ok(content) = fs::read_to_string(file_path) {
            let value = content.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }

        // Fall back to environment variable
        std::env::var(env_var).ok()
    }

    /// Discover pod information from Kubernetes Downward API volume mounts or environment variables
    ///
    /// This function first attempts to read pod identity from Downward API volume mounts
    /// at /etc/podinfo/{pod_name, pod_uid, pod_namespace}. This is critical for CRIU
    /// checkpoint/restore scenarios where environment variables contain stale values
    /// from the checkpoint source pod.
    ///
    /// If the Downward API files are not available, falls back to environment variables:
    /// - `POD_NAME`: Name of the pod (required)
    /// - `POD_UID`: UID of the pod (required for CR owner reference)
    /// - `POD_NAMESPACE`: Namespace of the pod (defaults to "default")
    pub fn from_env() -> Result<Self> {
        let podinfo_path = Path::new(DEFAULT_PODINFO_PATH);

        let pod_name = Self::read_from_file_or_env(&podinfo_path.join("pod_name"), "POD_NAME")
            .ok_or_else(|| anyhow::anyhow!("POD_NAME not available from file or environment"))?;

        let pod_uid = Self::read_from_file_or_env(&podinfo_path.join("pod_uid"), "POD_UID")
            .ok_or_else(|| anyhow::anyhow!("POD_UID not available from file or environment"))?;

        let pod_namespace =
            Self::read_from_file_or_env(&podinfo_path.join("pod_namespace"), "POD_NAMESPACE")
                .unwrap_or_else(|| {
                    tracing::warn!("POD_NAMESPACE not set, defaulting to 'default'");
                    "default".to_string()
                });

        // Log where we got the pod info from for debugging
        if podinfo_path.join("pod_name").exists() {
            tracing::info!(
                "Pod identity loaded from Downward API volume mount at {}",
                DEFAULT_PODINFO_PATH
            );
        } else {
            tracing::info!("Pod identity loaded from environment variables");
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_json_serialization_roundtrip() {
        // Verify that JSON serialization/deserialization preserves exact values
        let pod_names = [
            "worker-0",
            "worker-99999",
            "deployment-with-hash-suffix-a1b2c3d4e5f6",
            "fake-name-1-0-worker-nrdfv",
        ];

        for pod_name in &pod_names {
            let original_hash = hash_pod_name(pod_name);
            let json = serde_json::to_string(&original_hash).unwrap();
            let deserialized_hash: u64 = serde_json::from_str(&json).unwrap();

            assert_eq!(
                original_hash, deserialized_hash,
                "JSON roundtrip changed hash value for pod_name={:?}: {} -> {} (json: {})",
                pod_name, original_hash, deserialized_hash, json
            );
        }
    }

    #[test]
    fn test_hash_in_struct_serialization() {
        // Test serialization when the hash is embedded in a struct
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct WorkerInfo {
            instance_id: u64,
            name: String,
        }

        let pod_name = "fake-name-1-0-worker-nrdfv";
        let info = WorkerInfo {
            instance_id: hash_pod_name(pod_name),
            name: pod_name.to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: WorkerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info, deserialized);
    }
}
