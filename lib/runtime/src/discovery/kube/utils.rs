// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::core::v1::Pod;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Mask to clear top 11 bits, ensuring the instance ID can be safely
/// round-tripped through IEEE-754 f64 (JSON number).
const INSTANCE_ID_MASK: u64 = 0x001F_FFFF_FFFF_FFFFu64;

/// Build the CR name for a given pod and container.
///
/// The CR name uniquely identifies a container within a pod:
/// `"{pod_name}-{container_name}"`.
pub fn cr_name(pod_name: &str, container_name: &str) -> String {
    format!("{}-{}", pod_name, container_name)
}

/// Compute a deterministic instance ID from pod name and container name.
///
/// The ID is derived from `"{pod_name}:{container_name}"` and masked to
/// fit within 53 bits for safe JSON serialization.
pub fn instance_id(pod_name: &str, container_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    format!("{}:{}", pod_name, container_name).hash(&mut hasher);
    hasher.finish() & INSTANCE_ID_MASK
}

/// Extract ready containers from a Pod.
///
/// Returns `(instance_id, cr_name)` tuples for each container whose
/// `containerStatuses[].ready` is `true`. Init containers and sidecars
/// that never call `register()` are naturally excluded by the daemon's
/// CR correlation step (no matching CR → not included in snapshot).
pub(super) fn extract_ready_containers(pod: &Pod) -> Vec<(u64, String)> {
    let pod_name = match pod.metadata.name.as_deref() {
        Some(name) => name,
        None => return vec![],
    };

    let container_statuses = match pod
        .status
        .as_ref()
        .and_then(|s| s.container_statuses.as_ref())
    {
        Some(statuses) => statuses,
        None => return vec![],
    };

    container_statuses
        .iter()
        .filter(|cs| cs.ready)
        .map(|cs| {
            let id = instance_id(pod_name, &cs.name);
            let name = cr_name(pod_name, &cs.name);
            (id, name)
        })
        .collect()
}

/// Pod information extracted from environment
#[derive(Debug, Clone)]
pub(super) struct PodInfo {
    pub pod_name: String,
    pub container_name: String,
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
    /// - `CONTAINER_NAME`: Name of this container (required)
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

        let container_name = std::env::var("CONTAINER_NAME")
            .map_err(|_| anyhow::anyhow!("CONTAINER_NAME environment variable is required"))?;

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
            container_name,
            pod_namespace,
            pod_uid,
            system_port,
        })
    }

    /// CR name for this container: `"{pod_name}-{container_name}"`.
    pub fn cr_name(&self) -> String {
        cr_name(&self.pod_name, &self.container_name)
    }

    /// Instance ID for this container.
    pub fn instance_id(&self) -> u64 {
        instance_id(&self.pod_name, &self.container_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_id_deterministic() {
        let id1 = instance_id("worker-0", "engine-0");
        let id2 = instance_id("worker-0", "engine-0");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_instance_id_differs_by_container() {
        let id0 = instance_id("worker-0", "engine-0");
        let id1 = instance_id("worker-0", "engine-1");
        assert_ne!(id0, id1);
    }

    #[test]
    fn test_instance_id_differs_by_pod() {
        let id0 = instance_id("worker-0", "main");
        let id1 = instance_id("worker-1", "main");
        assert_ne!(id0, id1);
    }

    #[test]
    fn test_cr_name_format() {
        assert_eq!(cr_name("worker-0", "engine-0"), "worker-0-engine-0");
        assert_eq!(cr_name("my-pod", "main"), "my-pod-main");
    }

    #[test]
    fn test_instance_id_json_roundtrip() {
        let pod_names = [
            "worker-0",
            "worker-99999",
            "deployment-with-hash-suffix-a1b2c3d4e5f6",
            "fake-name-1-0-worker-nrdfv",
        ];

        for pod_name in &pod_names {
            let original = instance_id(pod_name, "main");
            let json = serde_json::to_string(&original).unwrap();
            let deserialized: u64 = serde_json::from_str(&json).unwrap();

            assert_eq!(
                original, deserialized,
                "JSON roundtrip changed value for pod_name={:?}: {} -> {} (json: {})",
                pod_name, original, deserialized, json
            );
        }
    }

    #[test]
    fn test_instance_id_in_struct_serialization() {
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct WorkerInfo {
            instance_id: u64,
            name: String,
        }

        let pod_name = "fake-name-1-0-worker-nrdfv";
        let info = WorkerInfo {
            instance_id: instance_id(pod_name, "main"),
            name: pod_name.to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: WorkerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info, deserialized);
    }
}
