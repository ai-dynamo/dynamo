// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use crate::config::environment_names::kube_discovery;

/// Mask to clear top 11 bits, ensuring the instance ID can be safely
/// round-tripped through IEEE-754 f64 (JSON number).
const INSTANCE_ID_MASK: u64 = 0x001F_FFFF_FFFF_FFFFu64;

/// The operator's standard name for single-container pods.
/// Containers with this name use pod-level identity even in container mode,
/// ensuring backward compatibility with pod-mode CRs.
const MAIN_CONTAINER_NAME: &str = "main";

/// Kubernetes discovery granularity determines whether identity is per-pod or per-container.
///
/// - `Pod`: one identity per pod (backward-compatible default). Uses EndpointSlice
///   reflector and `hash(pod_name)` for instance ID.
/// - `Container`: one identity per container. Uses Pod reflector. Containers named
///   "main" use pod-level identity for cross-compatibility. Other containers
///   (e.g., "engine-0") get `hash("{pod}-{container}")` identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KubeDiscoveryGranularity {
    Pod,
    Container,
}

impl KubeDiscoveryGranularity {
    /// Read granularity from `DYN_KUBE_DISCOVERY_GRANULARITY` env var.
    /// Defaults to `Pod` if unset or unrecognized.
    pub fn from_env() -> Self {
        match std::env::var(kube_discovery::DYN_KUBE_DISCOVERY_GRANULARITY).as_deref() {
            Ok("container") => Self::Container,
            _ => Self::Pod,
        }
    }

    /// CR name for this discovery participant.
    ///
    /// - Pod mode: returns `pod_name`.
    /// - Container mode with `container_name == "main"`: returns `pod_name`
    ///   (backward compat — "main" is the operator's default single-container name).
    /// - Container mode with any other name: returns `"{pod_name}-{container_name}"`.
    pub fn cr_name(&self, pod_name: &str, container_name: &str) -> String {
        match self {
            Self::Pod => pod_name.to_string(),
            Self::Container if container_name == MAIN_CONTAINER_NAME => pod_name.to_string(),
            Self::Container => format!("{}-{}", pod_name, container_name),
        }
    }

    /// Deterministic instance ID. Hashes the `cr_name` output so the two
    /// are always derived from the same key.
    pub fn instance_id(&self, pod_name: &str, container_name: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cr_name(pod_name, container_name).hash(&mut hasher);
        hasher.finish() & INSTANCE_ID_MASK
    }
}

/// Hash a pod name to get a consistent instance ID (pod-level granularity).
///
/// This is the backward-compatible function used by external consumers
/// (e.g., C bindings / EPP).
pub fn hash_pod_name(pod_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish() & INSTANCE_ID_MASK
}

/// Extract endpoint information from an EndpointSlice (pod-level granularity).
///
/// Returns `(instance_id, pod_name)` tuples for ready endpoints.
/// Used by the daemon in `Pod` granularity mode.
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

/// Extract ready containers from a Pod (container-level granularity).
///
/// Returns `(instance_id, cr_name)` tuples for each container whose
/// `containerStatuses[].ready` is `true`. Containers named "main" produce
/// pod-level identity for backward compatibility. Init containers and sidecars
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

    let g = KubeDiscoveryGranularity::Container;
    container_statuses
        .iter()
        .filter(|cs| cs.ready)
        .map(|cs| {
            let id = g.instance_id(pod_name, &cs.name);
            let name = g.cr_name(pod_name, &cs.name);
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
    pub granularity: KubeDiscoveryGranularity,
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
    /// - `CONTAINER_NAME`: Name of this container (required when granularity is Container)
    /// - `DYN_KUBE_DISCOVERY_GRANULARITY`: "container" or "pod" (default: "pod")
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

        let granularity = KubeDiscoveryGranularity::from_env();

        let container_name = match granularity {
            KubeDiscoveryGranularity::Container => {
                std::env::var("CONTAINER_NAME").map_err(|_| {
                    anyhow::anyhow!(
                        "CONTAINER_NAME is required when DYN_KUBE_DISCOVERY_GRANULARITY=container"
                    )
                })?
            }
            KubeDiscoveryGranularity::Pod => {
                std::env::var("CONTAINER_NAME").unwrap_or_else(|_| "main".to_string())
            }
        };

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
            granularity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_mode_instance_id_matches_hash_pod_name() {
        let pod_names = ["worker-0", "worker-99999", "fake-name-1-0-worker-nrdfv"];
        for pod_name in &pod_names {
            assert_eq!(
                KubeDiscoveryGranularity::Pod.instance_id(pod_name, "anything"),
                hash_pod_name(pod_name),
                "Pod mode instance_id must equal hash_pod_name for '{}'",
                pod_name
            );
        }
    }

    #[test]
    fn test_pod_mode_cr_name_is_pod_name() {
        assert_eq!(
            KubeDiscoveryGranularity::Pod.cr_name("worker-0", "engine-0"),
            "worker-0"
        );
    }

    #[test]
    fn test_container_mode_main_uses_pod_naming() {
        let g = KubeDiscoveryGranularity::Container;
        // "main" container should use pod-level naming
        assert_eq!(g.cr_name("worker-0", "main"), "worker-0");
        assert_eq!(g.instance_id("worker-0", "main"), hash_pod_name("worker-0"));
    }

    #[test]
    fn test_container_mode_non_main_uses_container_naming() {
        let g = KubeDiscoveryGranularity::Container;
        assert_eq!(g.cr_name("worker-0", "engine-0"), "worker-0-engine-0");
        // Should differ from pod-level hash
        assert_ne!(
            g.instance_id("worker-0", "engine-0"),
            hash_pod_name("worker-0")
        );
    }

    #[test]
    fn test_container_mode_instance_id_differs_by_container() {
        let g = KubeDiscoveryGranularity::Container;
        let id0 = g.instance_id("worker-0", "engine-0");
        let id1 = g.instance_id("worker-0", "engine-1");
        assert_ne!(id0, id1);
    }

    #[test]
    fn test_container_mode_instance_id_differs_by_pod() {
        let g = KubeDiscoveryGranularity::Container;
        let id0 = g.instance_id("worker-0", "engine-0");
        let id1 = g.instance_id("worker-1", "engine-0");
        assert_ne!(id0, id1);
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
            let original = hash_pod_name(pod_name);
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
            instance_id: hash_pod_name(pod_name),
            name: pod_name.to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: WorkerInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info, deserialized);
    }
}
