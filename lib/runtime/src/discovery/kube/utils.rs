// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use crate::config::environment_names::discovery;

const INSTANCE_ID_MASK: u64 = 0x001F_FFFF_FFFF_FFFFu64;
const MAIN_CONTAINER_NAME: &str = "main";

/// Kube discovery mode.
///
/// - `Pod`: default. One identity per pod.
/// - `Container`: each container independently registers with the discovery plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KubeDiscoveryMode {
    Pod,
    Container,
}

impl KubeDiscoveryMode {
    pub fn from_env() -> Result<Self> {
        match std::env::var(discovery::DYN_KUBE_DISCOVERY_MODE).as_deref() {
            Ok("container") => Ok(Self::Container),
            Ok("pod") | Err(_) => Ok(Self::Pod),
            Ok(other) => anyhow::bail!(
                "Invalid DYN_KUBE_DISCOVERY_MODE value '{}'. Valid values: 'pod', 'container'",
                other
            ),
        }
    }
}

/// A resolved discovery target identifying either a pod or a specific container within a pod.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KubeDiscoveryTarget {
    Pod(String),
    Container(String, String),
}

impl KubeDiscoveryTarget {
    /// CR name for this target, used as the DynamoWorkerMetadata resource name.
    pub fn cr_name(&self) -> String {
        match self {
            Self::Pod(pod_name) => pod_name.clone(),
            Self::Container(pod_name, container_name) if container_name == MAIN_CONTAINER_NAME => {
                pod_name.clone()
            }
            Self::Container(pod_name, container_name) => {
                format!("{}-{}", pod_name, container_name)
            }
        }
    }

    /// Deterministic instance ID derived from cr_name.
    pub fn instance_id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cr_name().hash(&mut hasher);
        hasher.finish() & INSTANCE_ID_MASK
    }

    pub fn pod_name(&self) -> &str {
        match self {
            Self::Pod(pod_name) | Self::Container(pod_name, _) => pod_name,
        }
    }
}

/// Hash a pod name to get a consistent instance ID (pod-level).
///
/// Used by C bindings (EPP) for pod-level worker ID mapping.
pub fn hash_pod_name(pod_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish() & INSTANCE_ID_MASK
}

/// Extract (instance_id, pod_name) tuples from an EndpointSlice for ready endpoints.
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

        let target = KubeDiscoveryTarget::Pod(pod_name.to_string());
        result.push((target.instance_id(), target.cr_name()));
    }

    result
}

/// Extract (instance_id, cr_name) tuples from a Pod for each ready container.
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
            let target = KubeDiscoveryTarget::Container(pod_name.to_string(), cs.name.clone());
            (target.instance_id(), target.cr_name())
        })
        .collect()
}

/// Pod information extracted from environment.
#[derive(Debug, Clone)]
pub(super) struct PodInfo {
    pub pod_name: String,
    pub pod_namespace: String,
    pub pod_uid: String,
    pub system_port: u16,
    /// Kube discovery mode for this process, read from DYN_KUBE_DISCOVERY_MODE.
    pub mode: KubeDiscoveryMode,
    /// Discovery target for this process, derived from mode + pod/container identity.
    pub target: KubeDiscoveryTarget,
}

const DEFAULT_PODINFO_PATH: &str = "/etc/podinfo";

impl PodInfo {
    fn read_from_file_or_env(file_path: &Path, env_var: &str) -> Option<String> {
        if let Ok(content) = fs::read_to_string(file_path) {
            let value = content.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
        std::env::var(env_var).ok()
    }

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

        let mode = KubeDiscoveryMode::from_env()?;

        let target = match mode {
            KubeDiscoveryMode::Pod => KubeDiscoveryTarget::Pod(pod_name.clone()),
            KubeDiscoveryMode::Container => {
                let container_name = std::env::var("CONTAINER_NAME").map_err(|_| {
                    anyhow::anyhow!(
                        "CONTAINER_NAME is required when DYN_KUBE_DISCOVERY_MODE=container"
                    )
                })?;
                KubeDiscoveryTarget::Container(pod_name.clone(), container_name)
            }
        };

        if podinfo_path.join("pod_name").exists() {
            tracing::info!(
                "Pod identity loaded from Downward API volume mount at {}",
                DEFAULT_PODINFO_PATH
            );
        } else {
            tracing::info!("Pod identity loaded from environment variables");
        }

        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        let system_port = config.system_port as u16;

        Ok(Self {
            pod_name,
            pod_namespace,
            pod_uid,
            system_port,
            mode,
            target,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_mode_backward_compat() {
        // Pod mode must produce the same instance_id as hash_pod_name
        // so existing deployments see no identity change on upgrade.
        let target = KubeDiscoveryTarget::Pod("worker-0".into());
        assert_eq!(target.instance_id(), hash_pod_name("worker-0"));
        assert_eq!(target.cr_name(), "worker-0");
    }

    #[test]
    fn test_container_mode_main_uses_pod_identity() {
        // A container named "main" uses pod-level identity so that
        // container-mode frontends can discover pod-mode workers.
        let target = KubeDiscoveryTarget::Container("worker-0".into(), "main".into());
        assert_eq!(target.instance_id(), hash_pod_name("worker-0"));
        assert_eq!(target.cr_name(), "worker-0");
    }

    #[test]
    fn test_container_mode_engine_gets_unique_identity() {
        // Non-main containers get per-container identity so that
        // failover engine containers are independently discoverable.
        let e0 = KubeDiscoveryTarget::Container("worker-0".into(), "engine-0".into());
        let e1 = KubeDiscoveryTarget::Container("worker-0".into(), "engine-1".into());
        assert_eq!(e0.cr_name(), "worker-0-engine-0");
        assert_eq!(e1.cr_name(), "worker-0-engine-1");
        assert_ne!(e0.instance_id(), e1.instance_id());
        assert_ne!(e0.instance_id(), hash_pod_name("worker-0"));
    }
}
