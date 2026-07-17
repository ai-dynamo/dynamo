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
const DEFAULT_MAIN_CONTAINER_NAME: &str = "main";

/// Environment variable overriding the name of the main container.
///
/// Set by the operator when the DGD renames the main container via
/// `mainContainerNameOverride`; the matching container keeps pod-level identity.
const DYN_MAIN_CONTAINER_NAME: &str = "DYN_MAIN_CONTAINER_NAME";

/// Pod annotation stamped by the operator recording the pod's effective
/// main-container name. Watchers resolve a remote pod's main container from
/// this annotation (falling back to `"main"`), never from their own
/// environment, so pods with different main-container names remain mutually
/// discoverable.
const MAIN_CONTAINER_ANNOTATION: &str = "nvidia.com/dynamo-main-container-name";

/// Name of the container granted pod-level identity in container discovery mode.
///
/// Reads `DYN_MAIN_CONTAINER_NAME` (trimmed), defaulting to `"main"` when
/// unset or blank.
fn main_container_name() -> String {
    match std::env::var(DYN_MAIN_CONTAINER_NAME) {
        Ok(name) if !name.trim().is_empty() => name.trim().to_string(),
        _ => DEFAULT_MAIN_CONTAINER_NAME.to_string(),
    }
}

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
    ///
    /// Resolves the main-container name from this process's own environment;
    /// use [`Self::cr_name_with_main`] when identifying containers of a
    /// *remote* pod (whose main-container name comes from the pod itself).
    pub fn cr_name(&self) -> String {
        self.cr_name_with_main(&main_container_name())
    }

    /// CR name for this target given an explicit main-container name.
    pub fn cr_name_with_main(&self, main_container_name: &str) -> String {
        match self {
            Self::Pod(pod_name) => pod_name.clone(),
            Self::Container(pod_name, container_name) if container_name == main_container_name => {
                pod_name.clone()
            }
            Self::Container(pod_name, container_name) => {
                format!("{}-{}", pod_name, container_name)
            }
        }
    }

    /// Deterministic instance ID derived from cr_name.
    pub fn instance_id(&self) -> u64 {
        self.instance_id_with_main(&main_container_name())
    }

    /// Deterministic instance ID given an explicit main-container name.
    pub fn instance_id_with_main(&self, main_container_name: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cr_name_with_main(main_container_name)
            .hash(&mut hasher);
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

/// Effective main-container name of a (remote) pod: the operator-stamped
/// annotation when present, the well-known `"main"` otherwise. Never consults
/// this process's own environment — a watcher must resolve each watched pod's
/// identity from that pod alone so mixed main-container names (including
/// rolling renames) stay mutually discoverable.
fn pod_main_container_name(pod: &Pod) -> String {
    pod.metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(MAIN_CONTAINER_ANNOTATION))
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| DEFAULT_MAIN_CONTAINER_NAME.to_string())
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

    let main_container_name = pod_main_container_name(pod);

    container_statuses
        .iter()
        .filter(|cs| cs.ready)
        .map(|cs| {
            let target = KubeDiscoveryTarget::Container(pod_name.to_string(), cs.name.clone());
            (
                target.instance_id_with_main(&main_container_name),
                target.cr_name_with_main(&main_container_name),
            )
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
        // With DYN_MAIN_CONTAINER_NAME unset, behavior is unchanged.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, None::<&str>, || {
            let target = KubeDiscoveryTarget::Container("worker-0".into(), "main".into());
            assert_eq!(target.instance_id(), hash_pod_name("worker-0"));
            assert_eq!(target.cr_name(), "worker-0");
        });
    }

    #[test]
    fn test_container_mode_engine_gets_unique_identity() {
        // Non-main containers get per-container identity so that
        // failover engine containers are independently discoverable.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, None::<&str>, || {
            let e0 = KubeDiscoveryTarget::Container("worker-0".into(), "engine-0".into());
            let e1 = KubeDiscoveryTarget::Container("worker-0".into(), "engine-1".into());
            assert_eq!(e0.cr_name(), "worker-0-engine-0");
            assert_eq!(e1.cr_name(), "worker-0-engine-1");
            assert_ne!(e0.instance_id(), e1.instance_id());
            assert_ne!(e0.instance_id(), hash_pod_name("worker-0"));
        });
    }

    #[test]
    fn test_container_mode_custom_main_name_uses_pod_identity() {
        // When DYN_MAIN_CONTAINER_NAME renames the main container, the
        // matching container keeps pod-level identity.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, Some("vllm-worker"), || {
            let target = KubeDiscoveryTarget::Container("worker-0".into(), "vllm-worker".into());
            assert_eq!(target.instance_id(), hash_pod_name("worker-0"));
            assert_eq!(target.cr_name(), "worker-0");
        });
    }

    #[test]
    fn test_whitespace_padded_main_container_name_env_is_trimmed() {
        // Shell quoting or YAML formatting can pad the env value; the
        // resolver trims it so the intended container keeps pod-level
        // identity.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, Some(" vllm-worker "), || {
            let target = KubeDiscoveryTarget::Container("worker-0".into(), "vllm-worker".into());
            assert_eq!(target.instance_id(), hash_pod_name("worker-0"));
            assert_eq!(target.cr_name(), "worker-0");
        });
    }

    #[test]
    fn test_container_mode_custom_main_name_demotes_literal_main() {
        // When the main container is renamed, a container literally named
        // "main" no longer gets pod-level identity, and other containers
        // remain container-scoped.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, Some("vllm-worker"), || {
            let main = KubeDiscoveryTarget::Container("worker-0".into(), "main".into());
            assert_eq!(main.cr_name(), "worker-0-main");
            assert_ne!(main.instance_id(), hash_pod_name("worker-0"));

            let engine = KubeDiscoveryTarget::Container("worker-0".into(), "engine-0".into());
            assert_eq!(engine.cr_name(), "worker-0-engine-0");
            assert_ne!(engine.instance_id(), hash_pod_name("worker-0"));
        });
    }

    fn ready_pod(name: &str, main_annotation: Option<&str>, containers: &[&str]) -> Pod {
        use k8s_openapi::api::core::v1::{ContainerStatus, PodStatus};
        use std::collections::BTreeMap;

        let mut pod = Pod::default();
        pod.metadata.name = Some(name.to_string());
        if let Some(main) = main_annotation {
            let mut annotations = BTreeMap::new();
            annotations.insert(MAIN_CONTAINER_ANNOTATION.to_string(), main.to_string());
            pod.metadata.annotations = Some(annotations);
        }
        pod.status = Some(PodStatus {
            container_statuses: Some(
                containers
                    .iter()
                    .map(|c| ContainerStatus {
                        name: c.to_string(),
                        ready: true,
                        ..Default::default()
                    })
                    .collect(),
            ),
            ..Default::default()
        });
        pod
    }

    #[test]
    fn test_remote_pods_resolve_main_from_their_own_annotation() {
        // Mixed main-container names on the same discovery plane: each pod's
        // identity comes from its own annotation, regardless of the observer's
        // environment (set to a third name here to prove independence).
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, Some("observer-name"), || {
            let custom = ready_pod("worker-0", Some("engine"), &["engine", "logger"]);
            let default = ready_pod("worker-1", None, &["main", "logger"]);

            let custom_ids = extract_ready_containers(&custom);
            assert_eq!(
                custom_ids,
                vec![
                    (hash_pod_name("worker-0"), "worker-0".to_string()),
                    (
                        KubeDiscoveryTarget::Container("worker-0".into(), "logger".into())
                            .instance_id_with_main("engine"),
                        "worker-0-logger".to_string()
                    ),
                ]
            );

            let default_ids = extract_ready_containers(&default);
            assert_eq!(
                default_ids,
                vec![
                    (hash_pod_name("worker-1"), "worker-1".to_string()),
                    (
                        KubeDiscoveryTarget::Container("worker-1".into(), "logger".into())
                            .instance_id_with_main("main"),
                        "worker-1-logger".to_string()
                    ),
                ]
            );
        });
    }

    #[test]
    fn test_remote_pod_annotation_demotes_literal_main_container() {
        // A renamed-main pod that also carries a container literally named
        // "main" (a sidecar) must scope that container per-container.
        let pod = ready_pod("worker-2", Some("engine"), &["engine", "main"]);
        let ids = extract_ready_containers(&pod);
        assert_eq!(
            ids,
            vec![
                (hash_pod_name("worker-2"), "worker-2".to_string()),
                (
                    KubeDiscoveryTarget::Container("worker-2".into(), "main".into())
                        .instance_id_with_main("engine"),
                    "worker-2-main".to_string()
                ),
            ]
        );
    }

    #[test]
    fn test_empty_main_container_name_env_falls_back_to_default() {
        // An empty or whitespace-only DYN_MAIN_CONTAINER_NAME behaves as unset.
        temp_env::with_var(DYN_MAIN_CONTAINER_NAME, Some(""), || {
            assert_eq!(main_container_name(), "main");
            let target = KubeDiscoveryTarget::Container("worker-0".into(), "main".into());
            assert_eq!(target.cr_name(), "worker-0");
        });
    }
}
