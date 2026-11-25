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

/// Pod readiness gate that holds a worker Pod NotReady until every required frontend can route
/// to it. A Pod that lists the gate is discovered as held from the moment its containers are
/// ready, so frontends can prepare its pipeline, and becomes routable when the Pod is Ready.
pub(super) const FRONTEND_ADMITTED_GATE: &str = "nvidia.com/dynamo-frontend-admitted";

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

/// Hash a (pod, container) pair to get the same per-container instance ID a
/// worker computes for itself under `DYN_KUBE_DISCOVERY_MODE=container` (see
/// `KubeDiscoveryTarget::Container`). A container named `"main"` collapses to
/// the pod-level identity (`hash_pod_name`), matching a container-mode
/// frontend's ability to discover pod-mode workers.
///
/// Used by the Rust EPP (`deploy/inference-gateway/ext-proc`) to resolve a
/// registered worker's per-container instance ID back to its pod's endpoint
/// when `DYN_KUBE_DISCOVERY_MODE=container` (e.g. intra-pod GMS failover,
/// where each engine container registers under its own container name).
pub fn hash_container_name(pod_name: &str, container_name: &str) -> u64 {
    KubeDiscoveryTarget::Container(pod_name.to_string(), container_name.to_string()).instance_id()
}

/// Extract (instance_id, cr_name, pod_uid) tuples from an EndpointSlice for ready endpoints.
///
/// Skips endpoints without a pod UID in `target_ref.uid` — never falls back to name-only
/// identity, which could match a new Pod incarnation to a previous one's metadata.
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String, String)> {
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

        let target_ref = match endpoint.target_ref.as_ref() {
            Some(r) => r,
            None => continue,
        };

        let pod_name = target_ref.name.as_deref().unwrap_or("");
        if pod_name.is_empty() {
            continue;
        }

        let pod_uid = match target_ref.uid.as_deref() {
            Some(uid) if !uid.is_empty() => uid.to_string(),
            _ => {
                tracing::debug!(
                    pod_name,
                    "EndpointSlice endpoint missing target_ref.uid, skipping"
                );
                continue;
            }
        };

        let target = KubeDiscoveryTarget::Pod(pod_name.to_string());
        result.push((target.instance_id(), target.cr_name(), pod_uid));
    }

    result
}

fn pod_condition_is_true(pod: &Pod, condition_type: &str) -> bool {
    pod.status
        .as_ref()
        .and_then(|status| status.conditions.as_ref())
        .is_some_and(|conditions| {
            conditions
                .iter()
                .any(|condition| condition.type_ == condition_type && condition.status == "True")
        })
}

/// Whether the Pod lists [`FRONTEND_ADMITTED_GATE`] among its readiness gates.
fn has_frontend_gate(pod: &Pod) -> bool {
    pod.spec
        .as_ref()
        .and_then(|spec| spec.readiness_gates.as_ref())
        .is_some_and(|gates| {
            gates
                .iter()
                .any(|gate| gate.condition_type == FRONTEND_ADMITTED_GATE)
        })
}

/// Extract (instance_id, cr_name, pod_uid, held) for a Pod that lists [`FRONTEND_ADMITTED_GATE`]
/// and whose containers are ready.
///
/// An EndpointSlice reports such a Pod not ready until the gate passes, and cannot tell that
/// from a Pod whose containers are not ready. The Pod can, so a gated Pod is read here for both
/// its held and its routable state. Returns `None` for a Pod without the gate, which leaves it
/// to its EndpointSlice, and for a terminating Pod, which an EndpointSlice also reports not
/// ready.
pub(super) fn extract_gated_pod(pod: &Pod) -> Option<(u64, String, String, bool)> {
    if !has_frontend_gate(pod)
        || pod.metadata.deletion_timestamp.is_some()
        || !pod_condition_is_true(pod, "ContainersReady")
    {
        return None;
    }
    // The Pod is routable when it is Ready, as its EndpointSlice reports it.
    let held = !pod_condition_is_true(pod, "Ready");
    let pod_name = pod.metadata.name.as_deref()?;
    let pod_uid = pod.metadata.uid.as_deref().filter(|uid| !uid.is_empty())?;
    let target = KubeDiscoveryTarget::Pod(pod_name.to_string());
    Some((
        target.instance_id(),
        target.cr_name(),
        pod_uid.to_string(),
        held,
    ))
}

/// Extract (instance_id, cr_name, pod_uid, held) tuples from a Pod for each ready container.
///
/// A ready container is held while its Pod lists [`FRONTEND_ADMITTED_GATE`] and the gate has
/// not passed. Otherwise it is routable whatever the Pod's own readiness, as before: once
/// the gate has passed, a container that is not ready does not hold the Pod's other
/// containers.
///
/// Skips pods without `metadata.uid` — never falls back to name-only identity.
pub(super) fn extract_ready_containers(pod: &Pod) -> Vec<(u64, String, String, bool)> {
    let pod_name = match pod.metadata.name.as_deref() {
        Some(name) => name,
        None => return vec![],
    };

    let pod_uid = match pod.metadata.uid.as_deref() {
        Some(uid) if !uid.is_empty() => uid.to_string(),
        _ => {
            tracing::debug!(pod_name, "Pod missing metadata.uid, skipping");
            return vec![];
        }
    };

    let container_statuses = match pod
        .status
        .as_ref()
        .and_then(|s| s.container_statuses.as_ref())
    {
        Some(statuses) => statuses,
        None => return vec![],
    };

    let held = has_frontend_gate(pod) && !pod_condition_is_true(pod, FRONTEND_ADMITTED_GATE);
    container_statuses
        .iter()
        .filter(|cs| cs.ready)
        .map(|cs| {
            let target = KubeDiscoveryTarget::Container(pod_name.to_string(), cs.name.clone());
            (
                target.instance_id(),
                target.cr_name(),
                pod_uid.clone(),
                held,
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
    /// Whether this process discovers worker Pods that wait on [`FRONTEND_ADMITTED_GATE`], as
    /// held instances. Read from DYN_KUBE_DISCOVERY_HELD_PODS. Off, such a Pod stays unknown
    /// until it is Ready.
    pub discover_held_pods: bool,
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
            discover_held_pods: crate::config::env_is_truthy(
                discovery::DYN_KUBE_DISCOVERY_HELD_PODS,
            ),
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

    #[test]
    fn test_hash_container_name_matches_target_instance_id() {
        // hash_container_name is the public entry point the Rust EPP uses; it
        // must stay in lockstep with the KubeDiscoveryTarget a registering
        // worker computes for itself, including the "main" pod-identity
        // fallback and per-engine uniqueness.
        assert_eq!(
            hash_container_name("worker-0", "main"),
            hash_pod_name("worker-0")
        );
        let e0 = hash_container_name("worker-0", "engine-0");
        let e1 = hash_container_name("worker-0", "engine-1");
        assert_ne!(e0, e1);
        assert_ne!(e0, hash_pod_name("worker-0"));
        assert_eq!(
            e0,
            KubeDiscoveryTarget::Container("worker-0".into(), "engine-0".into()).instance_id()
        );
    }

    use k8s_openapi::api::core::v1::{
        ContainerStatus, PodCondition, PodReadinessGate, PodSpec, PodStatus,
    };
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};

    /// A Pod named `worker-a` with one `main` container. `conditions` lists the Pod
    /// conditions that are `True`.
    fn pod(gated: bool, conditions: &[&str]) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some("worker-a".to_string()),
                uid: Some("pod-uid".to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                readiness_gates: gated.then(|| {
                    vec![PodReadinessGate {
                        condition_type: FRONTEND_ADMITTED_GATE.to_string(),
                    }]
                }),
                ..Default::default()
            }),
            status: Some(PodStatus {
                conditions: Some(
                    conditions
                        .iter()
                        .map(|condition| PodCondition {
                            type_: condition.to_string(),
                            status: "True".to_string(),
                            ..Default::default()
                        })
                        .collect(),
                ),
                container_statuses: Some(vec![ContainerStatus {
                    name: "main".to_string(),
                    ready: conditions.contains(&"ContainersReady"),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn gated_pod_is_held_from_containers_ready_until_pod_ready() {
        let instance_id = hash_pod_name("worker-a");
        let entry = |held| {
            Some((
                instance_id,
                "worker-a".to_string(),
                "pod-uid".to_string(),
                held,
            ))
        };

        // Containers not ready yet: nothing to discover.
        assert_eq!(extract_gated_pod(&pod(true, &[])), None);
        // Locally ready, waiting on the gate: held.
        assert_eq!(
            extract_gated_pod(&pod(true, &["ContainersReady"])),
            entry(true)
        );
        // The gate condition alone does not make the Pod Ready; the kubelet decides.
        assert_eq!(
            extract_gated_pod(&pod(true, &["ContainersReady", FRONTEND_ADMITTED_GATE])),
            entry(true)
        );
        assert_eq!(
            extract_gated_pod(&pod(true, &["ContainersReady", "Ready"])),
            entry(false)
        );

        // A terminating Pod is gone, as it is from its EndpointSlice.
        let mut terminating = pod(true, &["ContainersReady", "Ready"]);
        terminating.metadata.deletion_timestamp = Some(Time(Default::default()));
        assert_eq!(extract_gated_pod(&terminating), None);

        // A Pod without the gate is left to its EndpointSlice.
        assert_eq!(extract_gated_pod(&pod(false, &["ContainersReady"])), None);
        assert_eq!(
            extract_gated_pod(&pod(false, &["ContainersReady", "Ready"])),
            None
        );
    }

    #[test]
    fn ready_container_of_a_gated_pod_is_held_until_the_gate_passes() {
        let held = |pod: &Pod| {
            extract_ready_containers(pod)
                .into_iter()
                .map(|(_, _, _, held)| held)
                .collect::<Vec<_>>()
        };

        assert_eq!(held(&pod(true, &["ContainersReady"])), vec![true]);
        assert_eq!(
            held(&pod(
                true,
                &["ContainersReady", FRONTEND_ADMITTED_GATE, "Ready"]
            )),
            vec![false]
        );
        // Without the gate a ready container is routable whatever the Pod's readiness.
        assert_eq!(held(&pod(false, &["ContainersReady"])), vec![false]);
        assert!(held(&pod(true, &[])).is_empty());

        // A second container that is not ready keeps the Pod from being Ready. That holds the
        // ready container only until the gate has passed.
        let with_failed_sibling = |conditions: &[&str]| {
            let mut pod = pod(true, conditions);
            let statuses = pod
                .status
                .as_mut()
                .and_then(|status| status.container_statuses.as_mut())
                .unwrap();
            statuses[0].ready = true;
            statuses.push(ContainerStatus {
                name: "engine-1".to_string(),
                ready: false,
                ..Default::default()
            });
            pod
        };
        assert_eq!(held(&with_failed_sibling(&[])), vec![true]);
        assert_eq!(
            held(&with_failed_sibling(&[FRONTEND_ADMITTED_GATE])),
            vec![false]
        );
    }
}
