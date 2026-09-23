// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovers inference workers from pods selected by the standalone EPP's
//! [`InferencePool`](crate::inference_pool).
//!
//! Maintains an index of `Ready`, non-terminating pods using the pool's match
//! labels and target port. Workers are keyed by `hash_pod_name(pod_name)` for
//! selector registration and endpoint resolution. Under disaggregated topology
//! every entry also carries the worker's role, read from the configured pod
//! label; under aggregated topology the label is never read.
//!
//! # Required k8s RBAC
//!
//! Standalone mode only (`DYN_EPP_MODE=standalone`). Needs the following
//! permission granted to SA `dynamo-epp` in `examples/onramp/agg.yaml`:
//! - `pods:list/watch`

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use dynamo_runtime::discovery::hash_pod_name;
use k8s_openapi::api::core::v1::Pod;
use tokio::sync::watch;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::inference_pool::{PoolState, spawn_pool_watch};
use crate::worker_role::{RoleCounts, RoleLabelError, WorkerRole};

/// A discovered, `Ready` raw inference engine worker normalized for selector registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawWorker {
    /// Stable hash of the pod name; the selector catalog key.
    pub worker_id: u64,
    /// Kubernetes pod name.
    pub pod_name: String,
    /// Pod IP.
    pub pod_ip: String,
    /// Which catalog the worker belongs to.
    pub role: WorkerRole,
    /// OpenAI HTTP inference endpoint, `http://<ip>:<target_port>`.
    pub http_endpoint: String,
    /// Inference engine KV-event ZMQ PUB endpoints by global data-parallel rank,
    /// `tcp://<ip>:<kv_event_port + rank * stride>`. Empty for decode workers:
    /// the decode selector consumes no KV events, and a worker without an
    /// endpoint is still schedulable there.
    pub kv_events_endpoints: HashMap<u32, String>,
    /// Optional ZMQ REQ endpoint for live-stream gap replay. `None` for decode
    /// workers, for the same reason.
    pub replay_endpoint: Option<String>,
}

/// Why a pod produced no worker. `Ineligible` is ordinary churn (NotReady,
/// unselected, no IP) and the only outcome under aggregated topology; `Role`
/// is an operator error on an otherwise eligible pod.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PodRejection {
    Ineligible,
    Role(RoleLabelError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerEntry {
    worker: RawWorker,
    /// Scheme-less `ip:port` derived once from `worker.http_endpoint`.
    endpoint: String,
}

impl WorkerEntry {
    fn from_raw(worker: RawWorker) -> Self {
        let endpoint = strip_scheme(&worker.http_endpoint).to_string();
        Self { worker, endpoint }
    }
}

type WorkerIndex = HashMap<u64, WorkerEntry>;

/// The worker index plus per-role counts kept in step with it, so the
/// per-request emptiness check stays O(1). Derefs to the map for reads but
/// deliberately not `DerefMut`: every mutation goes through the inherent
/// methods that maintain the counts.
#[derive(Debug, Default, PartialEq, Eq)]
struct IndexState {
    workers: WorkerIndex,
    counts: RoleCounts,
}

impl std::ops::Deref for IndexState {
    type Target = WorkerIndex;

    fn deref(&self) -> &Self::Target {
        &self.workers
    }
}

impl IndexState {
    fn from_workers(workers: WorkerIndex) -> Self {
        let mut counts = RoleCounts::default();
        for entry in workers.values() {
            counts.add(entry.worker.role);
        }
        Self { workers, counts }
    }

    /// A role flip is an in-place update: drop the old role's count before
    /// adding the new one.
    fn insert(&mut self, worker_id: u64, entry: WorkerEntry) {
        let role = entry.worker.role;
        if let Some(previous) = self.workers.insert(worker_id, entry) {
            self.counts.remove(previous.worker.role);
        }
        self.counts.add(role);
    }

    fn remove(&mut self, worker_id: &u64) -> Option<WorkerEntry> {
        let removed = self.workers.remove(worker_id);
        if let Some(entry) = &removed {
            self.counts.remove(entry.worker.role);
        }
        removed
    }

    fn clear(&mut self) {
        self.workers.clear();
        self.counts = RoleCounts::default();
    }

    fn counts(&self) -> RoleCounts {
        self.counts
    }
}

/// Provides an index of `Ready` workers selected by the EPP's `InferencePool`.
#[derive(Clone)]
pub struct PodDiscovery {
    index: Arc<RwLock<IndexState>>,
    changes: watch::Receiver<u64>,
}

impl PodDiscovery {
    /// Start the InferencePool watch and a namespace-wide pod reflector. Returns
    /// a *live* readiness flag that is `true` only while the pod cache has synced
    /// (initial LIST done) **and** the `InferencePool` is resolved. It clears back
    /// to `false` if the pool is later deleted or edited into an unsupported spec
    /// (so nothing is routable), and recovers when both are healthy again — this
    /// is the gRPC health SERVING signal, so it must not latch true.
    pub async fn spawn(cfg: &EppStandaloneConfig) -> Result<(Self, Arc<AtomicBool>)> {
        use futures::StreamExt;
        use kube::{Api, Client, runtime::WatchStreamExt, runtime::reflector, runtime::watcher};

        let client = Client::try_default().await?;
        let namespace = cfg.namespace.clone();

        let (pool_rx, _pool_task) = spawn_pool_watch(
            client.clone(),
            namespace.clone(),
            cfg.inference_pool_name.clone(),
        )
        .await?;

        let pods: Api<Pod> = Api::namespaced(client, &namespace);
        let writer = reflector::store::Writer::default();
        let store = writer.as_reader();
        let ready = Arc::new(AtomicBool::new(false));
        let reflect = reflector::reflector(
            writer,
            watcher(pods, watcher::Config::default()).default_backoff(),
        );

        let (changes_tx, changes_rx) = watch::channel(0u64);
        let role_cfg = RoleDiscoveryConfig::from_config(cfg);
        let index: Arc<RwLock<IndexState>> = Arc::new(RwLock::new(IndexState::default()));

        tracing::info!(
            namespace = %namespace,
            pool = %cfg.inference_pool_name,
            kv_event_port = cfg.kv_event_port,
            "Starting namespace pod reflector for standalone mode"
        );

        let index_task = index.clone();
        let ready_task = ready.clone();
        tokio::spawn(async move {
            let mut pool_rx = pool_rx;
            tokio::pin!(reflect);
            let mut generation = 0u64;
            let mut last_counts = RoleCounts::default();
            let mut census_logged = false;
            let mut pod_synced = false;
            let mut relisting = false;

            enum Delta {
                Upsert(Pod),
                Remove(Pod),
                Rebuild,
                Skip,
                Stop,
            }

            loop {
                let delta = tokio::select! {
                    ev = reflect.next() => match ev {
                        None => {
                            tracing::warn!("Inference engine pod reflector stream ended unexpectedly");
                            Delta::Stop
                        }
                        Some(Ok(watcher::Event::Init | watcher::Event::InitApply(_))) => {
                            relisting = true;
                            Delta::Skip
                        }
                        Some(Ok(watcher::Event::InitDone)) => {
                            relisting = false;
                            pod_synced = true;
                            Delta::Rebuild
                        }
                        Some(Ok(watcher::Event::Apply(pod))) => Delta::Upsert(pod),
                        Some(Ok(watcher::Event::Delete(pod))) => Delta::Remove(pod),
                        Some(Err(e)) => {
                            tracing::warn!(error = %e, "Pod reflector watch error; retrying");
                            Delta::Skip
                        }
                    },
                    changed = pool_rx.changed() => {
                        if changed.is_err() {
                            tracing::warn!("InferencePool watch ended");
                            Delta::Stop
                        } else if defer_pool_rebuild(relisting, pool_rx.borrow().is_some()) {
                            Delta::Skip
                        } else {
                            Delta::Rebuild
                        }
                    }
                };

                let index_changed = match delta {
                    Delta::Stop => break,
                    Delta::Skip => continue,
                    Delta::Rebuild => {
                        rebuild_index(&store, pool_rx.borrow().as_ref(), &role_cfg, &index_task)
                    }
                    Delta::Upsert(pod) => {
                        upsert_pod(&index_task, &pod, pool_rx.borrow().as_ref(), &role_cfg)
                    }
                    Delta::Remove(pod) => remove_pod(&index_task, &pod),
                };

                let is_ready = pod_synced && pool_rx.borrow().is_some();

                // `last_counts` starts at zeros, so a role empty from process
                // start never crosses zero, and an all-empty index does not set
                // `index_changed` either; report the census once instead.
                // Aggregated has no census, so the block below is the
                // transition-only path there.
                let first_census = is_ready && !census_logged && role_cfg.role_label.is_some();

                // This loop is the only place with a before/after view of the
                // index, so the crossed-zero edge is reported here rather than
                // inside the pure mutators.
                if index_changed || first_census {
                    let counts = index_task.read().unwrap().counts();
                    if first_census {
                        census_logged = true;
                        log_role_census(&role_cfg, counts);
                    } else if index_changed {
                        log_role_count_transitions(&role_cfg, last_counts, counts);
                    }
                    last_counts = counts;
                }

                ready_task.store(is_ready, Ordering::Release);
                if index_changed {
                    generation = generation.wrapping_add(1);
                    let _ = changes_tx.send(generation);
                }
            }
            // Watch stream has ended, so stop advertising readiness and clear the index.
            ready_task.store(false, Ordering::Release);
            index_task.write().unwrap().clear();
        });

        Ok((
            Self {
                index,
                changes: changes_rx,
            },
            ready,
        ))
    }

    /// Every `Ready` worker, whatever its role.
    pub fn ready_workers(&self) -> Vec<RawWorker> {
        self.index
            .read()
            .unwrap()
            .values()
            .map(|entry| entry.worker.clone())
            .collect()
    }

    /// The `Ready` workers of one role, from a single read of the index.
    pub fn ready_workers_for(&self, role: WorkerRole) -> Vec<RawWorker> {
        self.index
            .read()
            .unwrap()
            .values()
            .filter(|entry| entry.worker.role == role)
            .map(|entry| entry.worker.clone())
            .collect()
    }

    /// Runs on every request, so it must not become a scan.
    pub fn has_ready_workers(&self, role: WorkerRole) -> bool {
        self.index.read().unwrap().counts().get(role) > 0
    }

    pub fn role_counts(&self) -> RoleCounts {
        self.index.read().unwrap().counts()
    }

    /// Any endpoint of `role`. O(n) in the worst case, but only the body-less
    /// path reaches it.
    pub fn resolve_any_endpoint(&self, role: WorkerRole) -> Option<String> {
        self.index
            .read()
            .unwrap()
            .values()
            .find(|entry| entry.worker.role == role)
            .map(|entry| entry.endpoint.clone())
    }

    /// The endpoint of `worker_id` if, and only if, it currently holds `role`.
    pub fn resolve_endpoint(&self, worker_id: u64, role: WorkerRole) -> Option<String> {
        self.index
            .read()
            .unwrap()
            .get(&worker_id)
            .filter(|entry| entry.worker.role == role)
            .map(|entry| entry.endpoint.clone())
    }

    pub fn ready_worker_ids_matching(
        &self,
        role: WorkerRole,
        pred: impl Fn(&str) -> bool,
    ) -> HashSet<u64> {
        let index = self.index.read().unwrap();
        index
            .iter()
            .filter(|(_, entry)| entry.worker.role == role && pred(entry.endpoint.as_str()))
            .map(|(worker_id, _)| *worker_id)
            .collect()
    }

    pub fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes.clone()
    }

    #[cfg(test)]
    pub(crate) fn for_test(workers: Vec<RawWorker>) -> (Self, watch::Sender<u64>) {
        let (changes_tx, changes) = watch::channel(0u64);
        (
            Self {
                index: Arc::new(RwLock::new(index_state_from(workers))),
                changes,
            },
            changes_tx,
        )
    }

    /// Replace the catalog; the caller wakes the adapter through the
    /// `watch::Sender` returned by [`Self::for_test`].
    #[cfg(test)]
    pub(crate) fn set_workers(&self, workers: Vec<RawWorker>) {
        *self.index.write().unwrap() = index_state_from(workers);
    }
}

#[cfg(test)]
fn index_state_from(workers: Vec<RawWorker>) -> IndexState {
    IndexState::from_workers(
        workers
            .into_iter()
            .map(|worker| (worker.worker_id, WorkerEntry::from_raw(worker)))
            .collect(),
    )
}

/// The roles to report in a startup census. Empty under aggregated topology:
/// one catalog makes "no workers" unambiguous already.
fn role_census(cfg: &RoleDiscoveryConfig, counts: RoleCounts) -> Vec<(WorkerRole, usize)> {
    if cfg.role_label.is_none() {
        return Vec::new();
    }
    [WorkerRole::Prefill, WorkerRole::Decode]
        .into_iter()
        .map(|role| (role, counts.get(role)))
        .collect()
}

/// Report every role's ready count once, when discovery first has a complete picture.
fn log_role_census(cfg: &RoleDiscoveryConfig, counts: RoleCounts) {
    for (role, ready) in role_census(cfg, counts) {
        if ready == 0 {
            tracing::warn!(
                role = role.as_str(),
                "Role has no ready workers at startup; requests needing it will fail until one \
                 appears. Check the worker-role label on this pool's pods."
            );
        } else {
            tracing::info!(role = role.as_str(), ready, "Role has ready workers");
        }
    }
}

/// Log the roles whose ready count crossed zero in either direction. A role
/// emptying is the condition an operator most needs to see, and health stays
/// SERVING through it by design, so a log line is the signal.
fn log_role_count_transitions(cfg: &RoleDiscoveryConfig, before: RoleCounts, after: RoleCounts) {
    let roles: &[WorkerRole] = if cfg.role_label.is_some() {
        &[WorkerRole::Prefill, WorkerRole::Decode]
    } else {
        &[WorkerRole::Aggregated]
    };

    for &role in roles {
        match (before.get(role), after.get(role)) {
            (0, 0) => {}
            (0, now) => tracing::info!(role = role.as_str(), ready = now, "Role has ready workers"),
            (_, 0) => tracing::warn!(
                role = role.as_str(),
                "Role has no ready workers; requests needing it will fail until one appears"
            ),
            _ => {}
        }
    }
}

/// Return `true` iff the pod is `Ready` and not terminating.
fn pod_is_ready(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }
    pod.status
        .as_ref()
        .and_then(|s| s.conditions.as_ref())
        .map(|conds| {
            conds
                .iter()
                .any(|c| c.type_ == "Ready" && c.status == "True")
        })
        .unwrap_or(false)
}

fn pod_matches(pod: &Pod, match_labels: &BTreeMap<String, String>) -> bool {
    let Some(labels) = pod.metadata.labels.as_ref() else {
        return match_labels.is_empty();
    };
    match_labels
        .iter()
        .all(|(k, v)| labels.get(k).map(|pv| pv == v).unwrap_or(false))
}

fn strip_scheme(endpoint: &str) -> &str {
    endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint)
}

fn pod_worker_id(pod: &Pod) -> Option<u64> {
    pod.metadata.name.as_deref().map(hash_pod_name)
}

fn defer_pool_rebuild(relisting: bool, pool_present: bool) -> bool {
    relisting && pool_present
}

/// A live pod whose label is removed or becomes unparseable is evicted, not
/// left stale. Warned rather than failed: a rolling update transiently
/// produces such pods.
fn upsert_pod(
    index: &RwLock<IndexState>,
    pod: &Pod,
    pool: Option<&PoolState>,
    cfg: &RoleDiscoveryConfig,
) -> bool {
    let Some(worker_id) = pod_worker_id(pod) else {
        return false;
    };
    let outcome = match pool {
        Some(pool) => raw_worker_from_pod(pod, pool, cfg),
        None => Err(PodRejection::Ineligible),
    };
    let entry = match outcome {
        Ok(worker) => Some(WorkerEntry::from_raw(worker)),
        Err(PodRejection::Ineligible) => None,
        Err(PodRejection::Role(error)) => {
            tracing::warn!(
                pod = pod.metadata.name.as_deref().unwrap_or("<unnamed>"),
                reason = error.reason(),
                %error,
                "Pod is pool-selected and Ready but has no usable worker role; excluding it"
            );
            None
        }
    };
    let mut index = index.write().unwrap();
    match entry {
        Some(entry) => {
            if index.get(&worker_id) == Some(&entry) {
                false
            } else {
                index.insert(worker_id, entry);
                true
            }
        }
        None => index.remove(&worker_id).is_some(),
    }
}

fn remove_pod(index: &RwLock<IndexState>, pod: &Pod) -> bool {
    pod_worker_id(pod)
        .and_then(|worker_id| index.write().unwrap().remove(&worker_id))
        .is_some()
}

fn rebuild_index(
    store: &kube::runtime::reflector::Store<Pod>,
    pool: Option<&PoolState>,
    cfg: &RoleDiscoveryConfig,
    index: &RwLock<IndexState>,
) -> bool {
    let mut fresh = WorkerIndex::new();
    let mut excluded = 0usize;
    if let Some(pool) = pool {
        for pod in store.state().iter() {
            match raw_worker_from_pod(pod, pool, cfg) {
                Ok(worker) => {
                    fresh.insert(worker.worker_id, WorkerEntry::from_raw(worker));
                }
                Err(PodRejection::Ineligible) => {}
                Err(_) => excluded += 1,
            }
        }
    }
    // Counted over the snapshot rather than per pod: relists would repeat a
    // per-pod warn forever.
    if excluded > 0 {
        tracing::warn!(
            excluded,
            eligible = fresh.len(),
            "Pool-selected Ready pods were excluded because their worker role could not be resolved"
        );
    }
    let fresh = IndexState::from_workers(fresh);
    let mut current = index.write().unwrap();
    if *current == fresh {
        false
    } else {
        *current = fresh;
        true
    }
}

/// KV-event endpoint layout for one pod's data-parallel ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvEventPorts {
    pub base: u16,
    pub stride: u16,
    pub data_parallel_size: u32,
}

impl KvEventPorts {
    fn endpoints(self, ip: IpAddr) -> HashMap<u32, String> {
        (0..self.data_parallel_size.max(1))
            .map(|rank| {
                let port = self
                    .base
                    .saturating_add(self.stride.saturating_mul(rank as u16));
                (rank, format!("tcp://{}", SocketAddr::new(ip, port)))
            })
            .collect()
    }
}

/// The per-pod inputs `raw_worker_from_pod` needs beyond the pool.
#[derive(Debug, Clone)]
pub(crate) struct RoleDiscoveryConfig {
    /// Label key carrying a worker's role, or `None` in aggregated topology,
    /// where the label is never read at all.
    role_label: Option<String>,
    kv_ports: KvEventPorts,
    replay_port: Option<u16>,
}

impl RoleDiscoveryConfig {
    pub(crate) fn from_config(cfg: &EppStandaloneConfig) -> Self {
        Self {
            role_label: cfg
                .topology_mode
                .is_disaggregated()
                .then(|| cfg.worker_role_label.clone()),
            kv_ports: KvEventPorts {
                base: cfg.kv_event_port,
                stride: cfg.kv_event_port_stride,
                data_parallel_size: cfg.data_parallel_size,
            },
            replay_port: cfg.replay_port,
        }
    }

    fn role_of(&self, pod: &Pod) -> Result<WorkerRole, RoleLabelError> {
        let Some(key) = self.role_label.as_deref() else {
            return Ok(WorkerRole::Aggregated);
        };
        let value = pod
            .metadata
            .labels
            .as_ref()
            .and_then(|labels| labels.get(key))
            .ok_or(RoleLabelError::Missing)?;
        WorkerRole::from_pod_label(value)
    }
}

/// Eligibility is decided before the role, so a NotReady prefill pod is
/// `Ineligible`, never a role error.
fn raw_worker_from_pod(
    pod: &Pod,
    pool: &PoolState,
    cfg: &RoleDiscoveryConfig,
) -> Result<RawWorker, PodRejection> {
    if !pod_is_ready(pod) || !pod_matches(pod, &pool.match_labels) {
        return Err(PodRejection::Ineligible);
    }
    let (Some(pod_name), Some(pod_ip)) = (
        pod.metadata.name.as_deref(),
        pod.status.as_ref().and_then(|s| s.pod_ip.as_deref()),
    ) else {
        return Err(PodRejection::Ineligible);
    };
    let Ok(ip) = pod_ip.parse::<IpAddr>() else {
        return Err(PodRejection::Ineligible);
    };
    let role = cfg.role_of(pod).map_err(PodRejection::Role)?;

    // A decode endpoint would open a subscription nothing publishes to, and a
    // dead subscriber is indistinguishable from a genuine cache miss.
    let subscribes_to_kv_events = role != WorkerRole::Decode;
    Ok(RawWorker {
        worker_id: hash_pod_name(pod_name),
        pod_name: pod_name.to_string(),
        pod_ip: pod_ip.to_string(),
        role,
        http_endpoint: format!("http://{}", SocketAddr::new(ip, pool.target_port)),
        kv_events_endpoints: if subscribes_to_kv_events {
            cfg.kv_ports.endpoints(ip)
        } else {
            HashMap::new()
        },
        replay_endpoint: subscribes_to_kv_events
            .then(|| {
                cfg.replay_port
                    .map(|p| format!("tcp://{}", SocketAddr::new(ip, p)))
            })
            .flatten(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_role::DEFAULT_WORKER_ROLE_LABEL;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
    use kube::api::ObjectMeta;

    fn pool() -> PoolState {
        PoolState {
            match_labels: BTreeMap::from([("app".to_string(), "vllm-qwen".to_string())]),
            target_port: 8000,
        }
    }

    fn pod(name: &str, ip: Option<&str>, ready: Option<bool>, labels: &[(&str, &str)]) -> Pod {
        let conditions = ready.map(|r| {
            vec![PodCondition {
                type_: "Ready".to_string(),
                status: if r { "True" } else { "False" }.to_string(),
                ..Default::default()
            }]
        });
        let label_map = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(label_map),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(|s| s.to_string()),
                conditions,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn single_rank(port: u16) -> KvEventPorts {
        KvEventPorts {
            base: port,
            stride: 1,
            data_parallel_size: 1,
        }
    }

    fn agg_cfg() -> RoleDiscoveryConfig {
        RoleDiscoveryConfig {
            role_label: None,
            kv_ports: single_rank(5557),
            replay_port: None,
        }
    }

    fn disagg_cfg() -> RoleDiscoveryConfig {
        RoleDiscoveryConfig {
            role_label: Some(DEFAULT_WORKER_ROLE_LABEL.to_string()),
            kv_ports: single_rank(5557),
            replay_port: None,
        }
    }

    fn with_replay(mut cfg: RoleDiscoveryConfig, port: u16) -> RoleDiscoveryConfig {
        cfg.replay_port = Some(port);
        cfg
    }

    fn role_pod(name: &str, ip: &str, role: &str) -> Pod {
        pod(
            name,
            Some(ip),
            Some(true),
            &[("app", "vllm-qwen"), (DEFAULT_WORKER_ROLE_LABEL, role)],
        )
    }

    #[test]
    fn data_parallel_ranks_publish_on_strided_ports() {
        let cfg = RoleDiscoveryConfig {
            kv_ports: KvEventPorts {
                base: 5557,
                stride: 2,
                data_parallel_size: 3,
            },
            ..agg_cfg()
        };
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            &cfg,
        )
        .expect("ready, selected pod should map");
        assert_eq!(
            w.kv_events_endpoints,
            HashMap::from([
                (0, "tcp://10.0.0.1:5557".to_string()),
                (1, "tcp://10.0.0.1:5559".to_string()),
                (2, "tcp://10.0.0.1:5561".to_string()),
            ])
        );
    }

    #[test]
    fn raw_worker_from_pod_maps_eligible_pods_and_rejects_the_rest() {
        // (role, http endpoint, rank-0 KV-event endpoint, replay endpoint)
        type Endpoints = (
            WorkerRole,
            &'static str,
            Option<&'static str>,
            Option<&'static str>,
        );
        type Expected = Result<Endpoints, PodRejection>;

        let mut terminating = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        terminating.metadata.deletion_timestamp = Some(Time(k8s_openapi::chrono::Utc::now()));

        let cases: Vec<(&str, Pod, RoleDiscoveryConfig, Expected)> = vec![
            (
                "ready selected pod maps to a worker",
                pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "vllm-qwen")],
                ),
                with_replay(agg_cfg(), 5560),
                Ok((
                    WorkerRole::Aggregated,
                    "http://10.0.0.1:8000",
                    Some("tcp://10.0.0.1:5557"),
                    Some("tcp://10.0.0.1:5560"),
                )),
            ),
            (
                "IPv6 pod IP is bracketed in every endpoint",
                pod(
                    "vllm-0",
                    Some("fd00::10"),
                    Some(true),
                    &[("app", "vllm-qwen")],
                ),
                with_replay(agg_cfg(), 5560),
                Ok((
                    WorkerRole::Aggregated,
                    "http://[fd00::10]:8000",
                    Some("tcp://[fd00::10]:5557"),
                    Some("tcp://[fd00::10]:5560"),
                )),
            ),
            (
                "malformed pod IP is ineligible",
                pod(
                    "vllm-0",
                    Some("not-an-ip"),
                    Some(true),
                    &[("app", "vllm-qwen")],
                ),
                agg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "pod outside the pool selector is ineligible",
                pod(
                    "other-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "something-else")],
                ),
                agg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "NotReady pod is ineligible",
                pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(false),
                    &[("app", "vllm-qwen")],
                ),
                agg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "terminating pod is ineligible",
                terminating,
                agg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "pod without an IP is ineligible",
                pod("vllm-0", None, Some(true), &[("app", "vllm-qwen")]),
                agg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "aggregated topology never reads the role label",
                role_pod("vllm-0", "10.0.0.1", "prefill"),
                agg_cfg(),
                Ok((
                    WorkerRole::Aggregated,
                    "http://10.0.0.1:8000",
                    Some("tcp://10.0.0.1:5557"),
                    None,
                )),
            ),
            (
                "prefill carries both KV-event endpoints",
                role_pod("p-0", "10.0.0.1", "prefill"),
                with_replay(disagg_cfg(), 5560),
                Ok((
                    WorkerRole::Prefill,
                    "http://10.0.0.1:8000",
                    Some("tcp://10.0.0.1:5557"),
                    Some("tcp://10.0.0.1:5560"),
                )),
            ),
            (
                "decode carries no KV-event endpoint even with a replay port",
                role_pod("d-0", "10.0.0.2", "decode"),
                with_replay(disagg_cfg(), 5560),
                Ok((WorkerRole::Decode, "http://10.0.0.2:8000", None, None)),
            ),
            (
                "missing role label is a role rejection",
                pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "vllm-qwen")],
                ),
                disagg_cfg(),
                Err(PodRejection::Role(RoleLabelError::Missing)),
            ),
            (
                "unparseable role label is a role rejection carrying the token",
                role_pod("vllm-0", "10.0.0.1", "gibberish"),
                disagg_cfg(),
                Err(PodRejection::Role(RoleLabelError::Invalid {
                    token: "gibberish".to_string(),
                })),
            ),
            (
                "NotReady outranks a valid role label",
                pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(false),
                    &[("app", "vllm-qwen"), (DEFAULT_WORKER_ROLE_LABEL, "prefill")],
                ),
                disagg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "unselected outranks a valid role label",
                pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "other"), (DEFAULT_WORKER_ROLE_LABEL, "prefill")],
                ),
                disagg_cfg(),
                Err(PodRejection::Ineligible),
            ),
            (
                "missing IP outranks a broken role label",
                pod(
                    "vllm-0",
                    None,
                    Some(true),
                    &[("app", "vllm-qwen"), (DEFAULT_WORKER_ROLE_LABEL, "bogus")],
                ),
                disagg_cfg(),
                Err(PodRejection::Ineligible),
            ),
        ];

        for (label, p, cfg, want) in cases {
            let got = raw_worker_from_pod(&p, &pool(), &cfg);
            match want {
                Ok((role, http, kv, replay)) => {
                    let w = got.unwrap_or_else(|e| panic!("{label}: {e:?}"));
                    let name = p.metadata.name.as_deref().unwrap();
                    assert_eq!(w.worker_id, hash_pod_name(name), "{label}");
                    assert_eq!(w.role, role, "{label}");
                    assert_eq!(w.http_endpoint, http, "{label}");
                    assert_eq!(
                        w.kv_events_endpoints.get(&0).map(String::as_str),
                        kv,
                        "{label}"
                    );
                    assert_eq!(
                        w.kv_events_endpoints.len(),
                        usize::from(kv.is_some()),
                        "{label}"
                    );
                    assert_eq!(w.replay_endpoint.as_deref(), replay, "{label}");
                }
                Err(rejection) => assert_eq!(got, Err(rejection), "{label}"),
            }
        }
    }

    #[test]
    fn role_census_covers_both_roles_or_is_empty_under_aggregated() {
        let prefill_only = RoleCounts {
            aggregated: 0,
            prefill: 2,
            decode: 0,
        };
        let aggregated_only = RoleCounts {
            aggregated: 3,
            prefill: 0,
            decode: 0,
        };
        let cases = [
            (
                "a role empty from process start is reported at 0",
                disagg_cfg(),
                prefill_only,
                vec![(WorkerRole::Prefill, 2), (WorkerRole::Decode, 0)],
            ),
            (
                "an all-empty pool reports both roles at 0",
                disagg_cfg(),
                RoleCounts::default(),
                vec![(WorkerRole::Prefill, 0), (WorkerRole::Decode, 0)],
            ),
            (
                "aggregated with no workers adds nothing",
                agg_cfg(),
                RoleCounts::default(),
                vec![],
            ),
            (
                "aggregated with workers adds nothing",
                agg_cfg(),
                aggregated_only,
                vec![],
            ),
        ];
        for (label, cfg, counts, want) in cases {
            assert_eq!(role_census(&cfg, counts), want, "{label}");
        }
    }

    fn store_from_pods(pods: Vec<Pod>) -> kube::runtime::reflector::Store<Pod> {
        use kube::runtime::watcher;
        let mut writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();
        writer.apply_watcher_event(&watcher::Event::Init);
        for p in pods {
            writer.apply_watcher_event(&watcher::Event::InitApply(p));
        }
        writer.apply_watcher_event(&watcher::Event::InitDone);
        store
    }

    #[test]
    fn rebuild_index_keeps_only_ready_selected_pods() {
        let cfg = with_replay(agg_cfg(), 5560);
        let store = store_from_pods(vec![
            pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            pod(
                "vllm-1",
                Some("10.0.0.2"),
                Some(false),
                &[("app", "vllm-qwen")],
            ),
            pod("other-0", Some("10.0.0.3"), Some(true), &[("app", "nope")]),
        ]);
        let index = RwLock::new(IndexState::default());
        assert!(rebuild_index(&store, Some(&pool()), &cfg, &index));
        assert!(!rebuild_index(&store, Some(&pool()), &cfg, &index));
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        let id = hash_pod_name("vllm-0");
        let entry = index.get(&id).expect("ready pod is indexed");
        assert_eq!(entry.worker.worker_id, id);
        assert_eq!(entry.endpoint, "10.0.0.1:8000");
    }

    #[test]
    fn rebuild_index_is_empty_without_pool() {
        let store = store_from_pods(vec![pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        )]);
        let index = RwLock::new(IndexState::default());
        assert!(!rebuild_index(&store, None, &agg_cfg(), &index));
        assert!(index.read().unwrap().is_empty());
    }

    #[test]
    fn upsert_and_remove_pod_mutate_index_incrementally() {
        let cfg = agg_cfg();
        let index = RwLock::new(IndexState::default());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        assert!(upsert_pod(&index, &ready, Some(&pool()), &cfg));
        assert!(!upsert_pod(&index, &ready, Some(&pool()), &cfg));
        assert_eq!(
            index.read().unwrap().get(&id).map(|e| e.endpoint.as_str()),
            Some("10.0.0.1:8000")
        );

        let not_ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(false),
            &[("app", "vllm-qwen")],
        );
        assert!(upsert_pod(&index, &not_ready, Some(&pool()), &cfg));
        assert!(!upsert_pod(&index, &not_ready, Some(&pool()), &cfg));
        assert!(!index.read().unwrap().contains_key(&id));

        assert!(upsert_pod(&index, &ready, Some(&pool()), &cfg));
        assert!(index.read().unwrap().contains_key(&id));
        assert!(remove_pod(&index, &ready));
        assert!(!remove_pod(&index, &ready));
        assert!(!index.read().unwrap().contains_key(&id));

        let unselected = pod("other-0", Some("10.0.0.2"), Some(true), &[("app", "other")]);
        assert!(!upsert_pod(&index, &unselected, Some(&pool()), &cfg));
    }

    #[test]
    fn upsert_pod_without_pool_drops_entry() {
        let cfg = agg_cfg();
        let index = RwLock::new(IndexState::default());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        assert!(upsert_pod(&index, &ready, Some(&pool()), &cfg));
        assert!(index.read().unwrap().contains_key(&id));
        assert!(upsert_pod(&index, &ready, None, &cfg));
        assert!(!upsert_pod(&index, &ready, None, &cfg));
        assert!(!index.read().unwrap().contains_key(&id));
    }

    #[test]
    fn pool_edit_during_relist_rebuilds_at_init_done_from_completed_store() {
        use kube::runtime::watcher;
        let cfg = agg_cfg();
        let vllm_0 = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        let vllm_1 = pod(
            "vllm-1",
            Some("10.0.0.2"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        let mut writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();
        writer.apply_watcher_event(&watcher::Event::Init);
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_0.clone()));
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_1.clone()));
        writer.apply_watcher_event(&watcher::Event::InitDone);
        let index = RwLock::new(IndexState::default());
        assert!(rebuild_index(&store, Some(&pool()), &cfg, &index));
        assert_eq!(index.read().unwrap().len(), 2);

        // A relist begins and only vllm-0 has been re-applied when the pool edit lands.
        writer.apply_watcher_event(&watcher::Event::Init);
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_0.clone()));
        assert_eq!(store.state().len(), 2);
        let mut updated_pool = pool();
        updated_pool.target_port = 9000;
        assert!(defer_pool_rebuild(true, true));
        assert_eq!(
            index
                .read()
                .unwrap()
                .get(&hash_pod_name("vllm-0"))
                .map(|entry| entry.endpoint.as_str()),
            Some("10.0.0.1:8000")
        );

        writer.apply_watcher_event(&watcher::Event::InitDone);
        assert!(rebuild_index(&store, Some(&updated_pool), &cfg, &index));
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(
            index
                .get(&hash_pod_name("vllm-0"))
                .map(|entry| entry.endpoint.as_str()),
            Some("10.0.0.1:9000")
        );
        assert!(!index.contains_key(&hash_pod_name("vllm-1")));
        assert!(!defer_pool_rebuild(true, false));
    }

    #[test]
    fn rebuild_index_drops_workers_absent_from_the_store() {
        let cfg = agg_cfg();
        let index = RwLock::new(IndexState::default());
        upsert_pod(
            &index,
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            Some(&pool()),
            &cfg,
        );
        upsert_pod(
            &index,
            &pod(
                "vllm-1",
                Some("10.0.0.2"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            Some(&pool()),
            &cfg,
        );
        assert_eq!(index.read().unwrap().len(), 2);
        let store = store_from_pods(vec![pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        )]);
        assert!(rebuild_index(&store, Some(&pool()), &cfg, &index));
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.contains_key(&hash_pod_name("vllm-0")));
        assert!(!index.contains_key(&hash_pod_name("vllm-1")));
    }

    fn raw_worker_with_endpoint(worker_id: u64, endpoint: &str) -> RawWorker {
        let name = format!("pod-{worker_id}");
        RawWorker {
            worker_id,
            pod_name: name.clone(),
            pod_ip: endpoint
                .rsplit_once(':')
                .map_or(endpoint, |(ip, _)| ip)
                .to_string(),
            role: WorkerRole::Aggregated,
            http_endpoint: format!("http://{endpoint}"),
            kv_events_endpoints: HashMap::from([(0, format!("tcp://{endpoint}"))]),
            replay_endpoint: None,
        }
    }

    fn discovery_with_endpoints(endpoints: HashMap<u64, String>) -> PodDiscovery {
        PodDiscovery::for_test(
            endpoints
                .into_iter()
                .map(|(id, endpoint)| raw_worker_with_endpoint(id, &endpoint))
                .collect(),
        )
        .0
    }

    #[test]
    fn ready_worker_ids_matching_filters_without_cloning() {
        let discovery = discovery_with_endpoints(HashMap::from([
            (1u64, "10.0.0.1:8000".to_string()),
            (2u64, "10.0.0.2:8000".to_string()),
            (3u64, "10.0.0.3:8000".to_string()),
        ]));
        let role = WorkerRole::Aggregated;
        let filtered =
            discovery.ready_worker_ids_matching(role, |endpoint| endpoint == "10.0.0.2:8000");
        assert_eq!(filtered, HashSet::from([2]));
        let all = discovery.ready_worker_ids_matching(role, |_| true);
        assert_eq!(all, HashSet::from([1, 2, 3]));
        assert!(
            discovery
                .ready_worker_ids_matching(role, |_| false)
                .is_empty()
        );
    }

    /// Recompute the counts from the map, i.e. the source of truth for the
    /// denormalized copy.
    fn recount(index: &IndexState) -> RoleCounts {
        let mut counts = RoleCounts::default();
        for entry in index.values() {
            counts.add(entry.worker.role);
        }
        counts
    }

    #[test]
    fn role_flip_keeps_worker_id_and_moves_counts() {
        let index = RwLock::new(IndexState::default());
        let id = hash_pod_name("vllm-0");
        assert!(upsert_pod(
            &index,
            &role_pod("vllm-0", "10.0.0.1", "prefill"),
            Some(&pool()),
            &disagg_cfg()
        ));
        assert_eq!(index.read().unwrap().counts().prefill, 1);

        // One in-place update: the id is derived from the pod name, so the
        // worker can never be in two catalogs.
        assert!(upsert_pod(
            &index,
            &role_pod("vllm-0", "10.0.0.1", "decode"),
            Some(&pool()),
            &disagg_cfg()
        ));
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.get(&id).unwrap().worker.role, WorkerRole::Decode);
        assert_eq!(index.counts().prefill, 0);
        assert_eq!(index.counts().decode, 1);
        assert_eq!(recount(&index), index.counts());
    }

    #[test]
    fn live_pod_losing_its_role_label_is_evicted() {
        let index = RwLock::new(IndexState::default());
        assert!(upsert_pod(
            &index,
            &role_pod("vllm-0", "10.0.0.1", "decode"),
            Some(&pool()),
            &disagg_cfg()
        ));
        let unlabelled = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        assert!(upsert_pod(
            &index,
            &unlabelled,
            Some(&pool()),
            &disagg_cfg()
        ));
        let index = index.read().unwrap();
        assert!(index.is_empty());
        assert_eq!(index.counts(), RoleCounts::default());
    }

    #[test]
    fn counts_stay_consistent_across_every_mutation_path() {
        let index = RwLock::new(IndexState::default());
        let store = store_from_pods(vec![
            role_pod("p-0", "10.0.0.1", "prefill"),
            role_pod("d-0", "10.0.0.2", "decode"),
            role_pod("d-1", "10.0.0.3", "decode"),
        ]);
        assert!(rebuild_index(&store, Some(&pool()), &disagg_cfg(), &index));
        {
            let index = index.read().unwrap();
            assert_eq!(index.counts(), recount(&index));
            assert_eq!(index.counts().prefill, 1);
            assert_eq!(index.counts().decode, 2);
        }

        assert!(upsert_pod(
            &index,
            &role_pod("p-1", "10.0.0.4", "prefill"),
            Some(&pool()),
            &disagg_cfg()
        ));
        assert_eq!(
            index.read().unwrap().counts(),
            recount(&index.read().unwrap())
        );

        assert!(remove_pod(&index, &role_pod("d-0", "10.0.0.2", "decode")));
        {
            let index = index.read().unwrap();
            assert_eq!(index.counts(), recount(&index));
            assert_eq!(index.counts().decode, 1);
        }

        // Removing every prefill pod leaves the decode count untouched.
        assert!(remove_pod(&index, &role_pod("p-0", "10.0.0.1", "prefill")));
        assert!(remove_pod(&index, &role_pod("p-1", "10.0.0.4", "prefill")));
        {
            let index = index.read().unwrap();
            assert_eq!(index.counts(), recount(&index));
            assert_eq!(index.counts().prefill, 0);
            assert_eq!(index.counts().decode, 1);
        }

        index.write().unwrap().clear();
        let index = index.read().unwrap();
        assert!(index.is_empty());
        assert_eq!(index.counts(), RoleCounts::default());
    }

    fn role_workers(workers: &[(&str, &str, WorkerRole)]) -> Vec<RawWorker> {
        workers
            .iter()
            .map(|(name, ip, role)| RawWorker {
                worker_id: hash_pod_name(name),
                pod_name: (*name).to_string(),
                pod_ip: (*ip).to_string(),
                role: *role,
                http_endpoint: format!("http://{ip}:8000"),
                kv_events_endpoints: if *role == WorkerRole::Decode {
                    HashMap::new()
                } else {
                    HashMap::from([(0, format!("tcp://{ip}:5557"))])
                },
                replay_endpoint: None,
            })
            .collect()
    }

    fn role_discovery(workers: &[(&str, &str, WorkerRole)]) -> PodDiscovery {
        PodDiscovery::for_test(role_workers(workers)).0
    }

    // --- the invariant: a prefill worker is never a destination -------------

    #[test]
    fn a_prefill_only_catalog_yields_nothing_for_decode() {
        let discovery = role_discovery(&[("p-0", "10.0.0.1", WorkerRole::Prefill)]);
        let prefill_id = hash_pod_name("p-0");

        assert!(!discovery.has_ready_workers(WorkerRole::Decode));
        assert!(discovery.resolve_any_endpoint(WorkerRole::Decode).is_none());
        assert!(
            discovery
                .ready_worker_ids_matching(WorkerRole::Decode, |_| true)
                .is_empty()
        );
        // Even holding the id outright does not resolve it as decode.
        assert!(
            discovery
                .resolve_endpoint(prefill_id, WorkerRole::Decode)
                .is_none()
        );
        // ...while the same catalog is visible to prefill.
        assert!(discovery.has_ready_workers(WorkerRole::Prefill));
        assert_eq!(
            discovery
                .resolve_endpoint(prefill_id, WorkerRole::Prefill)
                .as_deref(),
            Some("10.0.0.1:8000")
        );
    }

    #[test]
    fn mixed_catalog_reads_never_cross_roles() {
        let discovery = role_discovery(&[
            ("p-0", "10.0.0.1", WorkerRole::Prefill),
            ("d-0", "10.0.0.2", WorkerRole::Decode),
        ]);
        assert_eq!(
            discovery
                .resolve_any_endpoint(WorkerRole::Decode)
                .as_deref(),
            Some("10.0.0.2:8000")
        );
        assert_eq!(
            discovery.ready_worker_ids_matching(WorkerRole::Decode, |_| true),
            HashSet::from([hash_pod_name("d-0")])
        );
        assert_eq!(
            discovery.ready_worker_ids_matching(WorkerRole::Prefill, |_| true),
            HashSet::from([hash_pod_name("p-0")])
        );

        discovery.set_workers(role_workers(&[
            ("p-0", "10.0.0.1", WorkerRole::Prefill),
            ("d-0", "10.0.0.2", WorkerRole::Decode),
            ("d-1", "10.0.0.3", WorkerRole::Decode),
        ]));
        assert_eq!(discovery.role_counts().prefill, 1);
        assert_eq!(discovery.role_counts().decode, 2);
        let prefill = discovery.ready_workers_for(WorkerRole::Prefill);
        let decode = discovery.ready_workers_for(WorkerRole::Decode);
        assert_eq!(prefill.len(), 1);
        assert_eq!(decode.len(), 2);
        assert!(prefill.iter().all(|w| w.role == WorkerRole::Prefill));
        assert!(decode.iter().all(|w| w.role == WorkerRole::Decode));
        assert_eq!(discovery.ready_workers().len(), 3);

        discovery.set_workers(vec![]);
        assert_eq!(discovery.role_counts(), RoleCounts::default());
        assert!(!discovery.has_ready_workers(WorkerRole::Prefill));
    }
}
