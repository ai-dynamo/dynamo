// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovers inference workers from pods selected by the standalone EPP's
//! [`InferencePool`](crate::inference_pool).
//!
//! Maintains an index of `Ready`, non-terminating pods using the pool's match
//! labels and target port. Workers are keyed by `hash_pod_name(pod_name)` for
//! selector registration and endpoint resolution.

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
    /// Stage this worker serves. Always [`WorkerRole::Aggregated`] in aggregated
    /// topology; `Prefill` or `Decode` when the pool is role-split.
    pub role: WorkerRole,
    /// OpenAI HTTP inference endpoint, `http://<ip>:<target_port>`.
    pub http_endpoint: String,
    /// Inference engine KV-event ZMQ PUB endpoint, `tcp://<ip>:<kv_event_port>`.
    ///
    /// `None` for [`WorkerRole::Decode`]: that role's `SelectionService` runs
    /// with KV events off, so it neither subscribes nor requires an endpoint to
    /// call the worker schedulable.
    pub kv_events_endpoint: Option<String>,
    /// Optional ZMQ REQ endpoint for live-stream gap replay. `None` for
    /// [`WorkerRole::Decode`] for the same reason as `kv_events_endpoint`.
    pub replay_endpoint: Option<String>,
}

/// Why a pod produced no worker.
///
/// The two arms are deliberately distinct: an ineligible pod is normal and
/// silent, while an eligible pod the EPP cannot assign a role to is an operator
/// error worth surfacing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PodRejection {
    /// Not `Ready`, not pool-selected, unnamed, or without a parseable IP.
    /// Pre-existing behavior, and the only outcome possible in aggregated
    /// topology.
    Ineligible,
    /// Pool-selected and `Ready`, but the role label is missing or unparseable.
    Role(RoleLabelError),
}

/// One indexed worker: the materialized [`RawWorker`] for selector registration
/// plus its pre-stripped `ip:port`, so request-path reads (notably
/// [`PodDiscovery::resolve_endpoint`]) are O(1) lookups that never re-parse the
/// endpoint or clone the [`PoolState`].
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

/// Derived catalog of the `Ready`, pool-selected workers, keyed by `worker_id`.
type WorkerIndex = HashMap<u64, WorkerEntry>;

/// The worker index plus denormalized per-role counts.
///
/// The counts exist so [`PodDiscovery::has_ready_workers`] — which runs on every
/// request — stays O(1) after the index gains a role dimension.
///
/// [`Deref`](std::ops::Deref) exposes the map for reads, so every existing
/// lookup is unchanged. There is deliberately **no** `DerefMut`: mutating the
/// map directly would silently desynchronize `counts`, so all writes go through
/// the inherent methods below.
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
    /// Adopt a whole map, recomputing counts from it.
    fn from_workers(workers: WorkerIndex) -> Self {
        let mut counts = RoleCounts::default();
        for entry in workers.values() {
            counts.add(entry.worker.role);
        }
        Self { workers, counts }
    }

    fn insert(&mut self, worker_id: u64, entry: WorkerEntry) {
        let role = entry.worker.role;
        if let Some(previous) = self.workers.insert(worker_id, entry) {
            // A role flip is an in-place update: drop the old role's count
            // before adding the new one.
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

        // Namespace-wide pod watch; membership is decided in memory by the pool
        // selector so selector edits never require re-spawning this watch.
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
            // The pod cache is "synced" once the reflector's initial LIST lands
            // (InitDone); readiness stays gated on this AND pool presence below.
            let mut pod_synced = false;
            // True from `Init` until `InitDone`: a (re)list is buffering objects and
            // the live store still holds the pre-relist Pod set, so a pool edit must
            // not rebuild from it — defer to `InitDone` instead.
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
                        // During a relist the reflector emits Init + one InitApply
                        // per pod + InitDone.
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
                            // Wait for the relist to complete and InitDone to be emitted.
                            // To then rebuild the index from the completed pod list + latest PoolState.
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

                // This loop is the only place with a before/after view of the
                // index, so the crossed-zero edge is reported here rather than
                // inside the pure mutators.
                if index_changed {
                    let counts = index_task.read().unwrap().counts();
                    log_role_count_transitions(&role_cfg, last_counts, counts);
                    last_counts = counts;
                }

                // Readiness based on initial pods being synced and the pool being resolved.
                ready_task.store(pod_synced && pool_rx.borrow().is_some(), Ordering::Release);
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

    // Return all currently `Ready` workers selected by the pool, regardless of
    // role. Used by the aggregated reconcile path.
    pub fn ready_workers(&self) -> Vec<RawWorker> {
        self.index
            .read()
            .unwrap()
            .values()
            .map(|entry| entry.worker.clone())
            .collect()
    }

    /// Both role sets from a single read lock.
    ///
    /// Reading the two roles through separate calls would take two locks and so
    /// observe two generations; a role flip landing between them would put one
    /// `worker_id` into both desired sets, which is exactly the cross-catalog
    /// duplication the reconcile ordering exists to avoid.
    pub fn ready_workers_by_role(&self) -> RoleWorkerSets {
        let index = self.index.read().unwrap();
        let mut sets = RoleWorkerSets::default();
        for entry in index.values() {
            match entry.worker.role {
                WorkerRole::Prefill => sets.prefill.push(entry.worker.clone()),
                WorkerRole::Decode => sets.decode.push(entry.worker.clone()),
                WorkerRole::Aggregated => {}
            }
        }
        sets
    }

    /// Whether any `Ready`, pool-selected worker in `role` exists. O(1) via the
    /// denormalized counts — this runs on every request, so it must not become
    /// a scan when the index gains a role dimension.
    pub fn has_ready_workers(&self, role: WorkerRole) -> bool {
        self.index.read().unwrap().counts().get(role) > 0
    }

    /// Ready-worker count per role.
    pub fn role_counts(&self) -> RoleCounts {
        self.index.read().unwrap().counts()
    }

    /// Resolve any `Ready` worker in `role` to its `ip:port` endpoint, for
    /// body-less requests that route to an arbitrary worker without building an
    /// id set. O(n), but only on that path, which has no reservation.
    pub fn resolve_any_endpoint(&self, role: WorkerRole) -> Option<String> {
        self.index
            .read()
            .unwrap()
            .values()
            .find(|entry| entry.worker.role == role)
            .map(|entry| entry.endpoint.clone())
    }

    /// Resolve a `worker_id` to its current `ip:port` HTTP endpoint, refusing a
    /// worker whose role is not `role`.
    ///
    /// The role check is what makes "never route to a prefill endpoint" a
    /// compiler-and-runtime property rather than a convention: a caller holding
    /// a prefill id simply cannot turn it into a destination.
    pub fn resolve_endpoint(&self, worker_id: u64, role: WorkerRole) -> Option<String> {
        self.index
            .read()
            .unwrap()
            .get(&worker_id)
            .filter(|entry| entry.worker.role == role)
            .map(|entry| entry.endpoint.clone())
    }

    /// Worker IDs of the currently `Ready`, pool-selected workers whose `ip:port`
    /// endpoint satisfies `pred`, collected in a **single** index pass under one
    /// read lock — no intermediate full-ready set, and `pred` borrows the endpoint
    /// (no per-worker `String` clone). Used only on the subset-routing path.
    ///
    /// `pred` runs while the index read lock is held, so it must not re-enter
    /// `PodDiscovery` (`resolve_endpoint`, another read, …): `std::sync::RwLock`
    /// read locks are not guaranteed re-entrant and a nested acquire can deadlock.
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

    /// Swap the whole worker set, so a test can drive add / role-flip / remove
    /// through a real [`crate::TopologyAdapter`] without a cluster.
    ///
    /// The caller wakes the adapter by sending on the `watch::Sender` that
    /// [`Self::for_test`] handed back — the generation channel is owned there,
    /// not here.
    #[cfg(test)]
    pub(crate) fn set_workers(&self, workers: Vec<RawWorker>) {
        *self.index.write().unwrap() = index_state_from(workers);
    }
}

/// The `Ready` workers of each disaggregated role, taken from one snapshot.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoleWorkerSets {
    pub prefill: Vec<RawWorker>,
    pub decode: Vec<RawWorker>,
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

/// Log the roles whose ready count crossed zero in either direction.
///
/// A role emptying is the condition an operator most needs to see — it is the
/// difference between "the EPP is fine" and "every request for that role 503s"
/// — and health stays SERVING through it by design, so a log line is the signal.
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

/// Return `true` iff the pod carries every `match_labels` key with the equal
/// value (equality-based selector, matching `InferencePool.spec.selector`).
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

/// Stable index key for a pod (its `worker_id`). `None` for an unnamed pod.
fn pod_worker_id(pod: &Pod) -> Option<u64> {
    pod.metadata.name.as_deref().map(hash_pod_name)
}

/// Whether a pool update should wait for an in-progress pod relist to complete.
/// Pool removal is never deferred.
fn defer_pool_rebuild(relisting: bool, pool_present: bool) -> bool {
    relisting && pool_present
}

/// Apply a single pod delta to the index: upsert the worker if it is `Ready` and
/// pool-selected, otherwise drop any existing entry (a pod that went NotReady,
/// terminating, or unselected). Returns whether the derived index changed.
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
            // Eligible but unassignable: exclude it from every catalog. Warn
            // rather than fail, because a rolling update transiently produces
            // such pods; the operator-visible consequence of a wholly
            // mislabelled fleet is an empty role catalog and 503s.
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
        // Covers a live pod whose role label was removed or edited to something
        // unparseable: it is evicted, not left serving from a stale entry.
        None => index.remove(&worker_id).is_some(),
    }
}

/// Drop a deleted pod's worker from the index. Returns whether an entry existed.
fn remove_pod(index: &RwLock<IndexState>, pod: &Pod) -> bool {
    pod_worker_id(pod)
        .and_then(|worker_id| index.write().unwrap().remove(&worker_id))
        .is_some()
}

/// Recompute the whole index from the current pod store and pool selector.
/// Empty until the `InferencePool` has resolved. Used only for the initial LIST,
/// watch relists (which may have dropped pods without a `Delete`), and
/// pool-selector changes (which re-classify every pod). Returns whether the
/// rebuilt index differs from the current one.
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
                // Silent and expected: the pod is simply not one of ours.
                Err(PodRejection::Ineligible) => {}
                Err(_) => excluded += 1,
            }
        }
    }
    if excluded > 0 {
        // Counted over the snapshot rather than warned per pod, because this
        // path also runs on every watch relist and would otherwise repeat the
        // same warning for the same pods indefinitely.
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

/// Inputs the pure per-pod mapping needs, replacing the loose port pair that
/// used to be threaded through every index mutator.
#[derive(Debug, Clone)]
pub(crate) struct RoleDiscoveryConfig {
    /// Label key carrying a worker's role, or `None` in aggregated topology.
    /// When `None` the label is never read at all, which is what keeps
    /// aggregated discovery byte-identical to its previous behavior.
    role_label: Option<String>,
    kv_event_port: u16,
    replay_port: Option<u16>,
}

impl RoleDiscoveryConfig {
    pub(crate) fn from_config(cfg: &EppStandaloneConfig) -> Self {
        Self {
            role_label: cfg
                .topology_mode
                .is_disaggregated()
                .then(|| cfg.worker_role_label.clone()),
            kv_event_port: cfg.kv_event_port,
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

/// Build a [`RawWorker`] from a pod, or say why the pod produced none. Pure
/// function — unit-testable.
///
/// Eligibility is decided first and unchanged, so a `NotReady` prefill pod is
/// [`PodRejection::Ineligible`] rather than a role error: a rolling update must
/// not read as a misconfiguration.
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
    // The decode instance runs with KV events off, so handing decode workers an
    // endpoint would open a ZMQ subscription nothing ever publishes to — and a
    // dead subscriber is indistinguishable from a genuine cache miss.
    let subscribes_to_kv_events = role != WorkerRole::Decode;

    Ok(RawWorker {
        worker_id: hash_pod_name(pod_name),
        pod_name: pod_name.to_string(),
        pod_ip: pod_ip.to_string(),
        role,
        http_endpoint: format!("http://{}", SocketAddr::new(ip, pool.target_port)),
        kv_events_endpoint: subscribes_to_kv_events
            .then(|| format!("tcp://{}", SocketAddr::new(ip, cfg.kv_event_port))),
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

    /// Aggregated discovery: the role label is never read, so every eligible
    /// pod is `WorkerRole::Aggregated` no matter what labels it carries.
    fn agg_cfg() -> RoleDiscoveryConfig {
        RoleDiscoveryConfig {
            role_label: None,
            kv_event_port: 5557,
            replay_port: None,
        }
    }

    /// Disaggregated discovery keyed on the default role label.
    fn disagg_cfg() -> RoleDiscoveryConfig {
        RoleDiscoveryConfig {
            role_label: Some(DEFAULT_WORKER_ROLE_LABEL.to_string()),
            kv_event_port: 5557,
            replay_port: None,
        }
    }

    fn with_replay(mut cfg: RoleDiscoveryConfig, port: u16) -> RoleDiscoveryConfig {
        cfg.replay_port = Some(port);
        cfg
    }

    /// A pod carrying the default role label.
    fn role_pod(name: &str, ip: &str, role: &str) -> Pod {
        pod(
            name,
            Some(ip),
            Some(true),
            &[("app", "vllm-qwen"), (DEFAULT_WORKER_ROLE_LABEL, role)],
        )
    }

    #[test]
    fn ready_selected_pod_maps_to_worker() {
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            &with_replay(agg_cfg(), 5560),
        )
        .expect("ready, selected pod should map");
        assert_eq!(w.worker_id, hash_pod_name("vllm-0"));
        assert_eq!(w.http_endpoint, "http://10.0.0.1:8000");
        assert_eq!(w.kv_events_endpoint.as_deref(), Some("tcp://10.0.0.1:5557"));
        assert_eq!(w.replay_endpoint.as_deref(), Some("tcp://10.0.0.1:5560"));
    }

    #[test]
    fn ipv6_pod_ip_is_bracketed_in_all_endpoints() {
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("fd00::10"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            &with_replay(agg_cfg(), 5560),
        )
        .expect("ready, selected IPv6 pod should map");
        // SocketAddr brackets the IPv6 host so host and port are unambiguous.
        assert_eq!(w.http_endpoint, "http://[fd00::10]:8000");
        assert_eq!(
            w.kv_events_endpoint.as_deref(),
            Some("tcp://[fd00::10]:5557")
        );
        assert_eq!(w.replay_endpoint.as_deref(), Some("tcp://[fd00::10]:5560"));
    }

    #[test]
    fn malformed_pod_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "vllm-0",
                    Some("not-an-ip"),
                    Some(true),
                    &[("app", "vllm-qwen")]
                ),
                &pool(),
                &agg_cfg(),
            )
            .is_err()
        );
    }

    #[test]
    fn pod_not_matching_selector_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "other-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "something-else")]
                ),
                &pool(),
                &agg_cfg(),
            )
            .is_err()
        );
    }

    #[test]
    fn not_ready_pod_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(false),
                    &[("app", "vllm-qwen")]
                ),
                &pool(),
                &agg_cfg(),
            )
            .is_err()
        );
    }

    #[test]
    fn terminating_pod_is_skipped() {
        let mut p = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        p.metadata.deletion_timestamp = Some(Time(k8s_openapi::chrono::Utc::now()));
        assert!(raw_worker_from_pod(&p, &pool(), &agg_cfg()).is_err());
    }

    #[test]
    fn pod_without_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod("vllm-0", None, Some(true), &[("app", "vllm-qwen")]),
                &pool(),
                &agg_cfg(),
            )
            .is_err()
        );
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
        assert!(rebuild_index(
            &store,
            Some(&pool()),
            &with_replay(agg_cfg(), 5560),
            &index
        ));
        assert!(!rebuild_index(
            &store,
            Some(&pool()),
            &with_replay(agg_cfg(), 5560),
            &index
        ));

        // Only the ready, correctly-labeled pod is materialized.
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        let id = hash_pod_name("vllm-0");
        let entry = index.get(&id).expect("ready pod is indexed");
        assert_eq!(entry.worker.worker_id, id);
        // Endpoint is pre-stripped to a scheme-less ip:port.
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
        let index = RwLock::new(IndexState::default());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );

        // Ready + selected -> inserted, with a pre-stripped endpoint.
        assert!(upsert_pod(&index, &ready, Some(&pool()), &agg_cfg()));
        assert!(!upsert_pod(&index, &ready, Some(&pool()), &agg_cfg()));
        assert_eq!(
            index.read().unwrap().get(&id).map(|e| e.endpoint.as_str()),
            Some("10.0.0.1:8000")
        );

        // Same pod goes NotReady -> the upsert drops it (no stale entry).
        let not_ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(false),
            &[("app", "vllm-qwen")],
        );
        assert!(upsert_pod(&index, &not_ready, Some(&pool()), &agg_cfg()));
        assert!(!upsert_pod(&index, &not_ready, Some(&pool()), &agg_cfg()));
        assert!(!index.read().unwrap().contains_key(&id));

        // Re-add, then a Delete removes it.
        assert!(upsert_pod(&index, &ready, Some(&pool()), &agg_cfg()));
        assert!(index.read().unwrap().contains_key(&id));
        assert!(remove_pod(&index, &ready));
        assert!(!remove_pod(&index, &ready));
        assert!(!index.read().unwrap().contains_key(&id));

        // An unrelated namespace pod does not change the derived worker index.
        let unselected = pod("other-0", Some("10.0.0.2"), Some(true), &[("app", "other")]);
        assert!(!upsert_pod(&index, &unselected, Some(&pool()), &agg_cfg()));
    }

    #[test]
    fn upsert_pod_without_pool_drops_entry() {
        // A `None` pool (unresolved or deleted) means nothing is routable, so an
        // upsert must evict any existing entry rather than leave stale routing.
        let index = RwLock::new(IndexState::default());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );

        assert!(upsert_pod(&index, &ready, Some(&pool()), &agg_cfg()));
        assert!(index.read().unwrap().contains_key(&id));

        assert!(upsert_pod(&index, &ready, None, &agg_cfg()));
        assert!(!upsert_pod(&index, &ready, None, &agg_cfg()));
        assert!(!index.read().unwrap().contains_key(&id));
    }

    #[test]
    fn pool_edit_during_relist_rebuilds_at_init_done_from_completed_store() {
        use kube::runtime::watcher;

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

        // Initial LIST has both pods, and the index uses the original Pool port.
        writer.apply_watcher_event(&watcher::Event::Init);
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_0.clone()));
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_1.clone()));
        writer.apply_watcher_event(&watcher::Event::InitDone);
        let index = RwLock::new(IndexState::default());
        assert!(rebuild_index(&store, Some(&pool()), &agg_cfg(), &index));
        assert_eq!(index.read().unwrap().len(), 2);

        // During the next LIST, vllm-1 has disappeared. The reflector buffers
        // vllm-0, but its live Store still exposes the prior two-pod snapshot.
        writer.apply_watcher_event(&watcher::Event::Init);
        writer.apply_watcher_event(&watcher::Event::InitApply(vllm_0.clone()));
        assert_eq!(store.state().len(), 2);

        // A live Pool port edit must not rebuild from that stale Store. The
        // existing index remains the prior coherent snapshot until InitDone.
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

        // InitDone makes the staged list live. Its single rebuild uses both the
        // completed one-pod Store and the latest PoolState.
        writer.apply_watcher_event(&watcher::Event::InitDone);
        assert!(rebuild_index(
            &store,
            Some(&updated_pool),
            &agg_cfg(),
            &index
        ));
        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(
            index
                .get(&hash_pod_name("vllm-0"))
                .map(|entry| entry.endpoint.as_str()),
            Some("10.0.0.1:9000")
        );
        assert!(!index.contains_key(&hash_pod_name("vllm-1")));

        // Pool deletion/invalidity remains an immediate clear, even mid-relist.
        assert!(!defer_pool_rebuild(true, false));
    }

    #[test]
    fn rebuild_index_drops_workers_absent_from_the_store() {
        // A relist arrives as a fresh snapshot with no `Delete` events for pods
        // that disappeared during the disconnect, so the `InitDone`/pool-change
        // rebuild must *replace* the index, not merge into it — otherwise dead
        // workers linger and keep receiving traffic. Seed two workers, then
        // rebuild from a store that holds only one.
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
            &agg_cfg(),
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
            &agg_cfg(),
        );
        assert_eq!(index.read().unwrap().len(), 2);

        // Relist: vllm-1 vanished during the gap; only vllm-0 remains.
        let store = store_from_pods(vec![pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        )]);
        assert!(rebuild_index(&store, Some(&pool()), &agg_cfg(), &index));

        let index = index.read().unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.contains_key(&hash_pod_name("vllm-0")));
        assert!(!index.contains_key(&hash_pod_name("vllm-1")));
    }

    /// Build a `RawWorker` whose HTTP endpoint resolves to `endpoint` (so its
    /// stripped form equals `endpoint`), for index-backed unit tests.
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
            kv_events_endpoint: Some(format!("tcp://{endpoint}")),
            replay_endpoint: None,
        }
    }

    /// Build a `PodDiscovery` over a fixed index (no cluster) so we can unit-test
    /// the read-lock-based subset filter.
    fn discovery_with_endpoints(endpoints: HashMap<u64, String>) -> PodDiscovery {
        let index: WorkerIndex = endpoints
            .into_iter()
            .map(|(id, endpoint)| {
                (
                    id,
                    WorkerEntry::from_raw(raw_worker_with_endpoint(id, &endpoint)),
                )
            })
            .collect();
        let (_, changes) = watch::channel(0u64);
        PodDiscovery {
            index: Arc::new(RwLock::new(IndexState::from_workers(index))),
            changes,
        }
    }

    #[test]
    fn ready_worker_ids_matching_filters_without_cloning() {
        let discovery = discovery_with_endpoints(HashMap::from([
            (1u64, "10.0.0.1:8000".to_string()),
            (2u64, "10.0.0.2:8000".to_string()),
            (3u64, "10.0.0.3:8000".to_string()),
        ]));

        // Predicate borrows the endpoint; only worker 2 matches.
        let filtered = discovery.ready_worker_ids_matching(WorkerRole::Aggregated, |endpoint| {
            endpoint == "10.0.0.2:8000"
        });
        assert_eq!(filtered, HashSet::from([2]));

        // Match-all returns every ready worker (single pass, no input set).
        let all = discovery.ready_worker_ids_matching(WorkerRole::Aggregated, |_| true);
        assert_eq!(all, HashSet::from([1, 2, 3]));

        // No match -> empty.
        assert!(
            discovery
                .ready_worker_ids_matching(WorkerRole::Aggregated, |_| false)
                .is_empty()
        );
    }

    // --- role resolution ---------------------------------------------------

    /// Recompute counts from the map, so the denormalized copy can be checked
    /// against the source of truth after every mutation path.
    fn recount(index: &IndexState) -> RoleCounts {
        let mut counts = RoleCounts::default();
        for entry in index.values() {
            counts.add(entry.worker.role);
        }
        counts
    }

    #[test]
    fn aggregated_topology_ignores_the_role_label() {
        // Not merely "defaults to Aggregated": the label is never read, which is
        // what keeps aggregated discovery byte-identical to its old behavior.
        let w = raw_worker_from_pod(
            &role_pod("vllm-0", "10.0.0.1", "prefill"),
            &pool(),
            &agg_cfg(),
        )
        .expect("labelled pod is still eligible under aggregated");
        assert_eq!(w.role, WorkerRole::Aggregated);
        assert!(w.kv_events_endpoint.is_some());
    }

    #[test]
    fn disaggregated_topology_maps_the_two_roles() {
        for (label, expected) in [
            ("prefill", WorkerRole::Prefill),
            ("decode", WorkerRole::Decode),
        ] {
            let w = raw_worker_from_pod(
                &role_pod("vllm-0", "10.0.0.1", label),
                &pool(),
                &disagg_cfg(),
            )
            .expect("labelled pod should map");
            assert_eq!(w.role, expected);
        }
    }

    #[test]
    fn only_prefill_workers_carry_kv_event_endpoints() {
        // The decode instance runs with KV events off, so an endpoint there would
        // open a subscription nothing publishes to.
        let cfg = with_replay(disagg_cfg(), 5560);
        let prefill = raw_worker_from_pod(&role_pod("p-0", "10.0.0.1", "prefill"), &pool(), &cfg)
            .expect("prefill should map");
        assert_eq!(
            prefill.kv_events_endpoint.as_deref(),
            Some("tcp://10.0.0.1:5557")
        );
        assert_eq!(
            prefill.replay_endpoint.as_deref(),
            Some("tcp://10.0.0.1:5560")
        );

        let decode = raw_worker_from_pod(&role_pod("d-0", "10.0.0.2", "decode"), &pool(), &cfg)
            .expect("decode should map");
        assert!(decode.kv_events_endpoint.is_none());
        assert!(decode.replay_endpoint.is_none());
    }

    #[test]
    fn missing_role_label_is_a_role_rejection() {
        let p = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        assert_eq!(
            raw_worker_from_pod(&p, &pool(), &disagg_cfg()),
            Err(PodRejection::Role(RoleLabelError::Missing))
        );
    }

    #[test]
    fn unparseable_role_label_is_a_role_rejection() {
        let p = role_pod("vllm-0", "10.0.0.1", "gibberish");
        let Err(PodRejection::Role(error)) = raw_worker_from_pod(&p, &pool(), &disagg_cfg()) else {
            panic!("an unparseable role must be a role rejection");
        };
        assert_eq!(error.reason(), "role_label_invalid");
    }

    #[test]
    fn ineligibility_outranks_role_resolution() {
        // The boundary that keeps a rolling update from reading as a
        // misconfiguration: a NotReady prefill pod is ineligible, not a role error.
        let not_ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(false),
            &[("app", "vllm-qwen"), (DEFAULT_WORKER_ROLE_LABEL, "prefill")],
        );
        assert_eq!(
            raw_worker_from_pod(&not_ready, &pool(), &disagg_cfg()),
            Err(PodRejection::Ineligible)
        );

        // Likewise for a pod the pool does not select, even with a valid role.
        let unselected = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "other"), (DEFAULT_WORKER_ROLE_LABEL, "prefill")],
        );
        assert_eq!(
            raw_worker_from_pod(&unselected, &pool(), &disagg_cfg()),
            Err(PodRejection::Ineligible)
        );
    }

    // --- role changes and counts -------------------------------------------

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

        // One in-place update, not a remove-then-add: the id is derived from the
        // pod name, so it cannot exist in two catalogs at once.
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

        // Label edited away: the worker must go, not linger serving traffic.
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

        // rebuild_index
        assert!(rebuild_index(&store, Some(&pool()), &disagg_cfg(), &index));
        {
            let index = index.read().unwrap();
            assert_eq!(index.counts(), recount(&index));
            assert_eq!(index.counts().prefill, 1);
            assert_eq!(index.counts().decode, 2);
        }

        // upsert_pod (add)
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

        // remove_pod
        assert!(remove_pod(&index, &role_pod("d-0", "10.0.0.2", "decode")));
        {
            let index = index.read().unwrap();
            assert_eq!(index.counts(), recount(&index));
            assert_eq!(index.counts().decode, 1);
        }

        // clear
        index.write().unwrap().clear();
        let index = index.read().unwrap();
        assert!(index.is_empty());
        assert_eq!(index.counts(), RoleCounts::default());
    }

    #[test]
    fn deleting_every_prefill_pod_leaves_the_decode_catalog_intact() {
        let index = RwLock::new(IndexState::default());
        let store = store_from_pods(vec![
            role_pod("p-0", "10.0.0.1", "prefill"),
            role_pod("d-0", "10.0.0.2", "decode"),
        ]);
        assert!(rebuild_index(&store, Some(&pool()), &disagg_cfg(), &index));

        assert!(remove_pod(&index, &role_pod("p-0", "10.0.0.1", "prefill")));
        let index = index.read().unwrap();
        assert_eq!(index.counts().prefill, 0);
        assert_eq!(index.counts().decode, 1);
    }

    // --- the invariant: a prefill worker is never a destination -------------

    fn role_discovery(workers: &[(&str, &str, WorkerRole)]) -> PodDiscovery {
        let raw = workers
            .iter()
            .map(|(name, ip, role)| RawWorker {
                worker_id: hash_pod_name(name),
                pod_name: (*name).to_string(),
                pod_ip: (*ip).to_string(),
                role: *role,
                http_endpoint: format!("http://{ip}:8000"),
                kv_events_endpoint: (*role != WorkerRole::Decode)
                    .then(|| format!("tcp://{ip}:5557")),
                replay_endpoint: None,
            })
            .collect();
        PodDiscovery::for_test(raw).0
    }

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
        // Even holding the id outright, it cannot be turned into a destination.
        assert!(
            discovery
                .resolve_endpoint(prefill_id, WorkerRole::Decode)
                .is_none()
        );

        // ...while the same catalog is fully visible to prefill-scoped reads.
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
    }

    #[test]
    fn ready_workers_by_role_partitions_one_snapshot() {
        let discovery = role_discovery(&[
            ("p-0", "10.0.0.1", WorkerRole::Prefill),
            ("d-0", "10.0.0.2", WorkerRole::Decode),
            ("d-1", "10.0.0.3", WorkerRole::Decode),
        ]);

        let sets = discovery.ready_workers_by_role();
        assert_eq!(sets.prefill.len(), 1);
        assert_eq!(sets.decode.len(), 2);
        assert!(sets.prefill.iter().all(|w| w.role == WorkerRole::Prefill));
        assert!(sets.decode.iter().all(|w| w.role == WorkerRole::Decode));
    }

    #[test]
    fn set_workers_replaces_the_catalog_and_recomputes_counts() {
        let discovery = role_discovery(&[("p-0", "10.0.0.1", WorkerRole::Prefill)]);
        assert_eq!(discovery.role_counts().prefill, 1);

        discovery.set_workers(vec![]);
        assert_eq!(discovery.role_counts(), RoleCounts::default());
        assert!(!discovery.has_ready_workers(WorkerRole::Prefill));
    }
}
