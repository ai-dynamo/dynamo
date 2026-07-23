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

/// A discovered, `Ready` raw inference engine worker normalized for selector registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawWorker {
    /// Stable hash of the pod name; the selector catalog key.
    pub worker_id: u64,
    /// Kubernetes pod name.
    pub pod_name: String,
    /// Pod IP.
    pub pod_ip: String,
    /// OpenAI HTTP inference endpoint, `http://<ip>:<target_port>`.
    pub http_endpoint: String,
    /// Inference engine KV-event ZMQ PUB endpoint, `tcp://<ip>:<kv_event_port>`.
    pub kv_events_endpoint: String,
    /// Optional ZMQ REQ endpoint for live-stream gap replay.
    pub replay_endpoint: Option<String>,
}

/// One indexed worker: the materialized [`RawWorker`] for selector registration
/// plus its pre-stripped `ip:port`, so request-path reads (notably
/// [`PodDiscovery::resolve_endpoint`]) are O(1) lookups that never re-parse the
/// endpoint or clone the [`PoolState`].
#[derive(Debug, Clone)]
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

/// Provides an index of `Ready` workers selected by the EPP's `InferencePool`.
#[derive(Clone)]
pub struct PodDiscovery {
    index: Arc<RwLock<WorkerIndex>>,
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

        let kv_event_port = cfg.kv_event_port;
        let replay_port = cfg.replay_port;

        let index: Arc<RwLock<WorkerIndex>> = Arc::new(RwLock::new(WorkerIndex::new()));

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

                match delta {
                    Delta::Stop => break,
                    Delta::Skip => continue,
                    Delta::Rebuild => rebuild_index(
                        &store,
                        pool_rx.borrow().as_ref(),
                        kv_event_port,
                        replay_port,
                        &index_task,
                    ),
                    Delta::Upsert(pod) => upsert_pod(
                        &index_task,
                        &pod,
                        pool_rx.borrow().as_ref(),
                        kv_event_port,
                        replay_port,
                    ),
                    Delta::Remove(pod) => remove_pod(&index_task, &pod),
                }

                // Readiness based on initial pods being synced and the pool being resolved.
                ready_task.store(pod_synced && pool_rx.borrow().is_some(), Ordering::Release);
                generation = generation.wrapping_add(1);
                let _ = changes_tx.send(generation);
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

    // Return all currently `Ready` workers selected by the pool.
    pub fn ready_workers(&self) -> Vec<RawWorker> {
        self.index
            .read()
            .unwrap()
            .values()
            .map(|entry| entry.worker.clone())
            .collect()
    }

    // Return the IDs of all currently `Ready`, pool-selected workers.
    pub fn ready_worker_ids(&self) -> HashSet<u64> {
        self.index.read().unwrap().keys().copied().collect()
    }

    // Resolve a `worker_id` to its current `ip:port` HTTP endpoint.
    pub fn resolve_endpoint(&self, worker_id: u64) -> Option<String> {
        self.index
            .read()
            .unwrap()
            .get(&worker_id)
            .map(|entry| entry.endpoint.clone())
    }

    /// Retain the `worker_ids` whose current `ip:port` endpoint satisfies `pred`,
    /// under a **single** read lock and **without cloning** any endpoint (`pred`
    /// borrows it). Used on the subset-routing path so membership testing doesn't
    /// allocate a throwaway `String` per candidate. Unknown workers are dropped.
    pub fn filter_workers_by_endpoint(
        &self,
        worker_ids: &HashSet<u64>,
        pred: impl Fn(&str) -> bool,
    ) -> HashSet<u64> {
        let index = self.index.read().unwrap();
        worker_ids
            .iter()
            .copied()
            .filter(|worker_id| {
                index
                    .get(worker_id)
                    .is_some_and(|entry| pred(entry.endpoint.as_str()))
            })
            .collect()
    }

    pub fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes.clone()
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
/// terminating, or unselected).
fn upsert_pod(
    index: &RwLock<WorkerIndex>,
    pod: &Pod,
    pool: Option<&PoolState>,
    kv_event_port: u16,
    replay_port: Option<u16>,
) {
    let Some(worker_id) = pod_worker_id(pod) else {
        return;
    };
    let entry = pool
        .and_then(|pool| raw_worker_from_pod(pod, pool, kv_event_port, replay_port))
        .map(WorkerEntry::from_raw);
    let mut index = index.write().unwrap();
    match entry {
        Some(entry) => {
            index.insert(worker_id, entry);
        }
        None => {
            index.remove(&worker_id);
        }
    }
}

/// Drop a deleted pod's worker from the index. O(1).
fn remove_pod(index: &RwLock<WorkerIndex>, pod: &Pod) {
    if let Some(worker_id) = pod_worker_id(pod) {
        index.write().unwrap().remove(&worker_id);
    }
}

/// Recompute the whole index from the current pod store and pool selector.
/// Empty until the `InferencePool` has resolved. Used only for the initial LIST,
/// watch relists (which may have dropped pods without a `Delete`), and
/// pool-selector changes (which re-classify every pod).
fn rebuild_index(
    store: &kube::runtime::reflector::Store<Pod>,
    pool: Option<&PoolState>,
    kv_event_port: u16,
    replay_port: Option<u16>,
    index: &RwLock<WorkerIndex>,
) {
    let mut fresh = WorkerIndex::new();
    if let Some(pool) = pool {
        for pod in store.state().iter() {
            if let Some(worker) = raw_worker_from_pod(pod, pool, kv_event_port, replay_port) {
                fresh.insert(worker.worker_id, WorkerEntry::from_raw(worker));
            }
        }
    }
    *index.write().unwrap() = fresh;
}

/// Build a [`RawWorker`] from a pod, or `None` if it is not `Ready`, not
/// pool-selected, or lacks an IP/name. Pure function — unit-testable.
fn raw_worker_from_pod(
    pod: &Pod,
    pool: &PoolState,
    kv_event_port: u16,
    replay_port: Option<u16>,
) -> Option<RawWorker> {
    if !pod_is_ready(pod) || !pod_matches(pod, &pool.match_labels) {
        return None;
    }
    let pod_name = pod.metadata.name.as_deref()?;
    let pod_ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    let ip: IpAddr = pod_ip.parse().ok()?;

    Some(RawWorker {
        worker_id: hash_pod_name(pod_name),
        pod_name: pod_name.to_string(),
        pod_ip: pod_ip.to_string(),
        http_endpoint: format!("http://{}", SocketAddr::new(ip, pool.target_port)),
        kv_events_endpoint: format!("tcp://{}", SocketAddr::new(ip, kv_event_port)),
        replay_endpoint: replay_port.map(|p| format!("tcp://{}", SocketAddr::new(ip, p))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
            5557,
            Some(5560),
        )
        .expect("ready, selected pod should map");
        assert_eq!(w.worker_id, hash_pod_name("vllm-0"));
        assert_eq!(w.http_endpoint, "http://10.0.0.1:8000");
        assert_eq!(w.kv_events_endpoint, "tcp://10.0.0.1:5557");
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
            5557,
            Some(5560),
        )
        .expect("ready, selected IPv6 pod should map");
        // SocketAddr brackets the IPv6 host so host and port are unambiguous.
        assert_eq!(w.http_endpoint, "http://[fd00::10]:8000");
        assert_eq!(w.kv_events_endpoint, "tcp://[fd00::10]:5557");
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
                5557,
                None,
            )
            .is_none()
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
                5557,
                None,
            )
            .is_none()
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
                5557,
                None,
            )
            .is_none()
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
        assert!(raw_worker_from_pod(&p, &pool(), 5557, None).is_none());
    }

    #[test]
    fn pod_without_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod("vllm-0", None, Some(true), &[("app", "vllm-qwen")]),
                &pool(),
                5557,
                None,
            )
            .is_none()
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

        let index = RwLock::new(WorkerIndex::new());
        rebuild_index(&store, Some(&pool()), 5557, Some(5560), &index);

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
        let index = RwLock::new(WorkerIndex::new());
        rebuild_index(&store, None, 5557, None, &index);
        assert!(index.read().unwrap().is_empty());
    }

    #[test]
    fn upsert_and_remove_pod_mutate_index_incrementally() {
        let index = RwLock::new(WorkerIndex::new());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );

        // Ready + selected -> inserted, with a pre-stripped endpoint.
        upsert_pod(&index, &ready, Some(&pool()), 5557, None);
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
        upsert_pod(&index, &not_ready, Some(&pool()), 5557, None);
        assert!(!index.read().unwrap().contains_key(&id));

        // Re-add, then a Delete removes it.
        upsert_pod(&index, &ready, Some(&pool()), 5557, None);
        assert!(index.read().unwrap().contains_key(&id));
        remove_pod(&index, &ready);
        assert!(!index.read().unwrap().contains_key(&id));
    }

    #[test]
    fn upsert_pod_without_pool_drops_entry() {
        // A `None` pool (unresolved or deleted) means nothing is routable, so an
        // upsert must evict any existing entry rather than leave stale routing.
        let index = RwLock::new(WorkerIndex::new());
        let id = hash_pod_name("vllm-0");
        let ready = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );

        upsert_pod(&index, &ready, Some(&pool()), 5557, None);
        assert!(index.read().unwrap().contains_key(&id));

        upsert_pod(&index, &ready, None, 5557, None);
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
        let index = RwLock::new(WorkerIndex::new());
        rebuild_index(&store, Some(&pool()), 5557, None, &index);
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
        rebuild_index(&store, Some(&updated_pool), 5557, None, &index);
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
        let index = RwLock::new(WorkerIndex::new());
        upsert_pod(
            &index,
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            Some(&pool()),
            5557,
            None,
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
            5557,
            None,
        );
        assert_eq!(index.read().unwrap().len(), 2);

        // Relist: vllm-1 vanished during the gap; only vllm-0 remains.
        let store = store_from_pods(vec![pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        )]);
        rebuild_index(&store, Some(&pool()), 5557, None, &index);

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
            http_endpoint: format!("http://{endpoint}"),
            kv_events_endpoint: format!("tcp://{endpoint}"),
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
            index: Arc::new(RwLock::new(index)),
            changes,
        }
    }

    #[test]
    fn filter_workers_by_endpoint_matches_without_cloning() {
        let discovery = discovery_with_endpoints(HashMap::from([
            (1u64, "10.0.0.1:8000".to_string()),
            (2u64, "10.0.0.2:8000".to_string()),
            (3u64, "10.0.0.3:8000".to_string()),
        ]));
        let allowed: HashSet<u64> = [1, 2, 3].into_iter().collect();

        // Predicate borrows the endpoint; only worker 2 matches.
        let filtered =
            discovery.filter_workers_by_endpoint(&allowed, |endpoint| endpoint == "10.0.0.2:8000");
        assert_eq!(filtered, HashSet::from([2]));

        // A worker id with no endpoint in the snapshot is dropped.
        let allowed_with_unknown: HashSet<u64> = [1, 99].into_iter().collect();
        let filtered = discovery.filter_workers_by_endpoint(&allowed_with_unknown, |_| true);
        assert_eq!(filtered, HashSet::from([1]));
    }
}
