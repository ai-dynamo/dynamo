// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Each pool cluster's region, for pools that MCS lists: MCS names a pool's
//! cluster but not where the cluster is.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::time::Duration;

use dynamo_global_common::PoolLocation;
use futures::StreamExt;
use kube::api::{ApiResource, DynamicObject, GroupVersionKind};
use kube::runtime::watcher;
use kube::{Api, Client};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::cache::{WATCH_TIMEOUT_SECS, retried};

/// How long a list waits for the first read of Karmada's clusters, so that a
/// replica that just started does not list its pools under a stand-in region.
/// Once one wait runs out, lists stop waiting.
const FIRST_READ_WAIT: Duration = Duration::from_secs(5);

/// Where each pool cluster is, by the name MCS gives the cluster: with
/// Karmada, the `spec.region` of its member `Cluster`, then its configured
/// region, then the cluster's own name, as a stand-in that is logged once.
pub struct ClusterRegions {
    configured: HashMap<String, String>,
    karmada: Option<KarmadaRegions>,
    /// Clusters already logged as having no region.
    unplaced: Mutex<HashSet<String>>,
}

impl ClusterRegions {
    /// `karmada`, a client of the Karmada control plane, gives each member
    /// `Cluster`'s `spec.region`; the watch starts on first use and needs
    /// `get`, `list` and `watch` on `clusters.cluster.karmada.io`.
    /// `configured` maps a cluster's MCS name to its region otherwise.
    pub fn new(configured: HashMap<String, String>, karmada: Option<Client>) -> Self {
        Self {
            configured,
            karmada: karmada.map(KarmadaRegions::new),
            unplaced: Mutex::default(),
        }
    }

    /// Waits, up to [`FIRST_READ_WAIT`], until Karmada's clusters have been
    /// read once or failed to be.
    pub(super) async fn first_read(&self) {
        if let Some(karmada) = &self.karmada {
            karmada.first_read().await;
        }
    }

    pub(super) fn location(&self, cluster: &str) -> PoolLocation {
        let region = self
            .karmada
            .as_ref()
            .and_then(|karmada| karmada.region(cluster))
            .or_else(|| self.configured.get(cluster).cloned());
        let region = region.unwrap_or_else(|| {
            let mut unplaced = self.unplaced.lock().unwrap_or_else(PoisonError::into_inner);
            if unplaced.insert(cluster.to_owned()) {
                tracing::warn!(
                    cluster,
                    "no region is known for this pool cluster, so its pools are listed with the \
                     cluster's name as their region; set the region in the member Cluster's \
                     spec.region with Karmada, or with --cluster-region"
                );
            }
            cluster.to_owned()
        });
        PoolLocation {
            region,
            availability_zone: None,
            datacenter: None,
        }
    }
}

impl fmt::Debug for ClusterRegions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusterRegions")
            .field("configured", &self.configured)
            .field("karmada", &self.karmada.is_some())
            .finish_non_exhaustive()
    }
}

/// Each Karmada member `Cluster`'s `spec.region`, from one watch.
struct KarmadaRegions {
    clusters: Api<DynamicObject>,
    state: Arc<KarmadaState>,
    watching: OnceLock<()>,
    stop: CancellationToken,
}

struct KarmadaState {
    regions: RwLock<HashMap<String, String>>,
    /// Clusters read since the current relist began; they replace `regions`
    /// at `InitDone`.
    relist: Mutex<Option<HashMap<String, String>>>,
    /// The last watch error, so each new one is logged once.
    last_error: Mutex<Option<String>>,
    /// Whether the clusters have been read once, or failed to be.
    settled: watch::Sender<bool>,
    /// Lists no longer wait for the first read.
    gave_up_waiting: AtomicBool,
}

impl Default for KarmadaState {
    fn default() -> Self {
        Self {
            regions: RwLock::default(),
            relist: Mutex::default(),
            last_error: Mutex::default(),
            settled: watch::Sender::new(false),
            gave_up_waiting: AtomicBool::new(false),
        }
    }
}

impl KarmadaRegions {
    fn new(client: Client) -> Self {
        let gvk = GroupVersionKind::gvk("cluster.karmada.io", "v1alpha1", "Cluster");
        Self {
            clusters: Api::all_with(client, &ApiResource::from_gvk(&gvk)),
            state: Arc::new(KarmadaState::default()),
            watching: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    fn region(&self, cluster: &str) -> Option<String> {
        self.ensure_watching();
        self.state
            .regions
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(cluster)
            .cloned()
    }

    async fn first_read(&self) {
        self.ensure_watching();
        if self.state.gave_up_waiting.load(Ordering::Relaxed) {
            return;
        }
        let mut settled = self.state.settled.subscribe();
        let read = tokio::time::timeout(FIRST_READ_WAIT, settled.wait_for(|settled| *settled));
        if read.await.is_err() {
            self.state.gave_up_waiting.store(true, Ordering::Relaxed);
        }
    }

    /// Every caller runs in a Tokio runtime, so the spawn is safe.
    fn ensure_watching(&self) {
        self.watching.get_or_init(|| {
            tokio::spawn(watch_clusters(
                self.clusters.clone(),
                Arc::clone(&self.state),
                self.stop.clone(),
            ));
        });
    }
}

impl Drop for KarmadaRegions {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

impl KarmadaState {
    fn apply(&self, event: watcher::Event<DynamicObject>) {
        let mut relist = self.relist.lock().unwrap_or_else(PoisonError::into_inner);
        match event {
            // `retried` drops `Init`, so the first cluster starts a relist.
            watcher::Event::Init => {}
            watcher::Event::InitApply(cluster) => {
                if let Some((name, region)) = cluster_region(&cluster) {
                    relist.get_or_insert_with(HashMap::new).insert(name, region);
                }
            }
            watcher::Event::InitDone => {
                *self.regions.write().unwrap_or_else(PoisonError::into_inner) =
                    relist.take().unwrap_or_default();
                self.settle();
            }
            watcher::Event::Apply(cluster) => {
                let mut regions = self.regions.write().unwrap_or_else(PoisonError::into_inner);
                let name = cluster.metadata.name.clone().unwrap_or_default();
                match cluster_region(&cluster) {
                    Some((name, region)) => regions.insert(name, region),
                    None => regions.remove(&name),
                };
            }
            watcher::Event::Delete(cluster) => {
                if let Some(name) = &cluster.metadata.name {
                    self.regions
                        .write()
                        .unwrap_or_else(PoisonError::into_inner)
                        .remove(name);
                }
            }
        }
    }

    /// Keeps the regions last read; the watch retries with backoff.
    fn failed(&self, error: &watcher::Error) {
        *self.relist.lock().unwrap_or_else(PoisonError::into_inner) = None;
        let reason = error.to_string();
        let mut last = self
            .last_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if last.as_deref() != Some(reason.as_str()) {
            tracing::warn!(
                error = reason,
                "cannot read Karmada's Cluster objects for pool regions; regions come from \
                 --cluster-region or the cluster's name until they can be read"
            );
            *last = Some(reason);
        }
        self.settle();
    }

    fn settle(&self) {
        self.settled.send_replace(true);
    }
}

/// A member's name and region, or `None` without a usable `spec.region`.
fn cluster_region(cluster: &DynamicObject) -> Option<(String, String)> {
    let name = cluster.metadata.name.clone()?;
    let region = cluster.data.get("spec")?.get("region")?.as_str()?;
    let usable =
        !region.is_empty() && region.trim() == region && !region.chars().any(char::is_control);
    usable.then(|| (name, region.to_owned()))
}

/// Single consumer of the `Cluster` watch, so events apply in order.
async fn watch_clusters(
    clusters: Api<DynamicObject>,
    state: Arc<KarmadaState>,
    stop: CancellationToken,
) {
    // As the EndpointSlice watch: lists from the API server's cache, and a
    // server that stops answering fails within the client's read timeout.
    let config = watcher::Config {
        page_size: None,
        ..watcher::Config::default()
            .any_semantic()
            .timeout(WATCH_TIMEOUT_SECS)
    };
    let mut events = std::pin::pin!(retried(watcher(clusters, config)));
    loop {
        let event = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            event = events.next() => event,
        };
        match event {
            Some(Ok(event)) => {
                if state
                    .last_error
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .is_some()
                {
                    tracing::info!("reading Karmada's Cluster objects for pool regions again");
                }
                state.apply(event);
            }
            Some(Err(error)) => state.failed(&error),
            None => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn cluster(name: &str, spec: serde_json::Value) -> DynamicObject {
        serde_json::from_value(json!({
            "apiVersion": "cluster.karmada.io/v1alpha1",
            "kind": "Cluster",
            "metadata": {"name": name},
            "spec": spec,
        }))
        .unwrap()
    }

    fn region(state: &KarmadaState, name: &str) -> Option<String> {
        state.regions.read().unwrap().get(name).cloned()
    }

    #[test]
    fn a_relist_replaces_the_regions_and_events_update_them() {
        let state = KarmadaState::default();
        state.apply(watcher::Event::InitApply(cluster(
            "east",
            json!({"region": "us-east-2"}),
        )));
        // Unusable regions are left out.
        for (name, spec) in [
            ("no-region", json!({})),
            ("blank", json!({"region": " "})),
            ("padded", json!({"region": " us-west-2"})),
            ("number", json!({"region": 2})),
        ] {
            state.apply(watcher::Event::InitApply(cluster(name, spec)));
        }
        // Nothing changes until the relist is done.
        assert_eq!(region(&state, "east"), None);
        state.apply(watcher::Event::InitDone);
        assert_eq!(region(&state, "east").as_deref(), Some("us-east-2"));
        assert_eq!(state.regions.read().unwrap().len(), 1);

        state.apply(watcher::Event::Apply(cluster(
            "east",
            json!({"region": "us-east-1"}),
        )));
        assert_eq!(region(&state, "east").as_deref(), Some("us-east-1"));
        // A region that is cleared or becomes unusable is forgotten.
        state.apply(watcher::Event::Apply(cluster("east", json!({}))));
        assert_eq!(region(&state, "east"), None);
        state.apply(watcher::Event::Apply(cluster(
            "west",
            json!({"region": "us-west-2"}),
        )));
        state.apply(watcher::Event::Delete(cluster("west", json!({}))));
        assert_eq!(region(&state, "west"), None);

        // A relist drops clusters that are gone, and a failed one changes
        // nothing.
        state.apply(watcher::Event::Apply(cluster(
            "gone",
            json!({"region": "eu-west-1"}),
        )));
        state.apply(watcher::Event::InitApply(cluster(
            "east",
            json!({"region": "us-east-2"}),
        )));
        state.failed(&watcher::Error::NoResourceVersion);
        state.apply(watcher::Event::InitDone);
        assert_eq!(state.regions.read().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn a_configured_region_or_the_name_places_a_cluster() {
        let regions = ClusterRegions::new(
            HashMap::from([("east".to_owned(), "us-east-2".to_owned())]),
            None,
        );
        regions.first_read().await;
        assert_eq!(regions.location("east").region, "us-east-2");
        assert_eq!(regions.location("west").region, "west");
        assert_eq!(regions.location("west").availability_zone, None);
    }
}
