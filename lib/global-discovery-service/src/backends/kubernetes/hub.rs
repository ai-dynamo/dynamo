// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools from imported EndpointSlices: the Global Discovery Service's MCS
//! backend.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError};

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, PoolEvent, PoolListing, Snapshot, WatchEvent,
    WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::{PoolKey, PoolLocation, address_identity};
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::cache::{SliceDirectory, SliceIndex, SliceKey, ready_endpoints, tcp_port};
use super::mcs::{self, McsImplementation};
use super::regions::ClusterRegions;

/// One pool per (source cluster, namespace, DGD name) from imported
/// EndpointSlices, which MCS merges across clusters. One watch, started on
/// first use, feeds a cache that serves `list` and every watcher; lists are
/// `Exact` unless it keeps failing. [`ClusterRegions`] places each pool as it
/// is read, so a watcher sees a new region with the pool's next event.
#[derive(Debug)]
pub struct McsPoolDirectory {
    slices: SliceDirectory<PoolIndex>,
    regions: Arc<ClusterRegions>,
}

/// The Service in front of a pool's frontends, named `<DGD name><suffix>`
/// and exported like the PoolRelay Service. Its Ready endpoints on the named
/// TCP port become the pool's `frontends`, in one address family, IPv4 on a
/// dual-stack cluster: addresses that other clusters can reach, kept current
/// as frontend pods come and go.
#[derive(Clone, Debug)]
pub struct FrontendService {
    pub suffix: String,
    pub port_name: String,
}

impl FrontendService {
    /// A Service name must not match both suffixes, so neither may end with
    /// the other.
    pub fn check(&self, pool_service_suffix: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.suffix.is_empty()
                && !pool_service_suffix.is_empty()
                && !self.suffix.ends_with(pool_service_suffix)
                && !pool_service_suffix.ends_with(self.suffix.as_str()),
            "the frontend Service suffix {:?} and the PoolRelay Service suffix {:?} must differ, \
             and neither may end with the other",
            self.suffix,
            pool_service_suffix,
        );
        Ok(())
    }
}

impl McsPoolDirectory {
    /// `imports` reaches the imported slices: the hub with standard MCS; with
    /// Karmada, its control plane, or a member cluster with a `ServiceImport`
    /// per pool. Each PoolRelay Service is `<DGD name><pool_service_suffix>`.
    /// The client's connect and read timeouts should add up to under 90 s, so
    /// a server that stops answering shows up as failures.
    pub fn new(
        imports: Client,
        mcs: McsImplementation,
        pool_service_suffix: String,
        frontends: FrontendService,
        regions: ClusterRegions,
    ) -> anyhow::Result<Self> {
        frontends.check(&pool_service_suffix)?;
        Ok(Self {
            slices: SliceDirectory::new(
                Api::all(imports),
                mcs::imported_selector(mcs),
                PoolNaming {
                    mcs,
                    pool_service_suffix,
                    frontends,
                    portless: Mutex::default(),
                    frontendless: Mutex::default(),
                },
            ),
            regions: Arc::new(regions),
        })
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for McsPoolDirectory {
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        let mut snapshot = self.slices.list().await?;
        self.regions.first_read().await;
        self.regions.locate(&mut snapshot.items);
        Ok(snapshot)
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        let events = self.slices.list_and_watch(cancel_token).await?;
        self.regions.first_read().await;
        let regions = Arc::clone(&self.regions);
        Ok(Box::pin(events.map(move |event| {
            event.map(|event| match event {
                WatchEvent::Upserted(mut listing) => {
                    regions.locate(std::slice::from_mut(&mut listing));
                    WatchEvent::Upserted(listing)
                }
                WatchEvent::Resync(mut snapshot) => {
                    regions.locate(&mut snapshot.items);
                    WatchEvent::Resync(snapshot)
                }
                removed @ WatchEvent::Removed(_) => removed,
            })
        })))
    }
}

/// Turns imported slices into pool listings.
struct PoolNaming {
    mcs: McsImplementation,
    pool_service_suffix: String,
    frontends: FrontendService,
    /// Frontend slices without the named port, so each is warned about once.
    portless: Mutex<HashSet<String>>,
    /// Live pools without a frontend, so each is warned about once until it
    /// has a frontend again or stops being live. Kept here rather than in
    /// the index, since each read before the first sync builds a new index.
    frontendless: Mutex<HashSet<PoolKey>>,
}

impl PoolNaming {
    /// True when the live pool `key` has no frontend and was not already
    /// warned about.
    fn newly_frontendless(&self, key: &PoolKey, frontends: &[Url]) -> bool {
        let mut warned = self
            .frontendless
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !frontends.is_empty() {
            warned.remove(key);
            return false;
        }
        !warned.contains(key) && warned.insert(key.clone())
    }

    /// The pool `key` stopped being live.
    fn forget_frontendless(&self, key: &PoolKey) {
        self.frontendless
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(key);
    }

    /// Source cluster from [`mcs::source_cluster`]; DGD name from the
    /// Service name without `suffix`. `None` for imports of other Services,
    /// and for a slice from an unknown cluster, which is logged.
    fn pool_key(&self, slice: &EndpointSlice, suffix: &str) -> Option<PoolKey> {
        let service = mcs::service_name(self.mcs, slice)?;
        let dgd_name = service
            .strip_suffix(suffix)
            .filter(|name| !name.is_empty())?;
        let Some(cluster) = mcs::source_cluster(self.mcs, slice) else {
            mcs::warn_unknown_cluster(slice, service);
            return None;
        };
        let namespace = slice.metadata.namespace.as_deref()?;
        PoolKey::new(cluster, namespace, dgd_name).ok()
    }

    /// A PoolRelay slice, or a frontend slice with the named port.
    fn slice(&self, slice: &EndpointSlice) -> Option<SliceEntry> {
        if let Some(pool) = self.pool_key(slice, &self.pool_service_suffix) {
            return Some(SliceEntry {
                pool,
                ready: ready_addresses(slice),
                frontend_port: None,
                ipv4: slice.address_type == "IPv4",
                region: mcs::source_region(slice).and_then(PoolLocation::in_region),
            });
        }
        let pool = self.pool_key(slice, &self.frontends.suffix)?;
        let Some(port) = tcp_port(slice, &self.frontends.port_name) else {
            // The port name is misconfigured, or another Service's name ends
            // with the suffix. Every slice event would repeat the warning.
            let name = slice.metadata.name.clone().unwrap_or_default();
            let mut portless = self.portless.lock().unwrap_or_else(PoisonError::into_inner);
            if portless.insert(name.clone()) {
                tracing::warn!(
                    slice = name,
                    port_name = %self.frontends.port_name,
                    "frontend EndpointSlice has no TCP port of that name; its frontends are not listed"
                );
            }
            return None;
        };
        Some(SliceEntry {
            pool,
            ready: ready_addresses(slice),
            frontend_port: Some(port),
            ipv4: slice.address_type == "IPv4",
            region: mcs::source_region(slice).and_then(PoolLocation::in_region),
        })
    }
}

/// A location with an empty region, which only marks a pool whose slices name
/// no region until [`ClusterRegions`] places its cluster.
fn unplaced() -> PoolLocation {
    PoolLocation {
        region: String::new(),
        availability_zone: None,
        datacenter: None,
    }
}

/// Each Ready address in a slice. A dual-stack pod is in two slices, one per
/// address family.
fn ready_addresses(slice: &EndpointSlice) -> Vec<IpAddr> {
    ready_endpoints(slice)
        .flat_map(|endpoint| &endpoint.addresses)
        .filter_map(|address| address.parse().ok())
        .collect()
}

/// Formatted through `SocketAddr`, so IPv6 gets brackets.
fn frontend_url(address: SocketAddr) -> Option<Url> {
    Url::parse(&format!("http://{address}/")).ok()
}

struct SliceEntry {
    pool: PoolKey,
    /// Ready addresses.
    ready: Vec<IpAddr>,
    /// `Some` for a frontend slice: the port its endpoints serve HTTP on.
    frontend_port: Option<u16>,
    /// The slice holds IPv4 addresses.
    ipv4: bool,
    /// The source cluster's region, when the slice names a valid one.
    region: Option<PoolLocation>,
}

#[derive(Default)]
struct PoolState {
    slices: HashSet<SliceKey>,
    /// `Some` while the pool is live.
    listing: Option<PoolListing>,
}

#[derive(Default)]
struct PoolIndex {
    slices: HashMap<SliceKey, SliceEntry>,
    pools: HashMap<PoolKey, PoolState>,
}

impl SliceIndex for PoolIndex {
    type Item = PoolListing;
    type Key = PoolKey;
    type Rules = PoolNaming;
    type Entry = SliceEntry;

    fn entry(naming: &PoolNaming, slice: &EndpointSlice) -> Option<SliceEntry> {
        naming.slice(slice)
    }

    /// One slice changed or went away. Touches only that slice's pool.
    fn upsert(
        &mut self,
        naming: &PoolNaming,
        key: SliceKey,
        entry: Option<SliceEntry>,
    ) -> Vec<PoolEvent> {
        let old_pool = self.slices.remove(&key).map(|old| old.pool);
        let new_pool = entry.as_ref().map(|entry| entry.pool.clone());
        if let Some(entry) = entry {
            self.slices.insert(key.clone(), entry);
        }
        let mut events = Vec::new();
        if let Some(pool) = old_pool.filter(|pool| new_pool.as_ref() != Some(pool)) {
            if let Some(state) = self.pools.get_mut(&pool) {
                state.slices.remove(&key);
            }
            events.extend(self.rebuild(naming, &pool));
        }
        if let Some(pool) = new_pool {
            self.pools
                .entry(pool.clone())
                .or_default()
                .slices
                .insert(key);
            events.extend(self.rebuild(naming, &pool));
        }
        events
    }

    /// A relist: every slice at once. Rebuilds every pool; the cache
    /// publishes one `Resync` instead of per-pool events.
    fn replace(&mut self, naming: &PoolNaming, slices: HashMap<SliceKey, SliceEntry>) {
        self.slices = slices;
        for pool in self.pools.values_mut() {
            pool.slices.clear();
        }
        for (key, entry) in &self.slices {
            self.pools
                .entry(entry.pool.clone())
                .or_default()
                .slices
                .insert(key.clone());
        }
        let pools: Vec<PoolKey> = self.pools.keys().cloned().collect();
        for pool in &pools {
            self.rebuild(naming, pool);
        }
    }

    fn snapshot(&self) -> Snapshot<PoolListing> {
        Snapshot {
            items: self
                .pools
                .values()
                .filter_map(|pool| pool.listing.clone())
                .collect(),
            completeness: Completeness::Exact,
        }
    }
}

impl PoolIndex {
    /// Recomputes one pool from its slices. Returns the event to publish if
    /// the listing changed.
    fn rebuild(&mut self, naming: &PoolNaming, key: &PoolKey) -> Option<PoolEvent> {
        let state = self.pools.get(key)?;
        // Every Ready PoolRelay pod streams the pool's state: the operator runs
        // active-active replicas, each reports to every router, and routers
        // dedupe the overlapping reports. So the listing names every Ready
        // PoolRelay address. They are sorted, so every service instance lists
        // the same identities, and a rolling update lists the old and the new
        // pods while both are Ready. An IPv4-mapped IPv6 address counts as its
        // IPv4 address, as `address_identity` writes it. A dual-stack pod is
        // in two slices, one per address family.
        let mut relays: BTreeSet<IpAddr> = BTreeSet::new();
        let mut frontends: BTreeSet<SocketAddr> = BTreeSet::new();
        for entry in state
            .slices
            .iter()
            .filter_map(|slice| self.slices.get(slice))
        {
            for address in &entry.ready {
                if let Some(port) = entry.frontend_port {
                    frontends.insert(SocketAddr::new(*address, port));
                } else {
                    relays.insert(address.to_canonical());
                }
            }
        }
        // One address family, so a dual-stack frontend is listed once: IPv4
        // when the pool has an IPv4 frontend slice, Ready endpoints or not, so
        // that the family does not follow which slice was imported or became
        // Ready first. Slices need not name their pods, so a pod's two
        // addresses cannot always be matched.
        let ipv4 = state
            .slices
            .iter()
            .filter_map(|slice| self.slices.get(slice))
            .any(|entry| entry.frontend_port.is_some() && entry.ipv4);
        let mut frontends: Vec<Url> = frontends
            .into_iter()
            .filter(|address| address.is_ipv4() == ipv4)
            .filter_map(frontend_url)
            .collect();
        frontends.sort_unstable();

        // A pool is live while a PoolRelay endpoint is Ready. Ready can be
        // stale when a cluster loses contact with the MCS implementation, so
        // the runtime also drops a pool whose Relay connection is gone.
        let state = self.pools.get_mut(key)?;
        if relays.is_empty() {
            naming.forget_frontendless(key);
            let was_live = state.listing.take().is_some();
            if state.slices.is_empty() {
                self.pools.remove(key);
            }
            return was_live.then(|| WatchEvent::Removed(key.clone()));
        }
        // A region that a slice names comes first, from the first such slice in
        // key order, so every service instance lists the same one.
        let location = state
            .slices
            .iter()
            .filter_map(|slice| Some((slice, self.slices.get(slice)?.region.as_ref()?)))
            .min_by_key(|(slice, _)| *slice)
            .map_or_else(unplaced, |(_, region)| region.clone());
        let relay_identities = relays.into_iter().map(address_identity).collect();
        let listing = PoolListing {
            key: key.clone(),
            location,
            frontends,
            relay_identities,
        };
        if state.listing.as_ref() == Some(&listing) {
            return None;
        }
        if naming.newly_frontendless(key, &listing.frontends) {
            // Routers have nowhere to send the pool's requests. Usually its
            // frontend Service is not exported, or its frontends are not
            // Ready yet.
            let with_port = state
                .slices
                .iter()
                .filter_map(|slice| self.slices.get(slice))
                .any(|entry| entry.frontend_port.is_some());
            tracing::warn!(
                pool = ?key,
                frontend_service = %format!("{}{}", key.dgd_name(), naming.frontends.suffix),
                port_name = %naming.frontends.port_name,
                frontend_slices_with_port = with_port,
                "this pool is live but has no Ready frontend, so it is listed without frontends; \
                 check that its frontend Service is exported and its frontends are Ready"
            );
        }
        state.listing = Some(listing.clone());
        Some(WatchEvent::Upserted(listing))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use futures::{FutureExt, StreamExt};
    use std::time::Duration;

    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use kube::runtime::watcher;

    use super::super::cache::{RECOVERED_AFTER, STALE_AFTER, SliceCache, follow, slice_key};
    use super::mcs::{
        KARMADA_SLICE_MANAGER, KUBERNETES_SERVICE_NAME_LABEL, MANAGED_BY_LABEL, REGION_LABEL,
    };
    use super::*;

    const SUFFIX: &str = "-pool-relay";

    /// A Kubernetes API that lists `slices` and Karmada's `clusters`, or
    /// answers 404 for them without `clusters`, and holds every watch open.
    /// Also returns how many lists of clusters it has answered.
    async fn fake_api(
        slices: &[EndpointSlice],
        clusters: Option<serde_json::Value>,
    ) -> (Client, Arc<std::sync::atomic::AtomicUsize>) {
        let cluster_lists = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&cluster_lists);
        let lists = [
            serde_json::json!({
                "apiVersion": "discovery.k8s.io/v1", "kind": "EndpointSliceList",
                "metadata": {"resourceVersion": "1"}, "items": slices,
            }),
            serde_json::json!({
                "apiVersion": "cluster.karmada.io/v1alpha1", "kind": "ClusterList",
                "metadata": {"resourceVersion": "1"}, "items": clusters,
            }),
        ];
        let app = axum::Router::new().fallback(move |request: axum::extract::Request| {
            let [slices, clusters] = lists.clone();
            let counted = Arc::clone(&counted);
            async move {
                let query = request.uri().query().unwrap_or_default();
                if query.contains("watch=true") {
                    std::future::pending::<()>().await;
                }
                let karmada = request
                    .uri()
                    .path()
                    .starts_with("/apis/cluster.karmada.io/");
                if karmada {
                    counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                match (karmada, clusters["items"].is_null()) {
                    (true, true) => {
                        let status = serde_json::json!({
                            "kind": "Status", "apiVersion": "v1", "status": "Failure",
                            "reason": "NotFound", "code": 404,
                        });
                        (axum::http::StatusCode::NOT_FOUND, axum::Json(status))
                    }
                    (true, false) => (axum::http::StatusCode::OK, axum::Json(clusters)),
                    (false, _) => (axum::http::StatusCode::OK, axum::Json(slices)),
                }
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        let client = Client::try_from(kube::Config::new(url.parse().unwrap())).unwrap();
        (client, cluster_lists)
    }

    /// Lists pools from clusters `east`, `west` and `north` through
    /// `clusters`, with `--cluster-region` for `east` and `west`, and
    /// returns each cluster's listed region.
    async fn listed_regions(clusters: Option<serde_json::Value>) -> HashMap<String, String> {
        let slices = ["east", "west", "north"]
            .map(|cluster| slice(cluster, "mocker", "a1b2c", &[("10.43.0.1", true)]));
        let (client, _) = fake_api(&slices, clusters).await;
        let configured = [("east", "eu-west-9"), ("west", "us-west-2")].map(|(cluster, region)| {
            (cluster.to_owned(), PoolLocation::in_region(region).unwrap())
        });
        let regions = ClusterRegions::new(HashMap::from(configured), Some(client.clone()));
        let pools = McsPoolDirectory::new(
            client,
            McsImplementation::Karmada,
            SUFFIX.to_owned(),
            naming().frontends,
            regions,
        )
        .unwrap();
        let listed = tokio::time::timeout(Duration::from_secs(2), pools.list())
            .await
            .expect("a list waited for regions that cannot be read")
            .unwrap();
        listed
            .items
            .into_iter()
            .map(|pool| (pool.key.cluster_id().to_owned(), pool.location.region))
            .collect()
    }

    #[tokio::test]
    async fn a_listed_pool_takes_its_clusters_region() {
        let cluster = |name: &str, region: &str| serde_json::json!({"metadata": {"name": name}, "spec": {"region": region}});
        let regions = listed_regions(Some(serde_json::json!([cluster("east", "us-east-2")]))).await;
        // Karmada wins over the flag, which places the clusters Karmada
        // leaves out; a cluster neither places is named after itself.
        assert_eq!(regions["east"], "us-east-2");
        assert_eq!(regions["west"], "us-west-2");
        assert_eq!(regions["north"], "north");

        // A member cluster serves no Cluster objects: the flag places what it
        // can, and the list does not wait.
        let regions = listed_regions(None).await;
        assert_eq!(regions["east"], "eu-west-9");
        assert_eq!(regions["west"], "us-west-2");
        assert_eq!(regions["north"], "north");
    }

    #[tokio::test]
    async fn the_region_watch_stops_where_no_cluster_objects_are_served() {
        let (client, cluster_lists) = fake_api(&[], None).await;
        let regions = ClusterRegions::new(HashMap::new(), Some(client));
        regions.first_read().await;
        // kube's backoff would list again within about a second.
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert_eq!(cluster_lists.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    fn naming() -> PoolNaming {
        PoolNaming {
            mcs: McsImplementation::Karmada,
            pool_service_suffix: SUFFIX.to_owned(),
            frontends: FrontendService {
                suffix: "-frontend".to_owned(),
                port_name: "http".to_owned(),
            },
            portless: Mutex::default(),
            frontendless: Mutex::default(),
        }
    }

    /// A Karmada-imported slice of `<dgd>-frontend` from `cluster`, serving
    /// HTTP on the port named `http`.
    fn frontend_slice(cluster: &str, dgd: &str, endpoints: &[(&str, bool)]) -> EndpointSlice {
        let service = format!("{dgd}-frontend");
        let mut slice = slice(cluster, dgd, "f1", endpoints);
        slice.metadata.name = Some(format!("imported-{cluster}-{service}-f1"));
        slice.metadata.labels.as_mut().unwrap().insert(
            KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
            format!("derived-{service}"),
        );
        slice.ports = Some(vec![EndpointPort {
            name: Some("http".to_owned()),
            port: Some(8000),
            protocol: Some("TCP".to_owned()),
            ..Default::default()
        }]);
        slice
    }

    /// A Karmada-imported slice of `<dgd>-pool-relay` from `cluster`.
    fn slice(cluster: &str, dgd: &str, suffix: &str, endpoints: &[(&str, bool)]) -> EndpointSlice {
        let service = format!("{dgd}{SUFFIX}");
        EndpointSlice {
            address_type: "IPv4".to_owned(),
            metadata: ObjectMeta {
                name: Some(format!("imported-{cluster}-{service}-{suffix}")),
                namespace: Some("dynamo".to_owned()),
                labels: Some(BTreeMap::from([
                    (
                        MANAGED_BY_LABEL.to_owned(),
                        KARMADA_SLICE_MANAGER.to_owned(),
                    ),
                    (
                        KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
                        format!("derived-{service}"),
                    ),
                ])),
                ..Default::default()
            },
            endpoints: endpoints
                .iter()
                .map(|(address, ready)| Endpoint {
                    addresses: vec![(*address).to_owned()],
                    conditions: Some(EndpointConditions {
                        ready: Some(*ready),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ports: None,
        }
    }

    /// Like [`slice`], with each endpoint's pod named.
    fn pod_slice(
        cluster: &str,
        suffix: &str,
        address_type: &str,
        endpoints: &[(&str, &str)],
    ) -> EndpointSlice {
        let mut slice = slice(cluster, "mocker", suffix, &[]);
        slice.address_type = address_type.to_owned();
        slice.endpoints = endpoints
            .iter()
            .map(|(pod, address)| Endpoint {
                addresses: vec![(*address).to_owned()],
                target_ref: Some(ObjectReference {
                    kind: Some("Pod".to_owned()),
                    name: Some((*pod).to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        slice
    }

    /// Like [`pod_slice`], for `mocker-frontend`.
    fn frontend_pod_slice(
        suffix: &str,
        address_type: &str,
        endpoints: &[(&str, &str)],
    ) -> EndpointSlice {
        let mut slice = frontend_slice("west", "mocker", &[]);
        slice.metadata.name = Some(format!("imported-west-mocker-frontend-{suffix}"));
        slice.address_type = address_type.to_owned();
        slice.endpoints = pod_slice("west", suffix, address_type, endpoints).endpoints;
        slice
    }

    fn key(cluster: &str, dgd: &str) -> PoolKey {
        PoolKey::new(cluster, "dynamo", dgd).unwrap()
    }

    fn cache() -> Arc<SliceCache<PoolIndex>> {
        Arc::new(SliceCache::new(naming()))
    }

    fn sync(cache: &SliceCache<PoolIndex>, slices: Vec<EndpointSlice>) {
        cache.apply(watcher::Event::Init);
        for slice in slices {
            cache.apply(watcher::Event::InitApply(slice));
        }
        cache.apply(watcher::Event::InitDone);
    }

    fn completeness(cache: &SliceCache<PoolIndex>) -> Completeness {
        cache.snapshot().unwrap().completeness
    }

    #[tokio::test(start_paused = true)]
    async fn a_watch_that_keeps_failing_marks_its_last_list_incomplete() {
        let cache = cache();
        let ready = || slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]);
        sync(&cache, vec![ready()]);
        // A relist that keeps failing. kube sends `Init` before each attempt;
        // the watch drops it (see `retried`), and the cache does not count it
        // as a working watch either.
        for _ in 0..3 {
            cache.apply(watcher::Event::Init);
            cache.failed();
            tokio::time::advance(STALE_AFTER / 3).await;
        }
        assert_eq!(completeness(&cache), Completeness::Exact);
        cache.apply(watcher::Event::Init);
        assert_eq!(cache.failed(), STALE_AFTER);
        let stale = cache.snapshot().unwrap();
        assert_eq!(stale.items.len(), 1);
        assert_eq!(stale.completeness, Completeness::Estimated);

        // A relist that completes means the watch works again. Until then the
        // old view stays `Estimated`.
        cache.apply(watcher::Event::InitApply(ready()));
        assert_eq!(completeness(&cache), Completeness::Estimated);
        cache.apply(watcher::Event::InitDone);
        assert_eq!(completeness(&cache), Completeness::Exact);
    }

    #[tokio::test(start_paused = true)]
    async fn one_error_then_silence_is_a_resumed_watch() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        // kube resumes the watch without an event.
        cache.failed();
        tokio::time::advance(STALE_AFTER + Duration::from_secs(1)).await;
        assert_eq!(completeness(&cache), Completeness::Exact);
    }

    #[tokio::test(start_paused = true)]
    async fn a_failing_watch_that_falls_silent_counts_as_recovered() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        for _ in 0..3 {
            cache.failed();
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        assert_eq!(completeness(&cache), Completeness::Estimated);
        tokio::time::advance(RECOVERED_AFTER).await;
        assert_eq!(completeness(&cache), Completeness::Exact);
        // A later error starts a new run.
        assert_eq!(cache.failed(), Duration::ZERO);
        assert_eq!(completeness(&cache), Completeness::Exact);
    }

    fn listing(cache: &SliceCache<PoolIndex>, key: &PoolKey) -> Option<PoolListing> {
        cache
            .read()
            .index
            .pools
            .get(key)
            .and_then(|pool| pool.listing.clone())
    }

    #[test]
    fn pool_listing_from_imported_slice() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.66.175", true)])],
        );
        let listing = listing(&cache, &key("west", "mocker")).expect("live pool");
        // The slice names no region, so ClusterRegions places the pool.
        assert_eq!(listing.location.region, "");
        assert_eq!(listing.relay_identities, ["ip:10.43.66.175"]);
    }

    #[test]
    fn a_slices_region_label_places_its_pool() {
        let cache = cache();
        let pool = key("west", "mocker");
        let labelled = |suffix: &str, region: &str| {
            let mut slice = slice("west", "mocker", suffix, &[("10.43.66.175", true)]);
            if let Some(labels) = slice.metadata.labels.as_mut() {
                labels.insert(REGION_LABEL.to_owned(), region.to_owned());
            }
            slice
        };

        // The label names the pool's region; an empty one names none.
        sync(&cache, vec![labelled("b", "us-west-2")]);
        assert_eq!(listing(&cache, &pool).unwrap().location.region, "us-west-2");
        sync(&cache, vec![labelled("b", "")]);
        assert_eq!(listing(&cache, &pool).unwrap().location.region, "");

        // Slices that disagree list the first one's region in key order, so
        // every instance agrees, and its removal lists the next one's.
        sync(
            &cache,
            vec![labelled("b", "us-west-2"), labelled("a", "us-west-1")],
        );
        assert_eq!(listing(&cache, &pool).unwrap().location.region, "us-west-1");
        sync(&cache, vec![labelled("b", "us-west-2")]);
        assert_eq!(listing(&cache, &pool).unwrap().location.region, "us-west-2");
    }

    #[test]
    fn every_ready_relay_pod_is_listed_by_each_address() {
        let cache = cache();
        sync(
            &cache,
            vec![
                pod_slice(
                    "west",
                    "a1b2c",
                    "IPv4",
                    &[("relay-a", "10.43.0.7"), ("relay-b", "10.43.0.9")],
                ),
                pod_slice(
                    "west",
                    "d3e4f",
                    "IPv6",
                    &[("relay-a", "fd00::7"), ("relay-b", "fd00::1")],
                ),
            ],
        );
        let listing = listing(&cache, &key("west", "mocker")).expect("live pool");
        assert_eq!(
            listing.relay_identities,
            ["ip:10.43.0.7", "ip:10.43.0.9", "ip:fd00::1", "ip:fd00::7"]
        );
    }

    #[test]
    fn a_mapped_relay_address_is_listed_once() {
        let cache = cache();
        sync(
            &cache,
            vec![
                pod_slice("west", "a1b2c", "IPv4", &[("relay-a", "10.43.0.7")]),
                pod_slice("west", "d3e4f", "IPv6", &[("relay-a", "::ffff:10.43.0.7")]),
            ],
        );
        let listing = listing(&cache, &key("west", "mocker")).expect("live pool");
        assert_eq!(listing.relay_identities, ["ip:10.43.0.7"]);
    }

    #[test]
    fn a_relay_that_stops_being_ready_leaves_the_listing() {
        let cache = cache();
        let relays = |second_ready| {
            slice(
                "west",
                "mocker",
                "a1b2c",
                &[("10.43.0.2", true), ("10.43.0.9", second_ready)],
            )
        };
        sync(&cache, vec![relays(true)]);
        let mut events = cache.events.subscribe();

        cache.apply(watcher::Event::Apply(relays(false)));
        let Ok(WatchEvent::Upserted(listing)) = events.try_recv() else {
            panic!("the pool stays live with one Ready relay");
        };
        assert_eq!(listing.relay_identities, ["ip:10.43.0.2"]);
    }

    #[test]
    fn a_dual_stack_frontend_is_listed_once_at_its_ipv4_address() {
        let cache = cache();
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                frontend_pod_slice(
                    "f1",
                    "IPv4",
                    &[("frontend-a", "10.43.0.7"), ("frontend-b", "10.43.0.2")],
                ),
                frontend_pod_slice(
                    "f2",
                    "IPv6",
                    &[("frontend-a", "fd00::7"), ("frontend-b", "fd00::2")],
                ),
            ],
        );
        let ipv4 = [
            Url::parse("http://10.43.0.2:8000/").unwrap(),
            Url::parse("http://10.43.0.7:8000/").unwrap(),
        ];
        let frontends = |cache: &SliceCache<PoolIndex>| {
            listing(cache, &key("west", "mocker"))
                .expect("live pool")
                .frontends
        };
        assert_eq!(frontends(&cache), ipv4);

        // The same when the slices do not name their pods, so a pod's two
        // addresses cannot be matched.
        let unnamed = |suffix: &str, address_type: &str, addresses: [&str; 2]| {
            let mut slice = frontend_pod_slice(suffix, address_type, &[]);
            slice.endpoints = addresses
                .iter()
                .map(|address| Endpoint {
                    addresses: vec![(*address).to_owned()],
                    ..Default::default()
                })
                .collect();
            slice
        };
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                unnamed("f1", "IPv4", ["10.43.0.7", "10.43.0.2"]),
                unnamed("f2", "IPv6", ["fd00::7", "fd00::2"]),
            ],
        );
        assert_eq!(frontends(&cache), ipv4);
    }

    #[test]
    fn the_frontend_address_family_follows_the_slices_not_readiness() {
        // On a dual-stack cluster the IPv4 slice is imported, but none of its
        // endpoints is Ready yet, while the IPv6 slice's are: the pool keeps
        // the IPv4 family instead of listing IPv6 frontends.
        let cache = cache();
        let mut ipv4 = frontend_pod_slice("f1", "IPv4", &[("frontend-a", "10.43.0.7")]);
        for endpoint in &mut ipv4.endpoints {
            endpoint.conditions = Some(EndpointConditions {
                ready: Some(false),
                ..Default::default()
            });
        }
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                ipv4,
                frontend_pod_slice("f2", "IPv6", &[("frontend-a", "fd00::7")]),
            ],
        );
        let listing = listing(&cache, &key("west", "mocker")).expect("live pool");
        assert!(listing.frontends.is_empty(), "{:?}", listing.frontends);
    }

    #[test]
    fn frontends_come_from_the_frontend_service() {
        // A Service name must not match both suffixes.
        for (frontend, relay) in [
            ("-frontend", "-frontend"),
            ("-relay", "-pool-relay"),
            ("-pool-relay", "-relay"),
            ("", SUFFIX),
        ] {
            let service = FrontendService {
                suffix: frontend.to_owned(),
                port_name: "http".to_owned(),
            };
            assert!(service.check(relay).is_err(), "{frontend:?} {relay:?}");
        }
        assert!(naming().frontends.check(SUFFIX).is_ok());

        let cache = cache();
        let pool = key("west", "mocker");
        // Frontends alone do not make a pool live.
        sync(
            &cache,
            vec![frontend_slice(
                "west",
                "mocker",
                &[("10.43.0.2", true), ("10.43.0.3", false)],
            )],
        );
        assert!(listing(&cache, &pool).is_none());

        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.1", true)],
        )));
        let first = listing(&cache, &pool).expect("live pool");
        assert_eq!(
            first.frontends,
            [Url::parse("http://10.43.0.2:8000/").unwrap()]
        );
        // Frontend addresses are not PoolRelay identities.
        assert_eq!(first.relay_identities, ["ip:10.43.0.1"]);

        // A replaced frontend changes the listing. IPv6 gets brackets.
        let mut ipv6 = frontend_slice("west", "mocker", &[("fd00::2", true)]);
        ipv6.address_type = "IPv6".to_owned();
        cache.apply(watcher::Event::Apply(ipv6));
        let second = listing(&cache, &pool).unwrap();
        assert_ne!(second, first);
        assert_eq!(
            second.frontends,
            [Url::parse("http://[fd00::2]:8000/").unwrap()]
        );

        // Only the named TCP port counts, and not port 0.
        for (name, protocol, port) in [
            ("grpc", "TCP", 8000),
            ("http", "UDP", 8000),
            ("http", "TCP", 0),
        ] {
            let mut other = frontend_slice("west", "mocker", &[("10.43.0.4", true)]);
            let named = &mut other.ports.as_mut().unwrap()[0];
            named.name = Some(name.to_owned());
            named.protocol = Some(protocol.to_owned());
            named.port = Some(port);
            cache.apply(watcher::Event::Apply(other));
            assert!(
                listing(&cache, &pool).unwrap().frontends.is_empty(),
                "{name} {protocol} {port}"
            );
        }
    }

    #[test]
    fn a_pool_that_loses_its_frontend_slice_is_listed_without_frontends() {
        let cache = cache();
        let frontend = frontend_slice("west", "mocker", &[("10.43.0.2", true)]);
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                frontend.clone(),
            ],
        );
        let before = listing(&cache, &key("west", "mocker")).expect("live pool");
        let mut events = cache.events.subscribe();

        // The frontend Service's export was removed. Its Ready PoolRelay
        // keeps the pool live.
        cache.apply(watcher::Event::Delete(frontend));
        let Ok(WatchEvent::Upserted(after)) = events.try_recv() else {
            panic!("expected the pool without frontends");
        };
        assert!(after.frontends.is_empty());
        assert_eq!(after.relay_identities, before.relay_identities);
        assert!(events.try_recv().is_err());
    }

    #[test]
    fn a_live_pool_without_frontends_is_warned_about_once() {
        let naming = naming();
        let pool = key("west", "mocker");
        let relay = slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]);
        let frontend = frontend_slice("west", "mocker", &[("10.43.0.2", true)]);
        let entries = |slices: &[&EndpointSlice]| {
            slices
                .iter()
                .map(|slice| (slice_key(slice), naming.slice(slice).unwrap()))
                .collect::<HashMap<_, _>>()
        };
        let warned = || naming.frontendless.lock().unwrap().contains(&pool);

        // Each read before the first sync builds a new index from the same
        // rules, as `SliceDirectory::list` does. The first read records the
        // pool, so later ones do not warn again.
        for _ in 0..2 {
            PoolIndex::default().replace(&naming, entries(&[&relay]));
        }
        assert!(warned());
        assert!(!naming.newly_frontendless(&pool, &[]));

        // A frontend clears the record, so losing it warns again, and so does
        // the pool no longer being live.
        let mut index = PoolIndex::default();
        index.replace(&naming, entries(&[&relay, &frontend]));
        assert!(!warned());
        let _ = index.upsert(&naming, slice_key(&frontend), None);
        assert!(warned());
        let _ = index.upsert(&naming, slice_key(&relay), None);
        assert!(!warned());
    }

    #[test]
    fn skips_imports_that_are_not_pools() {
        let mut other = slice("west", "mocker", "a1b2c", &[("10.43.0.9", true)]);
        other.metadata.labels.as_mut().unwrap().insert(
            KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
            "derived-echo".to_owned(),
        );
        let cache = cache();
        sync(&cache, vec![other]);
        assert!(cache.snapshot().unwrap().items.is_empty());
    }

    #[test]
    fn same_export_from_two_clusters_is_two_pools() {
        let cache = cache();
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                slice("ohio", "mocker", "d3e4f", &[("10.42.0.1", true)]),
            ],
        );
        assert!(listing(&cache, &key("west", "mocker")).is_some());
        assert!(listing(&cache, &key("ohio", "mocker")).is_some());
    }

    #[test]
    fn pool_without_ready_relay_is_not_live() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", false)])],
        );
        assert!(cache.snapshot().unwrap().items.is_empty());
    }

    #[test]
    fn changes_publish_added_and_removed() {
        let cache = cache();
        sync(&cache, Vec::new());
        let mut events = cache.events.subscribe();

        let live = slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]);
        cache.apply(watcher::Event::Apply(live.clone()));
        assert!(matches!(
            events.try_recv().unwrap(),
            WatchEvent::Upserted(_)
        ));
        cache.apply(watcher::Event::Apply(live.clone()));
        assert!(events.try_recv().is_err());
        // The Relay stops being Ready.
        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.1", false)],
        )));
        assert!(
            matches!(events.try_recv().unwrap(), WatchEvent::Removed(k) if k == key("west", "mocker"))
        );
        // Ready again.
        cache.apply(watcher::Event::Apply(live.clone()));
        assert!(matches!(
            events.try_recv().unwrap(),
            WatchEvent::Upserted(_)
        ));
        cache.apply(watcher::Event::Delete(live));
        assert!(matches!(events.try_recv().unwrap(), WatchEvent::Removed(_)));
        assert!(cache.read().index.pools.is_empty());
    }

    #[test]
    fn pool_spans_several_slices() {
        let cache = cache();
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", false)]),
                slice("west", "mocker", "d3e4f", &[("10.43.0.2", true)]),
            ],
        );
        assert!(listing(&cache, &key("west", "mocker")).is_some());
        // Losing the slice with the Ready Relay ends the pool.
        cache.apply(watcher::Event::Delete(slice(
            "west",
            "mocker",
            "d3e4f",
            &[],
        )));
        assert!(listing(&cache, &key("west", "mocker")).is_none());
    }

    #[test]
    fn every_instance_lists_the_same_relays() {
        let pool = key("west", "mocker");
        let both = || {
            slice(
                "west",
                "mocker",
                "a1b2c",
                &[("10.43.0.9", true), ("10.43.0.2", true)],
            )
        };
        // One instance saw one PoolRelay alone first.
        let first = cache();
        sync(
            &first,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.9", true)])],
        );
        first.apply(watcher::Event::Apply(both()));
        // Another started later.
        let second = cache();
        sync(&second, vec![both()]);
        let listed =
            |cache: &SliceCache<PoolIndex>| listing(cache, &pool).unwrap().relay_identities;
        assert_eq!(listed(&first), listed(&second));
        assert_eq!(listed(&first), ["ip:10.43.0.2", "ip:10.43.0.9"]);
    }

    #[tokio::test(start_paused = true)]
    async fn watchers_learn_when_the_view_turns_stale_and_back() {
        let cache = cache();
        let ready = || slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]);
        sync(&cache, vec![ready()]);
        let mut events = cache.events.subscribe();
        let mut next_resync = || match events.try_recv() {
            Ok(WatchEvent::Resync(snapshot)) => Some(snapshot.completeness),
            Ok(_) => panic!("expected a Resync"),
            Err(_) => None,
        };
        for _ in 0..3 {
            cache.failed();
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        assert_eq!(next_resync(), Some(Completeness::Estimated));
        // Further errors change nothing.
        cache.failed();
        assert_eq!(next_resync(), None);

        // Recovered by silence.
        tokio::time::advance(RECOVERED_AFTER).await;
        cache.quiet();
        assert_eq!(next_resync(), Some(Completeness::Exact));
        assert_eq!(next_resync(), None);

        // Recovered by an event from the API server.
        for _ in 0..3 {
            cache.failed();
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        assert_eq!(next_resync(), Some(Completeness::Estimated));
        cache.apply(watcher::Event::Apply(ready()));
        assert_eq!(next_resync(), Some(Completeness::Exact));
        cache.quiet();
        assert_eq!(next_resync(), None);

        // Recovered by a relist, which announces only its own view.
        for _ in 0..3 {
            cache.failed();
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        assert_eq!(next_resync(), Some(Completeness::Estimated));
        cache.apply(watcher::Event::InitApply(ready()));
        assert_eq!(next_resync(), None);
        cache.apply(watcher::Event::InitDone);
        assert_eq!(next_resync(), Some(Completeness::Exact));
        assert_eq!(next_resync(), None);

        // An error after a quiet spell ends the old run first, though
        // `quiet` has not run.
        for _ in 0..3 {
            cache.failed();
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        assert_eq!(next_resync(), Some(Completeness::Estimated));
        tokio::time::advance(RECOVERED_AFTER).await;
        assert_eq!(cache.failed(), Duration::ZERO);
        assert_eq!(next_resync(), Some(Completeness::Exact));
        assert_eq!(next_resync(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn the_watch_ends_a_run_of_errors_that_falls_silent() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        let mut events = cache.events.subscribe();
        let (errors, watch) = futures::channel::mpsc::unbounded();
        let stop = CancellationToken::new();
        let following = tokio::spawn({
            let (cache, stop) = (Arc::clone(&cache), stop.clone());
            async move { follow(watch, "test", &cache, &stop).await }
        });
        for _ in 0..3 {
            errors
                .unbounded_send(Err(watcher::Error::NoResourceVersion))
                .unwrap();
            tokio::task::yield_now().await;
            tokio::time::advance(STALE_AFTER / 2).await;
        }
        let resync = tokio::time::timeout(Duration::from_secs(1), events.recv()).await;
        assert!(
            matches!(resync, Ok(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Estimated)
        );

        // kube resumed the watch without an event, so only the loop's timer
        // can end the run.
        tokio::time::advance(RECOVERED_AFTER).await;
        let resync = tokio::time::timeout(Duration::from_secs(1), events.recv()).await;
        assert!(
            matches!(resync, Ok(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Exact)
        );
        stop.cancel();
        following.await.unwrap();
    }

    #[test]
    fn relist_replaces_everything() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "old", "a1b2c", &[("10.43.0.1", true)])],
        );
        let mut events = cache.events.subscribe();
        cache.apply(watcher::Event::Init);
        cache.apply(watcher::Event::InitApply(slice(
            "west",
            "new",
            "d3e4f",
            &[("10.43.0.2", true)],
        )));
        // The old view stays until the relist completes.
        assert!(listing(&cache, &key("west", "old")).is_some());
        cache.apply(watcher::Event::InitDone);
        let WatchEvent::Resync(snapshot) = events.try_recv().unwrap() else {
            panic!("expected Resync");
        };
        let keys: Vec<PoolKey> = snapshot
            .items
            .into_iter()
            .map(|listing| listing.key)
            .collect();
        assert_eq!(keys, [key("west", "new")]);
        assert!(events.try_recv().is_err());
    }

    #[test]
    fn a_failed_relist_starts_over() {
        let cache = cache();
        sync(&cache, Vec::new());
        // A relist fails after its first slices, as a paged one could.
        cache.apply(watcher::Event::InitApply(slice(
            "west",
            "gone",
            "a1b2c",
            &[("10.43.0.1", true)],
        )));
        cache.failed();
        cache.apply(watcher::Event::InitApply(slice(
            "west",
            "kept",
            "d3e4f",
            &[("10.43.0.2", true)],
        )));
        cache.apply(watcher::Event::InitDone);
        assert!(listing(&cache, &key("west", "gone")).is_none());
        assert!(listing(&cache, &key("west", "kept")).is_some());
    }

    #[tokio::test]
    async fn watch_starts_with_added_then_resync() {
        let cache = cache();
        let cancel = CancellationToken::new();
        let mut stream = cache.subscribe(cancel.clone(), CancellationToken::new());
        // Polled once, so it subscribes before the first list completes.
        assert!(stream.next().now_or_never().is_none());
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));
        assert!(
            matches!(stream.next().await, Some(Ok(WatchEvent::Resync(s))) if s.items.len() == 1)
        );
        cache.apply(watcher::Event::Delete(slice(
            "west",
            "mocker",
            "a1b2c",
            &[],
        )));
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Removed(_)))
        ));
        cancel.cancel();
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn late_subscriber_gets_the_current_pools() {
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        let mut stream = cache.subscribe(CancellationToken::new(), CancellationToken::new());
        assert!(
            matches!(stream.next().await, Some(Ok(WatchEvent::Upserted(r))) if r.key == key("west", "mocker"))
        );
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
    }
}
