// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools from imported EndpointSlices, for router replicas and for the
//! Global Discovery Service's MCS store.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use crate::pool_discovery::{
    BuildPoolDiscovery, Completeness, McsImplementation, PeerCredentials, PoolDiscovery,
    PoolDiscoveryError, PoolEvent, PoolRecord, RelayAuthenticator, RelayIdentity, ReplicaDirectory,
    Revision, WatchEvent,
};
use async_trait::async_trait;
use dynamo_global_router_shared::PoolKey;
use dynamo_global_router_shared::PoolLocation;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::cache::{SliceDirectory, SliceIndex, SliceKey};
use super::mcs;

/// Startup for Kubernetes mode. `registrar` is `None`.
pub struct McsPoolDiscovery {
    /// This router's replicas, from the hub's Dynamo runtime discovery. Shared
    /// with the other modes; Kubernetes mode itself does not read it.
    pub replicas: Arc<ReplicaDirectory>,
    /// Where the imported pool slices are. The hub for standard MCS. For
    /// Karmada, the Karmada control plane: Karmada collects every exported
    /// Service's slices there, so no per-pool `ServiceImport` is needed. A
    /// member cluster with a `ServiceImport` per pool also works.
    pub imports: Client,
    pub mcs: McsImplementation,
    /// See `PoolDiscoveryMode::Kubernetes`.
    pub pool_service_suffix: String,
    pub relay_auth: RelayAuth,
    /// Keyed by About API cluster ID. ClusterProfile properties can fill it.
    pub locations: HashMap<String, PoolLocation>,
}

/// How the router names a Relay that dials in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayAuth {
    /// V1: a trusted network with no mesh. See
    /// [`TrustedNetworkAuthenticator`].
    TrustedNetwork,
    /// Mesh mTLS. Pool records carry no expected identity, because MCS does
    /// not export the Relay's SPIFFE ID.
    Mesh,
}

#[async_trait]
impl BuildPoolDiscovery for McsPoolDiscovery {
    /// Does not contact the API: the pool watch starts on first use.
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError> {
        let pools = McsPoolDirectory::new(
            self.imports,
            self.mcs,
            self.pool_service_suffix,
            self.relay_auth == RelayAuth::TrustedNetwork,
            self.locations,
        );
        let authenticator: Arc<dyn RelayAuthenticator> = match self.relay_auth {
            RelayAuth::TrustedNetwork => Arc::new(TrustedNetworkAuthenticator),
            RelayAuth::Mesh => Arc::new(MeshAuthenticator),
        };
        Ok(PoolDiscovery {
            pools: Arc::new(pools),
            replicas: self.replicas,
            registrar: None,
            authenticator,
        })
    }
}

/// One pool per (source cluster, namespace, DGD name), from imported
/// EndpointSlices. MCS merges same-named exports from several clusters into
/// one import, so pools are split by source cluster. Lists are `Exact`. Each
/// slice event updates one pool.
///
/// One watch per directory feeds a cache that serves `get`, `list` and
/// every `list_and_watch` stream. The watch starts on first use and stops
/// when the directory is dropped.
pub struct McsPoolDirectory(SliceDirectory<PoolIndex>);

/// The Service in front of a pool's frontends, named `<DGD name><suffix>`
/// and exported like the PoolRelay Service. Its Ready endpoints on the named
/// port become the pool's `frontends`: addresses that other clusters can
/// reach, kept current as frontend pods come and go.
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
    pub fn new(
        imports: Client,
        mcs: McsImplementation,
        pool_service_suffix: String,
        bind_relay_address: bool,
        locations: HashMap<String, PoolLocation>,
    ) -> Self {
        Self::create(
            imports,
            PoolNaming {
                mcs,
                pool_service_suffix,
                bind_relay_address,
                locations,
                frontends: None,
            },
        )
    }

    /// Also lists each pool's Ready frontends. A pool with frontends but no
    /// Ready PoolRelay is still not live.
    pub fn with_frontends(
        imports: Client,
        mcs: McsImplementation,
        pool_service_suffix: String,
        bind_relay_address: bool,
        locations: HashMap<String, PoolLocation>,
        frontends: FrontendService,
    ) -> anyhow::Result<Self> {
        frontends.check(&pool_service_suffix)?;
        Ok(Self::create(
            imports,
            PoolNaming {
                mcs,
                pool_service_suffix,
                bind_relay_address,
                locations,
                frontends: Some(frontends),
            },
        ))
    }

    fn create(imports: Client, naming: PoolNaming) -> Self {
        Self(SliceDirectory::new(
            Api::all(imports),
            mcs::imported_selector(naming.mcs),
            naming,
        ))
    }
}

#[async_trait]
impl Directory<PoolRecord, PoolKey> for McsPoolDirectory {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        self.0.check_connection().await
    }

    async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
        self.0.list().await
    }

    async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
        self.0.get(key).await
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
        self.0.list_and_watch(cancel_token).await
    }
}

/// Turns imported slices into pool records.
struct PoolNaming {
    mcs: McsImplementation,
    pool_service_suffix: String,
    /// Sets `relay_identity` to the PoolRelay endpoint's address, for
    /// [`TrustedNetworkAuthenticator`].
    bind_relay_address: bool,
    locations: HashMap<String, PoolLocation>,
    /// `Some` to list each pool's frontends too.
    frontends: Option<FrontendService>,
}

impl PoolNaming {
    /// Source cluster from [`mcs::source_cluster`]; DGD name from the
    /// Service name without `suffix`. `None` for imports of other Services.
    fn pool_key(&self, slice: &EndpointSlice, suffix: &str) -> Option<PoolKey> {
        let service = mcs::service_name(self.mcs, slice)?;
        let dgd_name = service
            .strip_suffix(suffix)
            .filter(|name| !name.is_empty())?;
        let cluster = mcs::source_cluster(self.mcs, slice)?;
        let namespace = slice.metadata.namespace.as_deref()?;
        PoolKey::new(cluster, namespace, dgd_name).ok()
    }

    /// A PoolRelay slice, or with `frontends`, a frontend slice with the
    /// named port.
    fn slice(&self, slice: &EndpointSlice) -> Option<SliceEntry> {
        if let Some(pool) = self.pool_key(slice, &self.pool_service_suffix) {
            return Some(SliceEntry {
                pool,
                ready: ready_addresses(slice),
                frontend_port: None,
            });
        }
        let frontends = self.frontends.as_ref()?;
        Some(SliceEntry {
            pool: self.pool_key(slice, &frontends.suffix)?,
            ready: ready_addresses(slice),
            frontend_port: Some(tcp_port(slice, &frontends.port_name)?),
        })
    }

    /// A cluster without a configured location gets one named after the
    /// cluster, so pools from a new cluster are not dropped.
    fn location(&self, cluster: &str) -> PoolLocation {
        self.locations
            .get(cluster)
            .cloned()
            .unwrap_or_else(|| PoolLocation {
                region: cluster.to_owned(),
                availability_zone: None,
                cluster: Some(cluster.to_owned()),
                datacenter: None,
            })
    }
}

/// A pool is live while its PoolRelay endpoint is Ready. Ready can be stale
/// when a cluster loses contact with the MCS implementation, so the runtime
/// also drops a pool whose Relay connection is gone.
fn ready_addresses(slice: &EndpointSlice) -> Vec<IpAddr> {
    slice
        .endpoints
        .iter()
        // The EndpointSlice API reads a missing `ready` as ready.
        .filter(|endpoint| {
            endpoint
                .conditions
                .as_ref()
                .and_then(|conditions| conditions.ready)
                .unwrap_or(true)
        })
        .flat_map(|endpoint| {
            endpoint
                .addresses
                .iter()
                .filter_map(|address| address.parse().ok())
        })
        .collect()
}

/// A slice's named TCP port. One slice lists one set of ports for all its
/// endpoints.
fn tcp_port(slice: &EndpointSlice, name: &str) -> Option<u16> {
    slice
        .ports
        .iter()
        .flatten()
        .find(|port| {
            port.name.as_deref() == Some(name)
                && port
                    .protocol
                    .as_deref()
                    .is_none_or(|protocol| protocol == "TCP")
        })
        .and_then(|port| port.port)
        .and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port != 0)
}

/// Formatted through `SocketAddr`, so IPv6 gets brackets.
fn frontend_url(address: IpAddr, port: u16) -> Option<Url> {
    Url::parse(&format!("http://{}/", SocketAddr::new(address, port))).ok()
}

struct SliceEntry {
    pool: PoolKey,
    ready: Vec<IpAddr>,
    /// `Some` for a frontend slice: the port its endpoints serve HTTP on.
    frontend_port: Option<u16>,
}

#[derive(Default)]
struct PoolState {
    slices: HashSet<SliceKey>,
    /// `Some` while the pool is live.
    record: Option<PoolRecord>,
    /// The Relay address in `record.relay_identity`.
    bound: Option<IpAddr>,
}

#[derive(Default)]
struct PoolIndex {
    slices: HashMap<SliceKey, SliceEntry>,
    pools: HashMap<PoolKey, PoolState>,
    last_revision: u64,
}

impl SliceIndex for PoolIndex {
    type Item = PoolRecord;
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

    fn snapshot(&self) -> Snapshot<PoolRecord> {
        Snapshot {
            items: self
                .pools
                .values()
                .filter_map(|pool| pool.record.clone())
                .collect(),
            completeness: Completeness::Exact,
        }
    }

    fn get(&self, key: &PoolKey) -> Option<PoolRecord> {
        self.pools.get(key).and_then(|pool| pool.record.clone())
    }
}

impl PoolIndex {
    /// Recomputes one pool from its slices. Returns the event to publish if
    /// the record changed.
    fn rebuild(&mut self, naming: &PoolNaming, key: &PoolKey) -> Option<PoolEvent> {
        let state = self.pools.get(key)?;
        let mut ready = Vec::new();
        let mut frontends = Vec::new();
        for entry in state
            .slices
            .iter()
            .filter_map(|slice| self.slices.get(slice))
        {
            match entry.frontend_port {
                None => ready.extend_from_slice(&entry.ready),
                Some(port) => frontends.extend(
                    entry
                        .ready
                        .iter()
                        .filter_map(|address| frontend_url(*address, port)),
                ),
            }
        }
        ready.sort_unstable();
        ready.dedup();
        frontends.sort_unstable();
        frontends.dedup();

        let state = self.pools.get_mut(key)?;
        let Some(&lowest) = ready.first() else {
            state.bound = None;
            let was_live = state.record.take().is_some();
            if state.slices.is_empty() {
                self.pools.remove(key);
            }
            return was_live.then(|| WatchEvent::Removed(key.clone()));
        };
        // Keep the bound Relay while it stays Ready, so a rolling update
        // that briefly shows two Ready Relays does not switch back and forth.
        let bound = naming.bind_relay_address.then(|| match state.bound {
            Some(current) if ready.contains(&current) => current,
            _ => lowest,
        });
        let mut record = PoolRecord {
            key: key.clone(),
            location: naming.location(key.site_id()),
            runtime_namespace: None,
            frontend_endpoint: None,
            model: None,
            capabilities: Vec::new(),
            frontends,
            capacity: Vec::new(),
            relay_identity: bound.map(address_identity),
            revision: Revision::default(),
        };
        if let Some(old) = &state.record {
            record.revision = old.revision;
            if *old == record {
                return None;
            }
        }
        self.last_revision += 1;
        record.revision = Revision(self.last_revision);
        state.bound = bound;
        state.record = Some(record.clone());
        Some(WatchEvent::Added(record))
    }
}

/// The identity both sides use for an address: `ip:<address>`.
pub fn address_identity(address: IpAddr) -> RelayIdentity {
    RelayIdentity(format!("ip:{address}"))
}

/// For a trusted network with no mesh. Accepts only
/// `PeerCredentials::PeerAddress` and names the caller with
/// [`address_identity`], which `pool_discovery::admit` matches against the
/// pool's Ready PoolRelay endpoint. Traffic between clusters must not be
/// source-NATed: exclude the peer ranges in the AKS ip-masq-agent or the AWS
/// VPC CNI, for example.
pub struct TrustedNetworkAuthenticator;

#[async_trait]
impl RelayAuthenticator for TrustedNetworkAuthenticator {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<RelayIdentity, PoolDiscoveryError> {
        match credentials {
            // A dual-stack listener reports IPv4 peers as `::ffff:a.b.c.d`;
            // slices list them as plain IPv4.
            PeerCredentials::PeerAddress(address) => Ok(address_identity(address.to_canonical())),
            _ => Err(PoolDiscoveryError::Unauthenticated),
        }
    }
}

/// Names the caller from the SPIFFE ID that mesh mTLS proved, and rejects
/// other credentials. Used with [`RelayAuth::Mesh`].
pub struct MeshAuthenticator;

#[async_trait]
impl RelayAuthenticator for MeshAuthenticator {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<RelayIdentity, PoolDiscoveryError> {
        match credentials {
            PeerCredentials::Spiffe(id) => Ok(RelayIdentity(id.clone())),
            _ => Err(PoolDiscoveryError::Unauthenticated),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use futures::{FutureExt, StreamExt};
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use kube::runtime::watcher;

    use super::super::cache::SliceCache;
    use super::mcs::{KARMADA_SLICE_MANAGER, KUBERNETES_SERVICE_NAME_LABEL, MANAGED_BY_LABEL};
    use super::*;

    const SUFFIX: &str = "-pool-relay";

    fn naming(bind_relay_address: bool) -> PoolNaming {
        PoolNaming {
            mcs: McsImplementation::Karmada,
            pool_service_suffix: SUFFIX.to_owned(),
            bind_relay_address,
            locations: HashMap::from([(
                "west".to_owned(),
                PoolLocation {
                    region: "us-west-2".to_owned(),
                    availability_zone: None,
                    cluster: Some("west".to_owned()),
                    datacenter: None,
                },
            )]),
            frontends: None,
        }
    }

    fn frontend_service(suffix: &str) -> FrontendService {
        FrontendService {
            suffix: suffix.to_owned(),
            port_name: "http".to_owned(),
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

    fn key(cluster: &str, dgd: &str) -> PoolKey {
        PoolKey::new(cluster, "dynamo", dgd).unwrap()
    }

    fn cache(bind_relay_address: bool) -> Arc<SliceCache<PoolIndex>> {
        Arc::new(SliceCache::new(naming(bind_relay_address)))
    }

    fn sync(cache: &SliceCache<PoolIndex>, slices: Vec<EndpointSlice>) {
        cache.apply(watcher::Event::Init);
        for slice in slices {
            cache.apply(watcher::Event::InitApply(slice));
        }
        cache.apply(watcher::Event::InitDone);
    }

    fn record(cache: &SliceCache<PoolIndex>, key: &PoolKey) -> Option<PoolRecord> {
        cache
            .read()
            .index
            .pools
            .get(key)
            .and_then(|pool| pool.record.clone())
    }

    #[test]
    fn pool_record_from_imported_slice() {
        let cache = cache(true);
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.66.175", true)])],
        );
        let record = record(&cache, &key("west", "mocker")).expect("live pool");
        assert_eq!(record.location.region, "us-west-2");
        assert_eq!(
            record.relay_identity,
            Some(RelayIdentity("ip:10.43.66.175".to_owned()))
        );
        assert_eq!((record.runtime_namespace, record.model), (None, None));
    }

    #[test]
    fn frontends_come_from_the_frontend_service() {
        let cache = Arc::new(SliceCache::new(PoolNaming {
            frontends: Some(frontend_service("-frontend")),
            ..naming(true)
        }));
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
        assert!(record(&cache, &pool).is_none());

        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.1", true)],
        )));
        let first = record(&cache, &pool).expect("live pool");
        assert_eq!(
            first.frontends,
            [Url::parse("http://10.43.0.2:8000/").unwrap()]
        );

        // A replaced frontend is a new revision. IPv6 gets brackets.
        cache.apply(watcher::Event::Apply(frontend_slice(
            "west",
            "mocker",
            &[("fd00::2", true)],
        )));
        let second = record(&cache, &pool).unwrap();
        assert!(second.revision > first.revision);
        assert_eq!(
            second.frontends,
            [Url::parse("http://[fd00::2]:8000/").unwrap()]
        );

        // A slice without the named port lists no frontends.
        let mut other_port = frontend_slice("west", "mocker", &[("10.43.0.4", true)]);
        other_port.ports.as_mut().unwrap()[0].name = Some("grpc".to_owned());
        cache.apply(watcher::Event::Apply(other_port));
        assert!(record(&cache, &pool).unwrap().frontends.is_empty());
    }

    #[test]
    fn frontend_suffix_must_not_overlap_the_relay_suffix() {
        assert!(frontend_service("-frontend").check(SUFFIX).is_ok());
        for suffix in ["", "-relay", SUFFIX, "-frontend-pool-relay"] {
            assert!(frontend_service(suffix).check(SUFFIX).is_err(), "{suffix}");
        }
    }

    #[test]
    fn unknown_cluster_gets_a_location_named_after_it() {
        let cache = cache(false);
        sync(
            &cache,
            vec![slice("ohio", "mocker", "a1b2c", &[("10.42.1.2", true)])],
        );
        let record = record(&cache, &key("ohio", "mocker")).expect("live pool");
        assert_eq!(record.location.region, "ohio");
        assert_eq!(record.location.cluster.as_deref(), Some("ohio"));
        assert_eq!(record.relay_identity, None);
    }

    #[test]
    fn skips_imports_that_are_not_pools() {
        let mut other = slice("west", "mocker", "a1b2c", &[("10.43.0.9", true)]);
        other.metadata.labels.as_mut().unwrap().insert(
            KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
            "derived-echo".to_owned(),
        );
        let cache = cache(true);
        sync(&cache, vec![other]);
        assert!(cache.snapshot().unwrap().items.is_empty());
    }

    #[test]
    fn same_export_from_two_clusters_is_two_pools() {
        let cache = cache(true);
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]),
                slice("ohio", "mocker", "d3e4f", &[("10.42.0.1", true)]),
            ],
        );
        assert!(record(&cache, &key("west", "mocker")).is_some());
        assert!(record(&cache, &key("ohio", "mocker")).is_some());
    }

    #[test]
    fn pool_without_ready_relay_is_not_live() {
        let cache = cache(true);
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", false)])],
        );
        assert!(cache.snapshot().unwrap().items.is_empty());
    }

    #[test]
    fn changes_publish_added_and_removed() {
        let cache = cache(true);
        sync(&cache, Vec::new());
        let mut events = cache.events.subscribe();

        let live = slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)]);
        cache.apply(watcher::Event::Apply(live.clone()));
        let WatchEvent::Added(first) = events.try_recv().unwrap() else {
            panic!("expected Added");
        };
        // The same slice again changes nothing.
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
        // Ready again: a newer revision.
        cache.apply(watcher::Event::Apply(live.clone()));
        let WatchEvent::Added(second) = events.try_recv().unwrap() else {
            panic!("expected Added");
        };
        assert!(second.revision > first.revision);
        // The slice is deleted.
        cache.apply(watcher::Event::Delete(live));
        assert!(matches!(events.try_recv().unwrap(), WatchEvent::Removed(_)));
        assert!(cache.read().index.pools.is_empty());
    }

    #[test]
    fn pool_spans_several_slices() {
        let cache = cache(false);
        sync(
            &cache,
            vec![
                slice("west", "mocker", "a1b2c", &[("10.43.0.1", false)]),
                slice("west", "mocker", "d3e4f", &[("10.43.0.2", true)]),
            ],
        );
        assert!(record(&cache, &key("west", "mocker")).is_some());
        // Losing the slice with the Ready Relay ends the pool.
        cache.apply(watcher::Event::Delete(slice(
            "west",
            "mocker",
            "d3e4f",
            &[],
        )));
        assert!(record(&cache, &key("west", "mocker")).is_none());
    }

    #[test]
    fn bound_relay_stays_during_a_rolling_update() {
        let cache = cache(true);
        let pool = key("west", "mocker");
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.9", true)])],
        );
        // The new Relay has a lower address and is Ready next to the old one.
        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.9", true), ("10.43.0.2", true)],
        )));
        assert_eq!(
            record(&cache, &pool).unwrap().relay_identity,
            Some(address_identity("10.43.0.9".parse().unwrap()))
        );
        // The old Relay leaves; the new one is bound.
        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.2", true)],
        )));
        assert_eq!(
            record(&cache, &pool).unwrap().relay_identity,
            Some(address_identity("10.43.0.2".parse().unwrap()))
        );
    }

    #[test]
    fn relist_replaces_everything() {
        let cache = cache(true);
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
        assert!(record(&cache, &key("west", "old")).is_some());
        cache.apply(watcher::Event::InitDone);
        let WatchEvent::Resync(snapshot) = events.try_recv().unwrap() else {
            panic!("expected Resync");
        };
        let keys: Vec<PoolKey> = snapshot
            .items
            .into_iter()
            .map(|record| record.key)
            .collect();
        assert_eq!(keys, [key("west", "new")]);
        assert!(events.try_recv().is_err());
    }

    #[tokio::test]
    async fn watch_starts_with_added_then_resync() {
        let cache = cache(true);
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
            Some(Ok(WatchEvent::Added(_)))
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
        let cache = cache(true);
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.0.1", true)])],
        );
        let mut stream = cache.subscribe(CancellationToken::new(), CancellationToken::new());
        assert!(
            matches!(stream.next().await, Some(Ok(WatchEvent::Added(r))) if r.key == key("west", "mocker"))
        );
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
    }

    #[tokio::test]
    async fn trusted_network_names_the_peer_address() {
        let ipv4: IpAddr = "10.43.66.175".parse().unwrap();
        let mapped: IpAddr = "::ffff:10.43.66.175".parse().unwrap();
        for peer in [ipv4, mapped] {
            let identity = TrustedNetworkAuthenticator
                .authenticate(&PeerCredentials::PeerAddress(peer))
                .await
                .unwrap();
            assert_eq!(identity, address_identity(ipv4));
        }
        assert!(matches!(
            TrustedNetworkAuthenticator
                .authenticate(&PeerCredentials::BearerToken("t".to_owned()))
                .await,
            Err(PoolDiscoveryError::Unauthenticated)
        ));
        assert_eq!(
            address_identity("fd00::2".parse().unwrap()),
            RelayIdentity("ip:fd00::2".to_owned())
        );
    }

    #[tokio::test]
    async fn mesh_accepts_only_spiffe() {
        let id = "spiffe://cluster.local/ns/dynamo/sa/relay".to_owned();
        assert_eq!(
            MeshAuthenticator
                .authenticate(&PeerCredentials::Spiffe(id.clone()))
                .await
                .unwrap(),
            RelayIdentity(id)
        );
        assert!(matches!(
            MeshAuthenticator
                .authenticate(&PeerCredentials::PeerAddress("10.0.0.1".parse().unwrap()))
                .await,
            Err(PoolDiscoveryError::Unauthenticated)
        ));
    }
}
