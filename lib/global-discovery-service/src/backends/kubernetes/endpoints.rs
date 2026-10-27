// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replicas from the EndpointSlices that MCS imports. The Global
//! Discovery Service serves them to PoolRelays as `GET /routers`. Router
//! replicas find each other through the Dynamo runtime, not through this.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};

use crate::pool_discovery::{
    Completeness, Directory, PoolDiscoveryError, ReplicaEndpoint, ReplicaEvent, ReplicaId,
    Snapshot, WatchEvent, WatchStream,
};
use async_trait::async_trait;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::cache::{SliceDirectory, SliceIndex, SliceKey};
use super::mcs::{self, McsImplementation};

/// A Service exported through MCS, such as the Global Router's. It need not
/// be headless: imported slices list every pod, and readers use slices, not
/// DNS.
#[derive(Clone, Debug)]
pub struct ExportedService {
    pub namespace: String,
    pub name: String,
    /// The named port its replicas are reached on, for example the router's
    /// state stream.
    pub port_name: String,
}

/// Ready replicas from a Service's imported slices, named
/// `<source cluster>/<pod>`, or `<source cluster>/<address>` when a slice
/// does not name the pod. Lists are `Exact`.
pub struct EndpointSliceReplicaDirectory(SliceDirectory<ReplicaIndex>);

impl EndpointSliceReplicaDirectory {
    /// The Service's slices as MCS imports them into the cluster `client`
    /// reaches.
    pub fn imported(client: Client, service: &ExportedService, mcs: McsImplementation) -> Self {
        Self(SliceDirectory::new(
            Api::namespaced(client, &service.namespace),
            mcs::service_selector(mcs, &service.name),
            ReplicaNaming {
                mcs,
                service: service.name.clone(),
                port_name: service.port_name.clone(),
            },
        ))
    }
}

#[async_trait]
impl Directory<ReplicaEndpoint, ReplicaId> for EndpointSliceReplicaDirectory {
    async fn list(&self) -> Result<Snapshot<ReplicaEndpoint>, PoolDiscoveryError> {
        self.0.list().await
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<ReplicaEndpoint, ReplicaId>, PoolDiscoveryError> {
        self.0.list_and_watch(cancel_token).await
    }
}

/// Turns one imported slice into replicas.
struct ReplicaNaming {
    mcs: McsImplementation,
    service: String,
    port_name: String,
}

impl ReplicaNaming {
    /// Ready endpoints only. The URL comes from the endpoint's address and
    /// the named port, formatted through `SocketAddr` so IPv6 gets brackets.
    /// Per-pod DNS names are not used: they exist only for pods with
    /// hostnames. `None` for a slice of another Service, a slice whose source
    /// cluster is unknown, or one without the named port.
    fn replicas(&self, slice: &EndpointSlice) -> Option<Vec<ReplicaEndpoint>> {
        if mcs::service_name(self.mcs, slice) != Some(self.service.as_str()) {
            return None;
        }
        let cluster = mcs::source_cluster(self.mcs, slice)?;
        let port = slice
            .ports
            .iter()
            .flatten()
            .find(|port| port.name.as_deref() == Some(self.port_name.as_str()))
            .and_then(|port| port.port)
            .and_then(|port| u16::try_from(port).ok())?;
        let replicas = slice
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
            .filter_map(|endpoint| {
                // An endpoint's addresses are interchangeable, so the first is
                // enough.
                let address: IpAddr = endpoint.addresses.first()?.parse().ok()?;
                let url = Url::parse(&format!("http://{}", SocketAddr::new(address, port))).ok()?;
                let name = endpoint
                    .target_ref
                    .as_ref()
                    .and_then(|target| target.name.clone())
                    .unwrap_or_else(|| address.to_string());
                Some(ReplicaEndpoint {
                    replica: ReplicaId(format!("{cluster}/{name}")),
                    url,
                })
            })
            .collect();
        Some(replicas)
    }
}

#[derive(Default)]
struct ReplicaIndex {
    /// The replicas each slice lists.
    slices: HashMap<SliceKey, Vec<ReplicaEndpoint>>,
    /// Every listed replica, with the slices that list it. A replica in
    /// several slices, one per address family for example, is served from the
    /// first slice in key order.
    replicas: HashMap<ReplicaId, BTreeMap<SliceKey, ReplicaEndpoint>>,
}

impl ReplicaIndex {
    fn served(&self, id: &ReplicaId) -> Option<&ReplicaEndpoint> {
        self.replicas.get(id)?.values().next()
    }
}

impl SliceIndex for ReplicaIndex {
    type Item = ReplicaEndpoint;
    type Key = ReplicaId;
    type Rules = ReplicaNaming;
    type Entry = Vec<ReplicaEndpoint>;

    fn entry(naming: &ReplicaNaming, slice: &EndpointSlice) -> Option<Vec<ReplicaEndpoint>> {
        naming.replicas(slice)
    }

    /// Touches only the replicas this slice lists now or listed before.
    fn upsert(
        &mut self,
        _naming: &ReplicaNaming,
        key: SliceKey,
        entry: Option<Vec<ReplicaEndpoint>>,
    ) -> Vec<ReplicaEvent> {
        let old = self.slices.remove(&key).unwrap_or_default();
        let new = entry.unwrap_or_default();
        let touched: HashSet<ReplicaId> = old
            .iter()
            .chain(&new)
            .map(|endpoint| endpoint.replica.clone())
            .collect();
        let before: Vec<(ReplicaId, Option<ReplicaEndpoint>)> = touched
            .into_iter()
            .map(|id| {
                let served = self.served(&id).cloned();
                (id, served)
            })
            .collect();
        for endpoint in &old {
            if let Some(listed) = self.replicas.get_mut(&endpoint.replica) {
                listed.remove(&key);
                if listed.is_empty() {
                    self.replicas.remove(&endpoint.replica);
                }
            }
        }
        for endpoint in &new {
            self.replicas
                .entry(endpoint.replica.clone())
                .or_default()
                .insert(key.clone(), endpoint.clone());
        }
        if !new.is_empty() {
            self.slices.insert(key, new);
        }
        before
            .into_iter()
            .filter_map(|(id, before)| match (before, self.served(&id)) {
                (Some(_), None) => Some(WatchEvent::Removed(id)),
                (before, Some(after)) if before.as_ref() != Some(after) => {
                    Some(WatchEvent::Added(after.clone()))
                }
                _ => None,
            })
            .collect()
    }

    fn replace(
        &mut self,
        _naming: &ReplicaNaming,
        entries: HashMap<SliceKey, Vec<ReplicaEndpoint>>,
    ) {
        self.replicas.clear();
        for (key, endpoints) in &entries {
            for endpoint in endpoints {
                self.replicas
                    .entry(endpoint.replica.clone())
                    .or_default()
                    .insert(key.clone(), endpoint.clone());
            }
        }
        self.slices = entries
            .into_iter()
            .filter(|(_, endpoints)| !endpoints.is_empty())
            .collect();
    }

    fn snapshot(&self) -> Snapshot<ReplicaEndpoint> {
        Snapshot {
            items: self
                .replicas
                .keys()
                .filter_map(|id| self.served(id).cloned())
                .collect(),
            completeness: Completeness::Exact,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use kube::runtime::watcher;

    use super::super::cache::SliceCache;
    use super::super::mcs::{
        KARMADA_SLICE_MANAGER, KUBERNETES_SERVICE_NAME_LABEL, MANAGED_BY_LABEL,
    };
    use super::*;

    const ROUTER: &str = "global-router";

    fn naming() -> ReplicaNaming {
        ReplicaNaming {
            mcs: McsImplementation::Karmada,
            service: ROUTER.to_owned(),
            port_name: "state".to_owned(),
        }
    }

    /// A Karmada-imported slice of the router Service from `cluster`, with
    /// endpoints as (address, Ready, pod).
    fn slice(
        cluster: &str,
        suffix: &str,
        endpoints: &[(&str, Option<bool>, Option<&str>)],
    ) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_owned(),
            metadata: ObjectMeta {
                name: Some(format!("imported-{cluster}-{ROUTER}-{suffix}")),
                namespace: Some("dynamo".to_owned()),
                labels: Some(BTreeMap::from([
                    (
                        MANAGED_BY_LABEL.to_owned(),
                        KARMADA_SLICE_MANAGER.to_owned(),
                    ),
                    (
                        KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
                        format!("derived-{ROUTER}"),
                    ),
                ])),
                ..Default::default()
            },
            ports: Some(vec![EndpointPort {
                name: Some("state".to_owned()),
                port: Some(50051),
                ..Default::default()
            }]),
            endpoints: endpoints
                .iter()
                .map(|(address, ready, pod)| Endpoint {
                    addresses: vec![(*address).to_owned()],
                    conditions: Some(EndpointConditions {
                        ready: *ready,
                        ..Default::default()
                    }),
                    target_ref: pod.map(|pod| ObjectReference {
                        kind: Some("Pod".to_owned()),
                        name: Some(pod.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn cache() -> Arc<SliceCache<ReplicaIndex>> {
        Arc::new(SliceCache::new(naming()))
    }

    fn sync(cache: &SliceCache<ReplicaIndex>, slices: Vec<EndpointSlice>) {
        cache.apply(watcher::Event::Init);
        for slice in slices {
            cache.apply(watcher::Event::InitApply(slice));
        }
        cache.apply(watcher::Event::InitDone);
    }

    fn id(name: &str) -> ReplicaId {
        ReplicaId(name.to_owned())
    }

    /// Served replicas as (id, URL), sorted.
    fn served(cache: &SliceCache<ReplicaIndex>) -> Vec<(String, String)> {
        let mut replicas: Vec<(String, String)> = cache
            .snapshot()
            .expect("synced")
            .items
            .into_iter()
            .map(|endpoint| (endpoint.replica.0, endpoint.url.to_string()))
            .collect();
        replicas.sort();
        replicas
    }

    #[test]
    fn ready_endpoints_become_replicas() {
        let cache = cache();
        sync(
            &cache,
            vec![slice(
                "ohio",
                "a1b2c",
                &[
                    ("10.42.0.5", Some(true), Some("global-router-0")),
                    // The EndpointSlice API reads a missing `ready` as ready.
                    ("10.42.0.6", None, None),
                    ("10.42.0.7", Some(false), Some("global-router-2")),
                ],
            )],
        );
        assert_eq!(
            served(&cache),
            [
                (
                    "ohio/10.42.0.6".to_owned(),
                    "http://10.42.0.6:50051/".to_owned()
                ),
                (
                    "ohio/global-router-0".to_owned(),
                    "http://10.42.0.5:50051/".to_owned()
                ),
            ]
        );
    }

    #[test]
    fn brackets_ipv6_addresses() {
        let cache = cache();
        let mut ipv6 = slice(
            "ohio",
            "a1b2c",
            &[("fd00::5", Some(true), Some("global-router-0"))],
        );
        ipv6.address_type = "IPv6".to_owned();
        sync(&cache, vec![ipv6]);
        assert_eq!(
            served(&cache),
            [(
                "ohio/global-router-0".to_owned(),
                "http://[fd00::5]:50051/".to_owned()
            )]
        );
    }

    #[test]
    fn skips_slices_without_the_port_or_of_another_service() {
        let mut no_port = slice("ohio", "a1b2c", &[("10.42.0.5", Some(true), None)]);
        no_port.ports = Some(Vec::new());
        let mut other = slice("ohio", "d3e4f", &[("10.42.0.6", Some(true), None)]);
        other.metadata.labels.as_mut().unwrap().insert(
            KUBERNETES_SERVICE_NAME_LABEL.to_owned(),
            "derived-echo".to_owned(),
        );
        let cache = cache();
        sync(&cache, vec![no_port, other]);
        assert!(served(&cache).is_empty());
    }

    #[test]
    fn changes_publish_added_and_removed() {
        let cache = cache();
        sync(&cache, Vec::new());
        let mut events = cache.events.subscribe();

        let one = slice(
            "ohio",
            "a1b2c",
            &[("10.42.0.5", Some(true), Some("global-router-0"))],
        );
        cache.apply(watcher::Event::Apply(one.clone()));
        assert!(
            matches!(events.try_recv().unwrap(), WatchEvent::Added(e) if e.replica == id("ohio/global-router-0"))
        );
        // The same slice again changes nothing.
        cache.apply(watcher::Event::Apply(one.clone()));
        assert!(events.try_recv().is_err());
        // The pod moves to a new address: the replica changes.
        cache.apply(watcher::Event::Apply(slice(
            "ohio",
            "a1b2c",
            &[("10.42.0.9", Some(true), Some("global-router-0"))],
        )));
        assert!(
            matches!(events.try_recv().unwrap(), WatchEvent::Added(e) if e.url.as_str() == "http://10.42.0.9:50051/")
        );
        // It stops being Ready.
        cache.apply(watcher::Event::Apply(slice(
            "ohio",
            "a1b2c",
            &[("10.42.0.9", Some(false), Some("global-router-0"))],
        )));
        assert!(
            matches!(events.try_recv().unwrap(), WatchEvent::Removed(r) if r == id("ohio/global-router-0"))
        );
        assert!(cache.read().index.replicas.is_empty());
    }

    #[test]
    fn replica_in_two_slices_stays_until_both_drop_it() {
        let cache = cache();
        let first = slice(
            "ohio",
            "a1b2c",
            &[("10.42.0.5", Some(true), Some("global-router-0"))],
        );
        let second = slice(
            "ohio",
            "d3e4f",
            &[("10.42.0.5", Some(true), Some("global-router-0"))],
        );
        sync(&cache, vec![first.clone(), second]);
        assert_eq!(served(&cache).len(), 1);
        let mut events = cache.events.subscribe();
        cache.apply(watcher::Event::Delete(first));
        assert!(events.try_recv().is_err());
        assert_eq!(served(&cache).len(), 1);
        cache.apply(watcher::Event::Delete(slice("ohio", "d3e4f", &[])));
        assert!(matches!(events.try_recv().unwrap(), WatchEvent::Removed(_)));
        assert!(served(&cache).is_empty());
    }

    #[test]
    fn replicas_from_two_clusters_keep_their_names_apart() {
        let cache = cache();
        sync(
            &cache,
            vec![
                slice(
                    "ohio",
                    "a1b2c",
                    &[("10.42.0.5", Some(true), Some("global-router-0"))],
                ),
                slice(
                    "west",
                    "d3e4f",
                    &[("10.43.0.5", Some(true), Some("global-router-0"))],
                ),
            ],
        );
        let ids: Vec<String> = served(&cache).into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, ["ohio/global-router-0", "west/global-router-0"]);
    }

    #[test]
    fn relist_replaces_everything() {
        let cache = cache();
        sync(
            &cache,
            vec![slice(
                "ohio",
                "a1b2c",
                &[("10.42.0.5", Some(true), Some("global-router-0"))],
            )],
        );
        let mut events = cache.events.subscribe();
        sync(
            &cache,
            vec![slice(
                "west",
                "d3e4f",
                &[("10.43.0.5", Some(true), Some("global-router-1"))],
            )],
        );
        let WatchEvent::Resync(snapshot) = events.try_recv().unwrap() else {
            panic!("expected Resync");
        };
        let ids: Vec<ReplicaId> = snapshot.items.into_iter().map(|e| e.replica).collect();
        assert_eq!(ids, [id("west/global-router-1")]);
        assert_eq!(
            served(&cache),
            [(
                "west/global-router-1".to_owned(),
                "http://10.43.0.5:50051/".to_owned()
            )]
        );
    }
}
