// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools from imported EndpointSlices: the Global Discovery Service's MCS
//! store.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::net::IpAddr;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, PoolEvent, PoolListing, Snapshot, WatchEvent,
    WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::{PoolKey, PoolLocation, address_identity};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use super::cache::{SliceDirectory, SliceIndex, SliceKey, ready_endpoints};
use super::mcs::{self, McsImplementation};

/// One pool per (source cluster, namespace, DGD name), from imported
/// EndpointSlices. MCS merges same-named exports from several clusters into
/// one import, so pools are split by source cluster. Lists are `Exact`. Each
/// slice event updates one pool.
///
/// One watch per directory feeds a cache that serves `list` and every
/// `list_and_watch` stream. The watch starts on first use and stops when the
/// directory is dropped.
#[derive(Debug)]
pub struct McsPoolDirectory(SliceDirectory<PoolIndex>);

impl McsPoolDirectory {
    /// `imports` reaches the cluster where the imported pool slices are: the
    /// hub for standard MCS. For Karmada, the Karmada control plane, which
    /// collects every exported Service's slices, so no per-pool
    /// `ServiceImport` is needed. A member cluster with a `ServiceImport` per
    /// pool also works. Each PoolRelay Service is named
    /// `<DGD name><pool_service_suffix>`.
    pub fn new(imports: Client, mcs: McsImplementation, pool_service_suffix: String) -> Self {
        Self(SliceDirectory::new(
            Api::all(imports),
            mcs::imported_selector(mcs),
            PoolNaming {
                mcs,
                pool_service_suffix,
            },
        ))
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for McsPoolDirectory {
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        self.0.list().await
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        self.0.list_and_watch(cancel_token).await
    }
}

/// Turns imported slices into pool listings.
struct PoolNaming {
    mcs: McsImplementation,
    pool_service_suffix: String,
}

impl PoolNaming {
    /// Source cluster from [`mcs::source_cluster`]; DGD name from the
    /// Service name without `pool_service_suffix`. `None` for imports that
    /// are not pools, and for a pool slice from an unknown cluster, which is
    /// logged.
    fn pool_key(&self, slice: &EndpointSlice) -> Option<PoolKey> {
        let service = mcs::service_name(self.mcs, slice)?;
        let dgd_name = service
            .strip_suffix(self.pool_service_suffix.as_str())
            .filter(|name| !name.is_empty())?;
        let Some(cluster) = mcs::source_cluster(self.mcs, slice) else {
            mcs::warn_unknown_cluster(slice, service);
            return None;
        };
        let namespace = slice.metadata.namespace.as_deref()?;
        PoolKey::new(cluster, namespace, dgd_name).ok()
    }

    fn slice(&self, slice: &EndpointSlice) -> Option<SliceEntry> {
        Some(SliceEntry {
            pool: self.pool_key(slice)?,
            relays: ready_relays(slice),
        })
    }
}

/// MCS knows only the pool's cluster, so its location is named after it.
fn location(cluster: &str) -> PoolLocation {
    PoolLocation {
        region: cluster.to_owned(),
        availability_zone: None,
        cluster: Some(cluster.to_owned()),
        datacenter: None,
    }
}

/// A pool is live while a PoolRelay endpoint is Ready. Ready can be stale
/// when a cluster loses contact with the MCS implementation, so the runtime
/// also drops a pool whose Relay connection is gone.
///
/// Each Ready address comes with its pod: the endpoint's `targetRef`, or the
/// address itself when the endpoint has none. A dual-stack pod is in two
/// slices, one per address family.
fn ready_relays(slice: &EndpointSlice) -> Vec<(String, IpAddr)> {
    ready_endpoints(slice)
        .flat_map(|endpoint| {
            let pod = endpoint
                .target_ref
                .as_ref()
                .and_then(|target| target.name.as_deref());
            endpoint.addresses.iter().filter_map(move |address| {
                let address: IpAddr = address.parse().ok()?;
                let pod = pod.map_or_else(|| address.to_string(), str::to_owned);
                Some((pod, address))
            })
        })
        .collect()
}

struct SliceEntry {
    pool: PoolKey,
    relays: Vec<(String, IpAddr)>,
}

#[derive(Default)]
struct PoolState {
    slices: HashSet<SliceKey>,
    /// `Some` while the pool is live.
    listing: Option<PoolListing>,
    /// The PoolRelay pod in `listing.relay_identities`.
    bound: Option<String>,
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
        _naming: &PoolNaming,
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
            events.extend(self.rebuild(&pool));
        }
        if let Some(pool) = new_pool {
            self.pools
                .entry(pool.clone())
                .or_default()
                .slices
                .insert(key);
            events.extend(self.rebuild(&pool));
        }
        events
    }

    /// A relist: every slice at once. Rebuilds every pool; the cache
    /// publishes one `Resync` instead of per-pool events.
    fn replace(&mut self, _naming: &PoolNaming, slices: HashMap<SliceKey, SliceEntry>) {
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
            self.rebuild(pool);
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
    fn rebuild(&mut self, key: &PoolKey) -> Option<PoolEvent> {
        let state = self.pools.get(key)?;
        let mut relays: BTreeMap<&str, BTreeSet<IpAddr>> = BTreeMap::new();
        for entry in state
            .slices
            .iter()
            .filter_map(|slice| self.slices.get(slice))
        {
            for (pod, address) in &entry.relays {
                relays.entry(pod).or_default().insert(*address);
            }
        }
        // Keep the bound pod while it stays Ready, so a rolling update that
        // briefly shows two Ready PoolRelays does not switch back and forth.
        // Otherwise bind the pod with the lowest address.
        let bound = state
            .bound
            .as_deref()
            .filter(|pod| relays.contains_key(*pod))
            .or_else(|| {
                relays
                    .iter()
                    .min_by_key(|(_, addresses)| addresses.first())
                    .map(|(pod, _)| *pod)
            })
            .map(|pod| {
                let identities: Vec<String> =
                    relays[pod].iter().copied().map(address_identity).collect();
                (pod.to_owned(), identities)
            });

        let state = self.pools.get_mut(key)?;
        let Some((pod, relay_identities)) = bound else {
            state.bound = None;
            let was_live = state.listing.take().is_some();
            if state.slices.is_empty() {
                self.pools.remove(key);
            }
            return was_live.then(|| WatchEvent::Removed(key.clone()));
        };
        let listing = PoolListing {
            key: key.clone(),
            location: location(key.site_id()),
            model: None,
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
            relay_identities,
        };
        state.bound = Some(pod);
        if state.listing.as_ref() == Some(&listing) {
            return None;
        }
        state.listing = Some(listing.clone());
        Some(WatchEvent::Added(listing))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use futures::{FutureExt, StreamExt};
    use std::time::Duration;

    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use kube::runtime::watcher;

    use super::super::cache::{RECOVERED_AFTER, STALE_AFTER, SliceCache};
    use super::mcs::{KARMADA_SLICE_MANAGER, KUBERNETES_SERVICE_NAME_LABEL, MANAGED_BY_LABEL};
    use super::*;

    const SUFFIX: &str = "-pool-relay";

    fn naming() -> PoolNaming {
        PoolNaming {
            mcs: McsImplementation::Karmada,
            pool_service_suffix: SUFFIX.to_owned(),
        }
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
        // A relist that keeps failing: kube sends `Init` before each attempt,
        // which is no sign that the watch works.
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

        // A listed slice means the watch works again.
        cache.apply(watcher::Event::InitApply(ready()));
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
        assert_eq!(listing.location.region, "west");
        assert_eq!(listing.location.cluster.as_deref(), Some("west"));
        assert_eq!(listing.relay_identities, ["ip:10.43.66.175"]);
        assert_eq!(listing.model, None);
    }

    #[test]
    fn a_dual_stack_relay_is_one_pod_with_both_addresses() {
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
        assert_eq!(listing.relay_identities, ["ip:10.43.0.7", "ip:fd00::7"]);
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
        assert!(matches!(events.try_recv().unwrap(), WatchEvent::Added(_)));
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
        // Ready again.
        cache.apply(watcher::Event::Apply(live.clone()));
        assert!(matches!(events.try_recv().unwrap(), WatchEvent::Added(_)));
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
    fn bound_relay_stays_during_a_rolling_update() {
        let cache = cache();
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
            listing(&cache, &pool).unwrap().relay_identities,
            [address_identity("10.43.0.9".parse().unwrap())]
        );
        // The old Relay leaves; the new one is bound.
        cache.apply(watcher::Event::Apply(slice(
            "west",
            "mocker",
            "a1b2c",
            &[("10.43.0.2", true)],
        )));
        assert_eq!(
            listing(&cache, &pool).unwrap().relay_identities,
            [address_identity("10.43.0.2".parse().unwrap())]
        );
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
        let cache = cache();
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
}
