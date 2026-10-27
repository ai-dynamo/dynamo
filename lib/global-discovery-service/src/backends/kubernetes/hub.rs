// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pools from imported EndpointSlices: the Global Discovery Service's MCS
//! store.

use std::collections::{HashMap, HashSet};
use std::net::IpAddr;

use crate::pool_discovery::{
    Completeness, Directory, PoolDiscoveryError, PoolEvent, PoolRecord, Revision, Snapshot,
    WatchEvent, WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::{PoolKey, PoolLocation, address_identity};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use super::cache::{SliceDirectory, SliceIndex, SliceKey};
use super::mcs::{self, McsImplementation};

/// One pool per (source cluster, namespace, DGD name), from imported
/// EndpointSlices. MCS merges same-named exports from several clusters into
/// one import, so pools are split by source cluster. Lists are `Exact`. Each
/// slice event updates one pool.
///
/// One watch per directory feeds a cache that serves `list` and every
/// `list_and_watch` stream. The watch starts on first use and stops when the
/// directory is dropped.
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
impl Directory<PoolRecord, PoolKey> for McsPoolDirectory {
    async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
        self.0.list().await
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
}

impl PoolNaming {
    /// Source cluster from [`mcs::source_cluster`]; DGD name from the
    /// Service name without `pool_service_suffix`. `None` for imports that
    /// are not pools.
    fn pool_key(&self, slice: &EndpointSlice) -> Option<PoolKey> {
        let service = mcs::service_name(self.mcs, slice)?;
        let dgd_name = service
            .strip_suffix(self.pool_service_suffix.as_str())
            .filter(|name| !name.is_empty())?;
        let cluster = mcs::source_cluster(self.mcs, slice)?;
        let namespace = slice.metadata.namespace.as_deref()?;
        PoolKey::new(cluster, namespace, dgd_name).ok()
    }

    fn slice(&self, slice: &EndpointSlice) -> Option<SliceEntry> {
        Some(SliceEntry {
            pool: self.pool_key(slice)?,
            ready: ready_addresses(slice),
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

struct SliceEntry {
    pool: PoolKey,
    ready: Vec<IpAddr>,
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
}

impl PoolIndex {
    /// Recomputes one pool from its slices. Returns the event to publish if
    /// the record changed.
    fn rebuild(&mut self, key: &PoolKey) -> Option<PoolEvent> {
        let state = self.pools.get(key)?;
        let mut ready: Vec<IpAddr> = state
            .slices
            .iter()
            .filter_map(|slice| self.slices.get(slice))
            .flat_map(|entry| entry.ready.iter().copied())
            .collect();
        ready.sort_unstable();
        ready.dedup();

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
        let bound = match state.bound {
            Some(current) if ready.contains(&current) => current,
            _ => lowest,
        };
        let mut record = PoolRecord {
            key: key.clone(),
            location: location(key.site_id()),
            model: None,
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: Vec::new(),
            relay_identity: Some(address_identity(bound)),
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
        state.bound = Some(bound);
        state.record = Some(record.clone());
        Some(WatchEvent::Added(record))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use futures::{FutureExt, StreamExt};
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use kube::runtime::watcher;

    use super::super::cache::SliceCache;
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
        let cache = cache();
        sync(
            &cache,
            vec![slice("west", "mocker", "a1b2c", &[("10.43.66.175", true)])],
        );
        let record = record(&cache, &key("west", "mocker")).expect("live pool");
        assert_eq!(record.location.region, "west");
        assert_eq!(record.location.cluster.as_deref(), Some("west"));
        assert_eq!(record.relay_identity.as_deref(), Some("ip:10.43.66.175"));
        assert_eq!(record.model, None);
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
        assert!(record(&cache, &key("west", "mocker")).is_some());
        assert!(record(&cache, &key("ohio", "mocker")).is_some());
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
        let cache = cache();
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
