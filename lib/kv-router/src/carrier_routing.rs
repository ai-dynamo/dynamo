// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carrier-feed bindings and lock-free carrier-depth lookup.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arc_swap::ArcSwap;
use dynamo_tokens::PositionalLineageHash;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::carrier_feed::{CarrierFeedReplica, FeedHolder, FeedKind, ManifestKey};
use crate::carrier_lookup::{CarrierLookup, CarrierLookupSource};
use crate::protocols::{WorkerId, WorkerWithDpRank};

pub const CARRIER_RUNTIME_DATA_KEY: &str = "kvbm_carrier";

/// Worker-advertised binding to a KVBM hub carrier feed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CarrierWorkerConfig {
    pub hub_url: String,
    pub manifest: String,
    pub block_size: u32,
    pub instance_ids: HashMap<u32, String>,
}

pub trait CarrierFeedConnector: Send + Sync {
    /// Start feeding `replica` from `hub_url` until `cancel` fires.
    fn connect(&self, hub_url: &str, replica: Arc<CarrierFeedReplica>, cancel: CancellationToken);
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CarrierMatchDetails {
    /// Every bound worker rank with its manifest kind.
    ///
    /// `None` means the replica has not seen the manifest yet.
    pub bound: FxHashMap<WorkerWithDpRank, Option<FeedKind>>,
    /// Deepest held PLH position plus one, per bound rank. Absent means zero.
    pub depth_blocks: FxHashMap<WorkerWithDpRank, u32>,
}

struct WorkerBinding {
    hub_url: String,
    manifest: ManifestKey,
    holders: FxHashMap<u32, FeedHolder>,
}

struct HubConnection {
    lookup: Arc<dyn CarrierLookup>,
    cancel: Option<CancellationToken>,
    ref_count: usize,
}

struct MutableState {
    workers: FxHashMap<WorkerId, WorkerBinding>,
    hubs: HashMap<String, HubConnection>,
}

struct CarrierGroup {
    lookup: Arc<dyn CarrierLookup>,
    manifest: ManifestKey,
    holders: FxHashMap<FeedHolder, WorkerWithDpRank>,
}

#[derive(Default)]
struct CarrierSnapshot {
    groups: Vec<CarrierGroup>,
}

pub struct CarrierRouter {
    connector: Option<Arc<dyn CarrierFeedConnector>>,
    lookup_source: Option<Arc<dyn CarrierLookupSource>>,
    state: Mutex<MutableState>,
    snapshot: ArcSwap<CarrierSnapshot>,
}

impl CarrierRouter {
    pub fn new(
        connector: Option<Arc<dyn CarrierFeedConnector>>,
        lookup_source: Option<Arc<dyn CarrierLookupSource>>,
    ) -> Self {
        Self {
            connector,
            lookup_source,
            state: Mutex::new(MutableState {
                workers: FxHashMap::default(),
                hubs: HashMap::new(),
            }),
            snapshot: ArcSwap::from_pointee(CarrierSnapshot::default()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.snapshot.load().groups.is_empty()
    }

    /// Replace a worker's binding. An invalid binding removes any prior binding.
    ///
    /// The manifest must be 64 hexadecimal characters, every instance ID must be a
    /// decimal `u128`, every configured rank must be in `dp_ranks`, and the block
    /// size must match `partition_block_size`. A router without a connector or
    /// lookup source is disabled and ignores upserts.
    pub fn upsert_worker(
        &self,
        worker_id: WorkerId,
        dp_ranks: Range<u32>,
        config: Option<&CarrierWorkerConfig>,
        partition_block_size: u32,
    ) -> Result<(), String> {
        if self.connector.is_none() && self.lookup_source.is_none() {
            return Ok(());
        }

        let Some(config) = config else {
            self.remove_worker(worker_id);
            return Ok(());
        };
        let parsed = match parse_binding(config, dp_ranks, partition_block_size) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.remove_worker(worker_id);
                return Err(error);
            }
        };

        let mut state = self.state.lock();
        let same_hub = state
            .workers
            .get(&worker_id)
            .is_some_and(|binding| binding.hub_url == parsed.hub_url);
        if !same_hub {
            self.remove_worker_locked(&mut state, worker_id);
        }

        let hub_url = parsed.hub_url.clone();
        if !state.hubs.contains_key(&hub_url) {
            let hub = if let Some(lookup) = self
                .lookup_source
                .as_ref()
                .and_then(|source| source.lookup(&hub_url))
            {
                HubConnection {
                    lookup,
                    cancel: None,
                    ref_count: 0,
                }
            } else if let Some(connector) = self.connector.as_ref() {
                let replica = Arc::new(CarrierFeedReplica::new(4096));
                let cancel = CancellationToken::new();
                connector.connect(&hub_url, Arc::clone(&replica), cancel.clone());
                let lookup: Arc<dyn CarrierLookup> = replica;
                HubConnection {
                    lookup,
                    cancel: Some(cancel),
                    ref_count: 0,
                }
            } else {
                self.rebuild_snapshot_locked(&state);
                return Err(format!("no carrier lookup for hub {hub_url}"));
            };
            state.hubs.insert(hub_url.clone(), hub);
        }
        let Some(hub) = state.hubs.get_mut(&hub_url) else {
            return Err(format!("no carrier lookup for hub {hub_url}"));
        };
        if !same_hub {
            hub.ref_count += 1;
        }
        state.workers.insert(
            worker_id,
            WorkerBinding {
                hub_url: parsed.hub_url,
                manifest: parsed.manifest,
                holders: parsed.holders,
            },
        );
        self.rebuild_snapshot_locked(&state);
        Ok(())
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        let mut state = self.state.lock();
        if self.remove_worker_locked(&mut state, worker_id) {
            self.rebuild_snapshot_locked(&state);
        }
    }

    /// Find carrier matches for PLHs ordered by ascending position.
    pub fn find_matches(&self, plhs: &[PositionalLineageHash]) -> Option<CarrierMatchDetails> {
        let snapshot = self.snapshot.load();
        if snapshot.groups.is_empty() {
            return None;
        }

        let mut matches = CarrierMatchDetails::default();
        for group in &snapshot.groups {
            let kind = group.lookup.kind(&group.manifest);
            for worker in group.holders.values() {
                matches.bound.insert(*worker, kind);
            }
            for (holder, hash) in group.lookup.deepest_by_holder(&group.manifest, plhs) {
                if let Some(worker) = group.holders.get(&holder) {
                    let depth =
                        u32::try_from(hash.position().saturating_add(1)).unwrap_or(u32::MAX);
                    matches.depth_blocks.insert(*worker, depth);
                }
            }
        }
        Some(matches)
    }

    fn remove_worker_locked(&self, state: &mut MutableState, worker_id: WorkerId) -> bool {
        let Some(binding) = state.workers.remove(&worker_id) else {
            return false;
        };
        let mut remove_hub = false;
        if let Some(hub) = state.hubs.get_mut(&binding.hub_url) {
            hub.ref_count = hub.ref_count.saturating_sub(1);
            remove_hub = hub.ref_count == 0;
        }
        if remove_hub && let Some(hub) = state.hubs.remove(&binding.hub_url) {
            if let Some(cancel) = hub.cancel {
                cancel.cancel();
            }
        }
        true
    }

    fn rebuild_snapshot_locked(&self, state: &MutableState) {
        let mut groups: HashMap<(String, ManifestKey), FxHashMap<FeedHolder, WorkerWithDpRank>> =
            HashMap::new();
        for (&worker_id, binding) in &state.workers {
            let holders = groups
                .entry((binding.hub_url.clone(), binding.manifest))
                .or_default();
            for (&dp_rank, &holder) in &binding.holders {
                holders.insert(holder, WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }

        let groups = groups
            .into_iter()
            .filter(|(_, holders)| !holders.is_empty())
            .filter_map(|((hub_url, manifest), holders)| {
                state.hubs.get(&hub_url).map(|hub| CarrierGroup {
                    lookup: Arc::clone(&hub.lookup),
                    manifest,
                    holders,
                })
            })
            .collect();
        self.snapshot.store(Arc::new(CarrierSnapshot { groups }));
    }
}

struct ParsedBinding {
    hub_url: String,
    manifest: ManifestKey,
    holders: FxHashMap<u32, FeedHolder>,
}

fn parse_binding(
    config: &CarrierWorkerConfig,
    dp_ranks: Range<u32>,
    partition_block_size: u32,
) -> Result<ParsedBinding, String> {
    if config.hub_url.trim().is_empty() {
        return Err("carrier hub_url must not be empty".to_string());
    }
    if config.block_size != partition_block_size {
        return Err(format!(
            "carrier block_size {} does not match partition block_size {}",
            config.block_size, partition_block_size
        ));
    }
    let manifest = parse_manifest(&config.manifest)?;
    let mut holders = FxHashMap::default();
    for (&dp_rank, instance_id) in &config.instance_ids {
        if !dp_ranks.contains(&dp_rank) {
            return Err(format!(
                "carrier instance rank {dp_rank} is outside worker data-parallel ranks"
            ));
        }
        if instance_id.is_empty() || !instance_id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "carrier instance ID `{instance_id}` must be decimal digits"
            ));
        }
        let holder = instance_id.parse::<u128>().map_err(|error| {
            format!("carrier instance ID `{instance_id}` is not a u128: {error}")
        })?;
        holders.insert(dp_rank, holder);
    }
    Ok(ParsedBinding {
        hub_url: config.hub_url.clone(),
        manifest,
        holders,
    })
}

fn parse_manifest(value: &str) -> Result<ManifestKey, String> {
    if value.len() != 64 {
        return Err("carrier manifest must contain 64 hexadecimal characters".to_string());
    }
    let bytes = value.as_bytes();
    let mut manifest = [0; 32];
    for (index, byte) in manifest.iter_mut().enumerate() {
        let high = hex_nibble(bytes[index * 2])?;
        let low = hex_nibble(bytes[index * 2 + 1])?;
        *byte = (high << 4) | low;
    }
    Ok(manifest)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("carrier manifest must contain only hexadecimal characters".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::carrier_feed::{CarrierFeedSnapshot, FeedApply, HolderSnapshot, ManifestSnapshot};
    use crate::carrier_lookup::{CarrierLookup, CarrierLookupSource};

    const BLOCK_SIZE: u32 = 4;

    #[derive(Clone, Default)]
    struct TestConnector {
        connections: Arc<Mutex<Vec<TestConnection>>>,
        connect_count: Arc<AtomicUsize>,
    }

    struct TestConnection {
        hub_url: String,
        replica: Arc<CarrierFeedReplica>,
        cancel: CancellationToken,
    }

    impl CarrierFeedConnector for TestConnector {
        fn connect(
            &self,
            hub_url: &str,
            replica: Arc<CarrierFeedReplica>,
            cancel: CancellationToken,
        ) {
            self.connect_count.fetch_add(1, Ordering::Relaxed);
            self.connections.lock().push(TestConnection {
                hub_url: hub_url.to_string(),
                replica,
                cancel,
            });
        }
    }

    struct TestLookup {
        manifest: ManifestKey,
        kind: FeedKind,
        matches: Vec<(FeedHolder, PositionalLineageHash)>,
    }

    impl CarrierLookup for TestLookup {
        fn kind(&self, manifest: &ManifestKey) -> Option<FeedKind> {
            (*manifest == self.manifest).then_some(self.kind)
        }

        fn deepest_by_holder(
            &self,
            manifest: &ManifestKey,
            _plhs: &[PositionalLineageHash],
        ) -> Vec<(FeedHolder, PositionalLineageHash)> {
            if *manifest == self.manifest {
                self.matches.clone()
            } else {
                Vec::new()
            }
        }
    }

    #[derive(Default)]
    struct TestLookupSource {
        lookup: Option<(String, Arc<dyn CarrierLookup>)>,
    }

    impl TestLookupSource {
        fn with_lookup(hub_url: &str, lookup: Arc<dyn CarrierLookup>) -> Self {
            Self {
                lookup: Some((hub_url.to_string(), lookup)),
            }
        }
    }

    impl CarrierLookupSource for TestLookupSource {
        fn lookup(&self, hub_url: &str) -> Option<Arc<dyn CarrierLookup>> {
            self.lookup
                .as_ref()
                .filter(|(source_hub_url, _)| source_hub_url == hub_url)
                .map(|(_, lookup)| Arc::clone(lookup))
        }
    }

    fn plhs(blocks: u32) -> Vec<PositionalLineageHash> {
        dynamo_kv_hashing::Request::builder()
            .tokens((0..blocks * BLOCK_SIZE).collect::<Vec<_>>())
            .build()
            .unwrap()
            .positional_lineage_hashes(BLOCK_SIZE)
            .unwrap()
    }

    fn config(hub_url: &str, manifest: u8, instance_ids: &[(u32, u128)]) -> CarrierWorkerConfig {
        CarrierWorkerConfig {
            hub_url: hub_url.to_string(),
            manifest: format!("{manifest:02x}").repeat(32),
            block_size: BLOCK_SIZE,
            instance_ids: instance_ids
                .iter()
                .map(|(rank, holder)| (*rank, holder.to_string()))
                .collect(),
        }
    }

    fn install(
        connection: &TestConnection,
        manifest: u8,
        kind: FeedKind,
        holders: Vec<(u128, Vec<PositionalLineageHash>)>,
    ) {
        let result = connection.replica.install_snapshot(CarrierFeedSnapshot {
            version: crate::carrier_feed::CARRIER_FEED_VERSION,
            epoch: 1,
            seq: 0,
            manifests: vec![ManifestSnapshot {
                manifest: [manifest; 32],
                kind,
                max_positions: 8,
                holders: holders
                    .into_iter()
                    .map(|(holder, hashes)| HolderSnapshot { holder, hashes })
                    .collect(),
            }],
        });
        assert_eq!(result, Ok(FeedApply::Applied));
    }

    #[test]
    fn carrier_matches_report_depth_and_kind() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();
        let connections = connector.connections.lock();
        install(&connections[0], 1, FeedKind::Carrier, vec![(5, plhs(3))]);

        let matches = router.find_matches(&plhs(4)).unwrap();
        let worker = WorkerWithDpRank::new(7, 0);
        assert_eq!(matches.bound[&worker], Some(FeedKind::Carrier));
        assert_eq!(matches.depth_blocks[&worker], 3);
    }

    #[test]
    fn local_lookup_bypasses_connector_and_reports_depth() {
        let connector = TestConnector::default();
        let lookup = Arc::new(TestLookup {
            manifest: [1; 32],
            kind: FeedKind::Carrier,
            matches: vec![(5, plhs(3).pop().unwrap())],
        });
        let source = Arc::new(TestLookupSource::with_lookup("http://hub-a", lookup));
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), Some(source));
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub-a", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();

        let matches = router.find_matches(&plhs(4)).unwrap();
        let worker = WorkerWithDpRank::new(7, 0);
        assert_eq!(matches.bound[&worker], Some(FeedKind::Carrier));
        assert_eq!(matches.depth_blocks[&worker], 3);
        assert_eq!(connector.connect_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn missing_local_lookup_falls_back_to_connector() {
        let connector = TestConnector::default();
        let source = Arc::new(TestLookupSource::default());
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), Some(source));

        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub-b", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();

        assert_eq!(connector.connect_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn missing_lookup_without_connector_does_not_bind_worker() {
        let source = Arc::new(TestLookupSource::default());
        let router = CarrierRouter::new(None, Some(source));

        assert_eq!(
            router.upsert_worker(
                7,
                0..1,
                Some(&config("http://hub", 1, &[(0, 5)])),
                BLOCK_SIZE,
            ),
            Err("no carrier lookup for hub http://hub".to_string())
        );
        assert!(router.is_empty());
    }

    #[test]
    fn removing_last_source_backed_worker_empties_router() {
        let lookup = Arc::new(TestLookup {
            manifest: [1; 32],
            kind: FeedKind::Carrier,
            matches: Vec::new(),
        });
        let source = Arc::new(TestLookupSource::with_lookup("http://hub", lookup));
        let router = CarrierRouter::new(None, Some(source));
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();

        router.remove_worker(7);

        assert!(router.is_empty());
    }

    #[test]
    fn multiple_hubs_and_manifests_are_grouped_independently() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub-a", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();
        router
            .upsert_worker(
                8,
                0..1,
                Some(&config("http://hub-b", 2, &[(0, 6)])),
                BLOCK_SIZE,
            )
            .unwrap();
        let connections = connector.connections.lock();
        assert_eq!(connections[0].hub_url, "http://hub-a");
        install(&connections[0], 1, FeedKind::Carrier, vec![(5, plhs(2))]);
        install(&connections[1], 2, FeedKind::Block, vec![(6, plhs(4))]);

        let matches = router.find_matches(&plhs(4)).unwrap();
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(7, 0)], 2);
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(8, 0)], 4);
    }

    #[test]
    fn multiple_manifests_share_one_hub_replica() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();
        router
            .upsert_worker(
                8,
                0..1,
                Some(&config("http://hub", 2, &[(0, 6)])),
                BLOCK_SIZE,
            )
            .unwrap();
        assert_eq!(connector.connect_count.load(Ordering::Relaxed), 1);
        let connections = connector.connections.lock();
        assert_eq!(
            connections[0]
                .replica
                .install_snapshot(CarrierFeedSnapshot {
                    version: crate::carrier_feed::CARRIER_FEED_VERSION,
                    epoch: 1,
                    seq: 0,
                    manifests: vec![
                        ManifestSnapshot {
                            manifest: [1; 32],
                            kind: FeedKind::Carrier,
                            max_positions: 8,
                            holders: vec![HolderSnapshot {
                                holder: 5,
                                hashes: plhs(1),
                            }],
                        },
                        ManifestSnapshot {
                            manifest: [2; 32],
                            kind: FeedKind::Block,
                            max_positions: 8,
                            holders: vec![HolderSnapshot {
                                holder: 6,
                                hashes: plhs(3),
                            }],
                        },
                    ],
                }),
            Ok(FeedApply::Applied)
        );
        let matches = router.find_matches(&plhs(3)).unwrap();
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(7, 0)], 1);
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(8, 0)], 3);
    }

    #[test]
    fn invalid_bindings_remove_existing_binding() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector)), None);
        let valid = config("http://hub", 1, &[(0, 5)]);
        router
            .upsert_worker(7, 0..1, Some(&valid), BLOCK_SIZE)
            .unwrap();
        let invalid_manifest = CarrierWorkerConfig {
            manifest: "gg".repeat(32),
            ..valid
        };
        assert!(
            router
                .upsert_worker(7, 0..1, Some(&invalid_manifest), BLOCK_SIZE)
                .is_err()
        );
        assert!(router.is_empty());
    }

    #[test]
    fn binding_validation_rejects_bad_instance_rank_id_and_block_size() {
        let router = CarrierRouter::new(Some(Arc::new(TestConnector::default())), None);
        let mut invalid = config("http://hub", 1, &[(2, 5)]);
        assert!(
            router
                .upsert_worker(7, 0..1, Some(&invalid), BLOCK_SIZE)
                .is_err()
        );
        invalid = config("http://hub", 1, &[(0, 5)]);
        invalid.block_size = BLOCK_SIZE + 1;
        assert!(
            router
                .upsert_worker(7, 0..1, Some(&invalid), BLOCK_SIZE)
                .is_err()
        );
        invalid = config("http://hub", 1, &[(0, 5)]);
        invalid.instance_ids.insert(0, "not-a-u128".to_string());
        assert!(
            router
                .upsert_worker(7, 0..1, Some(&invalid), BLOCK_SIZE)
                .is_err()
        );
    }

    #[test]
    fn unknown_manifest_kind_is_reported_without_a_depth() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        router
            .upsert_worker(
                7,
                0..1,
                Some(&config("http://hub", 1, &[(0, 5)])),
                BLOCK_SIZE,
            )
            .unwrap();
        let connections = connector.connections.lock();
        let result = connections[0]
            .replica
            .install_snapshot(CarrierFeedSnapshot {
                version: crate::carrier_feed::CARRIER_FEED_VERSION,
                epoch: 1,
                seq: 0,
                manifests: Vec::new(),
            });
        assert_eq!(result, Ok(FeedApply::Applied));
        let matches = router.find_matches(&plhs(1)).unwrap();
        assert_eq!(matches.bound[&WorkerWithDpRank::new(7, 0)], None);
        assert!(
            !matches
                .depth_blocks
                .contains_key(&WorkerWithDpRank::new(7, 0))
        );
    }

    #[test]
    fn replacing_and_removing_workers_refcounts_hubs() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        let first = config("http://hub", 1, &[(0, 5)]);
        let second = config("http://hub", 1, &[(0, 6)]);
        router
            .upsert_worker(7, 0..1, Some(&first), BLOCK_SIZE)
            .unwrap();
        router
            .upsert_worker(7, 0..1, Some(&second), BLOCK_SIZE)
            .unwrap();
        assert_eq!(connector.connect_count.load(Ordering::Relaxed), 1);
        router
            .upsert_worker(8, 0..1, Some(&first), BLOCK_SIZE)
            .unwrap();
        let cancel = connector.connections.lock()[0].cancel.clone();
        router.remove_worker(7);
        assert!(!cancel.is_cancelled());
        router.remove_worker(8);
        assert!(cancel.is_cancelled());
        assert!(router.is_empty());
    }

    #[test]
    fn disabled_router_ignores_upserts_and_empty_router_matches_nothing() {
        let router = CarrierRouter::new(None, None);
        let config = config("http://hub", 1, &[(0, 5)]);
        assert!(
            router
                .upsert_worker(7, 0..1, Some(&config), BLOCK_SIZE)
                .is_ok()
        );
        assert!(router.is_empty());
        assert!(router.find_matches(&plhs(1)).is_none());
    }

    #[test]
    fn multi_rank_bindings_map_holders_to_each_rank() {
        let connector = TestConnector::default();
        let router = CarrierRouter::new(Some(Arc::new(connector.clone())), None);
        let config = config("http://hub", 1, &[(2, 5), (3, 6)]);
        router
            .upsert_worker(7, 2..4, Some(&config), BLOCK_SIZE)
            .unwrap();
        let connections = connector.connections.lock();
        install(
            &connections[0],
            1,
            FeedKind::Carrier,
            vec![(5, plhs(1)), (6, plhs(3))],
        );
        let matches = router.find_matches(&plhs(3)).unwrap();
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(7, 2)], 1);
        assert_eq!(matches.depth_blocks[&WorkerWithDpRank::new(7, 3)], 3);
    }
}
