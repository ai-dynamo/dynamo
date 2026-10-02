// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned wire types and a local replica for a sequenced carrier feed.

use std::collections::VecDeque;
use std::sync::Arc;

use arc_swap::ArcSwap;
use dynamo_tokens::PositionalLineageHash;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::indexer::positional_carrier::{CarrierHit, PositionalCarrierIndex};

pub const CARRIER_FEED_TOPIC: &[u8] = b"kvbm.carrier_feed.v1";
pub const CARRIER_FEED_VERSION: u32 = 1;

/// A 32-byte cache manifest digest.
pub type ManifestKey = [u8; 32];
/// The hub instance ID of a holder.
pub type FeedHolder = u128;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FeedKind {
    Block,
    Carrier,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CarrierFeedOp {
    Insert(Vec<PositionalLineageHash>),
    Remove(Vec<PositionalLineageHash>),
    RemoveHolder,
    ReplaceHolder(Vec<PositionalLineageHash>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CarrierFeedFrame {
    pub version: u32,
    /// Random per hub process start. A change means the replica must resync.
    pub epoch: u64,
    /// Starts at 1 per epoch and advances by one for each frame.
    pub seq: u64,
    pub manifest: ManifestKey,
    pub kind: FeedKind,
    /// Capacity in positions for `manifest` when this frame was emitted.
    pub max_positions: u64,
    pub holder: FeedHolder,
    pub op: CarrierFeedOp,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CarrierFeedSnapshot {
    pub version: u32,
    pub epoch: u64,
    /// Frames through `seq` are reflected; the next frame has sequence `seq + 1`.
    pub seq: u64,
    pub manifests: Vec<ManifestSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestSnapshot {
    pub manifest: ManifestKey,
    pub kind: FeedKind,
    pub max_positions: u64,
    /// Holders are sorted by ID and their hashes by ascending position.
    pub holders: Vec<HolderSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HolderSnapshot {
    pub holder: FeedHolder,
    pub hashes: Vec<PositionalLineageHash>,
}

pub fn encode_frame(frame: &CarrierFeedFrame) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(frame)
}

pub fn decode_frame(bytes: &[u8]) -> Result<CarrierFeedFrame, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

pub fn encode_snapshot(
    snapshot: &CarrierFeedSnapshot,
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(snapshot)
}

pub fn decode_snapshot(bytes: &[u8]) -> Result<CarrierFeedSnapshot, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

pub struct ReplicaManifest {
    pub kind: FeedKind,
    pub index: PositionalCarrierIndex<FeedHolder>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedApply {
    Applied,
    Stale,
    NeedsSnapshot,
    UnsupportedVersion,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CarrierFeedError {
    #[error("unsupported carrier feed version")]
    UnsupportedVersion,
}

pub struct CarrierFeedReplica {
    manifests: ArcSwap<FxHashMap<ManifestKey, Arc<ReplicaManifest>>>,
    state: Mutex<ReplicaCursor>,
}

struct ReplicaCursor {
    synced: Option<(u64, u64)>,
    pending: VecDeque<CarrierFeedFrame>,
    max_pending: usize,
}

impl CarrierFeedReplica {
    pub fn new(max_pending: usize) -> Self {
        Self {
            manifests: ArcSwap::from_pointee(FxHashMap::default()),
            state: Mutex::new(ReplicaCursor {
                synced: None,
                pending: VecDeque::new(),
                max_pending,
            }),
        }
    }

    pub fn is_synced(&self) -> bool {
        self.state.lock().synced.is_some()
    }

    /// Returns the current `(epoch, sequence)` cursor.
    pub fn cursor(&self) -> Option<(u64, u64)> {
        self.state.lock().synced
    }

    pub fn apply(&self, frame: CarrierFeedFrame) -> FeedApply {
        if frame.version != CARRIER_FEED_VERSION {
            return FeedApply::UnsupportedVersion;
        }

        let mut state = self.state.lock();
        let Some((epoch, seq)) = state.synced else {
            Self::queue_pending(&mut state, frame);
            return FeedApply::NeedsSnapshot;
        };

        if frame.epoch != epoch || frame.seq > seq.saturating_add(1) {
            state.synced = None;
            state.pending.clear();
            Self::queue_pending(&mut state, frame);
            return FeedApply::NeedsSnapshot;
        }
        if frame.seq <= seq {
            return FeedApply::Stale;
        }

        self.apply_frame(&frame);
        state.synced = Some((epoch, frame.seq));
        FeedApply::Applied
    }

    pub fn install_snapshot(
        &self,
        snapshot: CarrierFeedSnapshot,
    ) -> Result<FeedApply, CarrierFeedError> {
        if snapshot.version != CARRIER_FEED_VERSION {
            return Err(CarrierFeedError::UnsupportedVersion);
        }

        let mut state = self.state.lock();
        let mut manifests = FxHashMap::default();
        for manifest in snapshot.manifests {
            let index = PositionalCarrierIndex::new(manifest.max_positions);
            for holder in manifest.holders {
                index.replace_holder(holder.holder, &holder.hashes);
            }
            manifests.insert(
                manifest.manifest,
                Arc::new(ReplicaManifest {
                    kind: manifest.kind,
                    index,
                }),
            );
        }
        self.manifests.store(Arc::new(manifests));

        let epoch = snapshot.epoch;
        let snapshot_seq = snapshot.seq;
        let mut pending = std::mem::take(&mut state.pending);
        state.synced = Some((epoch, snapshot_seq));
        while let Some(frame) = pending.pop_front() {
            if frame.epoch != epoch || frame.seq <= snapshot_seq {
                continue;
            }

            let Some((cursor_epoch, cursor_seq)) = state.synced else {
                state.pending.push_back(frame);
                state.pending.append(&mut pending);
                return Ok(FeedApply::NeedsSnapshot);
            };
            if frame.epoch != cursor_epoch || frame.seq != cursor_seq.saturating_add(1) {
                state.synced = None;
                state.pending.push_back(frame);
                state.pending.append(&mut pending);
                return Ok(FeedApply::NeedsSnapshot);
            }

            self.apply_frame(&frame);
            state.synced = Some((epoch, frame.seq));
        }

        Ok(FeedApply::Applied)
    }

    pub fn kind(&self, manifest: &ManifestKey) -> Option<FeedKind> {
        let manifests = self.manifests.load();
        manifests.get(manifest).map(|entry| entry.kind)
    }

    /// Query hashes must be ordered by ascending `position()`.
    pub fn deepest(
        &self,
        manifest: &ManifestKey,
        hashes: &[PositionalLineageHash],
    ) -> Option<CarrierHit<FeedHolder>> {
        let manifests = self.manifests.load();
        manifests.get(manifest)?.index.deepest(hashes)
    }

    /// Query hashes must be ordered by ascending `position()`.
    pub fn deepest_by_holder(
        &self,
        manifest: &ManifestKey,
        hashes: &[PositionalLineageHash],
    ) -> Vec<(FeedHolder, PositionalLineageHash)> {
        let manifests = self.manifests.load();
        manifests
            .get(manifest)
            .map(|entry| entry.index.deepest_by_holder(hashes))
            .unwrap_or_default()
    }

    fn queue_pending(state: &mut ReplicaCursor, frame: CarrierFeedFrame) {
        if state.max_pending == 0 {
            return;
        }
        state.pending.push_back(frame);
        while state.pending.len() > state.max_pending {
            state.pending.pop_front();
        }
    }

    fn apply_frame(&self, frame: &CarrierFeedFrame) {
        let manifests = self.manifests.load_full();
        let existing = manifests.get(&frame.manifest).cloned();
        let manifest = match existing {
            Some(manifest) if manifest.kind == frame.kind => {
                manifest.index.grow_to(frame.max_positions);
                manifest
            }
            _ => {
                let manifest = Arc::new(ReplicaManifest {
                    kind: frame.kind,
                    index: PositionalCarrierIndex::new(frame.max_positions),
                });
                self.manifests.rcu(|current| {
                    let mut next = (**current).clone();
                    next.insert(frame.manifest, Arc::clone(&manifest));
                    Arc::new(next)
                });
                manifest
            }
        };

        match &frame.op {
            CarrierFeedOp::Insert(hashes) => manifest.index.insert(frame.holder, hashes),
            CarrierFeedOp::Remove(hashes) => manifest.index.remove(frame.holder, hashes),
            CarrierFeedOp::RemoveHolder => manifest.index.remove_holder(frame.holder),
            CarrierFeedOp::ReplaceHolder(hashes) => {
                manifest.index.replace_holder(frame.holder, hashes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CARRIER_FEED_VERSION, CarrierFeedFrame, CarrierFeedOp, CarrierFeedReplica,
        CarrierFeedSnapshot, FeedApply, FeedKind, HolderSnapshot, ManifestSnapshot, decode_frame,
        decode_snapshot, encode_frame, encode_snapshot,
    };
    use dynamo_tokens::PositionalLineageHash;

    const BLOCK_SIZE: u32 = 4;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    fn plhs(blocks: u32) -> TestResult<Vec<PositionalLineageHash>> {
        Ok(dynamo_kv_hashing::Request::builder()
            .tokens((0..blocks * BLOCK_SIZE).collect())
            .build()?
            .positional_lineage_hashes(BLOCK_SIZE)?)
    }

    fn manifest_key(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn frame(
        epoch: u64,
        seq: u64,
        manifest: [u8; 32],
        kind: FeedKind,
        max_positions: u64,
        holder: u128,
        op: CarrierFeedOp,
    ) -> CarrierFeedFrame {
        CarrierFeedFrame {
            version: CARRIER_FEED_VERSION,
            epoch,
            seq,
            manifest,
            kind,
            max_positions,
            holder,
            op,
        }
    }

    fn snapshot(epoch: u64, seq: u64, manifests: Vec<ManifestSnapshot>) -> CarrierFeedSnapshot {
        CarrierFeedSnapshot {
            version: CARRIER_FEED_VERSION,
            epoch,
            seq,
            manifests,
        }
    }

    #[test]
    fn frame_and_snapshot_round_trip() -> TestResult<()> {
        let hashes = plhs(2)?;
        let frame = frame(
            7,
            3,
            manifest_key(1),
            FeedKind::Carrier,
            4,
            11,
            CarrierFeedOp::ReplaceHolder(hashes.clone()),
        );
        assert_eq!(decode_frame(&encode_frame(&frame)?)?, frame);

        let snapshot = snapshot(
            7,
            2,
            vec![ManifestSnapshot {
                manifest: manifest_key(1),
                kind: FeedKind::Carrier,
                max_positions: 4,
                holders: vec![HolderSnapshot { holder: 11, hashes }],
            }],
        );
        assert_eq!(decode_snapshot(&encode_snapshot(&snapshot)?)?, snapshot);
        Ok(())
    }

    #[test]
    fn buffers_frames_until_a_snapshot_is_installed() -> TestResult<()> {
        let hashes = plhs(2)?;
        let replica = CarrierFeedReplica::new(4);
        assert_eq!(
            replica.apply(frame(
                1,
                1,
                manifest_key(1),
                FeedKind::Block,
                2,
                10,
                CarrierFeedOp::Insert(vec![hashes[0]]),
            )),
            FeedApply::NeedsSnapshot
        );
        assert!(!replica.is_synced());
        assert_eq!(replica.deepest(&manifest_key(1), &hashes), None);
        assert!(
            replica
                .deepest_by_holder(&manifest_key(1), &hashes)
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn installs_snapshot_and_replays_only_the_buffered_tail() -> TestResult<()> {
        let hashes = plhs(3)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        assert_eq!(
            replica.apply(frame(
                2,
                1,
                key,
                FeedKind::Block,
                3,
                10,
                CarrierFeedOp::RemoveHolder,
            )),
            FeedApply::NeedsSnapshot
        );
        assert_eq!(
            replica.apply(frame(
                2,
                2,
                key,
                FeedKind::Block,
                3,
                20,
                CarrierFeedOp::Insert(vec![hashes[1]]),
            )),
            FeedApply::NeedsSnapshot
        );

        assert_eq!(
            replica.install_snapshot(snapshot(
                2,
                1,
                vec![ManifestSnapshot {
                    manifest: key,
                    kind: FeedKind::Block,
                    max_positions: 3,
                    holders: Vec::new(),
                }],
            ))?,
            FeedApply::Applied
        );
        assert_eq!(replica.cursor(), Some((2, 2)));
        assert_eq!(
            replica
                .deepest(&key, &hashes)
                .map(|hit| (hit.hash, hit.holders)),
            Some((hashes[1], vec![20]))
        );
        Ok(())
    }

    #[test]
    fn applies_in_order_and_ignores_duplicate_frames() -> TestResult<()> {
        let hashes = plhs(2)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(3, 0, Vec::new()))?;

        let frame = frame(
            3,
            1,
            key,
            FeedKind::Block,
            2,
            10,
            CarrierFeedOp::Insert(vec![hashes[0]]),
        );
        assert_eq!(replica.apply(frame.clone()), FeedApply::Applied);
        assert_eq!(replica.apply(frame), FeedApply::Stale);
        assert_eq!(replica.cursor(), Some((3, 1)));
        assert_eq!(replica.deepest(&key, &hashes).unwrap().hash, hashes[0]);
        Ok(())
    }

    #[test]
    fn a_gap_requires_a_snapshot_and_replays_the_tail_after_resync() -> TestResult<()> {
        let hashes = plhs(3)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(4, 0, Vec::new()))?;
        assert_eq!(
            replica.apply(frame(
                4,
                1,
                key,
                FeedKind::Block,
                3,
                10,
                CarrierFeedOp::Insert(vec![hashes[0]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(
            replica.apply(frame(
                4,
                3,
                key,
                FeedKind::Block,
                3,
                20,
                CarrierFeedOp::Insert(vec![hashes[2]]),
            )),
            FeedApply::NeedsSnapshot
        );
        assert!(!replica.is_synced());
        assert_eq!(
            replica.install_snapshot(snapshot(
                4,
                2,
                vec![ManifestSnapshot {
                    manifest: key,
                    kind: FeedKind::Block,
                    max_positions: 3,
                    holders: vec![HolderSnapshot {
                        holder: 10,
                        hashes: vec![hashes[0]],
                    }],
                }],
            ))?,
            FeedApply::Applied
        );
        assert_eq!(replica.cursor(), Some((4, 3)));
        assert_eq!(replica.deepest(&key, &hashes).unwrap().hash, hashes[2]);
        Ok(())
    }

    #[test]
    fn an_epoch_change_requires_a_snapshot() -> TestResult<()> {
        let hashes = plhs(1)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(5, 3, Vec::new()))?;
        assert_eq!(
            replica.apply(frame(
                6,
                1,
                key,
                FeedKind::Block,
                1,
                10,
                CarrierFeedOp::Insert(hashes.clone()),
            )),
            FeedApply::NeedsSnapshot
        );
        assert_eq!(
            replica.install_snapshot(snapshot(6, 0, Vec::new()))?,
            FeedApply::Applied
        );
        assert_eq!(replica.cursor(), Some((6, 1)));
        assert_eq!(replica.deepest(&key, &hashes).unwrap().hash, hashes[0]);
        Ok(())
    }

    #[test]
    fn unsupported_versions_do_not_change_replica_state() {
        let replica = CarrierFeedReplica::new(2);
        let mut frame = frame(
            1,
            1,
            manifest_key(1),
            FeedKind::Block,
            1,
            1,
            CarrierFeedOp::RemoveHolder,
        );
        frame.version += 1;
        assert_eq!(replica.apply(frame), FeedApply::UnsupportedVersion);
        assert_eq!(replica.cursor(), None);
        assert_eq!(
            replica.install_snapshot(CarrierFeedSnapshot {
                version: CARRIER_FEED_VERSION + 1,
                epoch: 1,
                seq: 0,
                manifests: Vec::new(),
            }),
            Err(super::CarrierFeedError::UnsupportedVersion)
        );
        assert_eq!(replica.cursor(), None);
    }

    #[test]
    fn manifests_are_isolated() -> TestResult<()> {
        let hashes = plhs(2)?;
        let first = manifest_key(1);
        let second = manifest_key(2);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(7, 0, Vec::new()))?;
        assert_eq!(
            replica.apply(frame(
                7,
                1,
                first,
                FeedKind::Block,
                2,
                10,
                CarrierFeedOp::Insert(vec![hashes[0]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(
            replica.apply(frame(
                7,
                2,
                second,
                FeedKind::Carrier,
                2,
                20,
                CarrierFeedOp::Insert(vec![hashes[1]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(replica.kind(&first), Some(FeedKind::Block));
        assert_eq!(replica.kind(&second), Some(FeedKind::Carrier));
        assert_eq!(replica.deepest(&first, &hashes).unwrap().hash, hashes[0]);
        assert_eq!(replica.deepest(&second, &hashes).unwrap().hash, hashes[1]);
        Ok(())
    }

    #[test]
    fn deepest_by_holder_returns_each_holders_deepest_hash() -> TestResult<()> {
        let hashes = plhs(3)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(
            7,
            0,
            vec![ManifestSnapshot {
                manifest: key,
                kind: FeedKind::Block,
                max_positions: 3,
                holders: vec![
                    HolderSnapshot {
                        holder: 30,
                        hashes: vec![hashes[0]],
                    },
                    HolderSnapshot {
                        holder: 10,
                        hashes: vec![hashes[0], hashes[1]],
                    },
                    HolderSnapshot {
                        holder: 20,
                        hashes: vec![hashes[2]],
                    },
                ],
            }],
        ))?;

        assert_eq!(
            replica.deepest_by_holder(&key, &hashes),
            vec![(10, hashes[1]), (20, hashes[2]), (30, hashes[0])]
        );
        Ok(())
    }

    #[test]
    fn a_kind_change_replaces_the_manifest_index() -> TestResult<()> {
        let hashes = plhs(2)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(
            8,
            0,
            vec![ManifestSnapshot {
                manifest: key,
                kind: FeedKind::Block,
                max_positions: 2,
                holders: vec![HolderSnapshot {
                    holder: 10,
                    hashes: vec![hashes[0]],
                }],
            }],
        ))?;
        assert_eq!(
            replica.apply(frame(
                8,
                1,
                key,
                FeedKind::Carrier,
                2,
                20,
                CarrierFeedOp::Insert(vec![hashes[1]]),
            )),
            FeedApply::Applied
        );

        assert_eq!(replica.kind(&key), Some(FeedKind::Carrier));
        assert_eq!(
            replica
                .deepest(&key, &hashes)
                .map(|hit| (hit.hash, hit.holders)),
            Some((hashes[1], vec![20]))
        );
        Ok(())
    }

    #[test]
    fn remove_and_replace_holder_update_the_replica() -> TestResult<()> {
        let hashes = plhs(3)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(
            8,
            0,
            vec![ManifestSnapshot {
                manifest: key,
                kind: FeedKind::Block,
                max_positions: 3,
                holders: vec![
                    HolderSnapshot {
                        holder: 10,
                        hashes: vec![hashes[0], hashes[1]],
                    },
                    HolderSnapshot {
                        holder: 20,
                        hashes: vec![hashes[0]],
                    },
                ],
            }],
        ))?;
        assert_eq!(
            replica.apply(frame(
                8,
                1,
                key,
                FeedKind::Block,
                3,
                10,
                CarrierFeedOp::ReplaceHolder(vec![hashes[2]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(
            replica
                .deepest(&key, &hashes)
                .map(|hit| (hit.hash, hit.holders)),
            Some((hashes[2], vec![10]))
        );
        assert_eq!(
            replica.apply(frame(
                8,
                2,
                key,
                FeedKind::Block,
                3,
                10,
                CarrierFeedOp::RemoveHolder,
            )),
            FeedApply::Applied
        );
        assert_eq!(
            replica
                .deepest(&key, &hashes)
                .map(|hit| (hit.hash, hit.holders)),
            Some((hashes[0], vec![20]))
        );
        assert_eq!(
            replica.apply(frame(
                8,
                3,
                key,
                FeedKind::Block,
                3,
                20,
                CarrierFeedOp::Remove(vec![hashes[0]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(replica.deepest(&key, &hashes), None);
        Ok(())
    }

    #[test]
    fn frames_grow_manifest_capacity() -> TestResult<()> {
        let hashes = plhs(3)?;
        let key = manifest_key(1);
        let replica = CarrierFeedReplica::new(4);
        replica.install_snapshot(snapshot(
            9,
            0,
            vec![ManifestSnapshot {
                manifest: key,
                kind: FeedKind::Block,
                max_positions: 1,
                holders: Vec::new(),
            }],
        ))?;
        assert_eq!(
            replica.apply(frame(
                9,
                1,
                key,
                FeedKind::Block,
                3,
                10,
                CarrierFeedOp::Insert(vec![hashes[2]]),
            )),
            FeedApply::Applied
        );
        assert_eq!(replica.deepest(&key, &hashes).unwrap().hash, hashes[2]);
        Ok(())
    }

    #[test]
    fn pending_buffer_drops_oldest_frames() -> TestResult<()> {
        let hashes = plhs(1)?;
        let replica = CarrierFeedReplica::new(2);
        for seq in 1..=3 {
            assert_eq!(
                replica.apply(frame(
                    10,
                    seq,
                    manifest_key(1),
                    FeedKind::Block,
                    1,
                    seq as u128,
                    CarrierFeedOp::Insert(hashes.clone()),
                )),
                FeedApply::NeedsSnapshot
            );
        }
        assert_eq!(
            replica
                .state
                .lock()
                .pending
                .iter()
                .map(|frame| frame.seq)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(
            replica.install_snapshot(snapshot(10, 0, Vec::new()))?,
            FeedApply::NeedsSnapshot
        );
        assert_eq!(
            replica
                .state
                .lock()
                .pending
                .iter()
                .map(|frame| frame.seq)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        Ok(())
    }
}
