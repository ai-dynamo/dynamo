// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GMS placement metadata index.
//!
//! This index is deliberately separate from the radix and lower-tier indexes:
//! routing overlap still comes from the existing KV event path, while GMS
//! placement events carry NIXL bootstrap descriptors keyed by the stable GMS
//! content hash. Routers use this metadata to stamp `gms_placement` onto a
//! selected request without querying the source worker on the request path.

use std::collections::HashMap;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use crate::protocols::{
    ExternalSequenceBlockHash, GmsPlacementBlock, GmsPlacementDescriptor, GmsPlacementEventData,
    GmsPlacementMatch, GmsPlacementStoreData, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlacementSource {
    source_nixl_agent_name: String,
    source_nixl_agent_metadata_hex: String,
    source_nixl_ip: Option<String>,
    source_nixl_listen_port: Option<u16>,
}

impl PlacementSource {
    fn from_store(data: &GmsPlacementStoreData) -> Self {
        Self {
            source_nixl_agent_name: data.source_nixl_agent_name.clone(),
            source_nixl_agent_metadata_hex: data.source_nixl_agent_metadata_hex.clone(),
            source_nixl_ip: data.source_nixl_ip.clone(),
            source_nixl_listen_port: data.source_nixl_listen_port,
        }
    }

    fn into_match(self, descriptors: Vec<Option<GmsPlacementDescriptor>>) -> GmsPlacementMatch {
        GmsPlacementMatch {
            source_nixl_agent_name: self.source_nixl_agent_name,
            source_nixl_agent_metadata_hex: self.source_nixl_agent_metadata_hex,
            source_nixl_ip: self.source_nixl_ip,
            source_nixl_listen_port: self.source_nixl_listen_port,
            descriptors,
        }
    }
}

#[derive(Debug, Clone)]
struct PlacementEntry {
    source: PlacementSource,
    descriptor: GmsPlacementDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlacementKey {
    worker: WorkerWithDpRank,
    content_hash_hex: String,
}

fn normalize_hash(hash: &str) -> Option<String> {
    let trimmed = hash.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

#[derive(Default)]
pub struct GmsPlacementIndex {
    entries: DashMap<PlacementKey, PlacementEntry, FxBuildHasher>,
}

impl GmsPlacementIndex {
    pub fn new() -> Self {
        Self {
            entries: DashMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn apply_event(&self, event: &RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let Some(placement) = event.gms_placement.as_ref() else {
            if matches!(event.event.data, KvCacheEventData::Cleared) {
                self.remove_worker_dp_rank(worker);
            }
            return;
        };

        match placement {
            GmsPlacementEventData::Stored(data) => self.store(worker, data),
            GmsPlacementEventData::Removed(data) => {
                for hash in &data.content_hashes_hex {
                    if let Some(content_hash_hex) = normalize_hash(hash) {
                        self.entries.remove(&PlacementKey {
                            worker,
                            content_hash_hex,
                        });
                    }
                }
            }
            GmsPlacementEventData::Cleared => self.remove_worker_dp_rank(worker),
        }
    }

    fn store(&self, worker: WorkerWithDpRank, data: &GmsPlacementStoreData) {
        if data.source_nixl_agent_name.is_empty() {
            return;
        }
        let source = PlacementSource::from_store(data);
        for block in &data.blocks {
            if !block.descriptor.sealed.unwrap_or(true) {
                continue;
            }
            let Some(content_hash_hex) = normalize_hash(&block.content_hash_hex) else {
                continue;
            };
            self.entries.insert(
                PlacementKey {
                    worker,
                    content_hash_hex,
                },
                PlacementEntry {
                    source: source.clone(),
                    descriptor: block.descriptor.clone(),
                },
            );
        }
    }

    pub fn lookup(
        &self,
        worker: WorkerWithDpRank,
        content_hashes_hex: &[String],
    ) -> Option<GmsPlacementMatch> {
        let mut source: Option<PlacementSource> = None;
        let mut descriptors = Vec::with_capacity(content_hashes_hex.len());
        let mut hits = 0usize;

        for hash in content_hashes_hex {
            let Some(content_hash_hex) = normalize_hash(hash) else {
                descriptors.push(None);
                continue;
            };
            let Some(entry) = self.entries.get(&PlacementKey {
                worker,
                content_hash_hex,
            }) else {
                descriptors.push(None);
                continue;
            };

            match &source {
                Some(existing) if *existing != entry.source => {
                    descriptors.push(None);
                }
                Some(_) => {
                    descriptors.push(Some(entry.descriptor.clone()));
                    hits += 1;
                }
                None => {
                    source = Some(entry.source.clone());
                    descriptors.push(Some(entry.descriptor.clone()));
                    hits += 1;
                }
            }
        }

        if hits == 0 {
            return None;
        }
        source.map(|source| source.into_match(descriptors))
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.entries
            .retain(|key, _| key.worker.worker_id != worker_id);
    }

    pub fn remove_worker_dp_rank(&self, worker: WorkerWithDpRank) {
        self.entries.retain(|key, _| key.worker != worker);
    }

    pub fn dump_events(&self) -> Vec<RouterEvent> {
        let mut grouped: HashMap<(WorkerWithDpRank, PlacementSource), Vec<GmsPlacementBlock>> =
            HashMap::new();
        for entry in self.entries.iter() {
            grouped
                .entry((entry.key().worker, entry.value().source.clone()))
                .or_default()
                .push(GmsPlacementBlock {
                    content_hash_hex: entry.key().content_hash_hex.clone(),
                    descriptor: entry.value().descriptor.clone(),
                });
        }

        let mut events = Vec::with_capacity(grouped.len());
        for (event_id, ((worker, source), blocks)) in grouped.into_iter().enumerate() {
            events.push(RouterEvent::with_gms_placement(
                worker.worker_id,
                KvCacheEvent {
                    event_id: event_id as u64,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None::<ExternalSequenceBlockHash>,
                        start_position: None,
                        blocks: Vec::new(),
                    }),
                    dp_rank: worker.dp_rank,
                },
                StorageTier::External,
                GmsPlacementEventData::Stored(GmsPlacementStoreData {
                    source_nixl_agent_name: source.source_nixl_agent_name,
                    source_nixl_agent_metadata_hex: source.source_nixl_agent_metadata_hex,
                    source_nixl_ip: source.source_nixl_ip,
                    source_nixl_listen_port: source.source_nixl_listen_port,
                    blocks,
                }),
            ));
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{GmsPlacementMemoryRegion, GmsPlacementRemoveData};

    fn descriptor(ptr: u64) -> GmsPlacementDescriptor {
        GmsPlacementDescriptor {
            remote_ptr: ptr,
            size: 128,
            tier: "host".to_string(),
            ranges: vec![GmsPlacementMemoryRegion {
                remote_ptr: ptr,
                size: 128,
                tier: "host".to_string(),
                layer: Some(0),
                offset: Some(0),
            }],
            generation: Some(7),
            sealed: Some(true),
        }
    }

    fn placement_event(worker: WorkerWithDpRank, hashes: &[&str]) -> RouterEvent {
        RouterEvent::with_gms_placement(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: Vec::new(),
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::External,
            GmsPlacementEventData::Stored(GmsPlacementStoreData {
                source_nixl_agent_name: "src".to_string(),
                source_nixl_agent_metadata_hex: "abcd".to_string(),
                source_nixl_ip: Some("10.0.0.1".to_string()),
                source_nixl_listen_port: Some(5555),
                blocks: hashes
                    .iter()
                    .enumerate()
                    .map(|(idx, hash)| GmsPlacementBlock {
                        content_hash_hex: (*hash).to_string(),
                        descriptor: descriptor(1000 + idx as u64),
                    })
                    .collect(),
            }),
        )
    }

    #[test]
    fn lookup_returns_ordered_descriptors_with_holes() {
        let index = GmsPlacementIndex::new();
        let worker = WorkerWithDpRank::new(7, 0);
        index.apply_event(&placement_event(worker, &["AA", "bb"]));

        let result = index
            .lookup(
                worker,
                &["aa".to_string(), "missing".to_string(), "BB".to_string()],
            )
            .unwrap();

        assert_eq!(result.source_nixl_agent_name, "src");
        assert!(result.descriptors[0].is_some());
        assert!(result.descriptors[1].is_none());
        assert!(result.descriptors[2].is_some());
    }

    #[test]
    fn remove_event_drops_hash() {
        let index = GmsPlacementIndex::new();
        let worker = WorkerWithDpRank::new(7, 0);
        index.apply_event(&placement_event(worker, &["aa"]));
        let remove = RouterEvent::with_gms_placement(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(crate::protocols::KvCacheRemoveData {
                    block_hashes: Vec::new(),
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::External,
            GmsPlacementEventData::Removed(GmsPlacementRemoveData {
                content_hashes_hex: vec!["AA".to_string()],
            }),
        );
        index.apply_event(&remove);

        assert!(index.lookup(worker, &["aa".to_string()]).is_none());
    }

    #[test]
    fn ordinary_clear_event_drops_worker_descriptors() {
        let index = GmsPlacementIndex::new();
        let worker = WorkerWithDpRank::new(7, 0);
        index.apply_event(&placement_event(worker, &["aa"]));

        let clear = RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Cleared,
                dp_rank: worker.dp_rank,
            },
        );
        index.apply_event(&clear);

        assert!(index.lookup(worker, &["aa".to_string()]).is_none());
    }
}
