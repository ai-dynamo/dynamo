// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::WorkerID;
use crate::block_manager::block::nixl::{BlockDescriptorList, SerializedNixlBlockSet};
use crate::block_manager::storage::Storage;
use crate::tokens::{SequenceHash, compute_hash_v2};

use anyhow::{Context, Result};
use async_trait::async_trait;
use dynamo_runtime::{
    component::Component,
    pipeline::{RouterMode, SingleIn, network::egress::push_router::PushRouter},
    protocols::annotated::Annotated,
};
use foyer::{
    BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder,
    HybridCachePolicy, PsyncIoEngineConfig,
};
use futures::{StreamExt, future::join_all};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::time::Duration;

const DEFAULT_METADATA_SHARDS: usize = 16;
pub const G3PB_NAMESPACE: &str = "kvbm-g3pb";
pub const G3PB_COMPONENT_NAME: &str = "peer-cache";
pub const G3PB_ENDPOINT_NAME: &str = "g3pb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPeer {
    pub worker_id: WorkerID,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPutBlock {
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbQueryHit {
    pub worker_id: WorkerID,
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbTransferBlock {
    pub meta: G3pbPutBlock,
    pub payload: Vec<u8>,
}

impl G3pbTransferBlock {
    fn validate_payload_size(&self) -> Result<(), G3pbError> {
        let actual_size_bytes = self.payload.len();
        if actual_size_bytes != self.meta.size_bytes {
            return Err(G3pbError::InvalidPayloadSize {
                sequence_hash: self.meta.sequence_hash,
                expected_size_bytes: self.meta.size_bytes,
                actual_size_bytes,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbOfferRequest {
    pub blocks: Vec<G3pbPutBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbHealthResponse {
    pub worker_id: WorkerID,
    pub listen: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbQueryRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbOfferResponse {
    pub accepted: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPutPayloadRequest {
    pub blocks: Vec<G3pbTransferBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbFetchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbFetchResponse {
    pub blocks: Vec<G3pbTransferBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G3pbRpcRequest {
    Health,
    PutBlocks(Vec<G3pbPutBlock>),
    Offer(G3pbOfferRequest),
    PutPayload(G3pbPutPayloadRequest),
    Query(G3pbQueryRequest),
    Fetch(G3pbFetchRequest),
    StagePut(G3pbStageBlocksRequest),
    CommitPut(G3pbCommitRequest),
    LoadRemote(G3pbLoadRemoteRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G3pbRpcResponse {
    Ack,
    Health(G3pbHealthResponse),
    Offer(G3pbOfferResponse),
    PutPayload(Vec<G3pbTransferBlock>),
    Query(Vec<G3pbQueryHit>),
    Fetch(G3pbFetchBlocksResponse),
    StagePut(G3pbStageBlocksResponse),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbStageBlocksRequest {
    pub blocks: Vec<G3pbPutBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbStageBlocksResponse {
    pub worker_id: WorkerID,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbCommitRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbLoadRemoteRequest {
    pub blockset: SerializedNixlBlockSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbFetchBlocksResponse {
    pub worker_id: WorkerID,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, thiserror::Error)]
pub enum G3pbError {
    #[error("no live G3PB peers are available")]
    NoPeers,
    #[error("owning G3PB peer {worker_id} is not available")]
    UnknownPeer { worker_id: WorkerID },
    #[error("requested G3PB blocks were not found on peer {worker_id}: {sequence_hashes:?}")]
    NotFound {
        worker_id: WorkerID,
        sequence_hashes: Vec<SequenceHash>,
    },
    #[error(
        "G3PB payload for sequence hash {sequence_hash} had size {actual_size_bytes}, expected {expected_size_bytes}"
    )]
    InvalidPayloadSize {
        sequence_hash: SequenceHash,
        expected_size_bytes: usize,
        actual_size_bytes: usize,
    },
}

#[async_trait]
pub trait G3pbPeerStorage: Send + Sync {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>);
    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash>;
    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError>;
    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit>;
    async fn fetch_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError>;
}

#[derive(Clone, Debug)]
struct InMemoryG3pbEntry {
    meta: G3pbPutBlock,
    payload: Option<Vec<u8>>,
}

#[derive(Default)]
pub struct InMemoryG3pbPeerStorage {
    blocks: RwLock<HashMap<SequenceHash, InMemoryG3pbEntry>>,
}

#[async_trait]
impl G3pbPeerStorage for InMemoryG3pbPeerStorage {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) {
        let mut guard = self.blocks.write().expect("g3pb peer storage poisoned");
        for block in blocks {
            let payload = guard
                .remove(&block.sequence_hash)
                .and_then(|entry| entry.payload);
            guard.insert(
                block.sequence_hash,
                InMemoryG3pbEntry {
                    meta: block,
                    payload,
                },
            );
        }
    }

    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        let mut seen = HashSet::new();
        blocks
            .iter()
            .filter_map(|block| {
                let sequence_hash = block.sequence_hash;
                (!guard.contains_key(&sequence_hash) && seen.insert(sequence_hash))
                    .then_some(sequence_hash)
            })
            .collect()
    }

    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError> {
        let mut guard = self.blocks.write().expect("g3pb peer storage poisoned");
        for block in blocks {
            block.validate_payload_size()?;
            guard.insert(
                block.meta.sequence_hash,
                InMemoryG3pbEntry {
                    meta: block.meta.clone(),
                    payload: Some(block.payload),
                },
            );
        }

        Ok(())
    }

    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        sequence_hashes
            .iter()
            .filter_map(|sequence_hash| {
                guard.get(sequence_hash).map(|entry| G3pbQueryHit {
                    worker_id,
                    sequence_hash: *sequence_hash,
                    size_bytes: entry.meta.size_bytes,
                    checksum: entry.meta.checksum,
                })
            })
            .collect()
    }

    async fn fetch_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        let mut missing = Vec::new();
        let mut blocks = Vec::with_capacity(sequence_hashes.len());

        for sequence_hash in sequence_hashes {
            match guard.get(sequence_hash) {
                Some(entry) => {
                    let Some(payload) = entry.payload.clone() else {
                        missing.push(*sequence_hash);
                        continue;
                    };
                    blocks.push(G3pbTransferBlock {
                        meta: entry.meta.clone(),
                        payload,
                    });
                }
                None => missing.push(*sequence_hash),
            }
        }

        if !missing.is_empty() {
            return Err(G3pbError::NotFound {
                worker_id,
                sequence_hashes: missing,
            });
        }

        Ok(blocks)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G3pbStorageConfig {
    pub g2_capacity_bytes: usize,
    pub foyer_memory_capacity_bytes: usize,
    pub foyer_disk_capacity_bytes: usize,
    pub foyer_dirs: Vec<PathBuf>,
    pub device_id: usize,
    pub g2_high_watermark_pct: usize, // Evict when G2 is this full
    pub g2_low_watermark_pct: usize,  // Evict until G2 is this full
    pub promotion_threshold: u64,     // Access count to promote to G2
    pub demotion_threshold: u64,      // Access count to demote to G3
}

impl G3pbStorageConfig {
    pub const DEFAULT_G2_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;
    pub const DEFAULT_FOYER_MEMORY_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;
    pub const DEFAULT_FOYER_DISK_CAPACITY_BYTES: usize = 4 * 1024 * 1024 * 1024;
    pub const DEFAULT_FOYER_DIR: &str = "/tmp/dynamo-g3pb-foyer";
    pub const DEFAULT_G2_HIGH_WATERMARK_PCT: usize = 90; // 90%
    pub const DEFAULT_G2_LOW_WATERMARK_PCT: usize = 70; // 70%
    pub const DEFAULT_PROMOTION_THRESHOLD: u64 = 10;
    pub const DEFAULT_DEMOTION_THRESHOLD: u64 = 5;

    pub fn new(foyer_dirs: Vec<PathBuf>, device_id: usize) -> Self {
        Self {
            g2_capacity_bytes: Self::DEFAULT_G2_CAPACITY_BYTES,
            foyer_memory_capacity_bytes: Self::DEFAULT_FOYER_MEMORY_CAPACITY_BYTES,
            foyer_disk_capacity_bytes: Self::DEFAULT_FOYER_DISK_CAPACITY_BYTES,
            foyer_dirs: if foyer_dirs.is_empty() {
                vec![PathBuf::from(Self::DEFAULT_FOYER_DIR)]
            } else {
                foyer_dirs
            },
            device_id,
            g2_high_watermark_pct: Self::DEFAULT_G2_HIGH_WATERMARK_PCT,
            g2_low_watermark_pct: Self::DEFAULT_G2_LOW_WATERMARK_PCT,
            promotion_threshold: Self::DEFAULT_PROMOTION_THRESHOLD,
            demotion_threshold: Self::DEFAULT_DEMOTION_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
struct G3pbCacheMetadata {
    size_bytes: usize,
    location: G3pbCacheLocation,
    priority: u32,
    returned_tick: u64,
    acquired_tick: u64,
    access_count: u64,
    last_access_tick: u64,
    payload_ready: bool,
}

/// Sharded metadata storage to reduce lock contention
/// Each shard has its own RwLock, allowing concurrent access to different shards
struct ShardedMetadata {
    shards: Vec<RwLock<HashMap<SequenceHash, G3pbCacheMetadata>>>,
    num_shards: usize,
}

impl ShardedMetadata {
    fn new(num_shards: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(RwLock::new(HashMap::new()));
        }
        Self { shards, num_shards }
    }

    /// Get the shard index for a given sequence hash
    fn shard_index(&self, sequence_hash: SequenceHash) -> usize {
        (sequence_hash as usize) % self.num_shards
    }

    /// Get a read guard for the shard containing the sequence hash
    fn read_shard(
        &self,
        sequence_hash: SequenceHash,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<SequenceHash, G3pbCacheMetadata>>> {
        let idx = self.shard_index(sequence_hash);
        self.shards[idx]
            .read()
            .map_err(|_| anyhow::anyhow!("metadata lock poisoned"))
    }

    /// Get a write guard for the shard containing the sequence hash
    fn write_shard(
        &self,
        sequence_hash: SequenceHash,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<SequenceHash, G3pbCacheMetadata>>> {
        let idx = self.shard_index(sequence_hash);
        self.shards[idx]
            .write()
            .map_err(|_| anyhow::anyhow!("metadata lock poisoned"))
    }

    /// Get read guards for multiple sequence hashes (may return duplicates for same shard)
    fn get(&self, sequence_hash: SequenceHash) -> Result<Option<G3pbCacheMetadata>> {
        let guard = self.read_shard(sequence_hash)?;
        Ok(guard.get(&sequence_hash).cloned())
    }

    fn contains_key(&self, sequence_hash: SequenceHash) -> Result<bool> {
        let guard = self.read_shard(sequence_hash)?;
        Ok(guard.contains_key(&sequence_hash))
    }

    fn insert(&self, sequence_hash: SequenceHash, metadata: G3pbCacheMetadata) -> Result<()> {
        let mut guard = self.write_shard(sequence_hash)?;
        guard.insert(sequence_hash, metadata);
        Ok(())
    }

    fn update<F>(&self, sequence_hash: SequenceHash, f: F) -> Result<()>
    where
        F: FnOnce(&mut G3pbCacheMetadata),
    {
        let mut guard = self.write_shard(sequence_hash)?;
        if let Some(metadata) = guard.get_mut(&sequence_hash) {
            f(metadata);
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<Vec<(SequenceHash, G3pbCacheMetadata)>> {
        let mut items = Vec::new();
        for shard in &self.shards {
            let guard = shard
                .read()
                .map_err(|_| anyhow::anyhow!("metadata lock poisoned"))?;
            items.extend(guard.iter().map(|(hash, meta)| (*hash, meta.clone())));
        }
        Ok(items)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum G3pbCacheLocation {
    G2 { offset: usize },
    Foyer,
}

pub struct G3pbCacheStorage {
    g2_storage: Arc<crate::block_manager::storage::PinnedStorage>,
    foyer_shards: Vec<HybridCache<SequenceHash, Vec<u8>>>,
    metadata: ShardedMetadata,
    tick_counter: AtomicU64,
    _g2_allocator: Arc<crate::block_manager::storage::PinnedAllocator>,
    g2_free_list: RwLock<VecDeque<(usize, usize)>>, // (offset, size)
    g2_allocated: AtomicU64,
    config: G3pbStorageConfig,
}

impl G3pbCacheStorage {
    pub async fn new(config: G3pbStorageConfig) -> Result<Self> {
        use crate::block_manager::storage::{PinnedAllocator, StorageAllocator};

        let g2_allocator = Arc::new(PinnedAllocator::new(config.device_id)?);
        let g2_storage = Arc::new(g2_allocator.allocate(config.g2_capacity_bytes)?);

        let num_foyer_shards = config.foyer_dirs.len().max(1);
        let foyer_memory_per_shard = (config.foyer_memory_capacity_bytes / num_foyer_shards).max(1);
        let foyer_disk_per_shard = (config.foyer_disk_capacity_bytes / num_foyer_shards).max(1);
        let mut foyer_shards = Vec::with_capacity(num_foyer_shards);
        for (idx, dir) in config.foyer_dirs.iter().enumerate() {
            std::fs::create_dir_all(dir)?;
            let device = FsDeviceBuilder::new(dir)
                .with_capacity(foyer_disk_per_shard)
                .build()?;
            let cache: HybridCache<SequenceHash, Vec<u8>> = HybridCacheBuilder::new()
                .with_name(format!("g3pb-foyer-{idx}"))
                .with_policy(HybridCachePolicy::WriteOnEviction)
                .memory(foyer_memory_per_shard)
                .storage()
                .with_io_engine_config(PsyncIoEngineConfig::new())
                .with_engine_config(BlockEngineConfig::new(device))
                .build()
                .await?;
            foyer_shards.push(cache);
        }

        let metadata = ShardedMetadata::new(DEFAULT_METADATA_SHARDS);

        Ok(Self {
            g2_storage,
            foyer_shards,
            metadata,
            tick_counter: AtomicU64::new(0),
            _g2_allocator: g2_allocator,
            g2_free_list: RwLock::new(VecDeque::new()),
            g2_allocated: AtomicU64::new(0),
            config,
        })
    }

    fn next_tick(&self) -> u64 {
        self.tick_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn foyer_shard_index(&self, sequence_hash: SequenceHash) -> usize {
        (sequence_hash as usize) % self.foyer_shards.len()
    }

    async fn foyer_insert(&self, sequence_hash: SequenceHash, payload: Vec<u8>) -> Result<()> {
        self.foyer_shards[self.foyer_shard_index(sequence_hash)].insert(sequence_hash, payload);
        Ok(())
    }

    async fn foyer_get(&self, sequence_hash: SequenceHash) -> Result<Option<Vec<u8>>> {
        Ok(self.foyer_shards[self.foyer_shard_index(sequence_hash)]
            .get(&sequence_hash)
            .await?
            .map(|entry| entry.value().clone()))
    }

    async fn allocate_in_g2(
        &self,
        size: usize,
    ) -> Result<(usize, Arc<crate::block_manager::storage::PinnedStorage>)> {
        loop {
            {
                let mut free_list = self
                    .g2_free_list
                    .write()
                    .map_err(|_| anyhow::anyhow!("free list lock poisoned"))?;
                while let Some((offset, free_size)) = free_list.pop_front() {
                    if free_size >= size {
                        if free_size > size {
                            free_list.push_back((offset + size, free_size - size));
                        }
                        return Ok((offset, self.g2_storage.clone()));
                    }
                }
            }

            let current_allocated = self.g2_allocated.fetch_add(size as u64, Ordering::Relaxed);
            let new_allocated = current_allocated + size as u64;

            if new_allocated > self.config.g2_capacity_bytes as u64 {
                self.evict_to_g2(size).await?;
                continue;
            }

            return Ok((current_allocated as usize, self.g2_storage.clone()));
        }
    }

    async fn free_in_g2(&self, offset: usize, size: usize) {
        let mut free_list = self.g2_free_list.write().expect("free list lock poisoned");
        free_list.push_back((offset, size));
    }

    async fn read_from_g2(&self, offset: usize, size: usize) -> Vec<u8> {
        unsafe {
            let ptr = Storage::as_ptr(&*self.g2_storage).add(offset);
            let slice = std::slice::from_raw_parts(ptr, size);
            slice.to_vec()
        }
    }

    async fn write_to_g2(&self, offset: usize, data: &[u8]) {
        // We need to use Arc::make_mut to get mutable access
        // For now, we'll use unsafe to get mutable access to the Arc's inner data
        // This is safe because we're the only owner of this Arc
        unsafe {
            let storage_ptr = Arc::as_ptr(&self.g2_storage)
                as *mut crate::block_manager::storage::cuda::PinnedStorage;
            let ptr = (*storage_ptr).as_mut_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    async fn get_stored_location(&self, sequence_hash: SequenceHash) -> Option<G3pbCacheLocation> {
        let meta = self.metadata.get(sequence_hash).ok()??;
        if !meta.payload_ready {
            return None;
        }
        match meta.location {
            G3pbCacheLocation::G2 { offset } => {
                Some(self.read_from_g2(offset, meta.size_bytes).await)
                    .map(|_| G3pbCacheLocation::G2 { offset })
            }
            G3pbCacheLocation::Foyer => self
                .foyer_get(sequence_hash)
                .await
                .ok()??
                .is_empty()
                .then_some(G3pbCacheLocation::Foyer)
                .or(Some(G3pbCacheLocation::Foyer)),
        }
    }

    async fn promote_to_g2(&self, sequence_hash: SequenceHash) -> Result<()> {
        let size = {
            let Some(meta) = self.metadata.get(sequence_hash)? else {
                return Ok(());
            };

            if matches!(meta.location, G3pbCacheLocation::Foyer) && meta.payload_ready {
                meta.size_bytes
            } else {
                return Ok(());
            }
        };

        let payload = self.foyer_get(sequence_hash).await?;

        if let Some(payload) = payload {
            if let Ok((g2_offset, _)) = self.allocate_in_g2(size).await {
                self.write_to_g2(g2_offset, &payload).await;
                self.metadata.update(sequence_hash, |meta| {
                    meta.location = G3pbCacheLocation::G2 { offset: g2_offset };
                    meta.acquired_tick = self.next_tick();
                })?;
            }
        }

        Ok(())
    }

    async fn demote_to_g3(&self, sequence_hash: SequenceHash) -> Result<()> {
        let (offset, size_bytes) = {
            let Some(meta) = self.metadata.get(sequence_hash)? else {
                return Ok(());
            };

            match meta.location {
                G3pbCacheLocation::G2 { offset } => (offset, meta.size_bytes),
                G3pbCacheLocation::Foyer => return Ok(()),
            }
        };
        let _ = size_bytes;
        self.metadata.update(sequence_hash, |meta| {
            if matches!(
                meta.location,
                G3pbCacheLocation::G2 {
                    offset: current_offset
                } if current_offset == offset
            ) {
                meta.location = G3pbCacheLocation::Foyer;
                meta.returned_tick = self.next_tick();
            }
        })?;
        self.free_in_g2(offset, size_bytes).await;

        Ok(())
    }

    async fn evict_if_needed(&self) -> Result<()> {
        let (to_demote, to_evict) = {
            let metadata = self.metadata.snapshot()?;

            let g2_size: usize = metadata
                .iter()
                .map(|(_, meta)| meta)
                .filter(|m| matches!(m.location, G3pbCacheLocation::G2 { .. }))
                .map(|m| m.size_bytes)
                .sum();

            let g2_capacity = self.config.g2_capacity_bytes;
            let high_watermark = g2_capacity * self.config.g2_high_watermark_pct / 100;
            if g2_size <= high_watermark {
                return Ok(());
            }

            let low_watermark = g2_capacity * self.config.g2_low_watermark_pct / 100;
            let target_size = low_watermark;

            let mut g2_blocks: Vec<_> = metadata
                .into_iter()
                .filter(|(_, meta)| matches!(meta.location, G3pbCacheLocation::G2 { .. }))
                .collect();
            g2_blocks.sort_by_key(|(_, meta)| (meta.returned_tick, meta.priority));

            let mut to_demote = Vec::new();
            let mut to_evict = Vec::new();
            let mut evicted_size = 0;

            for (hash, meta) in g2_blocks {
                if evicted_size >= g2_size - target_size {
                    break;
                }

                if let G3pbCacheLocation::G2 { offset } = meta.location {
                    evicted_size += meta.size_bytes;
                    if meta.access_count >= self.config.demotion_threshold {
                        to_demote.push(hash);
                    } else {
                        to_evict.push((hash, offset, meta.size_bytes));
                    }
                }
            }

            (to_demote, to_evict)
        };

        for sequence_hash in to_demote {
            let _ = self.demote_to_g3(sequence_hash).await;
        }

        for (hash, offset, size) in to_evict {
            self.metadata.update(hash, |meta| {
                meta.location = G3pbCacheLocation::Foyer;
                meta.returned_tick = self.next_tick();
            })?;
            self.free_in_g2(offset, size).await;
        }

        Ok(())
    }

    async fn evict_to_g2(&self, required_size: usize) -> Result<()> {
        let (to_evict, freed_size) = {
            let mut g2_blocks: Vec<_> = self
                .metadata
                .snapshot()?
                .into_iter()
                .filter(|(_, meta)| matches!(meta.location, G3pbCacheLocation::G2 { .. }))
                .collect();
            g2_blocks.sort_by_key(|(_, meta)| (meta.returned_tick, meta.priority));

            let mut to_evict = Vec::new();
            let mut freed_size = 0;

            for (hash, meta) in g2_blocks {
                if freed_size >= required_size {
                    break;
                }

                if let G3pbCacheLocation::G2 { offset } = meta.location {
                    freed_size += meta.size_bytes;
                    to_evict.push((hash, offset, meta.size_bytes));
                }
            }

            (to_evict, freed_size)
        };

        for (hash, offset, size) in to_evict {
            self.metadata.update(hash, |meta| {
                meta.location = G3pbCacheLocation::Foyer;
                meta.returned_tick = self.next_tick();
            })?;
            self.free_in_g2(offset, size).await;
        }

        if freed_size < required_size {
            return Err(anyhow::anyhow!(
                "Failed to evict enough space: needed {}, freed {}",
                required_size,
                freed_size
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl G3pbPeerStorage for G3pbCacheStorage {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) {
        let tick = self.next_tick();

        for block in blocks {
            let existing_location = self.get_stored_location(block.sequence_hash).await;

            let location = if let Some(existing) = existing_location {
                existing
            } else {
                G3pbCacheLocation::Foyer
            };

            self.metadata
                .insert(
                    block.sequence_hash,
                    G3pbCacheMetadata {
                        size_bytes: block.size_bytes,
                        location,
                        priority: 0,
                        returned_tick: tick,
                        acquired_tick: tick,
                        access_count: 0,
                        last_access_tick: tick,
                        payload_ready: existing_location.is_some(),
                    },
                )
                .expect("metadata lock poisoned");
        }
    }

    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash> {
        let mut accepted = Vec::new();
        let mut seen = HashSet::new();

        for block in blocks {
            let sequence_hash = block.sequence_hash;
            if !seen.insert(sequence_hash) {
                continue;
            }

            if !self
                .metadata
                .contains_key(sequence_hash)
                .expect("metadata lock poisoned")
            {
                accepted.push(sequence_hash);
            }
        }

        accepted
    }

    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError> {
        let tick = self.next_tick();

        for block in blocks {
            block.validate_payload_size()?;

            // Check if block already exists and collect operations
            let existing_location = self
                .metadata
                .get(block.meta.sequence_hash)
                .expect("metadata lock poisoned")
                .filter(|m| m.payload_ready)
                .map(|m| (m.location, m.size_bytes));

            self.foyer_insert(block.meta.sequence_hash, block.payload.clone())
                .await
                .map_err(|_| G3pbError::NotFound {
                    worker_id: 0,
                    sequence_hashes: vec![block.meta.sequence_hash],
                })?;

            if let Some((location, _)) = existing_location {
                match location {
                    G3pbCacheLocation::G2 { offset } => {
                        self.write_to_g2(offset, &block.payload).await;
                    }
                    G3pbCacheLocation::Foyer => {}
                }

                self.metadata
                    .update(block.meta.sequence_hash, |meta| {
                        meta.returned_tick = tick;
                        meta.last_access_tick = tick;
                        meta.payload_ready = true;
                    })
                    .expect("metadata lock poisoned");
            } else {
                let location = match self.allocate_in_g2(block.payload.len()).await {
                    Ok((offset, _)) => {
                        self.write_to_g2(offset, &block.payload).await;
                        G3pbCacheLocation::G2 { offset }
                    }
                    Err(_) => G3pbCacheLocation::Foyer,
                };

                self.metadata
                    .insert(
                        block.meta.sequence_hash,
                        G3pbCacheMetadata {
                            size_bytes: block.meta.size_bytes,
                            location,
                            priority: 0,
                            returned_tick: tick,
                            acquired_tick: tick,
                            access_count: 0,
                            last_access_tick: tick,
                            payload_ready: true,
                        },
                    )
                    .expect("metadata lock poisoned");
            }
        }

        let _ = self.evict_if_needed().await;

        Ok(())
    }

    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit> {
        let mut hits = Vec::new();

        // Group by shard to minimize lock acquisitions
        let mut hashes_by_shard: std::collections::HashMap<usize, Vec<SequenceHash>> =
            std::collections::HashMap::new();
        for &hash in sequence_hashes {
            let shard_idx = self.metadata.shard_index(hash);
            hashes_by_shard
                .entry(shard_idx)
                .or_insert_with(Vec::new)
                .push(hash);
        }

        // Query each shard separately
        for (shard_idx, shard_hashes) in &hashes_by_shard {
            let guard = self.metadata.shards[*shard_idx]
                .read()
                .expect("metadata lock poisoned");

            for sequence_hash in shard_hashes {
                if let Some(meta) = guard.get(sequence_hash) {
                    if meta.payload_ready {
                        hits.push(G3pbQueryHit {
                            worker_id,
                            sequence_hash: *sequence_hash,
                            size_bytes: meta.size_bytes,
                            checksum: None,
                        });
                    }
                }
            }
        }

        hits
    }

    async fn fetch_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let tick = self.next_tick();
        let mut missing = Vec::new();
        let mut blocks = Vec::with_capacity(sequence_hashes.len());
        let mut to_promote = Vec::new();

        // Group sequence hashes by shard to minimize lock acquisitions
        let mut hashes_by_shard: std::collections::HashMap<usize, Vec<SequenceHash>> =
            std::collections::HashMap::new();
        for &hash in sequence_hashes {
            let shard_idx = self.metadata.shard_index(hash);
            hashes_by_shard
                .entry(shard_idx)
                .or_insert_with(Vec::new)
                .push(hash);
        }

        // First pass: update access tracking and collect blocks to promote
        // Process each shard separately to allow concurrent access
        for (shard_idx, shard_hashes) in &hashes_by_shard {
            let mut guard = self.metadata.shards[*shard_idx]
                .write()
                .expect("metadata lock poisoned");

            for sequence_hash in shard_hashes {
                match guard.get_mut(sequence_hash) {
                    Some(meta) => {
                        // Update access tracking
                        meta.access_count += 1;
                        meta.last_access_tick = tick;
                        meta.acquired_tick = tick;

                        // Collect blocks to promote
                        if meta.access_count >= self.config.promotion_threshold
                            && matches!(meta.location, G3pbCacheLocation::Foyer)
                        {
                            to_promote.push(*sequence_hash);
                        }
                    }
                    None => missing.push(*sequence_hash),
                }
            }
        }

        // Process promotions
        for sequence_hash in to_promote {
            let _ = self.promote_to_g2(sequence_hash).await;
        }

        // Second pass: collect block locations
        let mut block_locations = Vec::new();
        for (shard_idx, shard_hashes) in &hashes_by_shard {
            let guard = self.metadata.shards[*shard_idx]
                .read()
                .expect("metadata lock poisoned");

            for sequence_hash in shard_hashes {
                if missing.contains(sequence_hash) {
                    continue;
                }

                if let Some(meta) = guard.get(sequence_hash) {
                    if !meta.payload_ready {
                        missing.push(*sequence_hash);
                        continue;
                    }
                    block_locations.push((*sequence_hash, meta.location, meta.size_bytes));
                }
            }
        }

        // Third pass: fetch blocks (outside lock)
        for (sequence_hash, location, size) in block_locations {
            let payload = match location {
                G3pbCacheLocation::G2 { offset } => self.read_from_g2(offset, size).await,
                G3pbCacheLocation::Foyer => {
                    match self
                        .foyer_get(sequence_hash)
                        .await
                        .map_err(|_| G3pbError::NotFound {
                            worker_id,
                            sequence_hashes: vec![sequence_hash],
                        })? {
                        Some(p) => p,
                        None => {
                            missing.push(sequence_hash);
                            continue;
                        }
                    }
                }
            };

            blocks.push(G3pbTransferBlock {
                meta: G3pbPutBlock {
                    sequence_hash,
                    size_bytes: size,
                    checksum: None,
                },
                payload,
            });
        }

        if !missing.is_empty() {
            return Err(G3pbError::NotFound {
                worker_id,
                sequence_hashes: missing,
            });
        }

        Ok(blocks)
    }
}

pub fn select_g3pb_owner(sequence_hash: SequenceHash, peers: &[G3pbPeer]) -> Option<G3pbPeer> {
    peers
        .iter()
        .cloned()
        .max_by_key(|peer| rendezvous_score(sequence_hash, peer.worker_id))
}

fn rendezvous_score(sequence_hash: SequenceHash, worker_id: WorkerID) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&sequence_hash.to_le_bytes());
    bytes[8..].copy_from_slice(&worker_id.to_le_bytes());
    compute_hash_v2(&bytes, 0)
}

fn route_items_by_g3pb_owner<T, F>(
    items: impl IntoIterator<Item = T>,
    peers: &[G3pbPeer],
    sequence_hash: F,
) -> Result<HashMap<WorkerID, Vec<T>>, G3pbError>
where
    F: Fn(&T) -> SequenceHash,
{
    let mut routed = HashMap::<WorkerID, Vec<T>>::new();

    for item in items {
        let owner = select_g3pb_owner(sequence_hash(&item), peers).ok_or(G3pbError::NoPeers)?;
        routed.entry(owner.worker_id).or_default().push(item);
    }

    Ok(routed)
}

fn filter_first_sequence_hash_matches<T, F>(
    items: Vec<T>,
    mut sequence_hash: F,
    accepted_hashes: &HashSet<SequenceHash>,
) -> Vec<T>
where
    F: FnMut(&T) -> SequenceHash,
{
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|item| {
            let sequence_hash = sequence_hash(item);
            accepted_hashes.contains(&sequence_hash) && seen.insert(sequence_hash)
        })
        .collect()
}

pub fn route_g3pb_sequence_hashes_by_owner(
    sequence_hashes: &[SequenceHash],
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<SequenceHash>>, G3pbError> {
    route_items_by_g3pb_owner(sequence_hashes.iter().copied(), peers, |sequence_hash| {
        *sequence_hash
    })
}

pub fn route_g3pb_put_blocks_by_owner(
    blocks: Vec<G3pbPutBlock>,
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<G3pbPutBlock>>, G3pbError> {
    route_items_by_g3pb_owner(blocks, peers, |block| block.sequence_hash)
}

pub fn route_g3pb_transfer_blocks_by_owner(
    blocks: Vec<G3pbTransferBlock>,
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<G3pbTransferBlock>>, G3pbError> {
    route_items_by_g3pb_owner(blocks, peers, |block| block.meta.sequence_hash)
}

#[derive(Clone)]
pub struct G3pbStorageAgent {
    worker_id: WorkerID,
    storage: Arc<dyn G3pbPeerStorage>,
    #[cfg(test)]
    query_delay: Option<Duration>,
}

impl G3pbStorageAgent {
    pub fn new(worker_id: WorkerID) -> Self {
        Self::new_with_storage(worker_id, Arc::new(InMemoryG3pbPeerStorage::default()))
    }

    pub fn new_with_storage(worker_id: WorkerID, storage: Arc<dyn G3pbPeerStorage>) -> Self {
        Self {
            worker_id,
            storage,
            #[cfg(test)]
            query_delay: None,
        }
    }

    pub fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    #[cfg(test)]
    fn with_query_delay(mut self, query_delay: Duration) -> Self {
        self.query_delay = Some(query_delay);
        self
    }

    pub async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) {
        self.storage.put_blocks(blocks).await;
    }

    pub async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash> {
        self.storage.offer_blocks(blocks).await
    }

    pub async fn offered_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Vec<G3pbPutBlock> {
        let accepted_hashes: HashSet<_> = self.offer_blocks(&blocks).await.into_iter().collect();
        filter_first_sequence_hash_matches(blocks, |block| block.sequence_hash, &accepted_hashes)
    }

    pub async fn offer_and_put_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Vec<G3pbPutBlock> {
        let accepted_blocks = self.offered_blocks(blocks).await;
        if !accepted_blocks.is_empty() {
            self.put_blocks(accepted_blocks.clone()).await;
        }
        accepted_blocks
    }

    pub async fn offered_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        for block in &blocks {
            block.validate_payload_size()?;
        }

        let metadata: Vec<_> = blocks.iter().map(|block| block.meta.clone()).collect();
        let accepted_hashes: HashSet<_> = self.offer_blocks(&metadata).await.into_iter().collect();

        Ok(filter_first_sequence_hash_matches(
            blocks,
            |block| block.meta.sequence_hash,
            &accepted_hashes,
        ))
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let accepted_blocks = self.offered_payload_blocks(blocks).await?;
        if !accepted_blocks.is_empty() {
            self.storage
                .put_payload_blocks(accepted_blocks.clone())
                .await?;
        }
        Ok(accepted_blocks)
    }

    pub async fn query_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G3pbQueryHit> {
        #[cfg(test)]
        if let Some(query_delay) = self.query_delay {
            tokio::time::sleep(query_delay).await;
        }

        self.storage
            .query_blocks(self.worker_id, sequence_hashes)
            .await
    }

    pub async fn fetch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        self.storage
            .fetch_blocks(self.worker_id, sequence_hashes)
            .await
    }
}

pub struct G3pbStorageClient {
    peers: Vec<G3pbPeer>,
    agents: HashMap<WorkerID, Arc<G3pbStorageAgent>>,
}

#[derive(Clone)]
pub struct G3pbRequestPlaneClient {
    router: PushRouter<G3pbRpcRequest, Annotated<G3pbRpcResponse>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G3pbPeerInstance {
    pub peer: G3pbPeer,
    pub instance_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct G3pbDiscoveredPeers {
    peers_by_worker: HashMap<WorkerID, G3pbPeerInstance>,
}

impl G3pbRequestPlaneClient {
    pub async fn new(component: Component) -> Result<Self> {
        let client = component.endpoint(G3PB_ENDPOINT_NAME).client().await?;
        client.wait_for_instances().await?;
        let router = PushRouter::from_client_no_fault_detection(client, RouterMode::Direct).await?;
        Ok(Self { router })
    }

    pub fn instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids()
    }

    async fn request(&self, instance_id: u64, request: G3pbRpcRequest) -> Result<G3pbRpcResponse> {
        let mut stream = self
            .router
            .direct(SingleIn::new(request), instance_id)
            .await?;
        let response = stream.next().await.ok_or_else(|| {
            anyhow::anyhow!("G3PB request to instance {instance_id} returned no response")
        })?;

        response.into_result()?.ok_or_else(|| {
            anyhow::anyhow!("G3PB request to instance {instance_id} returned an empty response")
        })
    }

    pub async fn health(&self, instance_id: u64) -> Result<G3pbHealthResponse> {
        match self.request(instance_id, G3pbRpcRequest::Health).await? {
            G3pbRpcResponse::Health(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB health response: {other:?}"),
        }
    }

    pub async fn put_blocks(&self, instance_id: u64, blocks: Vec<G3pbPutBlock>) -> Result<()> {
        match self
            .request(instance_id, G3pbRpcRequest::PutBlocks(blocks))
            .await?
        {
            G3pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G3PB put_blocks response: {other:?}"),
        }
    }

    pub async fn offer(
        &self,
        instance_id: u64,
        request: G3pbOfferRequest,
    ) -> Result<G3pbOfferResponse> {
        match self
            .request(instance_id, G3pbRpcRequest::Offer(request))
            .await?
        {
            G3pbRpcResponse::Offer(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB offer response: {other:?}"),
        }
    }

    pub async fn put_payload(
        &self,
        instance_id: u64,
        request: G3pbPutPayloadRequest,
    ) -> Result<Vec<G3pbTransferBlock>> {
        match self
            .request(instance_id, G3pbRpcRequest::PutPayload(request))
            .await?
        {
            G3pbRpcResponse::PutPayload(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB put_payload response: {other:?}"),
        }
    }

    pub async fn query(
        &self,
        instance_id: u64,
        request: G3pbQueryRequest,
    ) -> Result<Vec<G3pbQueryHit>> {
        match self
            .request(instance_id, G3pbRpcRequest::Query(request))
            .await?
        {
            G3pbRpcResponse::Query(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB query response: {other:?}"),
        }
    }

    pub async fn fetch(
        &self,
        instance_id: u64,
        request: G3pbFetchRequest,
    ) -> Result<G3pbFetchBlocksResponse> {
        match self
            .request(instance_id, G3pbRpcRequest::Fetch(request))
            .await?
        {
            G3pbRpcResponse::Fetch(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB fetch response: {other:?}"),
        }
    }

    pub async fn stage_put(
        &self,
        instance_id: u64,
        request: G3pbStageBlocksRequest,
    ) -> Result<G3pbStageBlocksResponse> {
        match self
            .request(instance_id, G3pbRpcRequest::StagePut(request))
            .await?
        {
            G3pbRpcResponse::StagePut(response) => Ok(response),
            other => anyhow::bail!("unexpected G3PB stage_put response: {other:?}"),
        }
    }

    pub async fn commit_put(&self, instance_id: u64, request: G3pbCommitRequest) -> Result<()> {
        match self
            .request(instance_id, G3pbRpcRequest::CommitPut(request))
            .await?
        {
            G3pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G3PB commit_put response: {other:?}"),
        }
    }

    pub async fn load_remote(
        &self,
        instance_id: u64,
        request: G3pbLoadRemoteRequest,
    ) -> Result<()> {
        match self
            .request(instance_id, G3pbRpcRequest::LoadRemote(request))
            .await?
        {
            G3pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G3PB load_remote response: {other:?}"),
        }
    }
}

impl G3pbDiscoveredPeers {
    pub fn from_health_responses(discovered: Vec<(u64, G3pbHealthResponse)>) -> Result<Self> {
        let mut peers_by_worker = HashMap::with_capacity(discovered.len());
        for (instance_id, health) in discovered {
            let peer = G3pbPeer {
                worker_id: health.worker_id,
                endpoint: health.listen,
            };
            let resolved = G3pbPeerInstance {
                peer: peer.clone(),
                instance_id,
            };

            if let Some(previous) = peers_by_worker.insert(health.worker_id, resolved) {
                anyhow::bail!(
                    "duplicate remote worker_id {} discovered at instance {} and {}",
                    health.worker_id,
                    previous.instance_id,
                    instance_id
                );
            }
        }

        Ok(Self { peers_by_worker })
    }

    pub fn is_empty(&self) -> bool {
        self.peers_by_worker.is_empty()
    }

    pub fn peers(&self) -> Vec<G3pbPeer> {
        let mut peers: Vec<_> = self
            .peers_by_worker
            .values()
            .map(|resolved| resolved.peer.clone())
            .collect();
        peers.sort_by_key(|peer| peer.worker_id);
        peers
    }

    pub fn instances(&self) -> Vec<G3pbPeerInstance> {
        let mut instances: Vec<_> = self.peers_by_worker.values().cloned().collect();
        instances.sort_by_key(|resolved| resolved.peer.worker_id);
        instances
    }

    pub fn instance_id(&self, worker_id: WorkerID) -> Result<u64> {
        self.peers_by_worker
            .get(&worker_id)
            .map(|resolved| resolved.instance_id)
            .ok_or_else(|| anyhow::anyhow!("missing backend instance for worker {worker_id}"))
    }

    pub async fn load_remote_blockset(
        &self,
        request_client: &G3pbRequestPlaneClient,
        blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        for resolved in self.instances() {
            request_client
                .load_remote(
                    resolved.instance_id,
                    G3pbLoadRemoteRequest {
                        blockset: blockset.clone(),
                    },
                )
                .await
                .with_context(|| {
                    format!(
                        "failed to publish local blockset to worker {}",
                        resolved.peer.worker_id
                    )
                })?;
        }

        Ok(())
    }
}

pub async fn discover_g3pb_peers(
    request_client: &G3pbRequestPlaneClient,
) -> Result<G3pbDiscoveredPeers> {
    let mut discovered = Vec::new();
    for instance_id in request_client.instance_ids() {
        let health = request_client.health(instance_id).await?;
        discovered.push((instance_id, health));
    }

    G3pbDiscoveredPeers::from_health_responses(discovered)
}

impl G3pbStorageClient {
    pub fn new(peers: Vec<G3pbPeer>, agents: HashMap<WorkerID, Arc<G3pbStorageAgent>>) -> Self {
        Self { peers, agents }
    }

    pub fn owner_for(&self, sequence_hash: SequenceHash) -> Result<G3pbPeer, G3pbError> {
        select_g3pb_owner(sequence_hash, &self.peers).ok_or(G3pbError::NoPeers)
    }

    pub async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Result<(), G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks, &self.peers)?;
        for (worker_id, blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            agent.put_blocks(blocks).await;
        }

        Ok(())
    }

    pub async fn offer_blocks(
        &self,
        blocks: Vec<G3pbPutBlock>,
    ) -> Result<Vec<G3pbPutBlock>, G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_blocks(&owner_blocks).await;
            accepted_by_owner.insert(worker_id, accepted.into_iter().collect());
        }

        Ok(filter_accepted_put_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_and_put_blocks(
        &self,
        blocks: Vec<G3pbPutBlock>,
    ) -> Result<Vec<G3pbPutBlock>, G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_and_put_blocks(owner_blocks).await;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_put_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let routed = route_g3pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offered_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_transfer_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let routed = route_g3pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_and_put_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_transfer_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn query_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> HashMap<SequenceHash, G3pbQueryHit> {
        let grouped =
            route_g3pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let queries = grouped
            .into_iter()
            .filter_map(|(worker_id, sequence_hashes)| {
                self.agents
                    .get(&worker_id)
                    .cloned()
                    .map(|agent| async move { agent.query_blocks(&sequence_hashes).await })
            });

        let mut hits = HashMap::new();
        for owner_hits in join_all(queries).await {
            for hit in owner_hits {
                hits.insert(hit.sequence_hash, hit);
            }
        }

        hits
    }

    pub async fn fetch_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G3pbTransferBlock> {
        let grouped =
            route_g3pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let mut fetched = Vec::new();
        for (worker_id, owner_hashes) in grouped {
            let Some(agent) = self.agents.get(&worker_id) else {
                continue;
            };

            match agent.fetch_blocks(&owner_hashes).await {
                Ok(mut blocks) => fetched.append(&mut blocks),
                Err(G3pbError::NotFound { .. }) => {
                    // G3PB is a cache. Fetch failures degrade to cache misses.
                }
                Err(G3pbError::InvalidPayloadSize { .. }) => {}
                Err(G3pbError::NoPeers | G3pbError::UnknownPeer { .. }) => {}
            }
        }

        fetched.sort_by_key(|block| {
            sequence_hashes
                .iter()
                .position(|sequence_hash| *sequence_hash == block.meta.sequence_hash)
                .unwrap_or(usize::MAX)
        });
        fetched
    }
}

fn filter_accepted_put_blocks<F>(
    blocks: Vec<G3pbPutBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<WorkerID, HashSet<SequenceHash>>,
) -> Vec<G3pbPutBlock>
where
    F: Fn(SequenceHash) -> Option<G3pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&sequence_hash));

                if is_accepted && seen.insert(sequence_hash) {
                    accepted_blocks.push(block);
                }

                (accepted_blocks, seen)
            },
        )
        .0
}

fn filter_accepted_transfer_blocks<F>(
    blocks: Vec<G3pbTransferBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<WorkerID, HashSet<SequenceHash>>,
) -> Vec<G3pbTransferBlock>
where
    F: Fn(SequenceHash) -> Option<G3pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.meta.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&sequence_hash));

                if is_accepted && seen.insert(sequence_hash) {
                    accepted_blocks.push(block);
                }

                (accepted_blocks, seen)
            },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use tokio::time::Instant;

    fn peers() -> Vec<G3pbPeer> {
        vec![
            G3pbPeer {
                worker_id: 10,
                endpoint: "tcp://peer-10".to_string(),
            },
            G3pbPeer {
                worker_id: 20,
                endpoint: "tcp://peer-20".to_string(),
            },
            G3pbPeer {
                worker_id: 30,
                endpoint: "tcp://peer-30".to_string(),
            },
        ]
    }

    #[test]
    fn owner_selection_is_deterministic() {
        let peers = peers();
        let sequence_hash = 0xdead_beef_u64;

        let owner_a = select_g3pb_owner(sequence_hash, &peers).unwrap();

        let mut reversed = peers.clone();
        reversed.reverse();
        let owner_b = select_g3pb_owner(sequence_hash, &reversed).unwrap();

        assert_eq!(owner_a, owner_b);
    }

    #[test]
    fn route_sequence_hashes_groups_by_owner_and_preserves_owner_order() {
        let peer_list = peers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64, 4_u64];

        let routed = route_g3pb_sequence_hashes_by_owner(&sequence_hashes, &peer_list).unwrap();

        for hashes in routed.values() {
            let mut expected = hashes.clone();
            expected.sort_by_key(|sequence_hash| {
                sequence_hashes
                    .iter()
                    .position(|candidate| candidate == sequence_hash)
                    .unwrap()
            });
            assert_eq!(*hashes, expected);
        }
    }

    #[test]
    fn discovered_peers_reject_duplicate_worker_ids() {
        let err = G3pbDiscoveredPeers::from_health_responses(vec![
            (
                101,
                G3pbHealthResponse {
                    worker_id: 10,
                    listen: "tcp://peer-10-a".to_string(),
                },
            ),
            (
                202,
                G3pbHealthResponse {
                    worker_id: 10,
                    listen: "tcp://peer-10-b".to_string(),
                },
            ),
        ])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("duplicate remote worker_id 10 discovered")
        );
    }

    #[test]
    fn discovered_peers_sort_instances_by_worker_id() {
        let discovered = G3pbDiscoveredPeers::from_health_responses(vec![
            (
                303,
                G3pbHealthResponse {
                    worker_id: 30,
                    listen: "tcp://peer-30".to_string(),
                },
            ),
            (
                101,
                G3pbHealthResponse {
                    worker_id: 10,
                    listen: "tcp://peer-10".to_string(),
                },
            ),
            (
                202,
                G3pbHealthResponse {
                    worker_id: 20,
                    listen: "tcp://peer-20".to_string(),
                },
            ),
        ])
        .unwrap();

        assert_eq!(
            discovered
                .instances()
                .into_iter()
                .map(|resolved| (resolved.peer.worker_id, resolved.instance_id))
                .collect::<Vec<_>>(),
            vec![(10, 101), (20, 202), (30, 303)]
        );
    }

    #[tokio::test]
    async fn agent_query_and_fetch_use_in_memory_peer_cache() {
        let agent = G3pbStorageAgent::new(7);

        agent
            .offer_and_put_payload_blocks(vec![
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: Some([3; 32]),
                    },
                    payload: vec![5, 6],
                },
            ])
            .await
            .unwrap();

        let hits = agent.query_blocks(&[11, 12, 13]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].worker_id, 7);
        assert_eq!(hits[0].size_bytes, 4);
        assert_eq!(hits[1].size_bytes, 2);

        let fetched = agent.fetch_blocks(&[11, 12]).await.unwrap();
        assert_eq!(
            fetched,
            vec![
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: Some([3; 32]),
                    },
                    payload: vec![5, 6],
                },
            ]
        );
    }

    #[tokio::test]
    async fn agent_fetch_reports_not_found() {
        let agent = G3pbStorageAgent::new(3);

        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 44,
                size_bytes: 8,
                checksum: None,
            }])
            .await;

        let err = agent.fetch_blocks(&[44]).await.unwrap_err();
        assert!(matches!(
            err,
            G3pbError::NotFound {
                worker_id: 3,
                sequence_hashes
            } if sequence_hashes == vec![44]
        ));
    }

    #[tokio::test]
    async fn agent_offer_only_accepts_missing_blocks() {
        let agent = G3pbStorageAgent::new(7);
        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 100,
                size_bytes: 16,
                checksum: None,
            }])
            .await;

        let accepted = agent
            .offered_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 16,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: None,
                },
            ])
            .await;

        assert_eq!(
            accepted,
            vec![G3pbPutBlock {
                sequence_hash: 200,
                size_bytes: 32,
                checksum: None,
            }]
        );
    }

    #[tokio::test]
    async fn agent_offer_and_put_registers_only_missing_blocks() {
        let agent = G3pbStorageAgent::new(7);
        let accepted = agent
            .offer_and_put_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 16,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: Some([9; 32]),
                },
                G3pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: Some([9; 32]),
                },
            ])
            .await;

        assert_eq!(accepted.len(), 2);

        let hits = agent.query_blocks(&[5, 6]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].sequence_hash, 5);
        assert_eq!(hits[1].sequence_hash, 6);
    }

    #[tokio::test]
    async fn agent_offer_payload_blocks_rejects_size_mismatch() {
        let agent = G3pbStorageAgent::new(7);

        let err = agent
            .offered_payload_blocks(vec![G3pbTransferBlock {
                meta: G3pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 4,
                    checksum: None,
                },
                payload: vec![1, 2, 3],
            }])
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            G3pbError::InvalidPayloadSize {
                sequence_hash: 5,
                expected_size_bytes: 4,
                actual_size_bytes: 3
            }
        ));
    }

    #[tokio::test]
    async fn client_put_and_query_route_by_owner() {
        let peer_list = peers();
        let owner = select_g3pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G3pbStorageAgent::new(owner.worker_id));
        let client = G3pbStorageClient::new(
            peer_list,
            HashMap::from_iter([(owner.worker_id, agent.clone())]),
        );

        client
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 64,
                checksum: Some([7; 32]),
            }])
            .await
            .unwrap();

        let hits = client.query_blocks(&[1234, 9999]).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[&1234].worker_id, owner.worker_id);
        assert_eq!(hits[&1234].size_bytes, 64);
    }

    #[tokio::test]
    async fn client_offer_routes_by_owner_and_preserves_input_order() {
        let peer_list = peers();
        let mut agents = HashMap::new();

        for peer in &peer_list {
            agents.insert(
                peer.worker_id,
                Arc::new(G3pbStorageAgent::new(peer.worker_id)),
            );
        }

        let existing_hash = 200_u64;
        let existing_owner = select_g3pb_owner(existing_hash, &peer_list).unwrap();
        agents[&existing_owner.worker_id]
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: existing_hash,
                size_bytes: 8,
                checksum: None,
            }])
            .await;

        let client = G3pbStorageClient::new(peer_list, agents);
        let accepted = client
            .offer_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: existing_hash,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 300,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: None,
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.sequence_hash)
                .collect::<Vec<_>>(),
            vec![100, 300]
        );
    }

    #[tokio::test]
    async fn client_offer_and_put_payload_rejects_duplicate_hashes_within_batch() {
        let peer_list = peers();
        let mut agents = HashMap::new();
        for peer in &peer_list {
            agents.insert(
                peer.worker_id,
                Arc::new(G3pbStorageAgent::new(peer.worker_id)),
            );
        }

        let client = G3pbStorageClient::new(peer_list.clone(), agents.clone());
        let accepted = client
            .offer_and_put_payload_blocks(vec![
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: None,
                    },
                    payload: vec![1, 2],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 2,
                        size_bytes: 2,
                        checksum: None,
                    },
                    payload: vec![3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: None,
                    },
                    payload: vec![5, 6],
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        let fetched = client.fetch_blocks(&[1, 2]).await;
        assert_eq!(
            fetched
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[tokio::test]
    async fn client_query_blocks_fans_out_across_owners_concurrently() {
        let peer_list = peers();
        let sequence_hashes = vec![100_u64, 200_u64, 300_u64];
        let mut agents = HashMap::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g3pb_owner(*sequence_hash, &peer_list).unwrap();
            agents.entry(owner.worker_id).or_insert_with(|| {
                Arc::new(
                    G3pbStorageAgent::new(owner.worker_id)
                        .with_query_delay(Duration::from_millis(75)),
                )
            });
            agents[&owner.worker_id]
                .put_blocks(vec![G3pbPutBlock {
                    sequence_hash: *sequence_hash,
                    size_bytes: 8,
                    checksum: None,
                }])
                .await;
        }

        let client = G3pbStorageClient::new(peer_list, agents);
        let start = Instant::now();
        let hits = client.query_blocks(&sequence_hashes).await;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), sequence_hashes.len());
        assert!(elapsed < Duration::from_millis(200));
    }

    #[tokio::test]
    async fn client_treats_missing_fetches_as_cache_miss() {
        let peer_list = peers();
        let owner = select_g3pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G3pbStorageAgent::new(owner.worker_id));
        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 4,
                checksum: None,
            }])
            .await;

        let client =
            G3pbStorageClient::new(peer_list, HashMap::from_iter([(owner.worker_id, agent)]));
        let fetched = client.fetch_blocks(&[1234]).await;
        assert!(fetched.is_empty());
    }

    #[tokio::test]
    async fn g3pb_cache_storage_supports_basic_operations() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let config = G3pbStorageConfig::new(vec![temp_dir.path().to_path_buf()], 0);

        let storage = Arc::new(G3pbCacheStorage::new(config).await?);
        let agent = G3pbStorageAgent::new_with_storage(77, storage.clone());

        agent
            .offer_and_put_payload_blocks(vec![G3pbTransferBlock {
                meta: G3pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: None,
                },
                payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }])
            .await?;

        let hits = agent.query_blocks(&[1001]).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].sequence_hash, 1001);
        assert_eq!(hits[0].size_bytes, 8);

        // Test offer blocks
        let accepted = agent
            .offer_blocks(&[
                G3pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 2001,
                    size_bytes: 16,
                    checksum: None,
                },
            ])
            .await;
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0], 2001);

        Ok(())
    }
}
