// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{G3pbError, G3pbPutBlock, G3pbQueryHit, G3pbTransferBlock};
use crate::block_manager::storage::Storage;
use crate::tokens::SequenceHash;

use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::time::Duration;

const DEFAULT_METADATA_SHARDS: usize = 16;

#[async_trait]
pub trait G3pbPeerStorage: Send + Sync {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>);
    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash>;
    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError>;
    async fn delete_blocks(&self, sequence_hashes: &[SequenceHash]) -> Result<(), G3pbError>;
    async fn query_blocks(
        &self,
        instance_id: u64,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit>;
    async fn fetch_blocks(
        &self,
        instance_id: u64,
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

    async fn delete_blocks(&self, sequence_hashes: &[SequenceHash]) -> Result<(), G3pbError> {
        let mut guard = self.blocks.write().expect("g3pb peer storage poisoned");
        for sequence_hash in sequence_hashes {
            guard.remove(sequence_hash);
        }
        Ok(())
    }

    async fn query_blocks(
        &self,
        instance_id: u64,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        sequence_hashes
            .iter()
            .filter_map(|sequence_hash| {
                guard.get(sequence_hash).map(|entry| G3pbQueryHit {
                    instance_id,
                    sequence_hash: *sequence_hash,
                    size_bytes: entry.meta.size_bytes,
                    checksum: entry.meta.checksum,
                })
            })
            .collect()
    }

    async fn fetch_blocks(
        &self,
        instance_id: u64,
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
                instance_id,
                sequence_hashes: missing,
            });
        }

        Ok(blocks)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G3pbStorageConfig {
    pub g2_capacity_bytes: usize,
    pub device_id: usize,
}

impl G3pbStorageConfig {
    pub const DEFAULT_G2_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;

    pub fn new(device_id: usize) -> Self {
        Self {
            g2_capacity_bytes: Self::DEFAULT_G2_CAPACITY_BYTES,
            device_id,
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

    fn remove(&self, sequence_hash: SequenceHash) -> Result<Option<G3pbCacheMetadata>> {
        let mut guard = self.write_shard(sequence_hash)?;
        Ok(guard.remove(&sequence_hash))
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
}

pub struct G3pbCacheStorage {
    g2_storage: Arc<crate::block_manager::storage::PinnedStorage>,
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

        let metadata = ShardedMetadata::new(DEFAULT_METADATA_SHARDS);

        Ok(Self {
            g2_storage,
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
                self.g2_allocated.fetch_sub(size as u64, Ordering::Relaxed);
                self.evict_for_space(size).await?;
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
        meta.payload_ready.then_some(meta.location)
    }

    async fn evict_for_space(&self, required_size: usize) -> Result<()> {
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

                let G3pbCacheLocation::G2 { offset } = meta.location;
                freed_size += meta.size_bytes;
                to_evict.push((hash, offset, meta.size_bytes));
            }

            (to_evict, freed_size)
        };

        for (hash, offset, size) in to_evict {
            let _ = self.metadata.remove(hash)?;
            self.free_in_g2(offset, size).await;
        }

        if freed_size < required_size {
            return Err(anyhow::anyhow!(
                "failed to evict enough G2 space: needed {}, freed {}",
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
                continue;
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

            if let Some((location, _)) = existing_location {
                match location {
                    G3pbCacheLocation::G2 { offset } => {
                        self.write_to_g2(offset, &block.payload).await;
                    }
                }

                self.metadata
                    .update(block.meta.sequence_hash, |meta| {
                        meta.returned_tick = tick;
                        meta.last_access_tick = tick;
                        meta.payload_ready = true;
                    })
                    .expect("metadata lock poisoned");
            } else {
                let (offset, _) = self.allocate_in_g2(block.payload.len()).await.map_err(|_| {
                    G3pbError::UnknownPeer { instance_id: 0 }
                })?;
                self.write_to_g2(offset, &block.payload).await;
                let location = G3pbCacheLocation::G2 { offset };

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

        Ok(())
    }

    async fn delete_blocks(&self, sequence_hashes: &[SequenceHash]) -> Result<(), G3pbError> {
        for sequence_hash in sequence_hashes {
            if let Some(metadata) = self
                .metadata
                .remove(*sequence_hash)
                .expect("metadata lock poisoned")
            {
                let G3pbCacheLocation::G2 { offset } = metadata.location;
                self.free_in_g2(offset, metadata.size_bytes).await;
            }
        }

        Ok(())
    }

    async fn query_blocks(
        &self,
        instance_id: u64,
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
                if let Some(meta) = guard.get(sequence_hash)
                    && meta.payload_ready
                {
                    hits.push(G3pbQueryHit {
                    instance_id,
                    sequence_hash: *sequence_hash,
                    size_bytes: meta.size_bytes,
                    checksum: None,
                });
                }
            }
        }

        hits
    }

    async fn fetch_blocks(
        &self,
        instance_id: u64,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let tick = self.next_tick();
        let mut missing = Vec::new();
        let mut blocks = Vec::with_capacity(sequence_hashes.len());

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
                    }
                    None => missing.push(*sequence_hash),
                }
            }
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
                instance_id,
                sequence_hashes: missing,
            });
        }

        Ok(blocks)
    }
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
pub struct G3pbStorageAgent {
    instance_id: u64,
    storage: Arc<dyn G3pbPeerStorage>,
    #[cfg(test)]
    query_delay: Option<Duration>,
}

impl G3pbStorageAgent {
    pub fn new(instance_id: u64) -> Self {
        Self::new_with_storage(instance_id, Arc::new(InMemoryG3pbPeerStorage::default()))
    }

    pub fn new_with_storage(instance_id: u64, storage: Arc<dyn G3pbPeerStorage>) -> Self {
        Self {
            instance_id,
            storage,
            #[cfg(test)]
            query_delay: None,
        }
    }

    pub fn instance_id(&self) -> u64 {
        self.instance_id
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
            .query_blocks(self.instance_id, sequence_hashes)
            .await
    }

    pub async fn fetch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        self.storage
            .fetch_blocks(self.instance_id, sequence_hashes)
            .await
    }

    pub async fn delete_blocks(&self, sequence_hashes: &[SequenceHash]) -> Result<(), G3pbError> {
        self.storage.delete_blocks(sequence_hashes).await
    }
}
