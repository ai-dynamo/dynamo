// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Each worker's device KV cache, driven by the router's bookings.
//!
//! The router decides where a request runs; this cache decides what that worker then
//! holds and publishes it as KV events, as an engine would. It follows a vLLM block
//! pool, simplified: a running request's blocks are pinned; at release they join the
//! free queue, tail first, so a prefix outlives its suffix; new blocks evict from the
//! head of the free queue when the cache is full.
//!
//! The cache tracks runs of blocks, not single blocks. A trace prompt repeats each
//! trace block's ID as its tokens, so two prompts that share a block also share the
//! rest of its trace block, up to where either ends. Each run of a prompt's blocks
//! inside one trace block is one entry, keyed by its first block, and is pinned,
//! released and evicted as a unit; every block belongs to exactly one entry. That
//! divides the cache's work by the blocks per trace block, while hits stay exact and
//! events still name every block. Each output block is an entry of its own.
//!
//! There is one tier, the device. A host tier (G2) or remote fetch (KVCR) would be
//! further caches with transfer rules between them.

use std::sync::Mutex;

use dynamo_kv_events::event::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
};
use dynamo_kv_hashing::routing::{
    BlockHashOptions, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
use rustc_hash::FxHashMap;
use serde::Serialize;

/// A prompt's complete blocks, hashed as the router hashes them.
pub struct PromptBlocks {
    pub local: Vec<u64>,
    pub sequence: Vec<u64>,
}

impl PromptBlocks {
    pub fn from_tokens(tokens: &[u32], block_size: u32) -> Self {
        let local = compute_block_hash_for_seq(tokens, block_size, BlockHashOptions::default());
        let sequence = compute_seq_hash_for_block(&local);
        Self {
            local: local.iter().map(|hash| hash.0).collect(),
            sequence,
        }
    }
}

/// The sequence and local hash of output block `k` of `request`, chained after `parent`.
/// Output blocks never match a later prompt, and two requests' outputs differ, so
/// both hashes are unique to the request.
pub fn decode_block_hashes(parent: u64, request: u64, k: u32) -> (u64, u64) {
    (
        mix(mix(parent, request), u64::from(k) + 1),
        mix(request, u64::from(k) + 1),
    )
}

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Totals over a run.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CacheStats {
    pub prompt_blocks: u64,
    /// Prompt blocks already on the booked worker at prefill.
    pub prompt_hits: u64,
    pub stored_blocks: u64,
    pub decode_blocks: u64,
    pub evicted_blocks: u64,
    /// Blocks stored while nothing more was evictable: running requests pinned more
    /// than the capacity.
    pub over_capacity: u64,
}

pub struct KvCaches {
    capacity: usize,
    dp_size: u32,
    /// Blocks per trace block: the most one entry holds.
    run_blocks: usize,
    workers: Vec<Mutex<WorkerCache>>,
}

impl KvCaches {
    /// Caches of `capacity` blocks for each DP rank of each worker. `run_blocks` is
    /// the blocks per trace block, or 1 where trace blocks do not align with blocks.
    pub fn new(workers: usize, dp_size: u32, capacity: usize, run_blocks: usize) -> Self {
        Self {
            capacity,
            dp_size,
            run_blocks: run_blocks.max(1),
            workers: (0..workers as u32 * dp_size)
                .map(|cache| Mutex::new(WorkerCache::new(cache % dp_size)))
                .collect(),
        }
    }

    /// The cache of worker `worker_id` (numbered from 1), DP rank `dp_rank`.
    pub fn index(&self, worker_id: u64, dp_rank: u32) -> usize {
        (worker_id as usize - 1) * self.dp_size as usize + dp_rank as usize
    }

    /// Pin the cached prefix of `prompt` and store the rest, appending each entry
    /// to `held`. `publish` receives the events while the worker's cache is locked,
    /// so each worker's events reach the index in the order the cache changed.
    /// Returns the prompt blocks already cached.
    pub fn prefill(
        &self,
        cache: usize,
        prompt: &PromptBlocks,
        held: &mut Vec<u32>,
        publish: impl FnOnce(Vec<KvCacheEvent>),
    ) -> usize {
        let mut worker = self.workers[cache].lock().unwrap();
        let sequence = &prompt.sequence;
        let mut cached = 0;
        while cached < sequence.len() {
            let Some(&entry) = worker.entries.get(&sequence[cached]) else {
                break;
            };
            worker.pin(entry);
            held.push(entry);
            // The prompt may end inside the entry.
            cached = (cached + worker.nodes[entry as usize].len as usize).min(sequence.len());
        }
        let mut evicted = Vec::new();
        let mut start = cached;
        while start < sequence.len() {
            let end = ((start / self.run_blocks + 1) * self.run_blocks).min(sequence.len());
            held.push(worker.allocate(
                &sequence[start..end],
                self.capacity,
                self.run_blocks,
                &mut evicted,
            ));
            start = end;
        }
        let new_blocks = sequence.len() - cached;
        let mut events = Vec::new();
        if !evicted.is_empty() {
            events.push(worker.removed(evicted));
        }
        if new_blocks > 0 {
            let blocks = sequence[cached..]
                .iter()
                .zip(&prompt.local[cached..])
                .map(|(&sequence, &local)| (sequence, local))
                .collect();
            let parent = cached.checked_sub(1).map(|i| sequence[i]);
            events.push(worker.stored(parent, blocks));
        }
        worker.stats.prompt_blocks += sequence.len() as u64;
        worker.stats.prompt_hits += cached as u64;
        worker.stats.stored_blocks += new_blocks as u64;
        publish(events);
        cached
    }

    /// Store output block `k` of `request`, chained after `parent`, the last block of
    /// its sequence so far (none for a prompt shorter than a block), and add it to
    /// `held`; `publish` as for `prefill`. Returns the new block's sequence hash.
    pub fn decode_block(
        &self,
        cache: usize,
        parent: Option<u64>,
        (request, k): (u64, u32),
        held: &mut Vec<u32>,
        publish: impl FnOnce(Vec<KvCacheEvent>),
    ) -> u64 {
        let mut worker = self.workers[cache].lock().unwrap();
        let (sequence, local) = decode_block_hashes(parent.unwrap_or(0), request, k);
        let mut evicted = Vec::new();
        held.push(worker.allocate(&[sequence], self.capacity, self.run_blocks, &mut evicted));
        let mut events = Vec::new();
        if !evicted.is_empty() {
            events.push(worker.removed(evicted));
        }
        events.push(worker.stored(parent, vec![(sequence, local)]));
        worker.stats.decode_blocks += 1;
        publish(events);
        sequence
    }

    /// Unpin a finished request's `held` entries; its tail becomes evictable first.
    pub fn release(&self, cache: usize, held: &[u32]) {
        self.workers[cache].lock().unwrap().release(held);
    }

    /// Totals over every worker.
    pub fn stats(&self) -> CacheStats {
        let mut total = CacheStats::default();
        for worker in &self.workers {
            let stats = worker.lock().unwrap().stats;
            total.prompt_blocks += stats.prompt_blocks;
            total.prompt_hits += stats.prompt_hits;
            total.stored_blocks += stats.stored_blocks;
            total.decode_blocks += stats.decode_blocks;
            total.evicted_blocks += stats.evicted_blocks;
            total.over_capacity += stats.over_capacity;
        }
        total
    }
}

const NIL: u32 = u32::MAX;

struct WorkerCache {
    /// The engine rank this cache belongs to; its events carry it.
    dp_rank: u32,
    /// Resident entries, by the sequence hash of their first block.
    entries: FxHashMap<u64, u32>,
    nodes: Vec<Node>,
    /// Each node's block hashes, `run_blocks` slots per node.
    hashes: Vec<u64>,
    /// Nodes of evicted entries, for reuse.
    free: Vec<u32>,
    /// Resident blocks.
    blocks: usize,
    /// The free queue: unpinned entries, least recently released at `head`.
    head: u32,
    tail: u32,
    next_event_id: u64,
    stats: CacheStats,
}

struct Node {
    len: u32,
    pins: u32,
    prev: u32,
    next: u32,
}

impl WorkerCache {
    fn new(dp_rank: u32) -> Self {
        Self {
            dp_rank,
            entries: FxHashMap::default(),
            nodes: Vec::new(),
            hashes: Vec::new(),
            free: Vec::new(),
            blocks: 0,
            head: NIL,
            tail: NIL,
            next_event_id: 0,
            stats: CacheStats::default(),
        }
    }

    fn pin(&mut self, entry: u32) {
        if self.nodes[entry as usize].pins == 0 {
            self.unlink(entry);
        }
        self.nodes[entry as usize].pins += 1;
    }

    /// A pinned entry for `run`, none of whose blocks is resident, evicting from the
    /// head of the free queue until the blocks fit.
    fn allocate(
        &mut self,
        run: &[u64],
        capacity: usize,
        run_blocks: usize,
        evicted: &mut Vec<u64>,
    ) -> u32 {
        while self.blocks + run.len() > capacity && self.head != NIL {
            let victim = self.head;
            self.unlink(victim);
            let start = victim as usize * run_blocks;
            let len = self.nodes[victim as usize].len as usize;
            let hashes = &self.hashes[start..start + len];
            self.entries.remove(&hashes[0]);
            // Tail first, as an engine evicts.
            evicted.extend(hashes.iter().rev());
            self.blocks -= len;
            self.free.push(victim);
        }
        if self.blocks + run.len() > capacity {
            self.stats.over_capacity += run.len() as u64;
        }
        let node = Node {
            len: run.len() as u32,
            pins: 1,
            prev: NIL,
            next: NIL,
        };
        let entry = match self.free.pop() {
            Some(entry) => {
                self.nodes[entry as usize] = node;
                entry
            }
            None => {
                self.nodes.push(node);
                self.hashes.resize(self.nodes.len() * run_blocks, 0);
                (self.nodes.len() - 1) as u32
            }
        };
        let start = entry as usize * run_blocks;
        self.hashes[start..start + run.len()].copy_from_slice(run);
        // Every block belongs to one entry. Replacing a resident entry's key would
        // strand it, and its eviction would remove blocks the cache still holds.
        let replaced = self.entries.insert(run[0], entry);
        assert!(
            replaced.is_none(),
            "block {:#x} is already resident",
            run[0]
        );
        self.blocks += run.len();
        entry
    }

    fn release(&mut self, held: &[u32]) {
        for &entry in held.iter().rev() {
            let node = &mut self.nodes[entry as usize];
            node.pins -= 1;
            if node.pins == 0 {
                self.push_back(entry);
            }
        }
    }

    fn unlink(&mut self, entry: u32) {
        let Node { prev, next, .. } = self.nodes[entry as usize];
        match prev {
            NIL => self.head = next,
            prev => self.nodes[prev as usize].next = next,
        }
        match next {
            NIL => self.tail = prev,
            next => self.nodes[next as usize].prev = prev,
        }
    }

    fn push_back(&mut self, entry: u32) {
        let tail = self.tail;
        let node = &mut self.nodes[entry as usize];
        node.prev = tail;
        node.next = NIL;
        match tail {
            NIL => self.head = entry,
            tail => self.nodes[tail as usize].next = entry,
        }
        self.tail = entry;
    }

    fn stored(&mut self, parent: Option<u64>, blocks: Vec<(u64, u64)>) -> KvCacheEvent {
        self.event(KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(ExternalSequenceBlockHash),
            start_position: None,
            blocks: blocks
                .into_iter()
                .map(|(sequence, local)| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(sequence),
                    tokens_hash: LocalBlockHash(local),
                    mm_extra_info: None,
                })
                .collect(),
        }))
    }

    fn removed(&mut self, blocks: Vec<u64>) -> KvCacheEvent {
        self.stats.evicted_blocks += blocks.len() as u64;
        self.event(KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: blocks.into_iter().map(ExternalSequenceBlockHash).collect(),
        }))
    }

    fn event(&mut self, data: KvCacheEventData) -> KvCacheEvent {
        self.next_event_id += 1;
        KvCacheEvent {
            event_id: self.next_event_id,
            data,
            dp_rank: self.dp_rank,
        }
    }
}
