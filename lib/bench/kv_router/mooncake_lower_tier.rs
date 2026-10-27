// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lower-tier (KVCR host tier) mode of `mooncake_bench`.
//!
//! A vLLM worker with KVCR publishes two host-tier streams, and the router indexes both in its
//! `LowerTierIndexers`:
//! - vLLM's CPU offload tier, owned by the worker: the mocker's `native_host_offload` writes each
//!   computed prompt block through to host and evicts LRU.
//! - KVCR's DRAM tier, owned by the worker's KVCR cache owner: [`derive_kvcr_layer`] copies every
//!   completed CPU-tier store into it, with its own capacity and LRU.
//!
//! Each query is the router's lookup (`LookupPipeline::find_matches_by_tier`): the CRTC device
//! match first, which seeds every worker's lower-tier walk.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use dynamo_kv_router::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::identity::{
    CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    StableDpSlotId,
};
use dynamo_kv_router::indexer::{
    KvIndexerInterface, KvIndexerMetrics, LowerTierIndexers, LowerTierQueryOptions, MatchDetails,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    LocalBlockHash, ResidencyProjection, ResidencyRoutingSnapshot, RouterHintSourceMetadata,
    StorageTier, WorkerWithDpRank,
};
use serde::Serialize;

use super::mooncake_shared::{WorkerTrace, WorkerTraceEntry};

/// A worker's KVCR cache owner: one per (worker, DP rank), as vLLM's state agent assigns them.
pub(crate) fn cache_owner(worker_id: u64) -> CacheOwnerId {
    let mut slot = [0; 16];
    slot[..8].copy_from_slice(&worker_id.to_be_bytes());
    CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(1),
        ),
        StableDpSlotId::new(slot, IdentitySource::Explicit),
    )
}

/// Each worker's owner projects to that worker and is a router-hint source, as the state agent
/// publishes owners once they are attached and ready.
pub(crate) fn routing_snapshot(
    worker_ids: impl IntoIterator<Item = u64>,
) -> anyhow::Result<ResidencyRoutingSnapshot> {
    let owners: Vec<_> = worker_ids
        .into_iter()
        .map(|worker_id| (cache_owner(worker_id), WorkerWithDpRank::new(worker_id, 0)))
        .collect();
    let projection = ResidencyProjection::new(owners.iter().copied())
        .map_err(|error| anyhow::anyhow!("projection: {error:?}"))?;
    Ok(ResidencyRoutingSnapshot::new(
        projection,
        owners.into_iter().map(|(owner, worker)| {
            (
                owner,
                RouterHintSourceMetadata {
                    source_control_endpoint: format!("tcp://worker-{}:9000", worker.worker_id),
                    worker_type: "decode".to_string(),
                    topology_domains: Default::default(),
                },
                Some(worker),
            )
        }),
    ))
}

/// One worker's KVCR DRAM tier: an LRU set of block hashes. `order` holds (tick, hash) per touch;
/// entries whose tick is no longer current are skipped on eviction.
struct KvcrCache {
    capacity: usize,
    tick: u64,
    resident: HashMap<u64, u64>,
    order: VecDeque<(u64, u64)>,
}

impl KvcrCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tick: 0,
            resident: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn touch(&mut self, hash: u64) {
        self.tick += 1;
        self.resident.insert(hash, self.tick);
        self.order.push_back((self.tick, hash));
        if self.order.len() > 4 * self.resident.len() + 1024 {
            let mut live: Vec<_> = self.resident.iter().map(|(h, t)| (*t, *h)).collect();
            live.sort_unstable();
            self.order = live.into();
        }
    }

    fn evict_overflow(&mut self) -> Vec<ExternalSequenceBlockHash> {
        let mut evicted = Vec::new();
        while self.resident.len() > self.capacity {
            let Some((tick, hash)) = self.order.pop_front() else {
                break;
            };
            if self.resident.get(&hash) == Some(&tick) {
                self.resident.remove(&hash);
                evicted.push(ExternalSequenceBlockHash(hash));
            }
        }
        evicted
    }
}

/// Adds the KVCR layer to one worker's timeline. Each completed CPU-tier store (a worker-owned
/// host-tier `Stored`) is copied into the worker's KVCR tier: its evictions become one `Removed`,
/// and each run of blocks KVCR didn't hold becomes one `Stored`, right after the CPU-tier event.
/// A copy refreshes recency tail to head, so a chain's tail is evicted before its head, and a
/// store longer than the tier keeps only its head.
pub(crate) fn derive_kvcr_layer(
    timeline: Vec<WorkerTrace>,
    capacity: usize,
) -> anyhow::Result<Vec<WorkerTrace>> {
    let mut cache = KvcrCache::new(capacity);
    let mut event_id = 0u64;
    let mut out = Vec::with_capacity(timeline.len());
    for entry in timeline {
        let copy = match &entry.entry {
            WorkerTraceEntry::Event {
                event,
                storage_tier,
                cache_owner: false,
            } if !storage_tier.is_gpu() => match &event.data {
                KvCacheEventData::Stored(store) => Some((store.clone(), event.dp_rank)),
                KvCacheEventData::Removed(_) => None,
                KvCacheEventData::Cleared => anyhow::bail!("host-tier clears are not modelled"),
            },
            _ => None,
        };
        let timestamp_us = entry.timestamp_us;
        out.push(entry);
        let Some((store, dp_rank)) = copy else {
            continue;
        };
        let blocks = &store.blocks[..store.blocks.len().min(capacity)];
        let missing: Vec<bool> = blocks
            .iter()
            .map(|block| !cache.resident.contains_key(&block.block_hash.0))
            .collect();
        for block in blocks.iter().rev() {
            cache.touch(block.block_hash.0);
        }
        let mut push = |data| {
            out.push(WorkerTrace {
                timestamp_us,
                entry: WorkerTraceEntry::Event {
                    event: KvCacheEvent {
                        event_id,
                        data,
                        dp_rank,
                    },
                    storage_tier: StorageTier::HostPinned,
                    cache_owner: true,
                },
            });
            event_id += 1;
        };
        let evicted = cache.evict_overflow();
        if !evicted.is_empty() {
            push(KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: evicted,
            }));
        }
        let mut start = 0;
        while start < missing.len() {
            if !missing[start] {
                start += 1;
                continue;
            }
            let end = (start..missing.len())
                .find(|&i| !missing[i])
                .unwrap_or(missing.len());
            push(KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: match start {
                    0 => store.parent_hash,
                    _ => Some(blocks[start - 1].block_hash),
                },
                start_position: store.start_position.map(|position| position + start as u32),
                blocks: blocks[start..end].to_vec(),
            }));
            start = end;
        }
    }
    Ok(out)
}

/// What the open loop needs for the lower tier: its indexers, the router's query options, and the
/// device match that seeds each lookup.
pub struct LowerTierRun<T> {
    pub indexers: LowerTierIndexers,
    pub options: LowerTierQueryOptions,
    pub device_details: fn(&T, &[LocalBlockHash], bool) -> MatchDetails,
}

impl LowerTierRun<ConcurrentRadixTreeCompressed> {
    /// As the router builds it (`Indexer::Concurrent`): one lane per primary event thread. It
    /// keeps the KV transfer chain when KVCR owners publish, as the router does for workers that
    /// advertise router hints.
    pub fn crtc(
        lanes: usize,
        block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
        kvcr: bool,
    ) -> Self {
        let indexers = LowerTierIndexers::new_with_metrics(lanes, block_size, metrics);
        // Start the host tier's lanes now, alongside the primary's. Created inside the run, they
        // would inherit the coordinator's pin to the query-issuer CPU.
        indexers.get_or_create(StorageTier::HostPinned);
        Self {
            indexers,
            options: LowerTierQueryOptions {
                retain_kv_transfer_chain: kvcr,
            },
            device_details: |crtc, sequence, retain| {
                crtc.find_match_details_impl_with_options(sequence, false, retain)
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Allocator {
    Glibc,
    Jemalloc,
}

/// Bytes the allocator has handed out and not freed: jemalloc's `stats.allocated` when jemalloc
/// is preloaded, else glibc's in-use plus mmapped heap.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn heap_allocated() -> anyhow::Result<Option<(Allocator, u64)>> {
    type Mallctl = unsafe extern "C" fn(
        *const libc::c_char,
        *mut libc::c_void,
        *mut libc::size_t,
        *mut libc::c_void,
        libc::size_t,
    ) -> libc::c_int;
    // SAFETY: dlsym only looks up a symbol among the loaded objects.
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"mallctl".as_ptr()) };
    if symbol.is_null() {
        // SAFETY: malloc_trim only returns free pages to the kernel, and mallinfo2 reads
        // allocator statistics; neither has preconditions.
        let info = unsafe {
            libc::malloc_trim(0);
            libc::mallinfo2()
        };
        let heap = (info.uordblks + info.hblkhd) as u64;
        return Ok(Some((Allocator::Glibc, heap)));
    }
    // SAFETY: the only `mallctl` a preloaded allocator exports is jemalloc's, with this signature.
    let mallctl: Mallctl = unsafe { std::mem::transmute(symbol) };
    // Writing `epoch` refreshes jemalloc's cached statistics.
    let mut epoch = 1u64;
    let mut epoch_len = std::mem::size_of::<u64>();
    // SAFETY: both pointers name `epoch`, a live u64 whose size is passed alongside.
    let rc = unsafe {
        mallctl(
            c"epoch".as_ptr(),
            (&raw mut epoch).cast(),
            &mut epoch_len,
            (&raw mut epoch).cast(),
            epoch_len,
        )
    };
    anyhow::ensure!(rc == 0, "mallctl epoch failed: {rc}");
    let mut allocated: libc::size_t = 0;
    let mut allocated_len = std::mem::size_of::<libc::size_t>();
    // SAFETY: reads one size_t into `allocated`, whose size is passed alongside.
    let rc = unsafe {
        mallctl(
            c"stats.allocated".as_ptr(),
            (&raw mut allocated).cast(),
            &mut allocated_len,
            std::ptr::null_mut(),
            0,
        )
    };
    anyhow::ensure!(rc == 0, "mallctl stats.allocated failed: {rc}");
    Ok(Some((Allocator::Jemalloc, allocated as u64)))
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn heap_allocated() -> anyhow::Result<Option<(Allocator, u64)>> {
    Ok(None)
}

pub(crate) struct Footprint {
    pub(crate) cpu_tier_memberships: usize,
    pub(crate) kvcr_memberships: usize,
    pub(crate) heap: Option<(Allocator, u64)>,
}

/// Counts the lower tier's (block, owner) memberships by domain from its dump, then measures its
/// heap as the drop in allocated bytes when it is dropped.
pub(crate) async fn footprint<T>(run: LowerTierRun<T>) -> anyhow::Result<Footprint> {
    let tier = run.indexers.get_or_create(StorageTier::HostPinned);
    let (mut cpu_tier_memberships, mut kvcr_memberships) = (0, 0);
    for event in tier.dump_events().await? {
        if let KvCacheEventData::Stored(store) = &event.event.data {
            match event.state_source {
                Some(_) => kvcr_memberships += store.blocks.len(),
                None => cpu_tier_memberships += store.blocks.len(),
            }
        }
    }
    drop(tier);
    let loaded = heap_allocated()?;
    drop(run);
    let dropped = heap_allocated()?;
    Ok(Footprint {
        cpu_tier_memberships,
        kvcr_memberships,
        heap: loaded
            .zip(dropped)
            .map(|((allocator, loaded), (_, dropped))| (allocator, loaded.saturating_sub(dropped))),
    })
}
