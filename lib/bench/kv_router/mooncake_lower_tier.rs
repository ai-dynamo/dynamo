// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lower-tier (KVCR host tier) mode of `mooncake_bench`.
//!
//! A vLLM worker with KVCR publishes two host-tier streams, and the router indexes both in its
//! `LowerTierIndexers`:
//! - The CPU offload tier, owned by the worker. Filled by write-through, it is vLLM's: the
//!   mocker's `native_host_offload` writes each computed prompt block through to host and evicts
//!   LRU. Filled by write-back ([`derive_write_back_layer`]), a block reaches it when the device
//!   evicts it, so a chain arrives tail first.
//! - KVCR's DRAM tier, owned by the worker's KVCR cache owner: [`derive_kvcr_layer`] copies every
//!   completed CPU-tier store into it, with its own capacity and LRU.
//!
//! Each query is the router's lookup (`LookupPipeline::find_matches_by_tier`): the CRTC device
//! match first, which seeds every worker's lower-tier walk.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use dynamo_kv_events::event::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
};
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
use dynamo_kv_indexer::host::{
    ConcurrentRadixTreeCompressed, KvIndexerInterface, KvIndexerMetrics, LowerTierIndexers,
    LowerTierQueryOptions, MatchDetails, ResidencyProjection, ResidencyRoutingSnapshot,
    RouterHintSourceMetadata,
};
use dynamo_router_protocol::identity::{
    CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
    RoutingScopeId, StableDpSlotId,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;
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
            ClusterId::new(1),
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

/// When a block reaches the CPU offload tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum HostFill {
    /// When it is computed, parent first: vLLM's CPU offload.
    #[default]
    WriteThrough,
    /// When the device evicts it, tail first: SGLang HiCache's `write_back`.
    WriteBack,
}

/// One worker's host tier: an LRU set of block hashes. `order` holds (tick, hash) per touch;
/// entries whose tick is no longer current are skipped on eviction.
struct HostLru {
    capacity: usize,
    tick: u64,
    resident: HashMap<u64, u64>,
    order: VecDeque<(u64, u64)>,
}

impl HostLru {
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
    let mut cache = HostLru::new(capacity);
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

/// A device block as its `Stored` event described it.
struct DeviceBlock {
    parent: Option<ExternalSequenceBlockHash>,
    position: Option<u32>,
    block: KvCacheStoredBlockData,
}

/// Builds one worker's CPU tier as a write-back cache, from a timeline with no host tier: a block
/// is written to host when the device evicts it. Each device `Removed` is followed by the host's
/// events for the evicted blocks it does not hold. A stretch of a chain goes out tail first, in
/// `Stored` events of at most `event_blocks` blocks, each in chain order, as SGLang's HiCache
/// publishes one store per evicted radix node, leaves first. The host evicts LRU, a chain's tail
/// first, and publishes the `Removed` ahead of the store that needed the room.
pub(crate) fn derive_write_back_layer(
    timeline: Vec<WorkerTrace>,
    capacity: usize,
    event_blocks: usize,
) -> anyhow::Result<Vec<WorkerTrace>> {
    anyhow::ensure!(
        (1..=capacity).contains(&event_blocks),
        "a write-back event holds 1 to {capacity} blocks, the host tier's capacity"
    );
    let mut host = HostLru::new(capacity);
    let mut device: HashMap<u64, DeviceBlock> = HashMap::new();
    let mut event_id = 0u64;
    let mut out = Vec::with_capacity(timeline.len());
    for entry in timeline {
        let evicted = match &entry.entry {
            WorkerTraceEntry::Event {
                event,
                storage_tier,
                ..
            } if storage_tier.is_gpu() => match &event.data {
                KvCacheEventData::Stored(store) => {
                    let mut parent = store.parent_hash;
                    for (index, block) in store.blocks.iter().enumerate() {
                        let position = store.start_position.map(|start| start + index as u32);
                        device.insert(
                            block.block_hash.0,
                            DeviceBlock {
                                parent,
                                position,
                                block: block.clone(),
                            },
                        );
                        parent = Some(block.block_hash);
                    }
                    None
                }
                KvCacheEventData::Removed(remove) => {
                    Some((remove.block_hashes.clone(), event.dp_rank))
                }
                KvCacheEventData::Cleared => None,
            },
            WorkerTraceEntry::Event { .. } => {
                anyhow::bail!("a write-back fill needs a timeline without host-tier events")
            }
            WorkerTraceEntry::Request(_) => None,
        };
        let timestamp_us = entry.timestamp_us;
        out.push(entry);
        let Some((evicted, dp_rank)) = evicted else {
            continue;
        };
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
                    cache_owner: false,
                },
            });
            event_id += 1;
        };

        // A block the host still holds is not written again; the eviction makes it recent.
        let mut writes = Vec::new();
        for hash in &evicted {
            if host.resident.contains_key(&hash.0) {
                host.touch(hash.0);
            } else if device.contains_key(&hash.0) {
                writes.push(hash.0);
            }
        }
        let written: HashSet<u64> = writes.iter().copied().collect();
        let parent_of = |hash: u64| device[&hash].parent.map(|parent| parent.0);
        let parents: HashSet<u64> = writes.iter().filter_map(|&hash| parent_of(hash)).collect();
        let mut taken = HashSet::new();
        // From each block with no evicted child, up its chain: one stretch, tail first.
        for &leaf in writes.iter().filter(|hash| !parents.contains(hash)) {
            let mut stretch = Vec::new();
            let mut next = Some(leaf);
            while let Some(hash) = next
                && written.contains(&hash)
                && taken.insert(hash)
            {
                stretch.push(hash);
                next = parent_of(hash);
            }
            for event in stretch.chunks(event_blocks) {
                // Tail to head, so the tail is the first of them to be evicted.
                for &hash in event {
                    host.touch(hash);
                }
                let removed = host.evict_overflow();
                if !removed.is_empty() {
                    push(KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: removed,
                    }));
                }
                let head = &device[&event[event.len() - 1]];
                push(KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: head.parent,
                    start_position: head.position,
                    blocks: event
                        .iter()
                        .rev()
                        .map(|hash| device[hash].block.clone())
                        .collect(),
                }));
            }
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

// These run in `tests/mooncake_trace.rs`, which includes this file. The benchmark binary builds
// in test mode without the harness, which drops the tests and leaves what they use unused.
#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use dynamo_bench::kv_router_common::replay::{
        WorkerReplayArtifacts, generate_replay_artifacts, process_mooncake_trace,
    };
    use dynamo_kv_indexer::host::ThreadPoolIndexer;
    use dynamo_mocker::replay::ReplayTimedKvEvent;

    use super::super::mooncake_open_loop::{
        OpenLoopConfig, prepare_mooncake_corpus, prepare_open_loop_trial, run_open_loop,
    };
    use super::super::mooncake_shared::{merge_worker_traces, prepare_scaled_benchmark};
    use super::*;

    const BLOCK_SIZE: u32 = 128;

    fn stored(parent: Option<u64>, start: u32, hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(ExternalSequenceBlockHash),
            start_position: Some(start),
            blocks: hashes
                .iter()
                .map(|&hash| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(hash),
                    tokens_hash: LocalBlockHash(hash + 100),
                    mm_extra_info: None,
                })
                .collect(),
        })
    }

    fn removed(hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .copied()
                .map(ExternalSequenceBlockHash)
                .collect(),
        })
    }

    #[test]
    fn write_back_layer_stores_evicted_chains_tail_first() -> anyhow::Result<()> {
        // The device holds the chain 1..=5 and evicts it leaf first in two batches (99 was never
        // stored), computes 3..=5 again and evicts them again.
        let device = [
            stored(None, 0, &[1, 2, 3, 4, 5]),
            removed(&[5, 4, 3]),
            removed(&[2, 1, 99]),
            stored(Some(2), 2, &[3, 4, 5]),
            removed(&[5, 4, 3]),
        ];
        let host_events = |capacity: usize, event_blocks: usize| -> anyhow::Result<_> {
            let kv_events = device
                .iter()
                .enumerate()
                .map(|(index, data)| ReplayTimedKvEvent {
                    event: KvCacheEvent {
                        event_id: index as u64,
                        data: data.clone(),
                        dp_rank: 0,
                    },
                    storage_tier: StorageTier::Device,
                    timestamp_us: 10 * index as u64,
                })
                .collect();
            let artifacts = vec![WorkerReplayArtifacts {
                requests: Vec::new(),
                output_signals: Vec::new(),
                kv_events,
            }];
            let merged = merge_worker_traces(artifacts, BLOCK_SIZE)?
                .with_write_back_layer(capacity, event_blocks)?;
            let prepared = prepare_scaled_benchmark(merged, 1_000);
            let events: Vec<KvCacheEventData> = prepared
                .worker_traces
                .iter()
                .next()
                .unwrap()
                .iter()
                .filter_map(|entry| match &entry.entry {
                    WorkerTraceEntry::Event {
                        event,
                        storage_tier: StorageTier::HostPinned,
                        cache_owner: false,
                    } => Some(event.data.clone()),
                    _ => None,
                })
                .collect();
            Ok((events, prepared.totals.cpu_tier))
        };

        // A host tier of 4 blocks, 2 blocks per store. Each eviction is written tail first; the
        // host drops its oldest block (5, then 2) to admit a store; 3 and 4, still on host at
        // the last eviction, are not written again.
        let (events, cpu_tier) = host_events(4, 2)?;
        assert_eq!(
            events,
            vec![
                stored(Some(3), 3, &[4, 5]),
                stored(Some(2), 2, &[3]),
                removed(&[5]),
                stored(None, 0, &[1, 2]),
                removed(&[2]),
                stored(Some(4), 4, &[5]),
            ]
        );
        assert_eq!(
            (
                cpu_tier.stored_events,
                cpu_tier.stored_blocks,
                cpu_tier.live_memberships
            ),
            (4, 6, 4)
        );
        // [4, 5] and [3] arrived before their parents; [1, 2] is a root and 5's parent was held.
        assert_eq!(
            (cpu_tier.before_parent_events, cpu_tier.before_parent_blocks),
            (2, 3)
        );
        assert_eq!(cpu_tier.stored_events_by_blocks, [2, 2, 0, 0]);

        // One block per store: every block of an eviction is its own event, tail first.
        let (events, cpu_tier) = host_events(8, 1)?;
        assert_eq!(
            events,
            vec![
                stored(Some(4), 4, &[5]),
                stored(Some(3), 3, &[4]),
                stored(Some(2), 2, &[3]),
                stored(Some(1), 1, &[2]),
                stored(None, 0, &[1]),
            ],
            "the host still holds 3..=5 at the last eviction"
        );
        assert_eq!(cpu_tier.before_parent_events, 4);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn open_loop_routes_a_write_back_fill() -> anyhow::Result<()> {
        let fixture = format!(
            "{}/testdata/mooncake_trace_1000.jsonl",
            env!("CARGO_MANIFEST_DIR")
        );
        // The fixture records one hash per 512-token trace block. The device evicts into a
        // host tier of the same size, 8 blocks per store, and KVCR copies it.
        let traces = process_mooncake_trace(&fixture, 512, 1, 1, 2, 42)?;
        let artifacts = generate_replay_artifacts(&traces, 1024, BLOCK_SIZE, None).await?;
        let merged = merge_worker_traces(artifacts, BLOCK_SIZE)?
            .with_write_back_layer(1024, 8)?
            .with_kvcr_layer(1024)?;
        let prepared = prepare_scaled_benchmark(merged, 1_000);
        let trial = prepare_open_loop_trial(prepare_mooncake_corpus(prepared, 1)?, 2)?;
        let indexer = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            BLOCK_SIZE,
        ));
        let result = run_open_loop(
            "concurrent-radix-tree-compressed",
            indexer,
            Some(LowerTierRun::crtc(2, BLOCK_SIZE, None, true)),
            trial,
            OpenLoopConfig {
                query_lanes: 2,
                issuer_threads: 3,
                spin_us: 50,
                issue_lag_diagnostic_threshold_us: 250,
                pre_run_quiescence_ms: 0,
                issuer_cpus: Vec::new(),
                query_issuer_cpu: None,
                backend_cpus: Vec::new(),
            },
        )
        .await?;

        assert!(
            result.failure_reasons.is_empty(),
            "{:?}",
            result.failure_reasons
        );
        let lower = result
            .lower_tier
            .expect("a lower-tier run reports its lower tier");
        assert!(
            lower.cpu_tier.before_parent_events > 0 && lower.kvcr.removed_events > 0,
            "a write-back fill stores blocks before their parents, and KVCR evicts"
        );
        assert_eq!(
            lower.cpu_tier_indexed_memberships,
            lower.cpu_tier.live_memberships
        );
        assert_eq!(lower.kvcr_indexed_memberships, lower.kvcr.live_memberships);
        Ok(())
    }
}
