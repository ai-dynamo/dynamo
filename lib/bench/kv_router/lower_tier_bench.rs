// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lower-tier (KVCR host tier) indexer benchmark on a replayed trace.
//!
//! Host-tier events are synthesized from a Mooncake-format trace: sessions are
//! assigned to workers in order of first appearance (a row without
//! `session_id` is its own session), and each worker runs an LRU host cache of
//! `--num-host-blocks` engine blocks. A request touches its resident blocks,
//! evicts least-recently-used blocks to make room (one `Removed` event), and
//! stores its missing blocks (one `Stored` event per contiguous run). Each
//! worker publishes as its own KVCR cache owner, projected to that worker and
//! registered as a router-hint source.
//!
//! - `--phase write`: apply every event to a fresh `LowerTierIndexers`
//!   HostPinned tier and time it. The indexer's footprint is the drop in
//!   allocated heap when it is dropped, per live (block, owner) membership.
//! - `--phase query`: replay again; before each request's events, time the
//!   router's lower-tier query for that request (hints on, no device tier, so
//!   every walk starts at the root). The result checksum must match across
//!   builds that compute the same routing answers.
//!
//! One measurement per process; the result goes to `--result-json-output`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufRead;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use dynamo_kv_router::identity::{
    CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    StableDpSlotId,
};
use dynamo_kv_router::indexer::{
    KvIndexerInterface, LowerTierIndexers, LowerTierMatchDetails, LowerTierQueryOptions,
    MatchDetails, query_lower_tiers_with_options,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, ResidencyProjection, ResidencyRoutingSnapshot,
    RouterEvent, RouterHintSourceMetadata, StorageTier, WorkerWithDpRank,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Phase {
    Write,
    Query,
}

#[derive(Parser, Debug)]
struct Args {
    /// Mooncake-format JSONL trace (`hash_ids`, optional `session_id`). Lines
    /// without `hash_ids`, such as a format header, are skipped.
    trace: String,

    /// Which measurement to run.
    #[clap(long, value_enum)]
    phase: Phase,

    /// Workers, each with its own host cache and cache owner.
    #[clap(long, default_value = "32")]
    workers: usize,

    /// Engine block size in tokens.
    #[clap(long, default_value = "16")]
    block_size: u32,

    /// Tokens per trace `hash_id`; must be a multiple of `--block-size`.
    #[clap(long, default_value = "64")]
    trace_block_size: u32,

    /// Host-tier capacity per worker, in engine blocks.
    #[clap(long, default_value = "65536")]
    num_host_blocks: usize,

    /// Indexer event lanes (the router uses its `--router-event-threads`).
    #[clap(long, default_value = "4")]
    lanes: usize,

    /// Where to write the result JSON.
    #[clap(long, default_value = "lower_tier_result.json")]
    result_json_output: String,

    /// Ignored - passed by cargo bench harness.
    #[arg(long, hide = true)]
    bench: bool,
}

#[derive(serde::Deserialize)]
struct Row {
    session_id: Option<String>,
    hash_ids: Option<Vec<u64>>,
}

/// splitmix64 finalizer over two words; stands in for the engine's block hashes.
fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn cache_owner(index: usize) -> CacheOwnerId {
    let mut slot = [0; 16];
    slot[..8].copy_from_slice(&(index as u64).to_be_bytes());
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

fn worker(index: usize) -> WorkerWithDpRank {
    WorkerWithDpRank::new(index as u64 + 1, 0)
}

/// Every cache owner projects to its worker and is a router-hint source.
fn routing_snapshot(workers: usize) -> anyhow::Result<ResidencyRoutingSnapshot> {
    let owners: Vec<_> = (0..workers).map(|i| (cache_owner(i), worker(i))).collect();
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

/// LRU set of resident external block hashes. `order` holds (tick, hash) per
/// touch; entries whose tick is no longer current are skipped on eviction.
struct HostCache {
    capacity: usize,
    tick: u64,
    resident: HashMap<u64, u64>,
    order: VecDeque<(u64, u64)>,
}

impl HostCache {
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

    fn evict_lru(&mut self) -> Option<u64> {
        while let Some((tick, hash)) = self.order.pop_front() {
            if self.resident.get(&hash) == Some(&tick) {
                self.resident.remove(&hash);
                return Some(hash);
            }
        }
        None
    }
}

struct HostWorker {
    owner: CacheOwnerId,
    cache: HostCache,
    next_event_id: u64,
}

impl HostWorker {
    fn event(&mut self, index: usize, data: KvCacheEventData) -> RouterEvent {
        let event = KvCacheEvent {
            event_id: self.next_event_id,
            dp_rank: 0,
            data,
        };
        self.next_event_id += 1;
        RouterEvent::with_cache_owner(
            worker(index).worker_id,
            event,
            StorageTier::HostPinned,
            self.owner,
        )
    }
}

#[derive(Default, Serialize)]
struct SynthStats {
    requests: usize,
    sessions: usize,
    request_blocks: usize,
    store_events: usize,
    stored_blocks: usize,
    remove_events: usize,
    removed_blocks: usize,
}

struct Request {
    locals: Vec<LocalBlockHash>,
    events: Vec<RouterEvent>,
}

/// Streams the trace through per-worker LRU host caches, one request at a time.
struct Synth {
    lines: std::io::Lines<std::io::BufReader<std::fs::File>>,
    blocks_per_hash: u64,
    workers: Vec<HostWorker>,
    session_workers: HashMap<String, usize>,
    stats: SynthStats,
}

impl Synth {
    fn open(args: &Args) -> anyhow::Result<Self> {
        anyhow::ensure!(
            args.trace_block_size.is_multiple_of(args.block_size),
            "--trace-block-size must be a multiple of --block-size"
        );
        let file = std::fs::File::open(&args.trace)?;
        Ok(Self {
            lines: std::io::BufReader::new(file).lines(),
            blocks_per_hash: (args.trace_block_size / args.block_size) as u64,
            workers: (0..args.workers)
                .map(|index| HostWorker {
                    owner: cache_owner(index),
                    cache: HostCache::new(args.num_host_blocks),
                    next_event_id: 0,
                })
                .collect(),
            session_workers: HashMap::new(),
            stats: SynthStats::default(),
        })
    }

    fn next_request(&mut self) -> anyhow::Result<Option<Request>> {
        let (session_id, hash_ids) = loop {
            let Some(line) = self.lines.next() else {
                self.stats.sessions = self.session_workers.len();
                return Ok(None);
            };
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: Row = serde_json::from_str(&line)?;
            if let Some(hash_ids) = row.hash_ids {
                break (row.session_id, hash_ids);
            }
        };
        let session = session_id.unwrap_or_else(|| format!("row-{}", self.stats.requests));
        let next_worker = self.session_workers.len() % self.workers.len();
        let index = *self.session_workers.entry(session).or_insert(next_worker);
        self.stats.requests += 1;
        let worker = &mut self.workers[index];

        let mut blocks: Vec<(LocalBlockHash, u64)> = Vec::new();
        let mut parent = 0u64;
        for hash_id in &hash_ids {
            for sub in 0..self.blocks_per_hash {
                let local = mix(*hash_id, sub + 1);
                parent = mix(parent ^ 0x5EED, local);
                blocks.push((LocalBlockHash(local), parent));
            }
        }
        anyhow::ensure!(
            blocks.len() <= worker.cache.capacity,
            "a request has more blocks than the host cache"
        );
        self.stats.request_blocks += blocks.len();
        let mut events = Vec::new();

        // Resident blocks become most recent first, so eviction never takes them.
        let mut missing = 0usize;
        for (_, hash) in &blocks {
            if worker.cache.resident.contains_key(hash) {
                worker.cache.touch(*hash);
            } else {
                missing += 1;
            }
        }
        let overflow =
            (worker.cache.resident.len() + missing).saturating_sub(worker.cache.capacity);
        if overflow > 0 {
            let evicted: Vec<_> = (0..overflow)
                .filter_map(|_| worker.cache.evict_lru())
                .map(ExternalSequenceBlockHash)
                .collect();
            self.stats.remove_events += 1;
            self.stats.removed_blocks += evicted.len();
            events.push(worker.event(
                index,
                KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: evicted,
                }),
            ));
        }

        // One Stored event per contiguous run of missing blocks.
        let mut run: Vec<KvCacheStoredBlockData> = Vec::new();
        let mut run_parent = None;
        for (pos, (local, hash)) in blocks.iter().enumerate() {
            if worker.cache.resident.contains_key(hash) {
                if !run.is_empty() {
                    self.stats.store_events += 1;
                    self.stats.stored_blocks += run.len();
                    let data = KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: run_parent,
                        start_position: None,
                        blocks: std::mem::take(&mut run),
                    });
                    events.push(worker.event(index, data));
                }
                continue;
            }
            if run.is_empty() {
                run_parent = pos
                    .checked_sub(1)
                    .map(|prev| ExternalSequenceBlockHash(blocks[prev].1));
            }
            worker.cache.touch(*hash);
            run.push(KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(*hash),
                tokens_hash: *local,
                mm_extra_info: None,
            });
        }
        if !run.is_empty() {
            self.stats.store_events += 1;
            self.stats.stored_blocks += run.len();
            let data = KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: run_parent,
                start_position: None,
                blocks: run,
            });
            events.push(worker.event(index, data));
        }

        Ok(Some(Request {
            locals: blocks.into_iter().map(|(local, _)| local).collect(),
            events,
        }))
    }

    /// Live (block, owner) memberships and distinct live blocks.
    fn live(&self) -> (usize, usize) {
        let memberships = self.workers.iter().map(|w| w.cache.resident.len()).sum();
        let distinct = self
            .workers
            .iter()
            .flat_map(|w| w.cache.resident.keys())
            .collect::<HashSet<_>>()
            .len();
        (memberships, distinct)
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum Allocator {
    Glibc,
    Jemalloc,
}

/// Bytes the allocator has handed out and not freed: jemalloc's `stats.allocated`
/// when jemalloc is preloaded, else glibc's in-use plus mmapped heap.
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
        // SAFETY: malloc_trim only returns free pages to the kernel, and mallinfo2
        // reads allocator statistics; neither has preconditions.
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

/// Process user plus system CPU seconds so far.
#[cfg(target_os = "linux")]
fn process_cpu_secs() -> Option<f64> {
    // SAFETY: rusage is plain integers, for which all-zero is a valid value.
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    // SAFETY: getrusage fills the struct it is given.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let secs = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    (rc == 0).then(|| secs(usage.ru_utime) + secs(usage.ru_stime))
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_secs() -> Option<f64> {
    None
}

fn new_indexers(args: &Args) -> anyhow::Result<LowerTierIndexers> {
    let indexers = LowerTierIndexers::new(args.lanes, args.block_size);
    indexers.set_residency_routing_snapshot(routing_snapshot(args.workers)?);
    Ok(indexers)
}

#[derive(Serialize)]
struct WriteResult {
    host_events: usize,
    apply_secs: f64,
    apply_cpu_secs: Option<f64>,
    /// Stored plus removed blocks: the per-block work an event lane does.
    apply_cpu_ns_per_block_op: Option<f64>,
    /// Live (block, owner) memberships per the synthesized host caches.
    expected_memberships: usize,
    /// The same, counted from the indexer's dump: equal unless it rejected events.
    indexed_memberships: usize,
    distinct_live_blocks: usize,
    indexer_bytes: Option<f64>,
    bytes_per_membership: Option<f64>,
    bytes_per_live_block: Option<f64>,
}

async fn write_phase(args: &Args, synth: &mut SynthStats) -> anyhow::Result<WriteResult> {
    let mut source = Synth::open(args)?;
    let mut events = Vec::new();
    while let Some(request) = source.next_request()? {
        events.extend(request.events);
    }
    let (expected_memberships, distinct_live_blocks) = source.live();
    *synth = std::mem::take(&mut source.stats);
    drop(source);
    let block_ops = (synth.stored_blocks + synth.removed_blocks) as f64;
    let host_events = events.len();

    let cpu_before = process_cpu_secs();
    let started = Instant::now();
    let indexers = new_indexers(args)?;
    let indexer = indexers.get_or_create(StorageTier::HostPinned);
    for event in events {
        indexer.apply_event(event).await;
    }
    indexer.flush_and_wait().await?;
    let apply_secs = started.elapsed().as_secs_f64();
    let apply_cpu_secs = cpu_before
        .zip(process_cpu_secs())
        .map(|(before, after)| after - before);

    let loaded = heap_allocated()?;
    let indexed_memberships = indexer
        .dump_events()
        .await?
        .iter()
        .map(|event| match &event.event.data {
            KvCacheEventData::Stored(store) => store.blocks.len(),
            _ => 0,
        })
        .sum();
    drop(indexer);
    drop(indexers);
    let dropped = heap_allocated()?;
    let indexer_bytes = loaded
        .zip(dropped)
        .map(|((_, loaded), (_, dropped))| loaded as f64 - dropped as f64);
    Ok(WriteResult {
        host_events,
        apply_secs,
        apply_cpu_secs,
        apply_cpu_ns_per_block_op: apply_cpu_secs.map(|secs| secs * 1e9 / block_ops),
        expected_memberships,
        indexed_memberships,
        distinct_live_blocks,
        indexer_bytes,
        bytes_per_membership: indexer_bytes.map(|bytes| bytes / expected_memberships as f64),
        bytes_per_live_block: indexer_bytes.map(|bytes| bytes / distinct_live_blocks as f64),
    })
}

/// Stable 64-bit digest of a query result, for cross-build equality.
fn fold_result(mut checksum: u64, details: &LowerTierMatchDetails) -> u64 {
    let mut hits: Vec<_> = details.hits.iter().map(|(w, h)| (*w, *h)).collect();
    hits.sort_unstable();
    for (worker, hit) in hits {
        checksum = mix(checksum, mix(worker.worker_id, hit as u64));
    }
    if let Some(candidates) = &details.kv_transfer_candidates {
        for hash in &candidates.block_hashes {
            checksum = mix(checksum, hash.0);
        }
        for (source, blocks) in &candidates.owner_prefix_blocks {
            let source = format!("{source:?}")
                .bytes()
                .fold(0xCBF2_9CE4_8422_2325, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100_0000_01B3)
                });
            checksum = mix(checksum, mix(source, *blocks as u64));
        }
    }
    checksum
}

fn nearest_rank(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = sorted
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Serialize)]
struct QueryResult {
    queries: usize,
    query_blocks: u64,
    /// Sum over queries of the deepest worker hit.
    matched_blocks: u64,
    mean_ns: f64,
    p50_ns: u64,
    p99_ns: u64,
    max_ns: u64,
    ns_per_matched_block: f64,
    phase_secs: f64,
    checksum: String,
}

async fn query_phase(args: &Args, synth: &mut SynthStats) -> anyhow::Result<QueryResult> {
    let indexers = new_indexers(args)?;
    let indexer = indexers.get_or_create(StorageTier::HostPinned);
    let options = LowerTierQueryOptions {
        retain_kv_transfer_chain: true,
    };
    let device = MatchDetails::default();
    let mut source = Synth::open(args)?;
    let mut query_ns: Vec<u64> = Vec::new();
    let mut matched_blocks = 0u64;
    let mut query_blocks = 0u64;
    let mut checksum = 0u64;

    let started = Instant::now();
    while let Some(request) = source.next_request()? {
        indexer.flush_and_wait().await?;
        let query_started = Instant::now();
        let result = query_lower_tiers_with_options(&indexers, &request.locals, &device, options);
        query_ns.push(query_started.elapsed().as_nanos() as u64);
        query_blocks += request.locals.len() as u64;
        if let Some(details) = result.get(&StorageTier::HostPinned) {
            matched_blocks += details.hits.values().copied().max().unwrap_or(0) as u64;
            checksum = fold_result(checksum, details);
        }
        for event in request.events {
            indexer.apply_event(event).await;
        }
    }
    indexer.flush_and_wait().await?;
    let phase_secs = started.elapsed().as_secs_f64();
    *synth = std::mem::take(&mut source.stats);
    anyhow::ensure!(!query_ns.is_empty(), "the trace has no requests");

    let total_ns: u64 = query_ns.iter().sum();
    let mut sorted = query_ns;
    sorted.sort_unstable();
    Ok(QueryResult {
        queries: sorted.len(),
        query_blocks,
        matched_blocks,
        mean_ns: total_ns as f64 / sorted.len() as f64,
        p50_ns: nearest_rank(&sorted, 50, 100),
        p99_ns: nearest_rank(&sorted, 99, 100),
        max_ns: sorted.last().copied().unwrap_or_default(),
        ns_per_matched_block: total_ns as f64 / matched_blocks.max(1) as f64,
        phase_secs,
        checksum: format!("{checksum:016x}"),
    })
}

/// Where a result came from, so a result file can be traced to its exact command and trace.
#[derive(Serialize)]
struct Provenance {
    argv: Vec<String>,
    binary: Option<String>,
    binary_sha256: Option<String>,
    trace_sha256: String,
}

fn provenance(args: &Args) -> anyhow::Result<Provenance> {
    let file_sha256 = |path: &std::path::Path| -> anyhow::Result<String> {
        let mut hasher = Sha256::new();
        std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    };
    let binary = std::env::current_exe().ok();
    Ok(Provenance {
        argv: std::env::args().collect(),
        binary_sha256: binary.as_deref().map(file_sha256).transpose()?,
        binary: binary.map(|path| path.display().to_string()),
        trace_sha256: file_sha256(std::path::Path::new(&args.trace))?,
    })
}

#[derive(Serialize)]
struct BenchResult {
    schema_version: u32,
    bench: &'static str,
    phase: Phase,
    /// The allocator whose statistics give the memory figures.
    allocator: Option<Allocator>,
    trace: String,
    workers: usize,
    block_size: u32,
    trace_block_size: u32,
    num_host_blocks: usize,
    lanes: usize,
    synth: SynthStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    write: Option<WriteResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<QueryResult>,
    provenance: Provenance,
}

async fn async_main(args: Args) -> anyhow::Result<()> {
    let mut synth = SynthStats::default();
    let (write, query) = match args.phase {
        Phase::Write => (Some(write_phase(&args, &mut synth).await?), None),
        Phase::Query => (None, Some(query_phase(&args, &mut synth).await?)),
    };
    let result = BenchResult {
        schema_version: RESULT_SCHEMA_VERSION,
        bench: "lower_tier",
        phase: args.phase,
        allocator: heap_allocated()?.map(|(allocator, _)| allocator),
        trace: args.trace.clone(),
        workers: args.workers,
        block_size: args.block_size,
        trace_block_size: args.trace_block_size,
        num_host_blocks: args.num_host_blocks,
        lanes: args.lanes,
        synth,
        write,
        query,
        provenance: provenance(&args)?,
    };
    std::fs::write(
        &args.result_json_output,
        serde_json::to_string_pretty(&result)?,
    )?;
    println!("Lower-tier result written to {}", args.result_json_output);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(args))
}
