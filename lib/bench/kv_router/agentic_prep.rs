// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic workload preparation for `mooncake_bench`: load the row pool, capture every
//! worker's closed-loop replay in virtual time, then merge and rescale into the same
//! prepared structure the Mooncake open-loop replay consumes.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dynamo_bench::kv_router_common::agentic::{
    AgenticCorpusConfig, AgenticPool, AgenticPrepReport, generate_agentic_artifacts,
};
use dynamo_bench::kv_router_common::replay::mock_engine_args_with_speedup;
use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, SglangArgs};

use super::mooncake_shared::{
    MergedMooncakeBenchmark, PreparedMooncakeBenchmark, WarmupEvent, WorkerTraceEntry,
    merge_worker_traces, prepare_scaled_benchmark_global,
};
use super::scaling_diag::PrepTimings;

/// Engine settings for the closed-loop capture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AgenticEngine {
    pub(crate) num_gpu_blocks: usize,
    pub(crate) block_size: u32,
    pub(crate) speedup_ratio: f64,
    /// `vllm` keeps the original capture; `sglang` uses the radix-cache engine with
    /// `page_size = block_size`.
    pub(crate) sglang: bool,
}

impl AgenticEngine {
    fn mock_engine_args(self) -> anyhow::Result<MockEngineArgs> {
        if !self.sglang {
            return mock_engine_args_with_speedup(
                self.num_gpu_blocks,
                self.block_size as usize,
                self.speedup_ratio,
            );
        }
        Ok(MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .sglang(Some(SglangArgs {
                page_size: Some(self.block_size as usize),
                ..SglangArgs::default()
            }))
            .num_gpu_blocks(self.num_gpu_blocks)
            .block_size(self.block_size as usize)
            .speedup_ratio(self.speedup_ratio)
            .enable_prefix_caching(true)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .build()?)
    }
}

/// [`prepare_agentic_benchmark`] through an on-disk corpus cache: when `cache` exists it is
/// loaded (its key must equal `key`, and its content digest must verify) instead of
/// re-capturing; otherwise the capture runs and, if a cache path is given, is written there
/// before the per-trial rescale. Returns the agentic provenance as JSON.
pub(crate) async fn prepare_agentic_benchmark_cached(
    pool_path: &Path,
    expected_pool_sha256: Option<&str>,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
    benchmark_duration_ms: u64,
    cache: Option<&Path>,
    key: serde_json::Value,
) -> anyhow::Result<(PreparedMooncakeBenchmark, PrepTimings, serde_json::Value)> {
    let mut timings = PrepTimings::default();
    if let Some(path) = cache.filter(|path| path.exists()) {
        let started = Instant::now();
        let (merged, warmup_events, mut report, digest) = corpus_cache::read(path, &key)?;
        timings.trace_load_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let mut prepared = prepare_scaled_benchmark_global(merged, benchmark_duration_ms);
        prepared.warmup_events = warmup_events;
        check_worker_order(&prepared)?;
        report["merged_corpus_digest"] = prepared_corpus_digest(&prepared).into();
        report["corpus_cache"] = serde_json::json!({
            "path": path.display().to_string(),
            "digest": digest,
            "loaded": true,
            "load_ms": timings.trace_load_ms,
        });
        timings.merge_and_rescale_ms = started.elapsed().as_secs_f64() * 1e3;
        return Ok((prepared, timings, report));
    }
    let (merged, warmup_events, mut report, capture_timings) =
        capture_agentic(pool_path, expected_pool_sha256, config, engine).await?;
    timings = capture_timings;
    let cache_json = match cache {
        Some(path) => {
            let started = Instant::now();
            let report_json = serde_json::to_value(&report)?;
            let digest = corpus_cache::write(path, &key, &report_json, &merged, &warmup_events)?;
            serde_json::json!({
                "path": path.display().to_string(),
                "digest": digest,
                "loaded": false,
                "write_ms": started.elapsed().as_secs_f64() * 1e3,
            })
        }
        None => serde_json::Value::Null,
    };
    let started = Instant::now();
    let mut prepared = prepare_scaled_benchmark_global(merged, benchmark_duration_ms);
    prepared.warmup_events = warmup_events;
    check_worker_order(&prepared)?;
    report.merged_corpus_digest = prepared_corpus_digest(&prepared);
    timings.merge_and_rescale_ms += started.elapsed().as_secs_f64() * 1e3;
    let mut report = serde_json::to_value(&report)?;
    report["corpus_cache"] = cache_json;
    Ok((prepared, timings, report))
}

/// Load the pool (failing closed on an expected SHA-256), capture, and merge (no rescale).
async fn capture_agentic(
    pool_path: &Path,
    expected_pool_sha256: Option<&str>,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
) -> anyhow::Result<(
    MergedMooncakeBenchmark,
    Vec<WarmupEvent>,
    AgenticPrepReport,
    PrepTimings,
)> {
    let mut timings = PrepTimings::default();
    let started = Instant::now();
    let (pool, pool_sha256) = AgenticPool::read(pool_path)?;
    if let Some(expected) = expected_pool_sha256
        && !expected.eq_ignore_ascii_case(&pool_sha256)
    {
        anyhow::bail!("agentic pool sha256 {pool_sha256} != expected {expected}");
    }
    timings.trace_load_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut report = AgenticPrepReport {
        workload: "agentic",
        pool_path: pool_path.display().to_string(),
        pool_sha256,
        pool_source_sha256: pool.source_sha256.clone(),
        pool_plays: pool.plays.len(),
        pool_requests: pool.request_count(),
        pool_block_size: pool.header.block_size,
        pool_source_digest: pool.header.source.digest.clone(),
        nested_timestamp_basis: pool.nested_timestamp_basis.clone(),
        ..AgenticPrepReport::default()
    };
    let started = Instant::now();
    let engine_args = engine.mock_engine_args()?;
    report.engine_type = if engine.sglang { "sglang" } else { "vllm" }.to_string();
    let (artifacts, warmups) =
        generate_agentic_artifacts(Arc::new(pool), config, engine_args, &mut report).await?;
    timings.simulation_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let warmup_events = merge_warmup_events(warmups);
    let merged = merge_worker_traces(artifacts, engine.block_size)?;
    timings.merge_and_rescale_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((merged, warmup_events, report, timings))
}

/// Load the pool (failing closed on an expected SHA-256), capture, merge, and rescale.
#[allow(dead_code)]
pub(crate) async fn prepare_agentic_benchmark(
    pool_path: &Path,
    expected_pool_sha256: Option<&str>,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
    benchmark_duration_ms: u64,
) -> anyhow::Result<(PreparedMooncakeBenchmark, PrepTimings, AgenticPrepReport)> {
    let (merged, warmup_events, mut report, mut timings) =
        capture_agentic(pool_path, expected_pool_sha256, config, engine).await?;
    let started = Instant::now();
    let mut prepared = prepare_scaled_benchmark_global(merged, benchmark_duration_ms);
    prepared.warmup_events = warmup_events;
    check_worker_order(&prepared)?;
    report.merged_corpus_digest = prepared_corpus_digest(&prepared);
    timings.merge_and_rescale_ms += started.elapsed().as_secs_f64() * 1e3;
    Ok((prepared, timings, report))
}

/// Merge per-worker warm-up events into one list ordered by (virtual time, worker, source
/// order); per-worker order is what the indexer needs, the global order mimics arrival.
fn merge_warmup_events(
    warmups: Vec<Vec<dynamo_mocker::replay::ReplayTimedKvEvent>>,
) -> Vec<WarmupEvent> {
    let mut keyed = Vec::with_capacity(warmups.iter().map(Vec::len).sum());
    for (worker, events) in warmups.into_iter().enumerate() {
        for (ordinal, event) in events.into_iter().enumerate() {
            keyed.push((event.timestamp_us, worker, ordinal, event));
        }
    }
    keyed.sort_unstable_by_key(|(timestamp_us, worker, ordinal, _)| {
        (*timestamp_us, *worker, *ordinal)
    });
    keyed
        .into_iter()
        .map(|(_, worker, _, event)| WarmupEvent {
            worker,
            event: event.event,
            storage_tier: event.storage_tier,
        })
        .collect()
}

/// Fail closed unless every worker's merged timeline is time-ordered.
pub(crate) fn check_worker_order(prepared: &PreparedMooncakeBenchmark) -> anyhow::Result<()> {
    for (worker, trace) in prepared.worker_traces.iter().enumerate() {
        if let Some(index) = trace
            .windows(2)
            .position(|pair| pair[1].timestamp_us < pair[0].timestamp_us)
        {
            anyhow::bail!(
                "worker {worker} merged timeline is out of order at entry {}",
                index + 1
            );
        }
    }
    Ok(())
}

/// xxh3 digest of a prepared corpus: every worker's timestamps, query hashes, and events.
/// Two preparations with equal digests replay identical operations.
pub(crate) fn prepared_corpus_digest(prepared: &PreparedMooncakeBenchmark) -> String {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut put = |value: u64| hasher.update(&value.to_le_bytes());
    put(prepared.benchmark_duration_ms);
    put(u64::from(prepared.block_size));
    for (worker, trace) in prepared.worker_traces.iter().enumerate() {
        put(worker as u64);
        put(trace.len() as u64);
        for entry in trace {
            put(entry.timestamp_us);
            match &entry.entry {
                WorkerTraceEntry::Request(hashes) => {
                    put(0);
                    put(hashes.len() as u64);
                    hashes.iter().for_each(|hash| put(hash.0));
                }
                WorkerTraceEntry::Event {
                    event,
                    storage_tier,
                } => {
                    put(1);
                    put(event.event_id);
                    put(u64::from(event.dp_rank));
                    put(u64::from(storage_tier.is_gpu()));
                    match &event.data {
                        KvCacheEventData::Stored(store) => {
                            put(2);
                            put(store.parent_hash.map_or(u64::MAX, |hash| hash.0));
                            put(store.start_position.map_or(u64::MAX, u64::from));
                            put(store.blocks.len() as u64);
                            for block in &store.blocks {
                                put(block.block_hash.0);
                                put(block.tokens_hash.0);
                            }
                        }
                        KvCacheEventData::Removed(remove) => {
                            put(3);
                            put(remove.block_hashes.len() as u64);
                            remove.block_hashes.iter().for_each(|hash| put(hash.0));
                        }
                        KvCacheEventData::Cleared => put(4),
                    }
                }
            }
        }
    }
    put(prepared.warmup_events.len() as u64);
    for warmup in &prepared.warmup_events {
        put(warmup.worker as u64);
        put(warmup.event.event_id);
        match &warmup.event.data {
            KvCacheEventData::Stored(store) => {
                put(2);
                put(store.parent_hash.map_or(u64::MAX, |hash| hash.0));
                put(store.blocks.len() as u64);
                for block in &store.blocks {
                    put(block.block_hash.0);
                    put(block.tokens_hash.0);
                }
            }
            KvCacheEventData::Removed(remove) => {
                put(3);
                put(remove.block_hashes.len() as u64);
                remove.block_hashes.iter().for_each(|hash| put(hash.0));
            }
            KvCacheEventData::Cleared => put(4),
        }
    }
    format!("{:016x}", hasher.digest())
}

/// EXPERIMENT ONLY (e2e indexer-contention campaign): load the corpus from `cache` (or capture
/// it, writing `cache` when given) and write one phantom-stream base per captured worker to
/// `out_dir`. Returns the stream manifest's summary.
pub(crate) async fn export_agentic_phantom_streams(
    pool_path: &Path,
    expected_pool_sha256: Option<&str>,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
    cache: Option<&Path>,
    key: serde_json::Value,
    out_dir: &Path,
) -> anyhow::Result<serde_json::Value> {
    let (merged, warmup, capture, cache_json) = match cache.filter(|path| path.exists()) {
        Some(path) => {
            let (merged, warmup, report, digest) = corpus_cache::read(path, &key)?;
            let cache_json = serde_json::json!({
                "path": path.display().to_string(),
                "digest": digest,
                "loaded": true,
            });
            (merged, warmup, report, cache_json)
        }
        None => {
            let (merged, warmup, report, _) =
                capture_agentic(pool_path, expected_pool_sha256, config, engine).await?;
            let report = serde_json::to_value(&report)?;
            let cache_json = match cache {
                Some(path) => {
                    let digest = corpus_cache::write(path, &key, &report, &merged, &warmup)?;
                    serde_json::json!({
                        "path": path.display().to_string(),
                        "digest": digest,
                        "loaded": false,
                    })
                }
                None => serde_json::Value::Null,
            };
            (merged, warmup, report, cache_json)
        }
    };
    let provenance = serde_json::json!({
        "exporter": "mooncake_bench --export-phantom-streams",
        "key": key,
        "capture": capture,
        "corpus_cache": cache_json,
    });
    phantom_export::write(merged, warmup, provenance, out_dir)
}

mod phantom_export {
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{bail, ensure};
    use dynamo_e2e_indexer_tools::stream::{
        BaseInfo, BaseStream, EventLists, Manifest, base_file_name, write_base,
    };
    use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier};

    use super::super::mooncake_shared::{
        MergedMooncakeBenchmark, WarmupEvent, WorkerTrace, WorkerTraceEntry,
    };

    fn push_raw(
        lists: &mut EventLists,
        event: &KvCacheEvent,
        tier: StorageTier,
    ) -> anyhow::Result<()> {
        ensure!(
            tier == StorageTier::Device && event.dp_rank == 0,
            "phantom streams support device-tier, DP-rank-0 events only"
        );
        match &event.data {
            KvCacheEventData::Stored(store) => {
                if store.start_position.is_some()
                    || store
                        .blocks
                        .iter()
                        .any(|block| block.mm_extra_info.is_some())
                {
                    bail!("phantom streams do not support positional or multimodal stores");
                }
                lists.push_store(
                    store.parent_hash.map(|hash| hash.0),
                    store
                        .blocks
                        .iter()
                        .map(|block| (block.block_hash.0, block.tokens_hash.0)),
                );
            }
            KvCacheEventData::Removed(remove) => {
                lists.push_remove(remove.block_hashes.iter().map(|hash| hash.0))
            }
            KvCacheEventData::Cleared => lists.push_clear(),
        }
        Ok(())
    }

    /// One base's timed trace and warm-up lists, taken once by an export thread.
    type BaseWork = (Vec<WorkerTrace>, EventLists);

    /// Timed events with one timestamp form one engine publish (one scheduler pass).
    fn base_stream(trace: Vec<WorkerTrace>, warmup: EventLists) -> anyhow::Result<BaseStream> {
        let mut stream = BaseStream {
            warmup,
            ..BaseStream::default()
        };
        let mut open_ts = None;
        for WorkerTrace {
            entry,
            timestamp_us,
        } in trace
        {
            match entry {
                WorkerTraceEntry::Request(hashes) => stream
                    .queries
                    .push(timestamp_us, hashes.iter().map(|hash| hash.0)),
                WorkerTraceEntry::Event {
                    event,
                    storage_tier,
                } => {
                    if open_ts != Some(timestamp_us) {
                        if let Some(ts) = open_ts {
                            stream.timed.close_list(ts);
                        }
                        open_ts = Some(timestamp_us);
                    }
                    push_raw(&mut stream.timed, &event, storage_tier)?;
                }
            }
        }
        if let Some(ts) = open_ts {
            stream.timed.close_list(ts);
        }
        Ok(stream)
    }

    pub(super) fn write(
        merged: MergedMooncakeBenchmark,
        warmup: Vec<WarmupEvent>,
        provenance: serde_json::Value,
        out_dir: &Path,
    ) -> anyhow::Result<serde_json::Value> {
        std::fs::create_dir_all(out_dir)?;
        let block_size = merged.block_size();
        let traces = merged.into_worker_traces();
        let workers = traces.len();

        // The merged warm-up is ordered by (virtual time, worker, source order), so a run of
        // one worker's events is one scheduler pass.
        let mut warmups = vec![EventLists::default(); workers];
        let mut previous: Option<usize> = None;
        for event in warmup {
            ensure!(
                event.worker < workers,
                "warm-up event for unknown worker {}",
                event.worker
            );
            if let Some(worker) = previous.filter(|&worker| worker != event.worker) {
                warmups[worker].close_list(0);
            }
            push_raw(&mut warmups[event.worker], &event.event, event.storage_tier)?;
            previous = Some(event.worker);
        }
        if let Some(worker) = previous {
            warmups[worker].close_list(0);
        }

        let work: Vec<Mutex<Option<BaseWork>>> = traces
            .into_iter()
            .zip(warmups)
            .map(|item| Mutex::new(Some(item)))
            .collect();
        let infos: Vec<Mutex<Option<BaseInfo>>> = (0..workers).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(workers.max(1));
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| -> anyhow::Result<()> {
                        loop {
                            let base = next.fetch_add(1, Ordering::Relaxed);
                            if base >= workers {
                                return Ok(());
                            }
                            let (trace, warmup) = work[base]
                                .lock()
                                .unwrap()
                                .take()
                                .expect("each base is taken once");
                            let stream = base_stream(trace, warmup)?;
                            let file = base_file_name(base);
                            write_base(&out_dir.join(&file), &stream)?;
                            *infos[base].lock().unwrap() = Some(BaseInfo::describe(file, &stream));
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("export thread panicked")?;
            }
            Ok(())
        })?;
        let infos: Vec<BaseInfo> = infos
            .into_iter()
            .map(|info| info.into_inner().unwrap().expect("every base was written"))
            .collect();
        let manifest = Manifest::new(block_size, infos, provenance);
        manifest.write(out_dir)?;
        let sum = |f: &dyn Fn(&BaseInfo) -> u64| manifest.bases.iter().map(f).sum::<u64>();
        Ok(serde_json::json!({
            "mode": "export_phantom_streams",
            "out_dir": out_dir.display().to_string(),
            "bases": manifest.bases.len(),
            "block_size": block_size,
            "t0_us": manifest.t0_us,
            "t1_us": manifest.t1_us,
            "warmup_write_blocks": sum(&|base| base.warmup.write_blocks()),
            "timed_events": sum(&|base| base.timed.events),
            "timed_lists": sum(&|base| base.timed.lists),
            "timed_stored_blocks": sum(&|base| base.timed.stored_blocks),
            "timed_removed_blocks": sum(&|base| base.timed.removed_blocks),
            "queries": sum(&|base| base.queries),
            "query_blocks": sum(&|base| base.query_blocks),
        }))
    }
}

/// Binary corpus cache: the merged, not yet rescaled per-worker timelines plus the warm-up
/// prefix and the capture provenance. Lookups are raw little-endian u64 arrays; KV events are
/// MessagePack. An xxh3 digest of everything after the magic is stored as the trailer.
mod corpus_cache {
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::Path;

    use anyhow::{Context, ensure};
    use dynamo_kv_router::protocols::{KvCacheEvent, LocalBlockHash, StorageTier};
    use xxhash_rust::xxh3::Xxh3;

    use super::super::mooncake_shared::{
        MergedMooncakeBenchmark, WarmupEvent, WorkerTrace, WorkerTraceEntry,
    };

    const MAGIC: &[u8; 8] = b"AGCORPC1";

    struct Writer<W: Write> {
        inner: W,
        hasher: Xxh3,
    }

    impl<W: Write> Writer<W> {
        fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            self.hasher.update(bytes);
            self.inner.write_all(bytes)
        }
        fn u64(&mut self, value: u64) -> std::io::Result<()> {
            self.put(&value.to_le_bytes())
        }
        fn blob(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            self.u64(bytes.len() as u64)?;
            self.put(bytes)
        }
    }

    struct Reader<R: Read> {
        inner: R,
        hasher: Xxh3,
    }

    impl<R: Read> Reader<R> {
        fn take(&mut self, bytes: &mut [u8]) -> std::io::Result<()> {
            self.inner.read_exact(bytes)?;
            self.hasher.update(bytes);
            Ok(())
        }
        fn u64(&mut self) -> std::io::Result<u64> {
            let mut bytes = [0u8; 8];
            self.take(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes))
        }
        fn blob(&mut self) -> anyhow::Result<Vec<u8>> {
            let len = usize::try_from(self.u64()?)?;
            ensure!(
                len < (1 << 34),
                "corpus cache blob length {len} is implausible"
            );
            let mut bytes = vec![0u8; len];
            self.take(&mut bytes)?;
            Ok(bytes)
        }
    }

    fn encode_event(event: &KvCacheEvent, tier: StorageTier) -> anyhow::Result<Vec<u8>> {
        Ok(rmp_serde::to_vec(&(event, tier))?)
    }

    fn decode_event(bytes: &[u8]) -> anyhow::Result<(KvCacheEvent, StorageTier)> {
        Ok(rmp_serde::from_slice(bytes)?)
    }

    /// Write the cache atomically (temp file + rename) and return its hex digest.
    pub(super) fn write(
        path: &Path,
        key: &serde_json::Value,
        report: &serde_json::Value,
        merged: &MergedMooncakeBenchmark,
        warmup: &[WarmupEvent],
    ) -> anyhow::Result<String> {
        let tmp = path.with_extension("partial");
        let file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating corpus cache {}", tmp.display()))?;
        let mut inner = BufWriter::with_capacity(1 << 24, file);
        inner.write_all(MAGIC)?;
        // The header (key and capture provenance, which holds wall times) is outside the
        // digest, so equal corpora captured by different binaries share one digest.
        let header = serde_json::to_vec(&serde_json::json!({ "key": key, "report": report }))?;
        inner.write_all(&(header.len() as u64).to_le_bytes())?;
        inner.write_all(&header)?;
        let mut out = Writer {
            inner,
            hasher: Xxh3::new(),
        };
        out.u64(u64::from(merged.block_size()))?;
        out.u64(merged.worker_traces().len() as u64)?;
        let mut hashes = Vec::new();
        for trace in merged.worker_traces() {
            out.u64(trace.len() as u64)?;
            for entry in trace {
                out.u64(entry.timestamp_us)?;
                match &entry.entry {
                    WorkerTraceEntry::Request(request) => {
                        out.u64(0)?;
                        out.u64(request.len() as u64)?;
                        hashes.clear();
                        for hash in request {
                            hashes.extend_from_slice(&hash.0.to_le_bytes());
                        }
                        out.put(&hashes)?;
                    }
                    WorkerTraceEntry::Event {
                        event,
                        storage_tier,
                    } => {
                        out.u64(1)?;
                        out.blob(&encode_event(event, *storage_tier)?)?;
                    }
                }
            }
        }
        out.u64(warmup.len() as u64)?;
        for event in warmup {
            out.u64(event.worker as u64)?;
            out.blob(&encode_event(&event.event, event.storage_tier)?)?;
        }
        let digest = out.hasher.digest();
        out.inner.write_all(&digest.to_le_bytes())?;
        out.inner.flush()?;
        drop(out);
        std::fs::rename(&tmp, path)?;
        Ok(format!("{digest:016x}"))
    }

    /// Read a cache, failing closed on a key or digest mismatch.
    pub(super) fn read(
        path: &Path,
        key: &serde_json::Value,
    ) -> anyhow::Result<(
        MergedMooncakeBenchmark,
        Vec<WarmupEvent>,
        serde_json::Value,
        String,
    )> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening corpus cache {}", path.display()))?;
        let mut inner = BufReader::with_capacity(1 << 24, file);
        let mut magic = [0u8; 8];
        inner.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "{} is not a corpus cache", path.display());
        let mut len = [0u8; 8];
        inner.read_exact(&mut len)?;
        let mut header = vec![0u8; usize::try_from(u64::from_le_bytes(len))?];
        inner.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header)?;
        let mut input = Reader {
            inner,
            hasher: Xxh3::new(),
        };
        ensure!(
            &header["key"] == key,
            "corpus cache key mismatch: cache {} vs requested {}",
            header["key"],
            key
        );
        let block_size = u32::try_from(input.u64()?)?;
        let workers = usize::try_from(input.u64()?)?;
        let mut traces = Vec::with_capacity(workers);
        let mut bytes = Vec::new();
        for _ in 0..workers {
            let len = usize::try_from(input.u64()?)?;
            let mut trace = Vec::with_capacity(len);
            for _ in 0..len {
                let timestamp_us = input.u64()?;
                let entry = match input.u64()? {
                    0 => {
                        let count = usize::try_from(input.u64()?)?;
                        bytes.resize(count * 8, 0);
                        input.take(&mut bytes)?;
                        WorkerTraceEntry::Request(
                            bytes
                                .chunks_exact(8)
                                .map(|chunk| {
                                    LocalBlockHash(u64::from_le_bytes(chunk.try_into().unwrap()))
                                })
                                .collect(),
                        )
                    }
                    1 => {
                        let (event, storage_tier) = decode_event(&input.blob()?)?;
                        WorkerTraceEntry::Event {
                            event,
                            storage_tier,
                        }
                    }
                    tag => anyhow::bail!("corpus cache entry tag {tag} is unknown"),
                };
                trace.push(WorkerTrace {
                    entry,
                    timestamp_us,
                });
            }
            traces.push(trace);
        }
        let warmup_len = usize::try_from(input.u64()?)?;
        let mut warmup = Vec::with_capacity(warmup_len);
        for _ in 0..warmup_len {
            let worker = usize::try_from(input.u64()?)?;
            let (event, storage_tier) = decode_event(&input.blob()?)?;
            warmup.push(WarmupEvent {
                worker,
                event,
                storage_tier,
            });
        }
        let digest = input.hasher.digest();
        let mut trailer = [0u8; 8];
        input.inner.read_exact(&mut trailer)?;
        ensure!(
            u64::from_le_bytes(trailer) == digest,
            "corpus cache {} failed its digest check",
            path.display()
        );
        let mut rest = [0u8; 1];
        ensure!(
            input.inner.read(&mut rest)? == 0,
            "corpus cache {} has trailing bytes",
            path.display()
        );
        Ok((
            MergedMooncakeBenchmark::from_parts(traces, block_size),
            warmup,
            header["report"].clone(),
            format!("{digest:016x}"),
        ))
    }
}
