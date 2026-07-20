// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-1 benchmark for the retained actor-owned DC CKF Relay producer.
//!
//! This intentionally stops at a black-hole delta receipt. Consumer ingestion and queries have a
//! separate benchmark so actor admission, mutation, dirty tracking, sequencing, and delivery stay
//! attributable. The actor core emits unsequenced bucket images; the actor-local publisher owns
//! the checked publication sequence and the complete transport envelope.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::mem::size_of;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Barrier, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use clap::Parser;
use dynamo_bench::kv_router_common::dc_ckf_shared::{
    DcCkfCapacityMetadata, DcCkfCorpusMetadata, DcCkfCorpusSpec, prepare_dc_ckf_corpus,
};
use dynamo_bench::kv_router_common::issuer::{
    contiguous_worker_issuer, pin_current_thread, pin_current_thread_to_cpus,
};
use dynamo_bench::kv_router_common::replay::generate_replay_artifacts;
use dynamo_kv_router::identity::{
    CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_kv_router::indexer::cuckoo::{
    CkfConfig, ConsumerInstanceId, DcCkfDelta, DcCkfDeltaSink, DcCkfPublicationBatch,
    DcCkfPublisher, DcCkfState, LaneLease, ProducerIdentity,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheEventError, RouterEvent,
};
use dynamo_mocker::replay::ReplayWorkerArtifacts;
use serde::Serialize;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_REPLAY_GPU_BLOCKS: usize = 65_536;
const DEFAULT_QUEUE_CAPACITY: usize = 256;
const DEFAULT_RESERVED_CONTROL_CORES: usize = 2;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Parser)]
#[command(about = "Benchmark the actor-owned DC CKF Relay producer and publisher")]
struct Args {
    /// Canonical Mooncake JSONL trace.
    #[arg(long)]
    trace_path: PathBuf,

    /// Expected trace SHA-256, with an optional `sha256:` prefix.
    #[arg(long)]
    trace_sha256: Option<String>,

    #[arg(long, default_value_t = 512)]
    block_size: usize,

    /// Checked multiplier for the effective prefix depth.
    #[arg(long, default_value_t = 1)]
    prefix_depth: usize,

    /// Checked copies in disjoint effective u32 hash spaces.
    #[arg(long, default_value_t = 1)]
    trace_duplication_factor: usize,

    #[arg(long, default_value_t = 4)]
    dc_count: usize,

    /// Multiplexed FIFO sources per DC; these are not OS threads.
    #[arg(long, default_value_t = 64)]
    logical_workers_per_dc: usize,

    /// Mock-engine blocks used only while deriving the frozen event corpus.
    #[arg(long, default_value_t = DEFAULT_REPLAY_GPU_BLOCKS)]
    replay_gpu_blocks: usize,

    /// Override the distinct-hash capacity of every DC CKF pool.
    #[arg(long)]
    pool_capacity: Option<usize>,

    /// Headroom above measured peak live distinct hashes when capacity is not overridden.
    #[arg(long, default_value_t = 20)]
    capacity_headroom_percent: usize,

    /// Worker-partitioned write-issuer sweep. Whole sources remain on one pinned issuer.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    issuer_threads: Vec<usize>,

    #[arg(long, default_value_t = DEFAULT_QUEUE_CAPACITY)]
    queue_capacity: usize,

    #[arg(long, default_value_t = 16)]
    cadence_count: usize,

    /// Soft publication timer. Zero disables the timer.
    #[arg(long, default_value_t = 1)]
    cadence_delay_ms: u64,

    /// Offered event admissions per second. Zero runs unpaced at saturation.
    #[arg(long, default_value_t = 0.0)]
    offered_load: f64,

    #[arg(long, default_value_t = 2_000)]
    warmup_ms: u64,

    #[arg(long, default_value_t = 5_000)]
    duration_ms: u64,

    /// Logical cores kept free for the issuer and runtime control.
    #[arg(long, default_value_t = DEFAULT_RESERVED_CONTROL_CORES)]
    reserved_control_cores: usize,

    #[arg(long, default_value = "dc_ckf_relay_result.json")]
    output_json: PathBuf,
}

#[derive(Debug, Clone)]
struct FrozenEvent {
    source_index: usize,
    dc_ordinal: usize,
    event: RouterEvent,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct EventCounts {
    processed: u64,
    errors: u64,
    capacity_omissions: u64,
}

impl EventCounts {
    fn checked_difference(self, earlier: Self) -> anyhow::Result<Self> {
        Ok(Self {
            processed: self
                .processed
                .checked_sub(earlier.processed)
                .context("processed event counter regressed")?,
            errors: self
                .errors
                .checked_sub(earlier.errors)
                .context("event error counter regressed")?,
            capacity_omissions: self
                .capacity_omissions
                .checked_sub(earlier.capacity_omissions)
                .context("capacity omission counter regressed")?,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct CoreMetrics {
    unknown_removals: u64,
    physical_touches: u64,
}

impl CoreMetrics {
    fn saturating_difference(self, earlier: Self) -> Self {
        Self {
            unknown_removals: self
                .unknown_removals
                .saturating_sub(earlier.unknown_removals),
            physical_touches: self
                .physical_touches
                .saturating_sub(earlier.physical_touches),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct PublicationCounts {
    deltas: u64,
    images: u64,
    resets: u64,
    sequence_errors: u64,
}

impl PublicationCounts {
    fn saturating_difference(self, earlier: Self) -> Self {
        Self {
            deltas: self.deltas.saturating_sub(earlier.deltas),
            images: self.images.saturating_sub(earlier.images),
            resets: self.resets.saturating_sub(earlier.resets),
            sequence_errors: self.sequence_errors.saturating_sub(earlier.sequence_errors),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CounterSnapshot {
    events: EventCounts,
    publication: PublicationCounts,
    core: CoreMetrics,
    publication_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Quantiles {
    operations: u64,
    samples: usize,
    min: u64,
    p50: u64,
    p95: u64,
    p99: u64,
    p999: u64,
    max: u64,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryMeasurement {
    bytes: usize,
    scope: &'static str,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct FinalPhysicalState {
    active_hashes: usize,
    occupied_buckets: usize,
    occupied_slots: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ActorRunResult {
    issuer_threads: usize,
    valid: bool,
    measured_seconds: f64,
    offered_load_events_per_second: f64,
    issued_events: u64,
    admitted_events: u64,
    admission_errors: u64,
    processed_events: u64,
    event_error_events: u64,
    capacity_omissions: u64,
    publication_errors: u64,
    throughput_events_per_second: f64,
    event_admission_latency_ns: Quantiles,
    per_lane_admission_wait_ns: Vec<PhysicalLaneAdmissionWait>,
    publication: PublicationCounts,
    terminal_sequences: Vec<u64>,
    core: CoreMetrics,
    final_physical_state: FinalPhysicalState,
    memory: MemoryMeasurement,
}

#[derive(Debug, Clone, Serialize)]
struct HostTopology {
    logical_cpus: usize,
    physical_cores: Option<usize>,
    reserved_control_cores: usize,
    actor_threads: usize,
    publisher_threads: usize,
    backend_cpus: Vec<usize>,
    reserved_issuer_cpus: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedCapacity {
    dc_ordinal: usize,
    measured_peak_active_distinct_hashes: usize,
    selected_capacity: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RunConfig {
    issuer_thread_sweep: Vec<usize>,
    queue_capacity: usize,
    cadence_count: usize,
    cadence_delay_ms: u64,
    offered_load_events_per_second: f64,
    warmup_ms: u64,
    duration_ms: u64,
    capacity_headroom_percent: usize,
    selected_capacity: Vec<SelectedCapacity>,
    logical_workers_are_multiplexed_sources: bool,
    cycle_close_clears: usize,
    event_latency_contract: &'static str,
    error_contract: &'static str,
    sequence_contract: &'static str,
}

#[derive(Debug, Serialize)]
struct RelayBenchmarkOutput {
    schema_version: u32,
    generated_at_unix_seconds: u64,
    host: HostTopology,
    config: RunConfig,
    corpus: DcCkfCorpusMetadata,
    measured_capacity: DcCkfCapacityMetadata,
    frozen_event_count: usize,
    actor_runs: Vec<ActorRunResult>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PhaseAdmission {
    issued: u64,
    admitted: u64,
    errors: u64,
}

struct PhaseResult {
    admission: PhaseAdmission,
    elapsed: Duration,
    latencies_ns: Vec<u64>,
    lane_waits: BTreeMap<usize, LaneWaitSamples>,
}

#[derive(Debug, Default)]
struct LaneWaitSamples {
    operations: u64,
    waits_ns: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct PhysicalLaneAdmissionWait {
    dc_ordinal: usize,
    wait_ns: Quantiles,
}

#[derive(Debug, Clone, Copy, Default)]
struct ActorCounters {
    events: EventCounts,
    publication_errors: u64,
}

#[derive(Debug, Default)]
struct BlackHoleDeltaSink {
    publication: PublicationCounts,
    observed_sequence: u64,
}

impl DcCkfDeltaSink for BlackHoleDeltaSink {
    type Error = Infallible;

    fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error> {
        let expected = self.observed_sequence.checked_add(1);
        if delta.base_sequence() != self.observed_sequence || Some(delta.sequence()) != expected {
            self.publication.sequence_errors = self.publication.sequence_errors.saturating_add(1);
        }
        self.observed_sequence = delta.sequence();
        self.publication.deltas = self.publication.deltas.saturating_add(1);
        self.publication.images = self
            .publication
            .images
            .saturating_add(delta.images().len() as u64);
        Ok(())
    }
}

type ActorPublisher = DcCkfPublisher<BlackHoleDeltaSink>;

#[derive(Debug, Clone, Copy)]
struct ActorSnapshot {
    counters: ActorCounters,
    publication: PublicationCounts,
    core: CoreMetrics,
    memory_bytes: usize,
    terminal_sequence: u64,
    physical_state: FinalPhysicalState,
}

enum ActorCommand {
    Event(RouterEvent),
    Barrier(std::sync::mpsc::Sender<ActorSnapshot>),
    Shutdown,
}

struct ActorHandle {
    sender: crossbeam_channel::Sender<ActorCommand>,
    worker: Option<JoinHandle<()>>,
}

impl ActorHandle {
    fn start(
        dc_ordinal: usize,
        block_size: u32,
        config: CkfConfig,
        queue_capacity: usize,
        publish_after: Option<Duration>,
    ) -> anyhow::Result<Self> {
        let (sender, receiver) = crossbeam_channel::bounded(queue_capacity);
        let worker = std::thread::Builder::new()
            .name(format!("dc-ckf-actor-{dc_ordinal}"))
            .spawn(move || actor_loop(receiver, dc_ordinal, block_size, config, publish_after))
            .context("failed to spawn DC CKF actor")?;
        Ok(Self {
            sender,
            worker: Some(worker),
        })
    }

    fn snapshot(&self) -> anyhow::Result<ActorSnapshot> {
        let (reply, response) = std::sync::mpsc::channel();
        self.sender
            .send(ActorCommand::Barrier(reply))
            .context("DC CKF actor disconnected before barrier")?;
        response
            .recv_timeout(CONTROL_TIMEOUT)
            .context("DC CKF actor barrier timed out")
    }

    fn shutdown(mut self) -> anyhow::Result<()> {
        self.sender
            .send(ActorCommand::Shutdown)
            .context("DC CKF actor disconnected during shutdown")?;
        self.worker
            .take()
            .expect("actor worker exists until shutdown")
            .join()
            .map_err(|_| anyhow::anyhow!("DC CKF actor panicked"))
    }
}

fn actor_loop(
    receiver: crossbeam_channel::Receiver<ActorCommand>,
    dc_ordinal: usize,
    block_size: u32,
    config: CkfConfig,
    publish_after: Option<Duration>,
) {
    let mut state = DcCkfState::new(config).expect("validated CKF actor configuration");
    let mut semantic_digest = [0; 16];
    semantic_digest[..u32::BITS as usize / 8].copy_from_slice(&block_size.to_le_bytes());
    let domain = IndexerDomainId::new(
        CacheSemanticsId::new(semantic_digest, IdentitySource::Explicit),
        RoutingScopeId::new([1; 16], IdentitySource::Explicit),
    );
    let identity = ProducerIdentity::new(
        PoolId::new(domain, DcId::new(dc_ordinal as u64)),
        1,
        1,
        state.format(),
    );
    let lane = u8::try_from(dc_ordinal).expect("validated DC lane ordinal");
    let lease = LaneLease::new(ConsumerInstanceId::new(1), lane, 1);
    let mut publisher = DcCkfPublisher::new(identity, 0, BlackHoleDeltaSink::default());
    publisher
        .snapshot_after_barrier(&mut state, lease)
        .expect("empty actor state has a valid barrier snapshot");
    let mut counters = ActorCounters::default();
    let mut next_flush: Option<Instant> = None;

    loop {
        let command = match next_flush {
            Some(deadline) => {
                let timeout = deadline.saturating_duration_since(Instant::now());
                match receiver.recv_timeout(timeout) {
                    Ok(command) => Some(command),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        publish_actor_batch(&mut publisher, &mut counters, state.flush());
                        next_flush = None;
                        None
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                }
            }
            None => match receiver.recv() {
                Ok(command) => Some(command),
                Err(_) => return,
            },
        };
        let Some(command) = command else {
            continue;
        };
        match command {
            ActorCommand::Event(event) => {
                let outcome = state.apply_event(event);
                counters.events.processed = counters.events.processed.saturating_add(1);
                if let Some(error) = outcome.first_error() {
                    counters.events.errors = counters.events.errors.saturating_add(1);
                    if matches!(error, KvCacheEventError::CapacityExhausted) {
                        counters.events.capacity_omissions =
                            counters.events.capacity_omissions.saturating_add(1);
                    }
                }
                let boundary = outcome.publication_boundary();
                publish_actor_batch(&mut publisher, &mut counters, outcome.into_publication());
                if boundary {
                    next_flush = None;
                } else if next_flush.is_none() {
                    next_flush = publish_after.and_then(|delay| Instant::now().checked_add(delay));
                }
            }
            ActorCommand::Barrier(reply) => {
                publish_actor_batch(&mut publisher, &mut counters, state.flush());
                next_flush = None;
                let _ = reply.send(actor_snapshot(&state, &publisher, counters));
            }
            ActorCommand::Shutdown => return,
        }
    }
}

fn publish_actor_batch(
    publisher: &mut ActorPublisher,
    counters: &mut ActorCounters,
    batch: Option<DcCkfPublicationBatch>,
) {
    let Some(batch) = batch else {
        return;
    };
    if publisher.publish(batch).is_err() {
        counters.publication_errors = counters.publication_errors.saturating_add(1);
    }
}

fn actor_snapshot(
    state: &DcCkfState,
    publisher: &ActorPublisher,
    mut counters: ActorCounters,
) -> ActorSnapshot {
    let stats = state.stats();
    let aggregation = stats.aggregation();
    let publication = stats.publication();
    let memory = stats.memory();
    counters.events.capacity_omissions = aggregation.capacity_failures();
    let memory_bytes = size_of::<DcCkfState>()
        .saturating_add(size_of::<ActorPublisher>())
        .saturating_add(memory.filter_bytes())
        .saturating_add(memory.dirty_tracking_bytes())
        .saturating_add(
            memory
                .member_set_capacity()
                .saturating_mul(size_of::<ExternalSequenceBlockHash>()),
        )
        .saturating_add(
            memory
                .refcount_capacity()
                .saturating_mul(size_of::<(ExternalSequenceBlockHash, u32)>()),
        )
        .saturating_add(
            memory
                .insertion_scratch_capacity()
                .saturating_mul(size_of::<usize>()),
        );
    ActorSnapshot {
        counters,
        publication: publisher.sink().publication,
        core: CoreMetrics {
            unknown_removals: aggregation.unknown_removals(),
            physical_touches: publication.physical_touches(),
        },
        memory_bytes,
        terminal_sequence: publisher.last_sequence(),
        physical_state: FinalPhysicalState {
            active_hashes: aggregation.unique_block_count(),
            occupied_buckets: aggregation.occupied_bucket_count(),
            occupied_slots: aggregation.occupied_slot_count(),
        },
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let host = discover_host_topology(&args)?;
    pin_current_thread_to_cpus(&host.backend_cpus)?;

    let mut corpus_spec =
        DcCkfCorpusSpec::new(&args.trace_path, args.dc_count, args.logical_workers_per_dc);
    corpus_spec.expected_sha256 = args.trace_sha256.clone();
    corpus_spec.trace_block_size = args.block_size;
    corpus_spec.prefix_depth_factor = args.prefix_depth;
    corpus_spec.trace_duplication_factor = args.trace_duplication_factor;
    let prepared = prepare_dc_ckf_corpus(&corpus_spec)?;
    let block_size = u32::try_from(args.block_size).context("block size exceeds u32")?;
    let artifacts = generate_replay_artifacts(
        &prepared.worker_traces,
        args.replay_gpu_blocks,
        block_size,
        None,
    )
    .await?;
    let measured_capacity = prepared.measure_capacity(&artifacts)?;
    let capacities = select_capacities(&args, &measured_capacity)?;
    let frozen: Arc<[FrozenEvent]> = freeze_events(&prepared.metadata, &artifacts)?.into();
    if frozen.is_empty() {
        bail!("frozen Mooncake corpus contains no KV events");
    }

    let mut actor_runs = Vec::with_capacity(args.issuer_threads.len());
    for &issuer_threads in &args.issuer_threads {
        actor_runs.push(
            run_actor_trial(
                &args,
                Arc::clone(&frozen),
                &capacities,
                issuer_threads,
                &host.reserved_issuer_cpus,
            )
            .await?,
        );
    }
    let output = RelayBenchmarkOutput {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_secs(),
        host,
        config: RunConfig {
            issuer_thread_sweep: args.issuer_threads.clone(),
            queue_capacity: args.queue_capacity,
            cadence_count: args.cadence_count,
            cadence_delay_ms: args.cadence_delay_ms,
            offered_load_events_per_second: args.offered_load,
            warmup_ms: args.warmup_ms,
            duration_ms: args.duration_ms,
            capacity_headroom_percent: args.capacity_headroom_percent,
            selected_capacity: measured_capacity
                .pools
                .iter()
                .zip(&capacities)
                .map(|(pool, &selected_capacity)| SelectedCapacity {
                    dc_ordinal: pool.dc_ordinal,
                    measured_peak_active_distinct_hashes: pool.measured_peak_active_distinct_hashes,
                    selected_capacity,
                })
                .collect(),
            logical_workers_are_multiplexed_sources: true,
            cycle_close_clears: prepared.metadata.sources.len(),
            event_latency_contract: "bounded queue-admission latency; ordinary acknowledgement is not mutation completion",
            error_contract: "event_error_events counts events reporting any first error; capacity_omissions counts omitted block operations and does not fence or stop replay",
            sequence_contract: "the serialized actor-local publisher owns one checked sequence per emitted bucket-image batch",
        },
        corpus: prepared.metadata,
        measured_capacity,
        frozen_event_count: frozen.len(),
        actor_runs,
    };
    std::fs::write(&args.output_json, serde_json::to_vec_pretty(&output)?)
        .with_context(|| format!("failed to write {}", args.output_json.display()))?;
    println!("wrote {}", args.output_json.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.dc_count == 0 || args.dc_count > 16 {
        bail!("dc_count must be in 1..=16");
    }
    if args.logical_workers_per_dc == 0 {
        bail!("logical_workers_per_dc must be positive");
    }
    if args.queue_capacity == 0 {
        bail!("queue_capacity must be positive");
    }
    if args.cadence_count == 0 {
        bail!("cadence_count must be positive");
    }
    if args.duration_ms == 0 {
        bail!("duration_ms must be positive");
    }
    if !args.offered_load.is_finite() || args.offered_load < 0.0 {
        bail!("offered_load must be finite and nonnegative");
    }
    if args.issuer_threads.is_empty() || args.issuer_threads.contains(&0) {
        bail!("issuer_threads must contain positive counts");
    }
    let source_count = args
        .dc_count
        .checked_mul(args.logical_workers_per_dc)
        .context("logical source count overflow")?;
    if args
        .issuer_threads
        .iter()
        .any(|&count| count > source_count)
    {
        bail!("issuer_threads cannot exceed the {source_count} logical sources");
    }
    Ok(())
}

fn discover_host_topology(args: &Args) -> anyhow::Result<HostTopology> {
    let allowed_cpus = allowed_cpu_ids()?;
    let logical_cpus = allowed_cpus.len();
    let actor_threads = args.dc_count;
    let max_issuers = *args
        .issuer_threads
        .iter()
        .max()
        .expect("validated nonempty issuer sweep");
    let required = actor_threads
        .checked_add(args.reserved_control_cores)
        .and_then(|value| value.checked_add(max_issuers))
        .context("required CPU count overflow")?;
    if required > logical_cpus {
        bail!(
            "requested topology oversubscribes the allocation: actor_threads={} + reserved_control_cores={} + issuer_threads={} = {}, logical_cpus={}; use a larger dedicated CPU allocation or reduce dc_count",
            actor_threads,
            args.reserved_control_cores,
            max_issuers,
            required,
            logical_cpus,
        );
    }
    let issuer_start = logical_cpus - max_issuers;
    Ok(HostTopology {
        logical_cpus,
        physical_cores: discover_physical_cores(&allowed_cpus),
        reserved_control_cores: args.reserved_control_cores,
        actor_threads,
        // Sequencing is serialized inside each actor thread; there are no extra publisher threads.
        publisher_threads: 0,
        backend_cpus: allowed_cpus[..issuer_start].to_vec(),
        reserved_issuer_cpus: allowed_cpus[issuer_start..].to_vec(),
    })
}

fn allowed_cpu_ids() -> anyhow::Result<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        let rc = unsafe {
            libc::sched_getaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &mut set as *mut libc::cpu_set_t,
            )
        };
        if rc != 0 {
            bail!("failed to discover the process CPU affinity mask");
        }
        let cpus = (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
            .collect::<Vec<_>>();
        if cpus.is_empty() {
            bail!("process CPU affinity mask is empty");
        }
        Ok(cpus)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let count = std::thread::available_parallelism()
            .context("failed to discover available logical CPUs")?
            .get();
        Ok((0..count).collect())
    }
}

fn discover_physical_cores(allowed_cpus: &[usize]) -> Option<usize> {
    if cfg!(target_os = "macos") {
        let output = Command::new("sysctl")
            .args(["-n", "hw.physicalcpu"])
            .output()
            .ok()?;
        return String::from_utf8(output.stdout).ok()?.trim().parse().ok();
    }
    if !cfg!(target_os = "linux") {
        return None;
    }
    let mut cores = BTreeSet::new();
    for cpu in allowed_cpus {
        let topology = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/topology"));
        let package = std::fs::read_to_string(topology.join("physical_package_id")).ok()?;
        let core = std::fs::read_to_string(topology.join("core_id")).ok()?;
        cores.insert((package.trim().to_string(), core.trim().to_string()));
    }
    (!cores.is_empty()).then_some(cores.len())
}

fn select_capacities(args: &Args, measured: &DcCkfCapacityMetadata) -> anyhow::Result<Vec<usize>> {
    measured
        .pools
        .iter()
        .map(|pool| {
            if let Some(capacity) = args.pool_capacity {
                if capacity == 0 {
                    bail!("pool_capacity must be positive");
                }
                return Ok(capacity);
            }
            let peak = pool.measured_peak_active_distinct_hashes;
            let numerator = peak
                .checked_mul(
                    100usize
                        .checked_add(args.capacity_headroom_percent)
                        .context("capacity headroom overflow")?,
                )
                .context("CKF capacity selection overflow")?;
            Ok(numerator.div_ceil(100).max(1))
        })
        .collect()
}

fn freeze_events(
    topology: &DcCkfCorpusMetadata,
    artifacts: &[ReplayWorkerArtifacts],
) -> anyhow::Result<Vec<FrozenEvent>> {
    if topology.sources.len() != artifacts.len() {
        bail!(
            "source/artifact mismatch: sources={}, artifacts={}",
            topology.sources.len(),
            artifacts.len()
        );
    }
    let mut ordered = Vec::new();
    for (source, artifact) in topology.sources.iter().zip(artifacts) {
        for (ordinal, timed) in artifact.kv_events.iter().enumerate() {
            ordered.push((
                timed.timestamp_us,
                source.source_index,
                ordinal,
                FrozenEvent {
                    source_index: source.source_index,
                    dc_ordinal: source.dc_ordinal,
                    event: RouterEvent::with_storage_tier(
                        source.member.worker_id,
                        timed.event.clone(),
                        timed.storage_tier,
                    ),
                },
            ));
        }
    }
    // Same-source FIFO is the contract. Source index only makes unrelated equal timestamps stable.
    ordered.sort_by_key(|(timestamp, source, ordinal, _)| (*timestamp, *source, *ordinal));
    let mut frozen = ordered
        .into_iter()
        .map(|(_, _, _, event)| event)
        .collect::<Vec<_>>();
    // The generated Mooncake event stream ends with live ownership. Close each sticky source with
    // one rank Clear so timed replay can cycle without accumulating state that the source trace
    // never described. This also keeps Clear cost in the authoritative Relay workload.
    frozen.extend(topology.sources.iter().map(|source| FrozenEvent {
        source_index: source.source_index,
        dc_ordinal: source.dc_ordinal,
        event: RouterEvent::new(
            source.member.worker_id,
            KvCacheEvent {
                event_id: u64::MAX,
                data: KvCacheEventData::Cleared,
                dp_rank: source.member.dp_rank,
            },
        ),
    }));
    Ok(frozen)
}

async fn run_actor_trial(
    args: &Args,
    events: Arc<[FrozenEvent]>,
    capacities: &[usize],
    issuer_threads: usize,
    issuer_cpus: &[usize],
) -> anyhow::Result<ActorRunResult> {
    let publish_after = publication_delay(args.cadence_delay_ms);
    let block_size = u32::try_from(args.block_size).context("block size exceeds u32")?;
    let mut actors = capacities
        .iter()
        .enumerate()
        .map(|(dc_ordinal, &capacity)| {
            ActorHandle::start(
                dc_ordinal,
                block_size,
                CkfConfig {
                    expected_blocks_per_dc: capacity,
                    publish_every_n_events: args.cadence_count,
                    ..CkfConfig::default()
                },
                args.queue_capacity,
                publish_after,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let senders: Arc<[crossbeam_channel::Sender<ActorCommand>]> = actors
        .iter()
        .map(|actor| actor.sender.clone())
        .collect::<Vec<_>>()
        .into();
    let _warmup = drive_phase(
        Arc::clone(&events),
        Duration::from_millis(args.warmup_ms),
        args.offered_load,
        false,
        issuer_threads,
        issuer_cpus,
        Arc::clone(&senders),
    )
    .await?;
    let baseline = actors
        .iter()
        .map(ActorHandle::snapshot)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let measured_started = Instant::now();
    let mut measured = drive_phase(
        events,
        Duration::from_millis(args.duration_ms),
        args.offered_load,
        true,
        issuer_threads,
        issuer_cpus,
        senders,
    )
    .await?;
    let final_snapshot = actors
        .iter()
        .map(ActorHandle::snapshot)
        .collect::<anyhow::Result<Vec<_>>>()?;
    measured.elapsed = measured_started.elapsed();

    let before = fold_actor_snapshots(&baseline);
    let after = fold_actor_snapshots(&final_snapshot);
    let event_counts = after.events.checked_difference(before.events)?;
    let publication = after.publication.saturating_difference(before.publication);
    let core = after.core.saturating_difference(before.core);
    let publication_errors = after
        .publication_errors
        .saturating_sub(before.publication_errors);
    let measured_seconds = measured.elapsed.as_secs_f64();
    let valid = measured.admission.errors == 0
        && publication_errors == 0
        && publication.sequence_errors == 0
        && measured.admission.admitted == event_counts.processed;
    let per_lane_admission_wait_ns = measured
        .lane_waits
        .into_iter()
        .map(|(dc_ordinal, samples)| PhysicalLaneAdmissionWait {
            dc_ordinal,
            wait_ns: quantiles(samples.waits_ns, samples.operations),
        })
        .collect();
    let result = ActorRunResult {
        issuer_threads,
        valid,
        measured_seconds,
        offered_load_events_per_second: args.offered_load,
        issued_events: measured.admission.issued,
        admitted_events: measured.admission.admitted,
        admission_errors: measured.admission.errors,
        processed_events: event_counts.processed,
        event_error_events: event_counts.errors,
        capacity_omissions: event_counts.capacity_omissions,
        publication_errors,
        throughput_events_per_second: event_counts.processed as f64 / measured_seconds,
        event_admission_latency_ns: quantiles(measured.latencies_ns, measured.admission.admitted),
        per_lane_admission_wait_ns,
        publication,
        terminal_sequences: final_snapshot
            .iter()
            .map(|snapshot| snapshot.terminal_sequence)
            .collect(),
        core,
        final_physical_state: fold_actor_physical_state(&final_snapshot),
        memory: MemoryMeasurement {
            bytes: final_snapshot
                .iter()
                .map(|snapshot| snapshot.memory_bytes)
                .sum(),
            scope: "estimated actor and publisher total from exposed capacities and packed storage",
        },
    };
    for actor in actors.drain(..) {
        actor.shutdown()?;
    }
    Ok(result)
}

struct IssuerPhaseResult {
    admission: PhaseAdmission,
    latencies_ns: Vec<u64>,
    lane_waits: BTreeMap<usize, LaneWaitSamples>,
}

async fn drive_phase(
    events: Arc<[FrozenEvent]>,
    duration: Duration,
    offered_load: f64,
    record_latency: bool,
    issuer_count: usize,
    issuer_cpus: &[usize],
    senders: Arc<[crossbeam_channel::Sender<ActorCommand>]>,
) -> anyhow::Result<PhaseResult> {
    if duration.is_zero() {
        return Ok(PhaseResult {
            admission: PhaseAdmission::default(),
            elapsed: Duration::ZERO,
            latencies_ns: Vec::new(),
            lane_waits: BTreeMap::new(),
        });
    }
    if issuer_count == 0 || issuer_count > issuer_cpus.len() {
        bail!(
            "issuer count {issuer_count} requires at least that many reserved CPUs; available={}",
            issuer_cpus.len()
        );
    }
    let issuer_cpus = issuer_cpus[..issuer_count].to_vec();
    let shards = partition_events(&events, issuer_count)?;
    tokio::task::spawn_blocking(move || {
        let ready = Arc::new(Barrier::new(issuer_count + 1));
        let start = Arc::new(Barrier::new(issuer_count + 1));
        let started = Arc::new(OnceLock::new());
        let (elapsed, outputs) = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(issuer_count);
            for (issuer, shard) in shards.into_iter().enumerate() {
                let events = Arc::clone(&events);
                let senders = Arc::clone(&senders);
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                let started = Arc::clone(&started);
                let cpu = issuer_cpus[issuer];
                handles.push(scope.spawn(move || -> anyhow::Result<IssuerPhaseResult> {
                    let pin_result = pin_current_thread(Some(cpu))
                        .map_err(|_| anyhow::anyhow!("failed to pin write issuer to CPU {cpu}"));
                    // Every issuer reaches both rendezvous points even if affinity setup fails,
                    // so the coordinator and healthy issuers cannot remain blocked forever.
                    ready.wait();
                    start.wait();
                    pin_result?;
                    let phase_started = *started.get().expect("coordinator publishes start");
                    let deadline = phase_started + duration;
                    let issuer_load = offered_load / issuer_count as f64;
                    let interval =
                        (issuer_load > 0.0).then(|| Duration::from_secs_f64(1.0 / issuer_load));
                    let mut next_issue = phase_started;
                    let mut cursor = 0usize;
                    let mut admission = PhaseAdmission::default();
                    let mut latencies_ns = Vec::new();
                    let mut lane_waits = BTreeMap::<usize, LaneWaitSamples>::new();
                    loop {
                        if let Some(interval) = interval {
                            let now = Instant::now();
                            if now >= deadline {
                                break;
                            }
                            if now < next_issue {
                                std::thread::sleep(next_issue.min(deadline) - now);
                            }
                            if Instant::now() >= deadline {
                                break;
                            }
                            next_issue = next_issue.checked_add(interval).unwrap_or(deadline);
                        } else if admission.issued.is_multiple_of(64) && Instant::now() >= deadline
                        {
                            break;
                        }
                        let frozen = &events[shard[cursor]];
                        cursor += 1;
                        if cursor == shard.len() {
                            cursor = 0;
                        }
                        admission.issued = admission.issued.saturating_add(1);
                        let admission_started = (record_latency
                            && admission.issued.is_multiple_of(64))
                        .then(Instant::now);
                        let lane = lane_waits.entry(frozen.dc_ordinal).or_default();
                        lane.operations = lane.operations.saturating_add(1);
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match senders[frozen.dc_ordinal]
                            .send_timeout(ActorCommand::Event(frozen.event.clone()), remaining)
                        {
                            Ok(()) => admission.admitted = admission.admitted.saturating_add(1),
                            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => break,
                            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                                admission.errors = admission.errors.saturating_add(1);
                                break;
                            }
                        }
                        if let Some(admission_started) = admission_started {
                            let wait = u64::try_from(admission_started.elapsed().as_nanos())
                                .unwrap_or(u64::MAX);
                            latencies_ns.push(wait);
                            lane.waits_ns.push(wait);
                        }
                    }
                    Ok(IssuerPhaseResult {
                        admission,
                        latencies_ns,
                        lane_waits,
                    })
                }));
            }
            ready.wait();
            let phase_started = Instant::now();
            started.set(phase_started).expect("start is assigned once");
            start.wait();
            let outputs = handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("write issuer panicked"))?
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok::<_, anyhow::Error>((phase_started.elapsed(), outputs))
        })?;
        let mut phase = PhaseResult {
            admission: PhaseAdmission::default(),
            elapsed,
            latencies_ns: Vec::new(),
            lane_waits: BTreeMap::new(),
        };
        for output in outputs {
            phase.admission.issued = phase
                .admission
                .issued
                .saturating_add(output.admission.issued);
            phase.admission.admitted = phase
                .admission
                .admitted
                .saturating_add(output.admission.admitted);
            phase.admission.errors = phase
                .admission
                .errors
                .saturating_add(output.admission.errors);
            phase.latencies_ns.extend(output.latencies_ns);
            for (dc, mut samples) in output.lane_waits {
                let aggregate = phase.lane_waits.entry(dc).or_default();
                aggregate.operations = aggregate.operations.saturating_add(samples.operations);
                aggregate.waits_ns.append(&mut samples.waits_ns);
            }
        }
        Ok(phase)
    })
    .await
    .context("write-issuer coordinator panicked")?
}

fn partition_events(
    events: &[FrozenEvent],
    issuer_count: usize,
) -> anyhow::Result<Vec<Vec<usize>>> {
    let source_count = events
        .iter()
        .map(|event| event.source_index)
        .max()
        .map_or(0, |source| source + 1);
    if issuer_count == 0 || issuer_count > source_count {
        bail!("issuer count {issuer_count} is outside 1..={source_count}");
    }
    let mut shards = vec![Vec::new(); issuer_count];
    for (index, event) in events.iter().enumerate() {
        let issuer = contiguous_worker_issuer(event.source_index, source_count, issuer_count);
        shards[issuer].push(index);
    }
    if let Some(empty) = shards.iter().position(Vec::is_empty) {
        bail!("issuer {empty} owns no logical source events");
    }
    Ok(shards)
}

fn publication_delay(delay_ms: u64) -> Option<Duration> {
    (delay_ms != 0).then(|| Duration::from_millis(delay_ms))
}

fn fold_actor_snapshots(snapshots: &[ActorSnapshot]) -> CounterSnapshot {
    snapshots
        .iter()
        .fold(CounterSnapshot::default(), |mut total, snapshot| {
            total.events.processed = total
                .events
                .processed
                .saturating_add(snapshot.counters.events.processed);
            total.events.errors = total
                .events
                .errors
                .saturating_add(snapshot.counters.events.errors);
            total.events.capacity_omissions = total
                .events
                .capacity_omissions
                .saturating_add(snapshot.counters.events.capacity_omissions);
            total.publication.deltas = total
                .publication
                .deltas
                .saturating_add(snapshot.publication.deltas);
            total.publication.images = total
                .publication
                .images
                .saturating_add(snapshot.publication.images);
            total.publication.resets = total
                .publication
                .resets
                .saturating_add(snapshot.publication.resets);
            total.publication.sequence_errors = total
                .publication
                .sequence_errors
                .saturating_add(snapshot.publication.sequence_errors);
            total.core.physical_touches = total
                .core
                .physical_touches
                .saturating_add(snapshot.core.physical_touches);
            total.core.unknown_removals = total
                .core
                .unknown_removals
                .saturating_add(snapshot.core.unknown_removals);
            total.publication_errors = total
                .publication_errors
                .saturating_add(snapshot.counters.publication_errors);
            total
        })
}

fn fold_actor_physical_state(snapshots: &[ActorSnapshot]) -> FinalPhysicalState {
    snapshots
        .iter()
        .fold(FinalPhysicalState::default(), |mut total, snapshot| {
            total.active_hashes = total
                .active_hashes
                .saturating_add(snapshot.physical_state.active_hashes);
            total.occupied_buckets = total
                .occupied_buckets
                .saturating_add(snapshot.physical_state.occupied_buckets);
            total.occupied_slots = total
                .occupied_slots
                .saturating_add(snapshot.physical_state.occupied_slots);
            total
        })
}

fn quantiles(mut values: Vec<u64>, operations: u64) -> Quantiles {
    if values.is_empty() {
        return Quantiles {
            operations,
            samples: 0,
            min: 0,
            p50: 0,
            p95: 0,
            p99: 0,
            p999: 0,
            max: 0,
        };
    }
    values.sort_unstable();
    Quantiles {
        operations,
        samples: values.len(),
        min: values[0],
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        p99: percentile(&values, 0.99),
        p999: percentile(&values, 0.999),
        max: *values.last().expect("nonempty latency vector"),
    }
}

fn percentile(values: &[u64], percentile: f64) -> u64 {
    let rank = ((values.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[rank]
}
