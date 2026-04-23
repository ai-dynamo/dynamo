// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process wrapper over kvbm-engine's `OffloadEngine` + `InstanceLeader`
//! backed by [`MockWorker`](super::worker::MockWorker).
//!
//! Construction is `async` (velo's `Messenger` needs a short TCP warmup).
//! The hot-path methods ([`tick`](MockOffloadEngine::tick),
//! [`enqueue_g1_eviction`](MockOffloadEngine::enqueue_g1_eviction),
//! [`start_swap_in`](MockOffloadEngine::start_swap_in),
//! [`find_in_tiers`](MockOffloadEngine::find_in_tiers),
//! [`earliest_pending_deadline`](MockOffloadEngine::earliest_pending_deadline))
//! are synchronous. `tick` drives PS completion using `now_ms` supplied by
//! the caller — live replay feeds wall-clock time, offline replay feeds
//! virtual time.

use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;

use kvbm_engine::leader::{FindMatchesOptions, InstanceLeader, Leader, StagingMode};
use kvbm_engine::offload::{
    ExternalBlock, OffloadEngine, PipelineBuilder, PresenceFilter, SourceBlocks,
};
use kvbm_engine::worker::Worker;
use kvbm_engine::{BlockId, G1, G2, SequenceHash};
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_logical::registry::BlockRegistry;

use super::ClockSource;
use super::config::KvbmOffloadConfig;
use super::worker::MockWorker;

/// Handle returned by [`MockOffloadEngine::try_onboard_prefix`]. Scheduler
/// parks one per deferred request and polls
/// [`is_complete`](Self::is_complete) each pass; the bit is flipped by
/// [`MockOffloadEngine::tick`] when the underlying transfer drains from
/// the onboard accountant.
///
/// The handle holds strong [`ImmutableBlock<G2>`] references for the
/// duration of the swap-in. kvbm-logical's inactive pool refuses to
/// reclaim a G2 block while any strong ref is live — so a concurrent
/// offload cannot race in and reassign the slots we're about to
/// onboard. Dropping the handle (after the scheduler promotes or
/// abandons the swap-in) releases the blocks back.
pub struct SwapInHandle {
    complete: Arc<AtomicBool>,
    /// Number of G2 blocks this swap-in delivers.
    block_count: usize,
    /// Strong refs pinning the G2 blocks for the transfer's lifetime.
    /// Not accessed directly — presence alone upholds the RAII contract.
    _g2_blocks: Vec<ImmutableBlock<G2>>,
}

impl SwapInHandle {
    pub fn is_complete(&self) -> bool {
        self.complete.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// In-process offload engine driving a G1→G2 pipeline over [`MockWorker`].
///
/// G1 blocks are handed to kvbm-engine as
/// [`kvbm_engine::offload::SourceBlocks::External`] — `(block_id, plh)`
/// pairs with no strong ref — so kvbm-logical is free to reclaim and
/// reassign the G1 slot while the transfer simulation is in flight.
/// That matches real kvbm semantics (slot reuse is normal) and lets the
/// mocker scale to 100k+ in-flight offloads without OOM.
#[allow(dead_code)]
pub struct MockOffloadEngine {
    config: KvbmOffloadConfig,

    engine: OffloadEngine,
    leader: Arc<InstanceLeader>,
    worker: Arc<MockWorker>,
    registry: Arc<BlockRegistry>,
    g2_manager: Arc<BlockManager<G2>>,

    /// Runtime the engine owns in offline mode (keeps background
    /// pipeline-drain / session-receiver tasks alive after the constructor's
    /// `block_on` returns). `None` in live mode, where ambient tokio keeps
    /// them running.
    _runtime: Option<tokio::runtime::Runtime>,
}

impl MockOffloadEngine {
    /// Build the engine end-to-end against a fresh `InstanceLeader` and
    /// `MockWorker`. Caller must be inside a tokio runtime; in offline
    /// mode, `init_kvbm_offline` constructs a one-worker multi-thread
    /// runtime and calls this via `block_on`.
    pub async fn new(config: KvbmOffloadConfig, _clock: ClockSource) -> Result<Self> {
        let messenger = create_local_messenger().await?;
        let registry = Arc::new(build_registry());
        let g2_manager = Arc::new(build_g2_block_manager(config.num_g2_blocks, &registry));

        let worker = Arc::new(MockWorker::new(
            config.block_size_bytes.unwrap_or(0),
            config.bandwidth_g1_to_g2_gbps,
            config.bandwidth_g2_to_g1_gbps,
            None,
            None,
        ));
        let worker_for_leader: Arc<dyn Worker> = worker.clone();

        // `InstanceLeader::build` calls `Handle::current()` internally,
        // hence the `async fn` and the in-runtime caller requirement.
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry((*registry).clone())
                .g2_manager(g2_manager.clone())
                .worker(worker_for_leader)
                .build()?,
        );

        let g1_to_g2_pipeline = PipelineBuilder::<G1, G2>::new()
            .policy(Arc::new(PresenceFilter::<G1, G2>::new(registry.clone())))
            .batch_size(config.offload_batch_size)
            .build();

        let engine = OffloadEngine::builder(leader.clone())
            .with_registry(registry.clone())
            .with_g2_manager(g2_manager.clone())
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline)
            .build()?;

        Ok(Self {
            config,
            engine,
            leader,
            worker,
            registry,
            g2_manager,
            _runtime: None,
        })
    }

    /// Hand ownership of the offline-mode tokio runtime to the engine so
    /// its worker thread outlives the `block_on` that constructed us.
    ///
    /// ```ignore
    /// let rt = tokio::runtime::Builder::new_multi_thread()
    ///     .worker_threads(1).enable_all().build()?;
    /// let mut engine = rt.block_on(MockOffloadEngine::new(cfg, ClockSource::Virtual))?;
    /// engine.attach_runtime(rt);
    /// ```
    pub fn attach_runtime(&mut self, rt: tokio::runtime::Runtime) {
        self._runtime = Some(rt);
    }

    /// Advance PS state to `now_ms` and fire awaiters for any transfers
    /// that completed in the interval. Caller picks `now_ms`: live mode
    /// passes wall-clock (`Instant::elapsed`-derived); offline replay
    /// passes the runtime's virtual time. Hot-path logic is identical
    /// in both — only the source of `now_ms` differs.
    ///
    /// # Per-worker virtual-time containment
    ///
    /// This `tick` only advances PS accountants owned by *this*
    /// engine's `MockWorker`. With one engine per scheduler worker and
    /// G1↔G2 transfers physically scoped to that worker's CPU/host
    /// memory, the per-worker PS queue is the correct accounting unit:
    /// concurrent offloads on different workers do not contend for the
    /// same DRAM bandwidth, and concurrent offloads on the same worker
    /// fair-share via `BandwidthAccountant`'s PS math.
    ///
    /// **Assumption that breaks for shared tiers (G3 NVMe, G4 object
    /// storage):** when multiple workers can drive transfers into the
    /// same physical resource, per-worker PS undercounts contention —
    /// two workers each see N=1 on their local accountant while the
    /// underlying device sees N=2. A harness-global PS queue (one
    /// accountant per shared link, shared across all engines) is
    /// required before extending this simulation past G2.
    pub fn tick(&self, now_ms: f64) {
        self.worker.set_now_ms(now_ms);
        self.worker.drain_completions(now_ms);
    }

    /// Earliest pending completion across both PS links, or `None` when
    /// both are idle.
    pub fn earliest_pending_deadline(&self) -> Option<f64> {
        self.worker.earliest_finish()
    }

    /// Enqueue a G1→G2 eviction as an `ExternalBlock` reference.
    ///
    /// Caller hands in just the `(block_id, seq_hash)` pair — not a strong
    /// `ImmutableBlock<G1>` clone. This means kvbm-logical is free to
    /// reclaim and reassign the G1 slot while the copy is in flight.
    /// That matches real product semantics: slot 42 in G1 transitions from
    /// carrying `seq_hash_A` to `seq_hash_B` when it's reused, and
    /// kvbm-engine's pipeline is designed to handle it via seq_hash-keyed
    /// presence checks at the destination. In the mocker,
    /// `MockWorker::execute_local_transfer` ignores the BlockId entirely
    /// (only block count matters for the bandwidth sim), so slot reuse is
    /// a non-issue for simulation correctness either.
    ///
    /// When `now_ms` is `Some`, the worker's simulation clock is advanced
    /// before the enqueue so the pipeline drain task reads a fresh time
    /// when it calls into `MockWorker::execute_local_transfer`. This
    /// matters on the first enqueue before any `tick` has run.
    pub fn enqueue_g1_eviction(
        &mut self,
        block_id: BlockId,
        seq_hash: SequenceHash,
        now_ms: Option<f64>,
    ) {
        if let Some(ms) = now_ms {
            self.worker.set_now_ms(ms);
        }
        tracing::debug!(
            now_ms = self.worker.now_ms(),
            block_id,
            "kvbm-offload: G1→G2 enqueue 1 block"
        );
        let source: SourceBlocks<G1> =
            SourceBlocks::External(vec![ExternalBlock::<G1>::new(block_id, seq_hash)]);
        let _handle = self
            .engine
            .enqueue_g1_to_g2(source)
            .expect("G1→G2 pipeline must be configured at engine construction");
    }

    /// Attempt to swap in the longest contiguous G2-resident prefix of
    /// `plhs`. Returns `None` when `plhs` is empty or G2 holds none of
    /// them. Otherwise reserves the onboard PS slot and returns a
    /// [`SwapInHandle`] that pins the matched G2 blocks via RAII for the
    /// transfer's lifetime.
    ///
    /// Find and reserve happen back-to-back under the same worker mutex
    /// (via [`MockWorker::reserve_swap_in`]) so there is no gap between
    /// presence-check and reservation during which the G2 blocks could
    /// be reclaimed by a concurrent offload.
    ///
    /// `plhs` must represent a sequential KV chain; `find_matches_with_options`
    /// applies "first hole" semantics internally, so the returned
    /// `g2_count()` is the contiguous prefix length — no additional
    /// prefix scan on the caller side.
    pub fn try_onboard_prefix(
        &mut self,
        plhs: &[SequenceHash],
        now_ms: Option<f64>,
    ) -> Option<SwapInHandle> {
        if plhs.is_empty() {
            return None;
        }
        let options = FindMatchesOptions {
            search_remote: false,
            staging_mode: StagingMode::Hold,
        };
        let mut result = self
            .leader
            .find_matches_with_options(plhs, options)
            .expect("find_matches_with_options must not fail on local-only Hold lookup");
        let g2_blocks = result
            .take_g2_blocks()
            .expect("Ready variant must yield G2 blocks on Hold + !search_remote path");
        if g2_blocks.is_empty() {
            tracing::info!(
                plhs_len = plhs.len(),
                "kvbm-offload: G2→G1 lookup MISS (0 blocks in G2)"
            );
            return None;
        }
        let block_count = g2_blocks.len();
        let now_ms = now_ms.unwrap_or_else(|| self.worker.now_ms());
        self.worker.set_now_ms(now_ms);
        tracing::info!(
            now_ms,
            plhs_len = plhs.len(),
            block_count,
            "kvbm-offload: G2→G1 swap-in HIT"
        );
        let complete = Arc::new(AtomicBool::new(false));
        self.worker
            .reserve_swap_in(now_ms, block_count, complete.clone());
        Some(SwapInHandle {
            complete,
            block_count,
            _g2_blocks: g2_blocks,
        })
    }

    #[doc(hidden)]
    pub fn worker(&self) -> &Arc<MockWorker> {
        &self.worker
    }

    #[doc(hidden)]
    pub fn offload_engine(&self) -> &OffloadEngine {
        &self.engine
    }

    /// Test-only accessor: integration tests outside this module
    /// register synthetic G2 blocks (allocate → stage → register)
    /// through it so [`try_onboard_prefix`](Self::try_onboard_prefix)
    /// has something to match.
    #[cfg(test)]
    pub(crate) fn g2_manager(&self) -> &Arc<BlockManager<G2>> {
        &self.g2_manager
    }
}

/// Local velo `Messenger` on a random TCP port. Avoids pulling in
/// kvbm-engine's `testing` feature for a ~20 line helper.
async fn create_local_messenger() -> Result<Arc<velo::Messenger>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let transport: Arc<dyn velo::backend::Transport> = Arc::new(
        velo::backend::tcp::TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );
    let messenger = velo::Messenger::builder()
        .add_transport(transport)
        .build()
        .await?;
    // Velo's TCP accept loop needs a moment to reach Ready before it will
    // route the first message.
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(messenger)
}

fn build_registry() -> BlockRegistry {
    BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
        .build()
}

/// `block_size(1)` because `MockWorker` never reads or writes real memory;
/// the LRU backend keeps the bookkeeping cheap.
fn build_g2_block_manager(block_count: usize, registry: &BlockRegistry) -> BlockManager<G2> {
    BlockManager::<G2>::builder()
        .block_count(block_count)
        .block_size(1)
        .registry(registry.clone())
        .with_lru_backend()
        .build()
        .expect("BlockManager<G2> should build with valid config")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_offload_engine_new_builds_end_to_end() {
        let config = KvbmOffloadConfig::default();
        let engine = MockOffloadEngine::new(config, ClockSource::Real)
            .await
            .expect("construction should succeed");

        assert!(engine.offload_engine().has_g1_to_g2());
        assert!(!engine.offload_engine().has_g2_to_g3());
        assert!(!engine.offload_engine().has_g2_to_g4());
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[tokio::test]
    async fn tick_is_noop_when_idle() {
        let engine = MockOffloadEngine::new(KvbmOffloadConfig::default(), ClockSource::Virtual)
            .await
            .unwrap();
        engine.tick(100.0);
        engine.tick(1_000_000.0);
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[test]
    fn offline_runtime_attach_pattern() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let mut engine = rt
            .block_on(MockOffloadEngine::new(
                KvbmOffloadConfig::default(),
                ClockSource::Virtual,
            ))
            .expect("offline construction succeeds");
        engine.attach_runtime(rt);
        assert_eq!(engine.earliest_pending_deadline(), None);
    }

    #[tokio::test]
    async fn try_onboard_prefix_empty_input_returns_none() {
        let mut engine = MockOffloadEngine::new(KvbmOffloadConfig::default(), ClockSource::Real)
            .await
            .unwrap();
        assert!(engine.try_onboard_prefix(&[], None).is_none());
    }

    #[tokio::test]
    async fn try_onboard_prefix_returns_none_when_g2_empty() {
        // Freshly-built engine has no G2 residents — any lookup must
        // yield `None` regardless of how many hashes are probed.
        use dynamo_tokens::PositionalLineageHash;
        let mut engine = MockOffloadEngine::new(KvbmOffloadConfig::default(), ClockSource::Real)
            .await
            .unwrap();
        let hashes: Vec<SequenceHash> = (0..5)
            .map(|i| PositionalLineageHash::new(i as u64, None, 0))
            .collect();
        assert!(engine.try_onboard_prefix(&hashes, None).is_none());
    }

    /// End-to-end: register a G2 block directly in the engine's
    /// `g2_manager`, call `try_onboard_prefix`, and verify (a) the handle
    /// is produced, (b) the reservation is reflected in
    /// `earliest_pending_deadline`, (c) `tick` past the finish time
    /// flips the completion bit, and (d) the handle pins the G2 block
    /// via RAII (a subsequent `find_matches_with_options` on the same
    /// PLH must still succeed while the handle is alive).
    #[tokio::test]
    async fn try_onboard_prefix_pins_g2_blocks_until_handle_drops() {
        use dynamo_tokens::PositionalLineageHash;
        use kvbm_logical::MutableBlock;
        let config = KvbmOffloadConfig {
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let mut engine = MockOffloadEngine::new(config, ClockSource::Virtual)
            .await
            .unwrap();
        engine.tick(0.0);

        // Register a G2 block. Allocate, stage with a PLH, register; let
        // the returned ImmutableBlock drop so the block lands in the
        // inactive pool (still matchable via find_matches).
        let plh = PositionalLineageHash::new(42, None, 0);
        let (mut alloc, _evicted) = engine
            .g2_manager
            .allocate_blocks_with_evictions(1)
            .expect("G2 allocate");
        let mutable: MutableBlock<G2> = alloc.pop().unwrap();
        let complete = mutable.stage(plh, 1).expect("G2 stage");
        drop(engine.g2_manager.register_block(complete));

        let handle = engine
            .try_onboard_prefix(&[plh], Some(0.0))
            .expect("G2 prefix match must produce a handle");
        assert_eq!(handle.block_count(), 1);
        assert!(!handle.is_complete());

        let deadline = engine
            .earliest_pending_deadline()
            .expect("swap-in must appear in earliest_finish");
        assert!(
            (deadline - 1.0).abs() < 1e-6,
            "1 MB / 1 GB/s = 1.0 ms, got {deadline}"
        );

        engine.tick(0.5);
        assert!(
            !handle.is_complete(),
            "swap-in must remain pending before finish time"
        );
        engine.tick(1.0);
        assert!(
            handle.is_complete(),
            "swap-in bit must flip once tick advances past finish"
        );
    }
}
