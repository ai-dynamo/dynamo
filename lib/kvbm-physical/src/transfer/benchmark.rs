// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional startup benchmarking cache for the planner selector (PR-7.5 / PR-7.5.1).
//!
//! [`BenchmarkCache`] maps [`BenchmarkKey`] (layout-pair shape) to a
//! [`BenchmarkOutcome`] that records which `Candidate` variant won when
//! benchmarked on this hardware.  The scorer consults the cache when
//! `TransferCapabilities::startup_benchmark` is enabled; on a cache miss
//! it falls back to the baseline [`score_candidate`] constants unchanged.
//!
//! # Correctness invariant
//!
//! The benchmark result only influences *selection* — it never changes
//! which code path is actually dispatched.  The winning `class_name` is a
//! `&'static str` that matches one of the existing `Candidate` variants;
//! dispatch always resolves through the real planner machinery.  A buggy
//! benchmark outcome cannot corrupt data.
//!
//! # Timing semantics
//!
//! [`BenchmarkCache::benchmark_pair`] measures end-to-end transfer time
//! per candidate:
//!
//! - **`DirectDma`**: `Instant::now()` → `dispatch_direct_dma_ops` →
//!   `stream.synchronize()` → elapsed. Both submit and DMA transfer time
//!   are included.
//! - **`TransformKernel`**: same pattern with `dispatch_transform_kernel`
//!   then `stream.synchronize()`. Includes pointer-table H2D upload and
//!   kernel execution.
//! - **`NixlDirectDma`**: `Instant::now()` → `create_xfer_req` +
//!   `post_xfer_req` → tight sync poll on `get_xfer_status()` until
//!   completion → elapsed. End-to-end including network transfer.
//!
//! All three routes are now end-to-end (not submit-only). The timings are
//! comparable within a candidate class; cross-class comparisons are valid
//! for candidate selection since the scorer only uses the winner label.
//!
//! # Eviction
//!
//! FIFO eviction with a hard cap of [`BENCHMARK_CACHE_CAP`] entries (256).
//! The access pattern for KV-cache startup benchmarking (one benchmark run
//! per layout-pair encountered at startup) makes recency tracking wasteful;
//! FIFO eviction is simpler and equivalent in practice.
//!
//! # Thread safety
//!
//! All methods acquire `Mutex<BenchmarkCacheInner>` for the duration of
//! their operation.  The benchmarking loop in
//! [`BenchmarkCache::benchmark_pair`] releases the lock before dispatching
//! (so only lookup/insert/evict hold it, not the timing-sensitive loop).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::{Result, bail};
use cudarc::driver::CudaStream;
use kvbm_common::LayoutSignature;

use crate::transfer::strategy::TransferStrategy;

/// Maximum number of benchmark outcomes retained in the cache.
///
/// 256 entries easily cover all layout-pair/dtype/route combinations seen
/// at a single node without significant memory pressure.
#[allow(dead_code)]
pub const BENCHMARK_CACHE_CAP: usize = 256;

/// Cache key for benchmark outcomes.
///
/// Keyed on layout *shape* (src + dst `LayoutSignature`), element dtype
/// (as byte width, feature-agnostic), and transfer route family.  Two
/// transfers with the same key describe structurally identical copy
/// operations on this hardware and are expected to produce the same
/// candidate ranking.
///
/// Using full [`LayoutSignature`]s (vs the compact `(descriptor_count,
/// total_bytes)` used by [`GraphCacheKey`]) gives more precise
/// discrimination: two layout pairs can share descriptor/byte counts
/// but differ in stride structure (e.g. page-size-16 vs page-size-32 with
/// half the blocks), producing different per-descriptor submit costs.
/// Benchmark outcomes are startup-time artefacts (not hot-path values),
/// so the allocation cost of cloning the signatures is acceptable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BenchmarkKey {
    /// Labelled-axis signature of the source layout.
    pub src_signature: LayoutSignature,
    /// Labelled-axis signature of the destination layout.
    pub dst_signature: LayoutSignature,
    /// Element size in bytes (`cfg.dtype_width_bytes`), used as a
    /// dtype discriminant.
    pub dtype_width_bytes: Option<u32>,
    /// Transfer route encoded as `TransferStrategy`'s discriminant
    /// integer so the type stays hashable without a custom impl.
    ///
    /// Mapping:
    ///   0 = CudaAsyncH2D, 1 = CudaAsyncD2H, 2 = CudaAsyncD2D,
    ///   10 = NixlRead, 11 = NixlWrite, 12 = NixlReadFlipped,
    ///   13 = NixlWriteFlipped, 255 = Other.
    pub route_discriminant: u8,
}

impl BenchmarkKey {
    /// Build a key from layout signatures + dtype + strategy.
    pub fn new(
        src_signature: LayoutSignature,
        dst_signature: LayoutSignature,
        dtype_width_bytes: Option<u32>,
        strategy: TransferStrategy,
    ) -> Self {
        let route_discriminant = strategy_discriminant(strategy);
        Self {
            src_signature,
            dst_signature,
            dtype_width_bytes,
            route_discriminant,
        }
    }
}

/// Encode `TransferStrategy` as a `u8` discriminant for use in `BenchmarkKey`.
///
/// Values are stable across PR revisions — new variants should use new
/// numbers, never reassign existing ones.
fn strategy_discriminant(s: TransferStrategy) -> u8 {
    match s {
        TransferStrategy::CudaAsyncH2D => 0,
        TransferStrategy::CudaAsyncD2H => 1,
        TransferStrategy::CudaAsyncD2D => 2,
        TransferStrategy::NixlRead => 10,
        TransferStrategy::NixlWrite => 11,
        TransferStrategy::NixlReadFlipped => 12,
        TransferStrategy::NixlWriteFlipped => 13,
        _ => 255,
    }
}

/// Result of benchmarking one `(BenchmarkKey, candidates)` pair.
///
/// `winner` is the `Candidate::class_name()` of the fastest candidate
/// observed on this hardware for this key.  The scorer adds
/// [`BENCHMARK_WINNER_BONUS`] to that candidate's base score, pushing it
/// above all peers with the same base score.
///
/// All fields are `Copy`-compatible so `BenchmarkCache::lookup` can return
/// an owned value (no ref from inside the Mutex).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkOutcome {
    /// `Candidate::class_name()` of the empirically fastest candidate.
    pub winner: &'static str,
    /// Minimum end-to-end latency observed for the winner across
    /// `runs_compared` benchmark trials (µs).  Zero if timing was
    /// not captured (scaffolding path).
    pub winner_latency_us: u64,
    /// Number of candidates compared during the benchmarking run.
    pub runs_compared: u8,
    /// Wall-clock time at which this outcome was recorded.
    pub recorded_at: SystemTime,
}

/// Score bonus applied to the `BenchmarkOutcome::winner` candidate.
///
/// +500 over any base score means the cached winner beats all other
/// candidates in the same family (base scores are 950–1100) and beats
/// a non-cached candidate in a higher-score family (e.g. a cached
/// DirectDma at 1500 beats a TransformKernel at 1100).
///
/// Correctness is unaffected: the scorer returns the same winning
/// *variant* but the dispatch machinery is identical regardless of which
/// variant was picked.
pub const BENCHMARK_WINNER_BONUS: i64 = 500;

// ─────────────────────────── Cache internals ─────────────────────────────────

struct BenchmarkCacheInner {
    entries: HashMap<BenchmarkKey, (u64, BenchmarkOutcome)>,
    #[allow(dead_code)]
    seq: u64,
}

impl BenchmarkCacheInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            seq: 0,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Evict the entry with the smallest insertion-order sequence number.
    #[allow(dead_code)]
    fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, (seq, _))| *seq)
            .map(|(k, _)| k.clone())
            .expect("non-empty map must have a min");
        self.entries.remove(&oldest_key);
    }
}

/// Thread-safe benchmark outcome cache.
///
/// Bounded at [`BENCHMARK_CACHE_CAP`] entries with FIFO eviction.  Lives
/// on [`TransferContext`] as an `Arc<BenchmarkCache>` (same pattern as
/// [`GraphCache`]) so it is shared across all clones of the context and
/// dropped with the last clone.
pub(crate) struct BenchmarkCache {
    inner: Mutex<BenchmarkCacheInner>,
}

impl BenchmarkCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(BenchmarkCacheInner::new()),
        }
    }

    /// Look up an outcome for `key`.
    ///
    /// Returns a cloned `BenchmarkOutcome` (not a reference) so the caller
    /// never needs to hold the Mutex across scoring.  Returns `None` on a
    /// cache miss.
    pub(crate) fn lookup(&self, key: &BenchmarkKey) -> Option<BenchmarkOutcome> {
        let guard = self.inner.lock().expect("BenchmarkCache mutex poisoned");
        guard.entries.get(key).map(|(_, outcome)| *outcome)
    }

    /// Insert an outcome.  Evicts the oldest entry if the cache is full.
    ///
    /// If an entry for `key` already exists it is replaced.
    #[allow(dead_code)]
    pub(crate) fn insert(&self, key: BenchmarkKey, outcome: BenchmarkOutcome) {
        let mut guard = self.inner.lock().expect("BenchmarkCache mutex poisoned");
        if guard.len() >= BENCHMARK_CACHE_CAP && !guard.entries.contains_key(&key) {
            guard.evict_oldest();
        }
        let seq = guard.seq;
        guard.seq += 1;
        guard.entries.insert(key, (seq, outcome));
    }

    /// Number of currently cached entries (for tests / telemetry).
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("BenchmarkCache mutex poisoned")
            .len()
    }

    /// Benchmark a set of candidates, record the winner in the cache, and
    /// return the outcome.
    ///
    /// # Supported variants (PR-7.5.1)
    ///
    /// - [`BenchmarkCandidate::DirectDma`]: measures end-to-end via
    ///   `memcpy_batch` + `stream.synchronize()`.
    /// - [`BenchmarkCandidate::TransformKernel`][]: `dispatch_transform_kernel`
    ///   + `stream.synchronize()`.
    /// - [`BenchmarkCandidate::NixlDirectDma`]: `create_xfer_req` +
    ///   `post_xfer_req` + sync polling on `get_xfer_status()`.
    ///
    /// # Timing semantics
    ///
    /// All routes measure end-to-end transfer time: the wall-clock elapsed
    /// from immediately before dispatch to immediately after the transfer
    /// completes on the device / network. This differs from PR-7.5's
    /// submit-only timing for `DirectDma` — see module doc §Timing semantics.
    ///
    /// # Stream ownership
    ///
    /// For CUDA routes the caller provides an `Arc<CudaStream>`.  Each trial
    /// ends with `stream.synchronize()`, so successive trials are isolated
    /// end-to-end. For NIXL routes the stream is not used.
    ///
    /// # Locality invariants (NIXL)
    ///
    /// - `NixlWrite` requires `src` to be local to the benchmarking agent.
    /// - `NixlRead` / `NixlReadFlipped` require `dst` to be local.
    ///
    /// Violation returns an error before any NIXL API is called.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `candidates` is empty.
    /// - Any dispatch fails.
    /// - Locality invariant is violated for a `NixlDirectDma` candidate.
    #[allow(dead_code)]
    pub(crate) fn benchmark_pair(
        self: &Arc<Self>,
        key: BenchmarkKey,
        candidates: Vec<BenchmarkCandidate>,
        stream: &Arc<CudaStream>,
    ) -> Result<BenchmarkOutcome> {
        if candidates.is_empty() {
            bail!("benchmark_pair: candidates list is empty");
        }

        let runs_compared = candidates.len().min(255) as u8;
        let mut best_class: &'static str = "";
        let mut best_latency_us = u64::MAX;

        for bc in &candidates {
            let (class_name, latency_us) = dispatch_benchmark_candidate(bc, stream)?;
            if latency_us < best_latency_us {
                best_latency_us = latency_us;
                best_class = class_name;
            }
        }

        let outcome = BenchmarkOutcome {
            winner: best_class,
            winner_latency_us: best_latency_us,
            runs_compared,
            recorded_at: SystemTime::now(),
        };
        self.insert(key, outcome);
        Ok(outcome)
    }
}

impl Default for BenchmarkCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────── BenchmarkCandidate — per-variant timing descriptors ─────────────
//
// `Candidate` (in `transfer::lower`) carries per-variant data that the
// benchmarking loop doesn't need. Rather than taking `&[Candidate]` and
// pattern-matching in the timing loop (which would spread dispatch concerns
// into `benchmark.rs`), we use a thin `BenchmarkCandidate` enum that the
// caller pre-decodes from a `Candidate`.
//
// Each variant carries exactly the data needed to dispatch one timed trial.

/// A pre-decoded candidate for end-to-end benchmarking.
///
/// Variants added in PR-7.5.1 extend the original `DirectDma`-only struct
/// (PR-7.5) to cover `TransformKernel` and NIXL direct DMA.
#[allow(dead_code)]
pub(crate) enum BenchmarkCandidate {
    /// CUDA memcpy batch DMA (cudaMemcpyBatchAsync / cudaMemcpyAsync).
    DirectDma {
        /// Copy descriptors from the planner.
        ops: Vec<CopyOp>,
    },

    /// Planner-driven permute kernel (operational ↔ universal, or NHD ↔ HND).
    TransformKernel {
        /// Kernel launch parameters resolved by the catalog.
        invocation: crate::transfer::kernel_catalog::KernelInvocation,
        /// Source physical layout (borrowing is not possible here — the
        /// benchmark loop outlives the planner call site).
        src: Arc<crate::layout::PhysicalLayout>,
        /// Destination physical layout.
        dst: Arc<crate::layout::PhysicalLayout>,
        /// `(src_block_id, dst_block_id)` pairs for this trial.
        block_pairs: Vec<(crate::BlockId, crate::BlockId)>,
    },

    /// NIXL direct DMA (cross-agent `create_xfer_req` + `post_xfer_req`).
    NixlDirectDma {
        /// Copy descriptors (coalesced op list from the planner).
        ops: Vec<CopyOp>,
        /// NIXL agent driving this transfer (the "local" agent).
        nixl_agent: dynamo_memory::nixl::NixlAgent,
        /// Source layout metadata (agent name, mem_type, device_id).
        src_agent_name: String,
        src_mem_type: dynamo_memory::nixl::MemType,
        src_device_id: u64,
        /// Destination layout metadata.
        dst_agent_name: String,
        dst_mem_type: dynamo_memory::nixl::MemType,
        dst_device_id: u64,
        /// Transfer direction (Read = pull, Write = push).
        xfer_op: dynamo_memory::nixl::XferOp,
        /// When `true`, swap src/dst descriptor lists at the NIXL layer
        /// (NixlReadFlipped / NixlWriteFlipped strategies).
        flip_descriptors: bool,
    },
}

impl BenchmarkCandidate {
    /// The `Candidate::class_name()` string for this variant.
    #[allow(dead_code)]
    pub(crate) fn class_name(&self) -> &'static str {
        match self {
            BenchmarkCandidate::DirectDma { .. } => "DirectDma",
            BenchmarkCandidate::TransformKernel { .. } => "TransformKernel",
            BenchmarkCandidate::NixlDirectDma { .. } => "DirectDma",
        }
    }
}

/// Thin copy descriptor re-exported for `BenchmarkCandidate`.
///
/// Mirrors the fields of [`crate::transfer::plan::CopyOp`] so
/// `benchmark.rs` doesn't need to import the whole `plan` module.
#[allow(dead_code)]
pub(crate) struct CopyOp {
    pub src_addr: usize,
    pub dst_addr: usize,
    pub size: usize,
}

impl From<&crate::transfer::plan::CopyOp> for CopyOp {
    fn from(op: &crate::transfer::plan::CopyOp) -> Self {
        Self {
            src_addr: op.src_addr,
            dst_addr: op.dst_addr,
            size: op.size,
        }
    }
}

// ──────────────────── Per-variant timed dispatch ──────────────────────────────

/// Dispatch one `BenchmarkCandidate` trial, returning `(class_name, duration_us)`.
///
/// The returned duration is end-to-end: from just before the dispatch call
/// to after the device/network confirms completion.
fn dispatch_benchmark_candidate(
    bc: &BenchmarkCandidate,
    stream: &Arc<CudaStream>,
) -> Result<(&'static str, u64)> {
    match bc {
        BenchmarkCandidate::DirectDma { ops } => {
            let t0 = std::time::Instant::now();
            dispatch_direct_dma_ops(ops, stream)?;
            stream.synchronize()?;
            Ok(("DirectDma", t0.elapsed().as_micros() as u64))
        }

        BenchmarkCandidate::TransformKernel {
            invocation,
            src,
            dst,
            block_pairs,
        } => {
            // Build the prepared plan OUTSIDE the timing window — the
            // bench measures raw kernel-dispatch + pointer-fill + device
            // sync, not the one-time plan construction (extract_universal
            // _base, scratch-pool allocator). This matches the pre-refactor
            // baseline where `dispatch_transform_kernel` was passed `None`
            // and built its arrays inline at dispatch time.
            let prepared = std::sync::Arc::new(
                crate::transfer::prepared::PreparedTransferPlan::build_transform(
                    *invocation,
                    src,
                    dst,
                )?,
            );
            let t0 = std::time::Instant::now();
            crate::transfer::executor::dispatch_transform_kernel(
                invocation,
                src,
                dst,
                block_pairs,
                None, // benchmark dispatches full-extent transfers
                stream,
                &prepared,
            )?;
            stream.synchronize()?;
            Ok(("TransformKernel", t0.elapsed().as_micros() as u64))
        }

        BenchmarkCandidate::NixlDirectDma {
            ops,
            nixl_agent,
            src_agent_name,
            src_mem_type,
            src_device_id,
            dst_agent_name,
            dst_mem_type,
            dst_device_id,
            xfer_op,
            flip_descriptors,
        } => dispatch_nixl_dma_ops_timed(
            ops,
            nixl_agent,
            src_agent_name,
            *src_mem_type,
            *src_device_id,
            dst_agent_name,
            *dst_mem_type,
            *dst_device_id,
            *xfer_op,
            *flip_descriptors,
        ),
    }
}

/// Dispatch a `DirectDma` op set by calling `memcpy_batch`.
///
/// Timing is done by the caller (`dispatch_benchmark_candidate`).
///
/// Mirrors `dispatch_ops_grouped_by_size` in `executor::planner` but
/// is intentionally a private copy here so `benchmark.rs` stays
/// self-contained and doesn't pull in the full planner module.  A future
/// refactor may merge them via a shared helper in `executor::memcpy`.
#[allow(dead_code)]
fn dispatch_direct_dma_ops(ops: &[CopyOp], stream: &Arc<CudaStream>) -> Result<()> {
    use kvbm_kernels::MemcpyBatchMode;
    use std::collections::BTreeMap;
    use std::ffi::c_void;

    let stream_raw = stream.cu_stream() as cudarc::runtime::sys::cudaStream_t;

    let mut by_size: BTreeMap<usize, (Vec<*const c_void>, Vec<*mut c_void>)> = BTreeMap::new();
    for op in ops {
        let e = by_size.entry(op.size).or_default();
        e.0.push(op.src_addr as *const c_void);
        e.1.push(op.dst_addr as *mut c_void);
    }

    for (size, (src_ptrs, dst_ptrs)) in by_size {
        if size == 0 {
            continue;
        }
        let status = unsafe {
            kvbm_kernels::memcpy_batch(
                src_ptrs.as_ptr(),
                dst_ptrs.as_ptr(),
                size,
                src_ptrs.len(),
                MemcpyBatchMode::BatchedWithFallback,
                stream_raw,
            )
        };
        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            bail!(
                "benchmark_pair: dispatch_direct_dma_ops failed: size={size}, \
                 num_copies={}, status={status:?}",
                src_ptrs.len()
            );
        }
    }
    Ok(())
}

/// Dispatch a `NixlDirectDma` trial end-to-end and return `(class_name, duration_us)`.
///
/// Builds `XferDescList` pairs, posts the transfer via `create_xfer_req` +
/// `post_xfer_req`, then polls `get_xfer_status()` in a tight sync loop
/// until the transfer completes.  Returns an error if the transfer or status
/// check fails, or if the locality invariant is violated.
///
/// # Locality invariants
///
/// - `NixlWrite` (push): `src` must be local to `nixl_agent`.
/// - `NixlRead` / `NixlReadFlipped` (pull): `dst` must be local to `nixl_agent`.
///
/// These match `execute_planner_nixl_transfer`'s invariants.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn dispatch_nixl_dma_ops_timed(
    ops: &[CopyOp],
    nixl_agent: &dynamo_memory::nixl::NixlAgent,
    src_agent_name: &str,
    src_mem_type: dynamo_memory::nixl::MemType,
    src_device_id: u64,
    dst_agent_name: &str,
    dst_mem_type: dynamo_memory::nixl::MemType,
    dst_device_id: u64,
    xfer_op: dynamo_memory::nixl::XferOp,
    flip_descriptors: bool,
) -> Result<(&'static str, u64)> {
    use dynamo_memory::nixl::{XferDescList, XferOp};

    // Locality check (mirrors execute_planner_nixl_transfer).
    let local_name = nixl_agent.name();
    match xfer_op {
        XferOp::Write => {
            if local_name != src_agent_name {
                bail!(
                    "benchmark_pair NixlDirectDma: Write (push) requires local src; \
                     src_agent={:?}, local_agent={:?}",
                    src_agent_name,
                    local_name,
                );
            }
        }
        XferOp::Read => {
            if local_name != dst_agent_name {
                bail!(
                    "benchmark_pair NixlDirectDma: Read (pull) requires local dst; \
                     dst_agent={:?}, local_agent={:?}",
                    dst_agent_name,
                    local_name,
                );
            }
        }
    }

    // Build descriptor lists.
    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;
    for op in ops {
        src_dl.add_desc(op.src_addr, op.size, src_device_id);
        dst_dl.add_desc(op.dst_addr, op.size, dst_device_id);
    }
    if flip_descriptors {
        std::mem::swap(&mut src_dl, &mut dst_dl);
    }

    let remote_agent = match xfer_op {
        XferOp::Write => dst_agent_name,
        XferOp::Read => src_agent_name,
    };

    // Start timing — includes both create+post and the transfer itself.
    let t0 = std::time::Instant::now();

    let xfer_req = nixl_agent.create_xfer_req(xfer_op, &src_dl, &dst_dl, remote_agent, None)?;
    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    // Sync poll until complete (identical to the tight loop in
    // llm/src/block_manager/block/transfer/nixl.rs).  This is fine at
    // benchmark/startup time — we are not on a tokio worker thread here,
    // and we don't need the notification-channel machinery.
    if still_pending {
        loop {
            match nixl_agent.get_xfer_status(&xfer_req) {
                Ok(status) if status.is_success() => break,
                Ok(_) => std::thread::yield_now(),
                Err(e) => bail!("benchmark_pair NixlDirectDma: get_xfer_status failed: {e}"),
            }
        }
    }

    let latency_us = t0.elapsed().as_micros() as u64;
    Ok(("DirectDma", latency_us))
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use kvbm_common::{AxisExtent, KvDim};

    fn make_sig() -> LayoutSignature {
        LayoutSignature::new(
            vec![
                (KvDim::Block, AxisExtent::full(4)),
                (KvDim::Page, AxisExtent::full(16)),
                (KvDim::HeadSize, AxisExtent::full(128)),
            ],
            vec![16 * 128 * 2, 128 * 2, 2],
            2,
            None,
        )
    }

    fn make_key(strategy: TransferStrategy) -> BenchmarkKey {
        let sig = make_sig();
        BenchmarkKey::new(sig.clone(), sig, Some(2), strategy)
    }

    fn make_outcome(winner: &'static str) -> BenchmarkOutcome {
        BenchmarkOutcome {
            winner,
            winner_latency_us: 42,
            runs_compared: 1,
            recorded_at: SystemTime::now(),
        }
    }

    // ── cache miss ───────────────────────────────────────────────────────────

    #[test]
    fn benchmark_cache_lookup_miss_returns_none() {
        let cache = BenchmarkCache::new();
        let key = make_key(TransferStrategy::CudaAsyncD2D);
        assert!(
            cache.lookup(&key).is_none(),
            "empty cache must return None on lookup"
        );
    }

    // ── insert + lookup ──────────────────────────────────────────────────────

    #[test]
    fn benchmark_cache_insert_then_lookup() {
        let cache = BenchmarkCache::new();
        let key = make_key(TransferStrategy::CudaAsyncD2D);
        let outcome = make_outcome("DirectDma");

        cache.insert(key.clone(), outcome);
        let got = cache
            .lookup(&key)
            .expect("cache must return the inserted outcome");
        assert_eq!(got.winner, "DirectDma");
        assert_eq!(got.winner_latency_us, 42);
        assert_eq!(got.runs_compared, 1);
    }

    // ── eviction ─────────────────────────────────────────────────────────────

    /// Inserting more than `BENCHMARK_CACHE_CAP` entries must not exceed
    /// the cap — FIFO eviction removes the oldest entries.
    #[test]
    fn benchmark_cache_eviction_bounded() {
        let cache = BenchmarkCache::new();

        // Insert CAP + 10 entries with distinct keys (vary the src signature).
        for i in 0..=(BENCHMARK_CACHE_CAP + 9) {
            let src_sig = LayoutSignature::new(
                vec![(KvDim::Block, AxisExtent::full(i + 1))],
                vec![2],
                2,
                None,
            );
            let dst_sig = make_sig();
            let key = BenchmarkKey::new(src_sig, dst_sig, Some(2), TransferStrategy::CudaAsyncD2D);
            cache.insert(key, make_outcome("DirectDma"));
        }

        assert!(
            cache.len() <= BENCHMARK_CACHE_CAP,
            "cache must not exceed BENCHMARK_CACHE_CAP={} entries, got {}",
            BENCHMARK_CACHE_CAP,
            cache.len()
        );
    }

    // ── strategy discriminant ────────────────────────────────────────────────

    /// Route discriminants must be stable across PR revisions.
    #[test]
    fn strategy_discriminants_are_stable() {
        assert_eq!(strategy_discriminant(TransferStrategy::CudaAsyncH2D), 0);
        assert_eq!(strategy_discriminant(TransferStrategy::CudaAsyncD2H), 1);
        assert_eq!(strategy_discriminant(TransferStrategy::CudaAsyncD2D), 2);
        assert_eq!(strategy_discriminant(TransferStrategy::NixlRead), 10);
        assert_eq!(strategy_discriminant(TransferStrategy::NixlWrite), 11);
        assert_eq!(strategy_discriminant(TransferStrategy::NixlReadFlipped), 12);
        assert_eq!(
            strategy_discriminant(TransferStrategy::NixlWriteFlipped),
            13
        );
    }

    // ── BenchmarkCandidate::class_name ────────────────────────────────────────

    /// Each variant must return the expected class_name string that
    /// matches `Candidate::class_name()` in `transfer::lower`.
    #[test]
    fn benchmark_candidate_class_names() {
        let direct = BenchmarkCandidate::DirectDma { ops: vec![] };
        assert_eq!(direct.class_name(), "DirectDma");

        // NixlDirectDma maps to "DirectDma" (same class, different route).
        let nixl = BenchmarkCandidate::NixlDirectDma {
            ops: vec![],
            nixl_agent: dynamo_memory::nixl::NixlAgent::new("bench-test-cls").expect("agent"),
            src_agent_name: "a".to_string(),
            src_mem_type: dynamo_memory::nixl::MemType::Dram,
            src_device_id: 0,
            dst_agent_name: "b".to_string(),
            dst_mem_type: dynamo_memory::nixl::MemType::Dram,
            dst_device_id: 0,
            xfer_op: dynamo_memory::nixl::XferOp::Read,
            flip_descriptors: false,
        };
        assert_eq!(nixl.class_name(), "DirectDma");
    }

    // ── NIXL locality invariant ───────────────────────────────────────────────

    /// A NixlDirectDma Write candidate with a non-local src must bail
    /// with a descriptive error before touching any NIXL API.
    ///
    /// This test is sync (no GPU required) and runs without UCX.
    #[test]
    fn benchmark_nixl_locality_write_requires_local_src() {
        let cache = Arc::new(BenchmarkCache::new());
        let _key = make_key(TransferStrategy::NixlWrite);

        // Fake stream — DirectDma is not dispatched; NIXL route doesn't use it.
        // We can't create a real CudaStream without a GPU here, so we exploit
        // the fact that the NIXL locality check fires before stream use.
        // Construct the candidate explicitly to trigger the check.

        // Agent named "local-agent" is local; src_agent_name is "remote-agent".
        let nixl_agent = dynamo_memory::nixl::NixlAgent::new("local-agent-write-check")
            .expect("NixlAgent::new must succeed");

        let _candidate = BenchmarkCandidate::NixlDirectDma {
            ops: vec![],
            nixl_agent: nixl_agent.clone(),
            src_agent_name: "remote-agent".to_string(),
            src_mem_type: dynamo_memory::nixl::MemType::Dram,
            src_device_id: 0,
            dst_agent_name: "local-agent-write-check".to_string(),
            dst_mem_type: dynamo_memory::nixl::MemType::Dram,
            dst_device_id: 0,
            xfer_op: dynamo_memory::nixl::XferOp::Write,
            flip_descriptors: false,
        };

        // dispatch_nixl_dma_ops_timed is private; exercise via the full
        // benchmark_pair path. We need a stream handle — borrow an existing
        // one is not possible without GPU, but the locality check happens
        // before any stream use. Use a null pointer transmute is risky; instead
        // call dispatch_nixl_dma_ops_timed directly via a helper shim.
        //
        // Direct call to the module-private function via the re-exposed
        // dispatch_benchmark_candidate path — we need a fake stream that
        // will never be dereferenced (the NIXL path doesn't use it).
        // The simplest safe approach: call dispatch_nixl_dma_ops_timed
        // directly since it's in the same module.
        let err = dispatch_nixl_dma_ops_timed(
            &[],
            &nixl_agent,
            "remote-agent", // src_agent_name (not local)
            dynamo_memory::nixl::MemType::Dram,
            0,
            "local-agent-write-check", // dst_agent_name
            dynamo_memory::nixl::MemType::Dram,
            0,
            dynamo_memory::nixl::XferOp::Write,
            false,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("Write (push) requires local src"),
            "expected locality error message, got: {msg}"
        );

        // Cache must remain empty — the error fired before any insert.
        assert_eq!(cache.len(), 0, "cache must be empty after locality error");
    }

    /// A NixlDirectDma Read candidate with a non-local dst must bail
    /// with a descriptive error.
    #[test]
    fn benchmark_nixl_locality_read_requires_local_dst() {
        let nixl_agent = dynamo_memory::nixl::NixlAgent::new("local-agent-read-check")
            .expect("NixlAgent::new must succeed");

        let err = dispatch_nixl_dma_ops_timed(
            &[],
            &nixl_agent,
            "local-agent-read-check", // src_agent_name
            dynamo_memory::nixl::MemType::Dram,
            0,
            "remote-agent", // dst_agent_name (not local)
            dynamo_memory::nixl::MemType::Dram,
            0,
            dynamo_memory::nixl::XferOp::Read,
            false,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("Read (pull) requires local dst"),
            "expected read locality error, got: {msg}"
        );
    }
}
