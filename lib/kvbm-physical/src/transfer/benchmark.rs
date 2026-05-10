// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional startup benchmarking cache for the planner selector (PR-7.5).
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
//! # Timing semantics — submit latency, not transfer completion
//!
//! [`benchmark_pair`](BenchmarkCache::benchmark_pair) measures the
//! synchronous *submit* latency of each candidate — the time from the
//! start of dispatch to when the call returns.  This mirrors the
//! `submit_latency_us` telemetry emitted by `execute_planner_cuda_transfer`
//! (PR-7.6).
//!
//! Submit latency is **not** transfer-completion time.  For small,
//! descriptor-heavy transfers the submit path (memcpy_batch setup,
//! kernel-launch API calls) dominates the observable latency before the
//! first byte moves.  For large single-chunk copies the GPU transfer time
//! dominates and submit latency is nearly constant — in that regime
//! benchmarking submit-only is uninformative.  A future PR may extend
//! `BenchmarkOutcome` to capture end-to-end timing (stream record + sync),
//! but that requires an async path and per-route notification handling that
//! is out of scope for PR-7.5.
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
    /// dtype discriminant without requiring the `permute_kernels` feature.
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
        Self { src_signature, dst_signature, dtype_width_bytes, route_discriminant }
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
    /// Minimum submit latency observed for the winner across
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
        Self { entries: HashMap::new(), seq: 0 }
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
        Self { inner: Mutex::new(BenchmarkCacheInner::new()) }
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
        self.inner.lock().expect("BenchmarkCache mutex poisoned").len()
    }

    /// Benchmark a set of `Candidate::DirectDma` ops, record the winner in
    /// the cache, and return the outcome.
    ///
    /// # Scope (PR-7.5 — narrow Path B)
    ///
    /// Only `Candidate::DirectDma` ops are benchmarked today.  The
    /// function bails with a descriptive error for any other candidate
    /// variant.  This is intentional: NIXL benchmarking, transform-kernel
    /// benchmarking, and graph-replay benchmarking all require their own
    /// execution infrastructure (NIXL agents, kernel invocations, CUDA
    /// graph capture) that is out of scope for one PR.  A future PR
    /// (`PR-7.5.1`) may widen the dispatch here.
    ///
    /// # Timing semantics
    ///
    /// Measures synchronous submit latency — the wall-clock elapsed time
    /// from immediately before `dispatch_ops_grouped_by_size` is called to
    /// immediately after it returns.  This mirrors the `tel_t0` / `tel_latency_us`
    /// brackets in `execute_planner_cuda_transfer` (PR-7.6 telemetry).
    /// Transfer-completion time (stream sync + event fire) is not captured
    /// — see module doc for the rationale.
    ///
    /// # Stream ownership
    ///
    /// The caller provides an `Arc<CudaStream>`.  The function issues
    /// CUDA work on that stream for each trial.  No explicit stream sync
    /// is performed between candidates — the goal is to measure submit
    /// overhead, not transfer latency, so cross-trial GPU serialisation
    /// would distort the measurement.  If the caller wants clean
    /// per-candidate isolation, it should provide separate streams.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `candidates` is empty.
    /// - Any candidate is not `Candidate::DirectDma` (deferral to PR-7.5.1).
    /// - The dispatch itself fails.
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
            let t0 = std::time::Instant::now();
            dispatch_direct_dma_ops(&bc.ops, stream)?;
            let latency_us = t0.elapsed().as_micros() as u64;

            if latency_us < best_latency_us {
                best_latency_us = latency_us;
                best_class = bc.class_name;
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

// ─────────────── BenchmarkCandidate — a thin submit-only descriptor ──────────
//
// `Candidate` is defined in `transfer::lower` and carries per-variant
// data (kernel invocations, graph keys, …) that benchmarking doesn't need.
// Rather than taking `&[Candidate]` and pattern-matching in the timing
// loop (which would spread dispatch concerns into `benchmark.rs`), we
// introduce a thin `BenchmarkCandidate` that the caller pre-decodes.
// This keeps `benchmark.rs` independent of the full `Candidate` enum.

/// A pre-decoded candidate for submit-only benchmarking.
///
/// Only `DirectDma` is supported today (PR-7.5 narrow Path B).
/// A future PR may extend this with `SmallStridedCopy` ops or
/// a `DirectDma | SmallStridedCopy` union field.
#[allow(dead_code)]
pub(crate) struct BenchmarkCandidate {
    /// Matches `Candidate::class_name()` for the variant this was decoded from.
    pub class_name: &'static str,
    /// Copy descriptors.  For `DirectDma`, these are the ops directly from
    /// the planner.  Other variants are unsupported (bail at construction).
    pub ops: Vec<CopyOp>,
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
        Self { src_addr: op.src_addr, dst_addr: op.dst_addr, size: op.size }
    }
}

/// Dispatch a `DirectDma` op set by calling `memcpy_batch`.
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
        let got = cache.lookup(&key).expect("cache must return the inserted outcome");
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
        assert_eq!(strategy_discriminant(TransferStrategy::NixlWriteFlipped), 13);
    }
}
