// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared transfer-plan cache.
//!
//! Two-tier cache keyed by `(src_handle, dst_handle, strategy, axis_slices)`
//! storing transform templates (G1↔G2 + remote operational↔universal via
//! the kernel catalog): the resolved [`KernelInvocation`], universal-side
//! base address and `bytes_per_block`, and a per-plan scratch pool of
//! pointer-array `Vec`s. Per-call work is reduced to filling the leased
//! scratch via [`PointerSink`] — steady-state same-or-smaller block lists
//! reuse capacity, so the hot path allocates only the device-side copy.
//!
//! Same-layout / sliced direct copies are not cached: their projection
//! cost is microseconds against millisecond-scale DMA/RDMA operations,
//! and the plan_copy + coalesce step is unaffected by caching.
//! Downstream NIXL/CUDA paths can still consume the resulting ops
//! through a [`DescSink`] (see [`NixlDescPairSink`], [`CopyOpVecSink`],
//! [`CountingDescSink`]).
//!
//! The cache values are wrapped in [`Arc`] so cache hits cost only a refcount
//! bump — scratch pools are shared by definition across concurrent calls.

use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use dynamo_memory::nixl::XferDescList;
use kvbm_common::{AxisIntersection, KvDim};

use crate::BlockId;
use crate::manager::LayoutHandle;
use crate::transfer::PhysicalLayout;
use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};
use crate::transfer::plan::CopyOp;
use crate::transfer::strategy::TransferStrategy;

// ============================================================================
// Cache keys
// ============================================================================

/// Hashable form of [`AxisIntersection`] for prepared-plan cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AxisIntersectionKey {
    dim: KvDim,
    src_start: usize,
    src_end: usize,
    dst_start: usize,
    dst_end: usize,
}

impl From<&AxisIntersection> for AxisIntersectionKey {
    fn from(value: &AxisIntersection) -> Self {
        Self {
            dim: value.dim,
            src_start: value.src_local.start,
            src_end: value.src_local.end,
            dst_start: value.dst_local.start,
            dst_end: value.dst_local.end,
        }
    }
}

/// Cache key for a reusable prepared transfer template.
///
/// Handle stability invariant: `LayoutHandle` is immutable for the lifetime
/// of a [`TransferManager`] — once issued, a handle always identifies the
/// same physical layout shape. This invariant is what makes the cached
/// `AnnotatedLayout` / `KernelInvocation` / `univ_base` values safe to reuse.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PreparedPlanKey {
    src_handle: LayoutHandle,
    dst_handle: LayoutHandle,
    strategy: StrategyKey,
    axis_slices: Vec<AxisIntersectionKey>,
}

impl PreparedPlanKey {
    pub(crate) fn new(
        src_handle: LayoutHandle,
        dst_handle: LayoutHandle,
        strategy: TransferStrategy,
        axis_slices: &[AxisIntersection],
    ) -> Self {
        Self {
            src_handle,
            dst_handle,
            strategy: StrategyKey::from(strategy),
            axis_slices: axis_slices.iter().map(AxisIntersectionKey::from).collect(),
        }
    }

    fn is_local_for(&self, worker_id: u64) -> bool {
        self.src_handle.worker_id() == worker_id && self.dst_handle.worker_id() == worker_id
    }

    fn approximate_heap_bytes(&self) -> usize {
        size_of::<PreparedPlanKey>() + self.axis_slices.len() * size_of::<AxisIntersectionKey>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StrategyKey {
    Memcpy,
    CudaH2D,
    CudaD2H,
    CudaD2D,
    NixlRead,
    NixlWrite,
    NixlReadFlipped,
    NixlWriteFlipped,
    Invalid,
}

impl From<TransferStrategy> for StrategyKey {
    fn from(value: TransferStrategy) -> Self {
        match value {
            TransferStrategy::Memcpy => Self::Memcpy,
            TransferStrategy::CudaAsyncH2D => Self::CudaH2D,
            TransferStrategy::CudaAsyncD2H => Self::CudaD2H,
            TransferStrategy::CudaAsyncD2D => Self::CudaD2D,
            TransferStrategy::NixlRead => Self::NixlRead,
            TransferStrategy::NixlWrite => Self::NixlWrite,
            TransferStrategy::NixlReadFlipped => Self::NixlReadFlipped,
            TransferStrategy::NixlWriteFlipped => Self::NixlWriteFlipped,
            TransferStrategy::Invalid => Self::Invalid,
        }
    }
}

// ============================================================================
// Push-style sinks
// ============================================================================

/// Sink for raw pointer addresses. Implementors capture pointers emitted
/// while walking a [`PreparedTransferPlan::Transform`] template.
pub trait PointerSink {
    fn push(&mut self, addr: usize);
    fn reserve(&mut self, _additional: usize) {}
}

impl PointerSink for Vec<usize> {
    fn push(&mut self, addr: usize) {
        Vec::push(self, addr);
    }
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }
}

/// Allocation-free pointer sink used for telemetry / benchmark counting.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingPointerSink {
    pub count: usize,
}

impl PointerSink for CountingPointerSink {
    fn push(&mut self, _addr: usize) {
        self.count += 1;
    }
}

/// Sink for `(src_addr, dst_addr, size)` descriptor triples. Used by
/// downstream NIXL/CUDA paths to consume planner output without an
/// intermediate `Vec<CopyOp>` (or via a `Vec` sink when one is still useful).
pub trait DescSink {
    fn push(&mut self, src_addr: usize, dst_addr: usize, size: usize);
    fn reserve(&mut self, _additional: usize) {}
}

/// Allocation-free descriptor sink for telemetry / benchmark counting.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingDescSink {
    pub count: usize,
    pub total_bytes: usize,
}

impl DescSink for CountingDescSink {
    fn push(&mut self, _src_addr: usize, _dst_addr: usize, size: usize) {
        self.count += 1;
        self.total_bytes += size;
    }
}

/// `DescSink` that fills a `Vec<CopyOp>` — preserves the legacy planner
/// shape for paths that still need an op vector (small-strided copies,
/// CUDA batch dispatch, coalescing).
pub struct CopyOpVecSink<'a>(pub &'a mut Vec<CopyOp>);

impl DescSink for CopyOpVecSink<'_> {
    fn push(&mut self, src_addr: usize, dst_addr: usize, size: usize) {
        self.0.push(CopyOp {
            src_addr,
            dst_addr,
            size,
        });
    }
    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }
}

/// `DescSink` that fills a NIXL `XferDescList` pair in lockstep. Each
/// `push` adds one descriptor to each side with the configured device IDs.
pub struct NixlDescPairSink<'a, 'b> {
    pub src: &'a mut XferDescList<'b>,
    pub dst: &'a mut XferDescList<'b>,
    pub src_device_id: u64,
    pub dst_device_id: u64,
}

impl DescSink for NixlDescPairSink<'_, '_> {
    fn push(&mut self, src_addr: usize, dst_addr: usize, size: usize) {
        self.src.add_desc(src_addr, size, self.src_device_id);
        self.dst.add_desc(dst_addr, size, self.dst_device_id);
    }
}

// ============================================================================
// Transform scratch pool
// ============================================================================

/// Per-prepared-plan scratch tuple: `op_ptrs` and `univ_ptrs` host arrays
/// reused across calls.
#[derive(Debug, Default)]
pub(crate) struct TransformScratch {
    pub op_ptrs: Vec<usize>,
    pub univ_ptrs: Vec<usize>,
}

/// Bounded pool of [`TransformScratch`] buffers. Concurrent transfers
/// on the same prepared plan each acquire a lease; the lease returns the
/// buffer (cleared but keeping capacity) on drop, so steady-state same-
/// or-smaller block lists never reallocate.
///
/// Capacity is unbounded by entry count — in practice the pool grows to
/// the maximum concurrency of the transfer dispatcher and stays there.
#[derive(Debug, Default)]
pub(crate) struct TransformScratchPool {
    inner: Mutex<Vec<TransformScratch>>,
    peak_op_capacity: AtomicUsize,
    peak_univ_capacity: AtomicUsize,
    high_water_lease_count: AtomicUsize,
    current_lease_count: AtomicUsize,
}

impl TransformScratchPool {
    pub(crate) fn acquire(self: &Arc<Self>) -> TransformScratchLease {
        let scratch = self.inner.lock().unwrap().pop().unwrap_or_default();
        let n = self.current_lease_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.high_water_lease_count.fetch_max(n, Ordering::Relaxed);
        TransformScratchLease {
            scratch: Some(scratch),
            pool: self.clone(),
        }
    }

    /// Peak observed capacity of any returned `op_ptrs` Vec. Used by
    /// observability / bench reporting.
    #[allow(dead_code)] // observability surface; consumed by tests + future bench reporting
    pub fn peak_op_capacity(&self) -> usize {
        self.peak_op_capacity.load(Ordering::Relaxed)
    }

    /// Peak observed capacity of any returned `univ_ptrs` Vec.
    #[allow(dead_code)]
    pub fn peak_univ_capacity(&self) -> usize {
        self.peak_univ_capacity.load(Ordering::Relaxed)
    }

    /// Highest concurrent-lease count seen since pool creation. Equal to
    /// the steady-state pool depth.
    #[allow(dead_code)]
    pub fn high_water_lease_count(&self) -> usize {
        self.high_water_lease_count.load(Ordering::Relaxed)
    }

    fn approximate_bytes(&self) -> usize {
        let guard = self.inner.lock().unwrap();
        guard
            .iter()
            .map(|s| (s.op_ptrs.capacity() + s.univ_ptrs.capacity()) * size_of::<usize>())
            .sum()
    }
}

/// RAII lease for a [`TransformScratch`] from a [`TransformScratchPool`].
/// The scratch is cleared on drop and returned to the pool.
pub(crate) struct TransformScratchLease {
    scratch: Option<TransformScratch>,
    pool: Arc<TransformScratchPool>,
}

impl TransformScratchLease {
    /// Disjoint mutable references to both scratch Vecs at once. Use when
    /// you need to hand both into a sink-emitting call.
    pub(crate) fn both_mut(&mut self) -> (&mut Vec<usize>, &mut Vec<usize>) {
        let s = self.scratch.as_mut().expect("lease alive");
        (&mut s.op_ptrs, &mut s.univ_ptrs)
    }
    pub(crate) fn op_ptrs(&self) -> &[usize] {
        &self.scratch.as_ref().expect("lease alive").op_ptrs
    }
    pub(crate) fn univ_ptrs(&self) -> &[usize] {
        &self.scratch.as_ref().expect("lease alive").univ_ptrs
    }
    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn op_ptrs_mut(&mut self) -> &mut Vec<usize> {
        &mut self.scratch.as_mut().expect("lease alive").op_ptrs
    }
    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn univ_ptrs_mut(&mut self) -> &mut Vec<usize> {
        &mut self.scratch.as_mut().expect("lease alive").univ_ptrs
    }
}

impl Drop for TransformScratchLease {
    fn drop(&mut self) {
        if let Some(mut s) = self.scratch.take() {
            self.pool
                .peak_op_capacity
                .fetch_max(s.op_ptrs.capacity(), Ordering::Relaxed);
            self.pool
                .peak_univ_capacity
                .fetch_max(s.univ_ptrs.capacity(), Ordering::Relaxed);
            s.op_ptrs.clear();
            s.univ_ptrs.clear();
            self.pool.inner.lock().unwrap().push(s);
            self.pool
                .current_lease_count
                .fetch_sub(1, Ordering::Relaxed);
        }
    }
}

// ============================================================================
// PreparedTransferPlan
// ============================================================================

/// Cached transfer template — value type for the prepared-plan cache.
///
/// Carries the resolved [`KernelInvocation`], the universal-side base
/// address and bytes-per-block (for `Universal*From*` kinds — `None`
/// for the op↔op transpose where both sides are walked), and a
/// per-plan scratch pool of host pointer-array `Vec`s.
///
/// Same-layout / sliced direct copies do not produce a prepared plan
/// — `lookup_prepared_plan` returns `None` for those, and
/// `plan_and_lower` projects `AnnotatedLayout`s inline. The cache
/// therefore only stores transform templates.
#[derive(Debug)]
pub(crate) struct PreparedTransferPlan {
    pub(crate) invocation: KernelInvocation,
    /// Universal-side single-allocation base address. `Some(_)` for
    /// `UniversalFromBlock`/`BlockFromUniversal`; `None` for
    /// `NhdHndTranspose` (both sides operational).
    univ_base: Option<usize>,
    /// Universal-side bytes-per-block (constant; from `LayoutConfig`).
    /// Paired with `univ_base`.
    univ_bytes_per_block: Option<usize>,
    /// Per-plan scratch pool of host pointer-array `Vec`s.
    scratch_pool: Arc<TransformScratchPool>,
}

impl PreparedTransferPlan {
    /// Build a plan from the dispatched `KernelInvocation` and the
    /// source/destination physical layouts.
    pub(crate) fn build_transform(
        invocation: KernelInvocation,
        src: &PhysicalLayout,
        dst: &PhysicalLayout,
    ) -> Result<Self> {
        let (univ_base, univ_bytes_per_block) = match invocation.kind {
            KernelKind::NhdHndTranspose => (None, None),
            KernelKind::UniversalFromBlock => extract_universal_base(dst)?,
            KernelKind::BlockFromUniversal => extract_universal_base(src)?,
        };
        Ok(Self {
            invocation,
            univ_base,
            univ_bytes_per_block,
            scratch_pool: Arc::new(TransformScratchPool::default()),
        })
    }

    /// Acquire a scratch lease from the plan's pool.
    pub(crate) fn acquire_transform_scratch(&self) -> Result<TransformScratchLease> {
        Ok(self.scratch_pool.acquire())
    }

    /// Fill the operational-side pointer array (`block × layer × outer`)
    /// and the universal-side pointer array (`block`) for a
    /// universal-kind transform (`UniversalFromBlock` / `BlockFromUniversal`).
    ///
    /// `op_layout` is the operational-side `PhysicalLayout` (caller picks
    /// based on `invocation.kind`). `op_block_ids` and `univ_block_ids`
    /// are the projected block-id slices from `block_pairs`.
    ///
    /// `layer_range` restricts the layer iteration to a contiguous
    /// subrange. `None` walks the full extent (`0..invocation.num_layers`).
    /// The universal-side pointer array is unaffected by the layer range
    /// — the kernel uses `nl_full`/`nl_offset` to address the slice
    /// within each universal block.
    pub(crate) fn emit_universal_kind_pointers<O, U>(
        &self,
        op_layout: &PhysicalLayout,
        op_block_ids: &[BlockId],
        univ_block_ids: &[BlockId],
        layer_range: Option<&std::ops::Range<usize>>,
        op_sink: &mut O,
        univ_sink: &mut U,
    ) -> Result<()>
    where
        O: PointerSink,
        U: PointerSink,
    {
        let (univ_base, univ_bytes_per_block) = match (self.univ_base, self.univ_bytes_per_block) {
            (Some(b), Some(bpb)) => (b, bpb),
            _ => bail!(
                "emit_universal_kind_pointers: plan has no universal base (kind={:?}) — \
                 caller routed an op↔op transpose into the universal path",
                self.invocation.kind
            ),
        };
        if !matches!(
            self.invocation.kind,
            KernelKind::UniversalFromBlock | KernelKind::BlockFromUniversal
        ) {
            bail!(
                "emit_universal_kind_pointers: invocation kind {:?} is not a universal kind",
                self.invocation.kind
            );
        }
        let nl_full = self.invocation.num_layers;
        let layer_iter = match layer_range {
            Some(r) => {
                if r.end > nl_full || r.start > r.end {
                    bail!(
                        "emit_universal_kind_pointers: layer_range {:?} out of bounds for \
                         invocation.num_layers={}",
                        r,
                        nl_full,
                    );
                }
                r.clone()
            }
            None => 0..nl_full,
        };
        let nl_subset = layer_iter.len();
        let no = self.invocation.outer_dim;
        op_sink.reserve(op_block_ids.len() * nl_subset * no);
        univ_sink.reserve(univ_block_ids.len());
        for &block_id in op_block_ids {
            for layer in layer_iter.clone() {
                for outer in 0..no {
                    let region = op_layout
                        .layout()
                        .memory_region(block_id, layer, outer)
                        .map_err(|e| {
                            anyhow!(
                                "emit_universal_kind_pointers: failed to read operational \
                                 chunk (block={block_id}, layer={layer}, outer={outer}): {e:?}"
                            )
                        })?;
                    op_sink.push(region.addr());
                }
            }
        }
        for &block_id in univ_block_ids {
            univ_sink.push(univ_base + block_id * univ_bytes_per_block);
        }
        Ok(())
    }

    /// Fill the src/dst pointer arrays (both `block × layer × outer`) for
    /// an `NhdHndTranspose` (op↔op) transform. Both sides are walked via
    /// `Layout::memory_region`.
    ///
    /// `layer_range` restricts the layer iteration to a contiguous
    /// subrange on both sides. `None` walks the full extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_oo_transpose_pointers<S, D>(
        &self,
        src_layout: &PhysicalLayout,
        dst_layout: &PhysicalLayout,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<&std::ops::Range<usize>>,
        src_sink: &mut S,
        dst_sink: &mut D,
    ) -> Result<()>
    where
        S: PointerSink,
        D: PointerSink,
    {
        if !matches!(self.invocation.kind, KernelKind::NhdHndTranspose) {
            bail!(
                "emit_oo_transpose_pointers: invocation kind {:?} is not NhdHndTranspose",
                self.invocation.kind
            );
        }
        if src_block_ids.len() != dst_block_ids.len() {
            bail!(
                "emit_oo_transpose_pointers: block id list length mismatch (src={}, dst={})",
                src_block_ids.len(),
                dst_block_ids.len()
            );
        }
        let nl_full = self.invocation.num_layers;
        let layer_iter = match layer_range {
            Some(r) => {
                if r.end > nl_full || r.start > r.end {
                    bail!(
                        "emit_oo_transpose_pointers: layer_range {:?} out of bounds for \
                         invocation.num_layers={}",
                        r,
                        nl_full,
                    );
                }
                r.clone()
            }
            None => 0..nl_full,
        };
        let no = self.invocation.outer_dim;
        let chunks_per_block = layer_iter.len() * no;
        src_sink.reserve(src_block_ids.len() * chunks_per_block);
        dst_sink.reserve(dst_block_ids.len() * chunks_per_block);
        emit_op_table(src_layout, src_block_ids, &layer_iter, no, src_sink, "src")?;
        emit_op_table(dst_layout, dst_block_ids, &layer_iter, no, dst_sink, "dst")?;
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn kernel_kind(&self) -> KernelKind {
        self.invocation.kind
    }

    fn approximate_heap_bytes(&self) -> usize {
        size_of::<KernelInvocation>() + self.scratch_pool.approximate_bytes()
    }

    /// Test-only constructor — assembles a plan from synthetic field
    /// values without driving the kernel catalog or projecting a real
    /// `PhysicalLayout`. Used by cache-only unit tests in
    /// `tests/prepared_plan.rs` that exercise `PreparedPlanCache`
    /// structural behaviour without emitting pointer arrays.
    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn synthetic_for_tests(
        invocation: KernelInvocation,
        univ_base: Option<usize>,
        univ_bytes_per_block: Option<usize>,
        scratch_pool: Arc<TransformScratchPool>,
    ) -> Self {
        Self {
            invocation,
            univ_base,
            univ_bytes_per_block,
            scratch_pool,
        }
    }
}

fn emit_op_table<S: PointerSink>(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    layer_iter: &std::ops::Range<usize>,
    no: usize,
    sink: &mut S,
    side: &'static str,
) -> Result<()> {
    for &block_id in block_ids {
        for layer in layer_iter.clone() {
            for outer in 0..no {
                let region = layout
                    .layout()
                    .memory_region(block_id, layer, outer)
                    .map_err(|e| {
                        anyhow!(
                            "emit_op_table[{side}]: failed to read chunk \
                             (block={block_id}, layer={layer}, outer={outer}): {e:?}"
                        )
                    })?;
                sink.push(region.addr());
            }
        }
    }
    Ok(())
}

fn extract_universal_base(univ_layout: &PhysicalLayout) -> Result<(Option<usize>, Option<usize>)> {
    let buffers = univ_layout.layout().memory_regions();
    if buffers.len() != 1 {
        bail!(
            "extract_universal_base: universal side expects 1 Buffer, got {}",
            buffers.len()
        );
    }
    let base = buffers[0].addr();
    let bpb = univ_layout.layout().config().bytes_per_block();
    Ok((Some(base), Some(bpb)))
}

// ============================================================================
// Cache
// ============================================================================

/// Public read-only view of [`PreparedPlanCache`] state. Exposed so
/// external callers (benchmark harnesses, observability) can inspect
/// cache behaviour without touching cache internals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedPlanCacheStats {
    pub local_hits: usize,
    pub local_misses: usize,
    pub local_entries: usize,
    pub remote_hits: usize,
    pub remote_misses: usize,
    pub remote_entries: usize,
    pub approximate_bytes: usize,
}

/// Compact prepared-plan cache: unbounded local map (G1↔G2 lifetime) plus a
/// bounded remote LRU (remote G2↔G2 handle pairs).
///
/// Values are stored as `Arc<PreparedTransferPlan>` so cache hits cost only
/// a refcount bump — scratch pools are shared by construction across
/// concurrent transfers.
pub(crate) struct PreparedPlanCache {
    enabled: bool,
    local: Mutex<HashMap<PreparedPlanKey, Arc<PreparedTransferPlan>>>,
    remote: Mutex<BoundedLru>,
    local_hits: AtomicUsize,
    local_misses: AtomicUsize,
    remote_hits: AtomicUsize,
    remote_misses: AtomicUsize,
}

impl PreparedPlanCache {
    pub(crate) fn new(enabled: bool, remote_capacity: usize) -> Self {
        Self {
            enabled,
            local: Mutex::new(HashMap::new()),
            remote: Mutex::new(BoundedLru::new(remote_capacity)),
            local_hits: AtomicUsize::new(0),
            local_misses: AtomicUsize::new(0),
            remote_hits: AtomicUsize::new(0),
            remote_misses: AtomicUsize::new(0),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn get_or_insert_with<F>(
        &self,
        worker_id: u64,
        key: PreparedPlanKey,
        build: F,
    ) -> Result<Arc<PreparedTransferPlan>>
    where
        F: FnOnce() -> Result<PreparedTransferPlan>,
    {
        if !self.enabled {
            return Ok(Arc::new(build()?));
        }

        if key.is_local_for(worker_id) {
            if let Some(plan) = self.local.lock().unwrap().get(&key).cloned() {
                self.local_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(plan);
            }
            self.local_misses.fetch_add(1, Ordering::Relaxed);
            let plan = Arc::new(build()?);
            self.local.lock().unwrap().insert(key, plan.clone());
            return Ok(plan);
        }

        if let Some(plan) = self.remote.lock().unwrap().get(&key) {
            self.remote_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(plan);
        }
        self.remote_misses.fetch_add(1, Ordering::Relaxed);
        let plan = Arc::new(build()?);
        self.remote.lock().unwrap().insert(key, plan.clone());
        Ok(plan)
    }

    pub(crate) fn stats(&self) -> PreparedPlanCacheStats {
        let local = self.local.lock().unwrap();
        let remote = self.remote.lock().unwrap();
        PreparedPlanCacheStats {
            local_hits: self.local_hits.load(Ordering::Relaxed),
            local_misses: self.local_misses.load(Ordering::Relaxed),
            local_entries: local.len(),
            remote_hits: self.remote_hits.load(Ordering::Relaxed),
            remote_misses: self.remote_misses.load(Ordering::Relaxed),
            remote_entries: remote.len(),
            approximate_bytes: approximate_map_bytes(&local) + remote.approximate_bytes(),
        }
    }
}

fn approximate_map_bytes(map: &HashMap<PreparedPlanKey, Arc<PreparedTransferPlan>>) -> usize {
    map.iter()
        .map(|(key, plan)| key.approximate_heap_bytes() + plan.approximate_heap_bytes())
        .sum()
}

struct BoundedLru {
    capacity: usize,
    map: HashMap<PreparedPlanKey, Arc<PreparedTransferPlan>>,
    order: VecDeque<PreparedPlanKey>,
}

impl BoundedLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &PreparedPlanKey) -> Option<Arc<PreparedTransferPlan>> {
        let plan = self.map.get(key).cloned()?;
        self.touch(key);
        Some(plan)
    }

    fn insert(&mut self, key: PreparedPlanKey, plan: Arc<PreparedTransferPlan>) {
        if self.capacity == 0 {
            return;
        }
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), plan);
            self.touch(&key);
            return;
        }
        while self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, plan);
    }

    fn touch(&mut self, key: &PreparedPlanKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn approximate_bytes(&self) -> usize {
        approximate_map_bytes(&self.map)
            + self.order.len() * size_of::<PreparedPlanKey>()
            + self
                .order
                .iter()
                .map(PreparedPlanKey::approximate_heap_bytes)
                .sum::<usize>()
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind};

    fn transform_plan() -> PreparedTransferPlan {
        PreparedTransferPlan {
            invocation: KernelInvocation {
                kind: KernelKind::UniversalFromBlock,
                num_layers: 1,
                outer_dim: 1,
                page_size: 1,
                num_heads: 1,
                head_dim: 1,
                dtype: kvbm_kernels::TensorDataType::F16,
                block_layout: kvbm_kernels::BlockLayout::NHD,
            },
            univ_base: Some(0x1000),
            univ_bytes_per_block: Some(64),
            scratch_pool: Arc::new(TransformScratchPool::default()),
        }
    }

    fn key(src_worker: u64, src_layout: u16, dst_worker: u64, dst_layout: u16) -> PreparedPlanKey {
        PreparedPlanKey::new(
            LayoutHandle::new(src_worker, src_layout),
            LayoutHandle::new(dst_worker, dst_layout),
            TransferStrategy::NixlRead,
            &[],
        )
    }

    #[test]
    fn remote_lru_bounds_entries() {
        let cache = PreparedPlanCache::new(true, 2);
        let worker = 7;
        for i in 0..4 {
            let k = key(100 + i, 0, worker, 1);
            cache
                .get_or_insert_with(worker, k, || Ok(transform_plan()))
                .expect("insert");
        }
        assert_eq!(cache.stats().remote_entries, 2);
        assert_eq!(cache.stats().remote_misses, 4);
        assert!(cache.stats().approximate_bytes > 0);
    }

    #[test]
    fn local_plan_hits_after_first_build() {
        let cache = PreparedPlanCache::new(true, 2);
        let worker = 7;
        let k = key(worker, 0, worker, 1);
        cache
            .get_or_insert_with(worker, k.clone(), || Ok(transform_plan()))
            .expect("first");
        cache
            .get_or_insert_with(worker, k, || panic!("must hit"))
            .expect("second");
        let stats = cache.stats();
        assert_eq!(stats.local_misses, 1);
        assert_eq!(stats.local_hits, 1);
        assert_eq!(stats.local_entries, 1);
        assert!(stats.approximate_bytes > 0);
    }

    #[test]
    fn arc_shares_scratch_pool_across_cache_hits() {
        let cache = PreparedPlanCache::new(true, 2);
        let worker = 7;
        let k = key(worker, 0, worker, 1);
        let first = cache
            .get_or_insert_with(worker, k.clone(), || Ok(transform_plan()))
            .unwrap();
        let second = cache
            .get_or_insert_with(worker, k, || panic!("must hit"))
            .unwrap();
        // Same Arc backing — scratch pools and pointer addresses match.
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn scratch_lease_reuses_capacity_across_drops() {
        let pool = Arc::new(TransformScratchPool::default());
        {
            let mut lease = pool.acquire();
            lease.op_ptrs_mut().reserve(128);
            lease.univ_ptrs_mut().reserve(16);
            // Push something so the Vecs are non-empty before drop.
            for i in 0..32 {
                lease.op_ptrs_mut().push(i as usize);
            }
            for i in 0..4 {
                lease.univ_ptrs_mut().push(i as usize);
            }
        }
        assert_eq!(pool.peak_op_capacity(), 128);
        assert_eq!(pool.peak_univ_capacity(), 16);
        // Second lease pops the previously released scratch with kept
        // capacity but cleared length.
        let mut lease = pool.acquire();
        assert_eq!(lease.op_ptrs().len(), 0);
        assert_eq!(lease.univ_ptrs().len(), 0);
        assert!(lease.op_ptrs_mut().capacity() >= 128);
        assert!(lease.univ_ptrs_mut().capacity() >= 16);
    }

    #[test]
    fn counting_pointer_sink_counts() {
        let mut sink = CountingPointerSink::default();
        sink.push(1);
        sink.push(2);
        sink.push(3);
        assert_eq!(sink.count, 3);
    }

    #[test]
    fn counting_desc_sink_accumulates() {
        let mut sink = CountingDescSink::default();
        sink.push(0, 0, 10);
        sink.push(0, 0, 20);
        assert_eq!(sink.count, 2);
        assert_eq!(sink.total_bytes, 30);
    }

    #[test]
    fn copy_op_vec_sink_fills_vec() {
        let mut ops = Vec::<CopyOp>::new();
        let mut sink = CopyOpVecSink(&mut ops);
        sink.push(1, 2, 10);
        sink.push(3, 4, 20);
        assert_eq!(
            ops,
            vec![
                CopyOp {
                    src_addr: 1,
                    dst_addr: 2,
                    size: 10
                },
                CopyOp {
                    src_addr: 3,
                    dst_addr: 4,
                    size: 20
                }
            ]
        );
    }
}
