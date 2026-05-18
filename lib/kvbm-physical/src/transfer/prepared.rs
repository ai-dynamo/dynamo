// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepared transfer-plan cache.
//!
//! The cache stores compact, handle-keyed templates only. It deliberately does
//! not store block-list-specific pointer tables or `CopyOp` vectors: CUDA and
//! NIXL backends still need final per-call materialisation, so the useful cache
//! boundary is the address-free/static layout work.

use std::collections::{HashMap, VecDeque};
use std::mem::{size_of, size_of_val};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use kvbm_common::{AxisIntersection, KvDim};

use crate::manager::LayoutHandle;
use crate::transfer::kernel_catalog::KernelInvocation;
use crate::transfer::plan::AnnotatedLayout;
use crate::transfer::strategy::TransferStrategy;

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

/// Cached template for a prepared transfer.
#[derive(Clone)]
pub(crate) enum PreparedTransferPlan {
    /// Semantic-layout transform through the kernel catalog.
    Transform { invocation: KernelInvocation },
    /// Same-layout/sliced direct copy. The projected layouts are cached so
    /// repeated calls avoid layout-view projection and `AnnotatedLayout`
    /// construction; block-list-specific `CopyOp`s are still emitted per call.
    Direct {
        src: AnnotatedLayout,
        dst: AnnotatedLayout,
    },
}

impl PreparedTransferPlan {
    fn approximate_heap_bytes(&self) -> usize {
        match self {
            PreparedTransferPlan::Transform { .. } => size_of::<KernelInvocation>(),
            PreparedTransferPlan::Direct { src, dst } => {
                approximate_annotated_layout_heap_bytes(src)
                    + approximate_annotated_layout_heap_bytes(dst)
            }
        }
    }
}

fn approximate_annotated_layout_heap_bytes(layout: &AnnotatedLayout) -> usize {
    size_of_val(layout.regions())
        + size_of_val(layout.dim_layout().dims())
        + size_of_val(layout.dim_layout().sizes())
        + size_of_val(layout.byte_strides().as_bytes())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PreparedPlanCacheStats {
    pub local_hits: usize,
    pub local_misses: usize,
    pub local_entries: usize,
    pub remote_hits: usize,
    pub remote_misses: usize,
    pub remote_entries: usize,
    pub approximate_bytes: usize,
}

/// Small two-tier cache: unbounded local entries (G1↔G2 lifetime) plus a
/// bounded remote LRU (remote G2↔G2 handle pairs).
pub(crate) struct PreparedPlanCache {
    enabled: bool,
    local: Mutex<HashMap<PreparedPlanKey, PreparedTransferPlan>>,
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

    pub(crate) fn get_or_insert_with<F>(
        &self,
        worker_id: u64,
        key: PreparedPlanKey,
        build: F,
    ) -> Result<PreparedTransferPlan>
    where
        F: FnOnce() -> Result<PreparedTransferPlan>,
    {
        if !self.enabled {
            return build();
        }

        if key.is_local_for(worker_id) {
            if let Some(plan) = self.local.lock().unwrap().get(&key).cloned() {
                self.local_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(plan);
            }
            self.local_misses.fetch_add(1, Ordering::Relaxed);
            let plan = build()?;
            self.local.lock().unwrap().insert(key, plan.clone());
            return Ok(plan);
        }

        if let Some(plan) = self.remote.lock().unwrap().get(&key) {
            self.remote_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(plan);
        }
        self.remote_misses.fetch_add(1, Ordering::Relaxed);
        let plan = build()?;
        self.remote.lock().unwrap().insert(key, plan.clone());
        Ok(plan)
    }

    #[allow(dead_code)]
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

fn approximate_map_bytes(map: &HashMap<PreparedPlanKey, PreparedTransferPlan>) -> usize {
    map.iter()
        .map(|(key, plan)| key.approximate_heap_bytes() + plan.approximate_heap_bytes())
        .sum()
}

struct BoundedLru {
    capacity: usize,
    map: HashMap<PreparedPlanKey, PreparedTransferPlan>,
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

    fn get(&mut self, key: &PreparedPlanKey) -> Option<PreparedTransferPlan> {
        let plan = self.map.get(key).cloned()?;
        self.touch(key);
        Some(plan)
    }

    fn insert(&mut self, key: PreparedPlanKey, plan: PreparedTransferPlan) {
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

    fn plan() -> PreparedTransferPlan {
        PreparedTransferPlan::Transform {
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
                .get_or_insert_with(worker, k, || Ok(plan()))
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
            .get_or_insert_with(worker, k.clone(), || Ok(plan()))
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
}
