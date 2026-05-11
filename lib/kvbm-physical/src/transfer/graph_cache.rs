// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe CUDA graph exec handle cache for replay-based transfers.
//!
//! [`GraphCache`] maps [`GraphCacheKey`] (transfer *shape*) to a
//! [`ManagedExecHandle`] that wraps a `cudaGraphExec_t` (as the
//! driver-API equivalent `CUgraphExec`). A single captured exec handle
//! can be relaunched many times with per-launch address rebinding
//! (`cuGraphExecMemcpyNodeSetParams`) for block-ID pairs whose
//! descriptor count / byte volume / route match the key.
//!
//! # Lifecycle
//!
//! The cache is created inside [`TransferContext`] as an
//! `Arc<Mutex<GraphCache>>`, so it lives exactly as long as the
//! context.  When a [`ManagedExecHandle`] is dropped (either via
//! eviction or when the cache itself drops), `cuGraphExecDestroy` is
//! called unconditionally.
//!
//! # Eviction
//!
//! FIFO eviction with a hard cap of [`GRAPH_CACHE_CAP`] entries (64).
//! Eviction removes the entry with the lowest insertion-order index.
//! LRU was considered but the access pattern for KV-cache transfers
//! (same shape many times in a row) makes FIFO equivalent in practice,
//! and simpler — no timestamp updates on the hot path.
//!
//! # Thread safety
//!
//! All public methods acquire `Mutex<GraphCacheInner>` for the duration
//! of their operation.  The lock is only held during the lookup /
//! insertion / eviction logic — the (rare) graph instantiation outside
//! the lock is done by the caller, which then calls `insert` to store
//! the result.

use std::collections::HashMap;
use std::sync::Mutex;

use cudarc::driver::sys as cu_sys;

use crate::transfer::lower::GraphCacheKey;

/// Maximum number of exec handles retained in the cache.
///
/// 64 entries cover a wide variety of shapes (different block counts,
/// H2D vs D2H vs D2D, multiple dtypes) while staying well within
/// typical GPU driver memory budgets.  Each captured exec is a modest
/// CUDA driver allocation; 64 of them on a single device is negligible.
pub const GRAPH_CACHE_CAP: usize = 64;

/// An RAII wrapper for a raw `CUgraphExec` handle.
///
/// `Drop` calls `cuGraphExecDestroy`.  Must not be cloned — the exec
/// handle is a unique resource.  Held inside `GraphCache` behind a
/// `Mutex`.
pub(crate) struct ManagedExecHandle {
    pub(crate) exec: cu_sys::CUgraphExec,
    /// The raw `CUgraph` from which this exec was instantiated. Stored
    /// so we can call `cuGraphGetNodes` to retrieve node handles for
    /// per-launch address rebinding.
    pub(crate) graph: cu_sys::CUgraph,
    /// Ordered list of graph node handles, one per memcpy op captured.
    /// Built once at instantiation time; used at every replay to
    /// rebind src/dst addresses.
    pub(crate) nodes: Vec<cu_sys::CUgraphNode>,
}

// SAFETY: `CUgraphExec` and `CUgraph` are opaque pointers managed by
// the CUDA driver; they are safe to send across threads as long as
// all access is serialised — which the surrounding `Mutex` ensures.
unsafe impl Send for ManagedExecHandle {}
unsafe impl Sync for ManagedExecHandle {}

impl Drop for ManagedExecHandle {
    fn drop(&mut self) {
        // Destroy exec first, then the template graph.
        if !self.exec.is_null() {
            let exec = std::mem::replace(&mut self.exec, std::ptr::null_mut());
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(exec);
            }
        }
        if !self.graph.is_null() {
            let graph = std::mem::replace(&mut self.graph, std::ptr::null_mut());
            unsafe {
                let _ = cu_sys::cuGraphDestroy(graph);
            }
        }
    }
}

struct GraphCacheInner {
    /// The exec handles, keyed by transfer shape.
    entries: HashMap<GraphCacheKey, (u64, ManagedExecHandle)>,
    /// Monotonically increasing insertion counter for FIFO ordering.
    seq: u64,
}

impl GraphCacheInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            seq: 0,
        }
    }

    /// Number of currently cached entries.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Evict the entry with the smallest insertion-order sequence number
    /// (the oldest entry).  Called when `len() == GRAPH_CACHE_CAP` before
    /// an insert.
    fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Find the key with the minimum seq value.
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, (seq, _))| *seq)
            .map(|(k, _)| k.clone())
            .expect("non-empty map must have a min");
        self.entries.remove(&oldest_key);
    }
}

/// Thread-safe CUDA graph exec handle cache.
///
/// Each entry maps a [`GraphCacheKey`] (transfer shape) to a
/// [`ManagedExecHandle`] (instantiated `CUgraphExec` with its node
/// list for rebinding). Bounded at [`GRAPH_CACHE_CAP`] entries with
/// FIFO eviction.
pub(crate) struct GraphCache {
    inner: Mutex<GraphCacheInner>,
}

impl GraphCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(GraphCacheInner::new()),
        }
    }

    /// Returns `true` if the cache contains an entry for `key`.
    #[allow(dead_code)]
    pub(crate) fn contains(&self, key: &GraphCacheKey) -> bool {
        let guard = self.inner.lock().expect("GraphCache mutex poisoned");
        guard.entries.contains_key(key)
    }

    /// Retrieve the node list for `key`, if present.
    ///
    /// Returns a cloned `Vec<CUgraphNode>` so the caller can rebind
    /// addresses without holding the lock during the CUDA API calls.
    /// The exec handle is returned as a raw pointer (not a reference)
    /// so the caller can call `cuGraphLaunch` without re-locking.
    ///
    /// **Caller contract**: do NOT call `cuGraphExecDestroy` on the
    /// returned exec; its lifetime is managed by the cache entry.  The
    /// handle is valid until the cache entry is evicted — callers must
    /// finish the rebind + launch before releasing the lock on the
    /// surrounding `TransferContext` stream (which serialises access to
    /// the same exec handle).
    pub(crate) fn get_exec_and_nodes(
        &self,
        key: &GraphCacheKey,
    ) -> Option<(cu_sys::CUgraphExec, Vec<cu_sys::CUgraphNode>)> {
        let guard = self.inner.lock().expect("GraphCache mutex poisoned");
        guard
            .entries
            .get(key)
            .map(|(_, h)| (h.exec, h.nodes.clone()))
    }

    /// Insert a new entry.  Evicts the oldest if the cache is full.
    ///
    /// If an entry for `key` already exists it is replaced (the old
    /// exec is destroyed via `ManagedExecHandle::drop`).
    pub(crate) fn insert(&self, key: GraphCacheKey, handle: ManagedExecHandle) {
        let mut guard = self.inner.lock().expect("GraphCache mutex poisoned");
        if guard.len() >= GRAPH_CACHE_CAP && !guard.entries.contains_key(&key) {
            guard.evict_oldest();
        }
        let seq = guard.seq;
        guard.seq += 1;
        guard.entries.insert(key, (seq, handle));
    }

    /// Number of currently cached entries (for tests / telemetry).
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().expect("GraphCache mutex poisoned").len()
    }
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}
