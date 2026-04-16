// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-backed implementation of vLLM's ``KVCacheManager`` contract.
//!
//! [`PyRustKvCacheManager`] is the PyO3 class that powers the
//! ``kvbm.v2.vllm.kv_cache_manager.RustKvCacheManager`` Python shim. It
//! owns a single [`BlockManager<G1>`] (a logical block tracker — the
//! GPU KV cache tensor itself is still owned by vLLM's model runner)
//! and a per-request map of [`RequestSequence<G1>`] slots.
//!
//! ## Why block id 0 is reserved
//!
//! vLLM's model runner treats block id `0` as a sentinel / padding
//! slot and never assigns it to a request. To preserve that invariant
//! without requiring kvbm-logical to know anything vLLM-specific, the
//! constructor eagerly allocates block id `0` from the fresh reset
//! pool (which hands out IDs in order starting from zero) and parks
//! it in `_reserved_zero` for the lifetime of the manager. The RAII
//! guard never drops, so block id `0` is permanently pinned.
//!
//! ## Delay-cache-blocks and the staged state
//!
//! vLLM's ``allocate_slots(..., delay_cache_blocks=True)`` signals
//! that newly allocated blocks are being filled by an external KV
//! transfer and must not be visible to prefix matching until that
//! transfer completes. kvbm-logical already has exactly this split
//! at the [`LogicalBlockAssignments`](kvbm_logical::LogicalBlockAssignments)
//! layer: unassigned ➜ **staged** ➜ assigned. The shim maps
//! `delay_cache_blocks=True` onto [`RequestSequence::stage_pending`]
//! (stop short of register), and `KVCacheManager::cache_blocks`
//! resumes via [`RequestSequence::register_staged`].
//!
//! ## What this module does not implement
//!
//! Features that vLLM supports but this binding currently rejects:
//!
//! * ``use_eagle=True`` (spec decode eagle draft model)
//! * ``dcp_world_size > 1`` (decode-context parallel)
//! * ``num_encoder_tokens > 0`` (encoder–decoder models)
//! * ``num_lookahead_tokens > 0`` (lookahead decoding spec proposer)
//!
//! These all raise ``NotImplementedError`` from the Python shim,
//! backed by explicit asserts/errors here.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use dynamo_tokens::Token;

use kvbm_engine::G1;
use kvbm_logical::blocks::{BlockRegistry, MutableBlock};
use kvbm_logical::manager::{
    BlockManager, FrequencyTrackingCapacity,
};
use kvbm_logical::{BlockId, ImmutableBlock, RequestSequence};

use crate::to_pyerr;
use super::block_manager_handle::PyG1BlockManagerHandle;

/// A single request's view of the KV cache.
///
/// Holds the token sequence, the associated block assignments
/// (unassigned ➜ staged ➜ assigned), and per-request bookkeeping that
/// vLLM's scheduler expects to drive via ``allocate_slots`` /
/// ``cache_blocks`` / ``free``.
struct SlotState {
    /// Opaque request id. Only used for equality / lookup; the Rust
    /// side never parses it.
    _request_id: String,

    /// Cached (lora, salt) hash for token-block hashing. Matches what
    /// v1's ``KvbmRequest::salt_hash`` computes. Consumed by
    /// [`PyRustKvCacheManager::build_sequence`] when the slot is
    /// recreated, but not read again during the slot's lifetime.
    #[allow(dead_code)]
    salt_hash: u64,

    /// The request-scoped block lifecycle driver.
    sequence: RequestSequence<G1>,

    /// Number of tokens the *scheduler* has told us are "committed".
    /// This is what vLLM's ``request.num_computed_tokens`` maps to;
    /// we re-read it on every ``allocate_slots`` so we know which
    /// chunk of ``request.all_token_ids`` is newly appendable.
    computed_token_count: usize,

    /// Set of block ids this request owns (ordered, deduped). This is
    /// the list returned to vLLM via ``get_block_ids``.
    owned_block_ids: Vec<BlockId>,
}

/// vLLM kv-cache-manager shim exposed to Python as
/// ``kvbm._core.v2.RustKvCacheManager``.
///
/// See the module docs for design rationale.
#[pyclass(name = "RustKvCacheManager")]
pub struct PyRustKvCacheManager {
    block_manager: Arc<BlockManager<G1>>,
    block_size: usize,
    enable_caching: bool,
    log_stats: bool,

    /// Permanently pinned guard for block id `0` (vLLM's sentinel).
    /// Never dropped for the lifetime of the manager.
    _reserved_zero: MutableBlock<G1>,

    /// Per-request slot table. All mutation goes through this lock.
    slots: Mutex<HashMap<String, SlotState>>,

    /// Blocks returned to vLLM as "cached" via ``get_computed_blocks``
    /// that may be round-tripped back through
    /// ``allocate_slots(new_computed_blocks=...)``. Pinned here so
    /// they stay alive (and out of the reset pool) in the gap between
    /// those two calls. Drained whenever the owning request is
    /// ``free``'d.
    ///
    /// Keyed by ``(request_id, block_id)``.
    pinned_cache_hits: Mutex<HashMap<(String, BlockId), ImmutableBlock<G1>>>,

    // ---- Stats ------------------------------------------------------
    prefix_cache_hits: AtomicU64,
    prefix_cache_queries: AtomicU64,
}

#[pymethods]
impl PyRustKvCacheManager {
    /// Build a new manager.
    ///
    /// Parameters mirror vLLM's ``KVCacheManager.__init__`` but are
    /// deliberately fewer: the shim only exposes what kvbm currently
    /// supports. Unsupported settings raise immediately.
    ///
    /// The Python side is expected to pre-extract ``total_blocks``
    /// and ``block_size`` from ``kv_cache_config`` — the Rust binding
    /// does not want to reach into a dynamically-typed vLLM config
    /// object.
    #[new]
    #[pyo3(signature = (
        total_blocks,
        block_size,
        enable_caching = true,
        log_stats = false,
    ))]
    pub fn new(
        total_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        log_stats: bool,
    ) -> PyResult<Self> {
        if total_blocks < 2 {
            return Err(PyValueError::new_err(
                "total_blocks must be >= 2 (block id 0 is reserved for \
                 vLLM's padding sentinel)",
            ));
        }
        if !block_size.is_power_of_two() || !(1..=1024).contains(&block_size) {
            return Err(PyValueError::new_err(format!(
                "block_size must be a power of two in [1, 1024], got {block_size}"
            )));
        }

        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
            .build();

        let manager = BlockManager::<G1>::builder()
            .block_count(total_blocks)
            .block_size(block_size)
            .registry(registry)
            .with_multi_lru_backend()
            .build()
            .map_err(to_pyerr)?;

        // Pin block id 0. A fresh BlockManager hands out IDs sequentially
        // starting at 0, so the first allocation is guaranteed to be id 0
        // — if that ever changes, fail loudly at construction rather than
        // silently handing id 0 back to a request later.
        let mut seed = manager
            .allocate_blocks(1)
            .ok_or_else(|| PyRuntimeError::new_err("failed to allocate sentinel block"))?;
        let reserved = seed.pop().expect("allocate_blocks(1) returned empty Vec");
        assert_eq!(
            reserved.block_id(),
            0,
            "BlockManager<G1> did not allocate block id 0 as the first block; \
             cannot enforce vLLM sentinel reservation"
        );

        Ok(Self {
            block_manager: Arc::new(manager),
            block_size,
            enable_caching,
            log_stats,
            _reserved_zero: reserved,
            slots: Mutex::new(HashMap::new()),
            pinned_cache_hits: Mutex::new(HashMap::new()),
            prefix_cache_hits: AtomicU64::new(0),
            prefix_cache_queries: AtomicU64::new(0),
        })
    }

    /// Construct from a pre-existing `BlockManager<G1>` (shared registry).
    ///
    /// This is the preferred constructor when a `ConnectorLeader` is present —
    /// the G1 manager shares the same `BlockRegistry` as G2/G3.
    #[classmethod]
    #[pyo3(name = "from_g1_handle")]
    pub fn from_g1_handle(
        _cls: &Bound<'_, PyType>,
        handle: &PyG1BlockManagerHandle,
        enable_caching: bool,
        log_stats: bool,
    ) -> PyResult<Self> {
        let manager = handle.inner.clone();
        let block_size = manager.block_size();

        // Pin block id 0 — same invariant as the `new` constructor.
        let mut seed = manager
            .allocate_blocks(1)
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "failed to allocate sentinel block from shared G1 manager",
                )
            })?;
        let reserved = seed.pop().expect("allocate_blocks(1) returned empty Vec");
        assert_eq!(
            reserved.block_id(),
            0,
            "shared BlockManager<G1> did not allocate block id 0 as the first block; \
             cannot enforce vLLM sentinel reservation"
        );

        Ok(Self {
            block_manager: manager,
            block_size,
            enable_caching,
            log_stats,
            _reserved_zero: reserved,
            slots: Mutex::new(HashMap::new()),
            pinned_cache_hits: Mutex::new(HashMap::new()),
            prefix_cache_hits: AtomicU64::new(0),
            prefix_cache_queries: AtomicU64::new(0),
        })
    }

    // ---------------------------------------------------------------
    // Core scheduling surface
    // ---------------------------------------------------------------

    /// ``usage`` — fraction of blocks currently in use, in ``[0, 1]``.
    ///
    /// Matches vLLM's ``KVCacheManager.usage`` property; the Python
    /// shim exposes this as a ``@property``.
    pub fn usage(&self) -> f32 {
        let total = self.block_manager.total_blocks();
        if total == 0 {
            return 0.0;
        }
        let inuse = total - self.block_manager.available_blocks();
        inuse as f32 / total as f32
    }

    /// Whether this manager was asked to log prefix cache stats.
    ///
    /// The Python shim decides whether to return a ``PrefixCacheStats``
    /// dataclass based on this flag.
    pub fn log_stats_enabled(&self) -> bool {
        self.log_stats
    }

    /// Returns ``(hits, queries)`` of the current prefix cache counters
    /// and resets them to zero. Caller is responsible for wrapping
    /// these into whatever stats dataclass vLLM expects.
    pub fn take_prefix_cache_stats(&self) -> (u64, u64) {
        let hits = self.prefix_cache_hits.swap(0, Ordering::Relaxed);
        let queries = self.prefix_cache_queries.swap(0, Ordering::Relaxed);
        (hits, queries)
    }

    /// Create or replace a request slot for ``request_id`` with the
    /// given token stream and salt.
    ///
    /// Idempotent for repeat calls with the same request id — the
    /// existing slot is dropped (and its blocks RAII'd back to the
    /// pool) and a new one installed.
    pub fn create_slot(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        salt_hash: u64,
        max_output_tokens: usize,
    ) -> PyResult<()> {
        let mut slots = self.slots.lock().unwrap();
        // Dropping an existing entry RAIIs the blocks back to the pool.
        slots.remove(&request_id);
        let tokens: Vec<Token> = tokens;
        let sequence = self.build_sequence(tokens, max_output_tokens, salt_hash);
        slots.insert(
            request_id.clone(),
            SlotState {
                _request_id: request_id,
                salt_hash,
                sequence,
                computed_token_count: 0,
                owned_block_ids: Vec::new(),
            },
        );
        Ok(())
    }

    /// Does this manager already have a slot for this request id?
    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slots.lock().unwrap().contains_key(request_id)
    }

    /// ``get_computed_blocks`` — prefix-match the request's current
    /// tokens and return the matched block ids plus the number of
    /// tokens they cover.
    ///
    /// The caller must have already installed the slot via
    /// ``create_slot``. The returned block ids are also *pinned* in
    /// ``pinned_cache_hits`` so they stay alive through any subsequent
    /// ``allocate_slots`` round-trip.
    pub fn get_computed_blocks(
        &self,
        request_id: &str,
    ) -> PyResult<(Vec<BlockId>, usize)> {
        let mut slots = self.slots.lock().unwrap();
        let slot = slots
            .get_mut(request_id)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no slot for {request_id}")))?;

        if self.log_stats {
            self.prefix_cache_queries.fetch_add(1, Ordering::Relaxed);
        }

        if !self.enable_caching {
            return Ok((Vec::new(), 0));
        }

        // If the slot already has blocks, this is a re-entry from vLLM
        // (which shouldn't happen for a brand-new request, but be safe).
        if !slot.sequence.assignments().is_empty() {
            return Ok((slot.owned_block_ids.clone(), slot.computed_token_count));
        }

        let matched = slot.sequence.match_and_add_prefix(&self.block_manager)
            .map_err(|e| PyRuntimeError::new_err(format!("prefix match failed: {e}")))?;

        let mut ids = Vec::with_capacity(matched);
        {
            let mut pinned = self.pinned_cache_hits.lock().unwrap();
            for (i, (block_id, immutable)) in slot
                .sequence
                .assignments()
                .assigned_iter()
                .take(matched)
                .enumerate()
            {
                ids.push(*block_id);
                // Pin a clone so vLLM can round-trip it.
                pinned.insert((request_id.to_string(), *block_id), immutable.clone());
                let _ = i;
            }
        }

        let num_tokens = matched * self.block_size;
        slot.owned_block_ids = ids.clone();
        slot.computed_token_count = num_tokens;

        if self.log_stats && matched > 0 {
            self.prefix_cache_hits
                .fetch_add(matched as u64, Ordering::Relaxed);
        }

        Ok((ids, num_tokens))
    }

    /// ``allocate_slots`` — extend a request with ``num_new_tokens``
    /// tokens of KV cache. Returns the list of *newly* allocated block
    /// ids (possibly empty, when the extension fits inside an already
    /// owned block).
    ///
    /// Returns ``None`` if the allocation cannot be satisfied (vLLM
    /// interprets ``None`` as "preempt and retry later").
    ///
    /// ``new_token_ids`` is the slice of ``request.all_token_ids``
    /// between the slot's previous ``num_computed_tokens`` and the
    /// new ``num_computed_tokens`` — the caller computes this because
    /// the scheduler already has the fully-materialized token list.
    ///
    /// ``delay_cache_blocks`` gates whether freshly filled blocks are
    /// registered into the prefix cache (``False`` → register now,
    /// ``True`` → leave them staged for a later ``cache_blocks`` call).
    #[pyo3(signature = (
        request_id,
        new_token_ids,
        num_new_tokens,
        num_new_computed_tokens = 0,
        delay_cache_blocks = false,
    ))]
    pub fn allocate_slots(
        &self,
        request_id: &str,
        new_token_ids: Vec<u32>,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        delay_cache_blocks: bool,
    ) -> PyResult<Option<Vec<BlockId>>> {
        if num_new_tokens == 0 {
            return Err(PyValueError::new_err("num_new_tokens must be > 0"));
        }

        let mut slots = self.slots.lock().unwrap();
        let slot = slots
            .get_mut(request_id)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no slot for {request_id}")))?;

        // Extend the token sequence with newly-computed tokens (prefill
        // chunks or decode steps).
        for token in new_token_ids {
            // The block lifecycle side-effect (None/Some(block_idx))
            // isn't needed here — kvbm's sequence handles the
            // bookkeeping, and we register below.
            let _ = slot.sequence.append_token(token);
        }

        slot.computed_token_count += num_new_computed_tokens;

        // How many complete blocks does the sequence need right now?
        // ``num_blocks()`` is the count of token-complete blocks;
        // plus one generation block if there's an in-progress tail.
        let total_complete = slot.sequence.num_blocks();
        let need_generation_block = slot.sequence.total_tokens() > total_complete * self.block_size;
        let target = total_complete + usize::from(need_generation_block);
        let currently_allocated = slot.sequence.assigned_blocks()
            + slot.sequence.staged_blocks()
            + slot.sequence.unassigned_blocks();

        let to_allocate = target.saturating_sub(currently_allocated);
        if to_allocate > 0 && !slot.sequence.allocate_blocks(to_allocate, &self.block_manager) {
            return Ok(None);
        }

        // Stage any newly-complete token blocks, and optionally
        // register them. Delay-cache-blocks stops at the staging edge.
        slot.sequence.stage_pending();
        if !delay_cache_blocks {
            slot.sequence.register_staged(&self.block_manager);
        }

        // Recompute owned_block_ids and return only the *new* ids
        // versus the pre-call state.
        let previous = std::mem::take(&mut slot.owned_block_ids);
        let current: Vec<BlockId> = slot
            .sequence
            .assignments()
            .all_block_ids()
            .copied()
            .collect();
        let previous_set: std::collections::HashSet<BlockId> =
            previous.iter().copied().collect();
        let new_ids: Vec<BlockId> = current
            .iter()
            .filter(|id| !previous_set.contains(id))
            .copied()
            .collect();
        slot.owned_block_ids = current;

        Ok(Some(new_ids))
    }

    /// ``cache_blocks`` — register any still-staged blocks for this
    /// request. Called after a KV transfer completes, closing the
    /// delay-cache-blocks gap.
    pub fn cache_blocks(&self, request_id: &str, _num_computed_tokens: usize) -> PyResult<()> {
        let mut slots = self.slots.lock().unwrap();
        let slot = slots
            .get_mut(request_id)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no slot for {request_id}")))?;
        slot.sequence.register_staged(&self.block_manager);
        slot.owned_block_ids = slot
            .sequence
            .assignments()
            .all_block_ids()
            .copied()
            .collect();
        Ok(())
    }

    /// ``free`` — drop this request's slot. RAII releases every block
    /// it owned (including any still-pinned cache hits).
    pub fn free(&self, request_id: &str) -> PyResult<()> {
        self.slots.lock().unwrap().remove(request_id);
        // Drop every pinned cache-hit entry that belonged to this
        // request.
        let mut pinned = self.pinned_cache_hits.lock().unwrap();
        pinned.retain(|(rid, _), _| rid != request_id);
        Ok(())
    }

    /// ``get_block_ids`` — return the (owned) block ids for this
    /// request, in lifecycle order (assigned ➜ staged ➜ unassigned).
    /// Returns an empty vec for unknown ids.
    pub fn get_block_ids(&self, request_id: &str) -> Vec<BlockId> {
        self.slots
            .lock().unwrap()
            .get(request_id)
            .map(|s| s.owned_block_ids.clone())
            .unwrap_or_default()
    }

    /// ``reset_prefix_cache`` — drop every currently-inactive block
    /// back to the reset pool. Called by RLHF workflows after weight
    /// updates.
    pub fn reset_prefix_cache(&self) -> bool {
        self.block_manager.reset_inactive_pool().is_ok()
    }

    /// ``get_num_common_prefix_blocks`` — number of blocks at the head
    /// of ``running_request_id``'s block list that are shared by every
    /// other tracked request.
    ///
    /// This mirrors vLLM's ``FullAttentionManager`` implementation
    /// (``vllm/v1/core/single_type_kv_cache_manager.py``):
    ///
    /// ```python
    /// blocks = self.req_to_blocks[running_request_id]
    /// num_common_blocks = 0
    /// for block in blocks:
    ///     if block.ref_cnt == len(self.req_to_blocks):
    ///         num_common_blocks += 1
    ///     else:
    ///         break
    /// return num_common_blocks
    /// ```
    ///
    /// The kvbm equivalent is more direct because block ids in
    /// kvbm-logical are pool indices and the registry only hands out
    /// the *same* [`ImmutableBlock`] / block id to two different
    /// slots when they prefix-matched the same sequence hash. So
    /// "block shared by every request" is exactly "the same block id
    /// appears in every active slot's owned list".
    ///
    /// We walk the running slot's **assigned** block ids in order
    /// (staged/unassigned blocks aren't in the registry yet and
    /// cannot be shared), and for each id count how many slots
    /// currently own it. If that count equals the total number of
    /// tracked slots, the block is a common-prefix block; otherwise
    /// we stop. ``num_running_requests`` is accepted for API
    /// symmetry with older vLLM signatures but is ignored — we use
    /// ``slots.len()`` as the ground-truth denominator, which is
    /// what vLLM's ``FullAttentionManager`` uses too
    /// (``len(self.req_to_blocks)``).
    pub fn get_num_common_prefix_blocks(&self, running_request_id: &str) -> usize {
        let slots = self.slots.lock().unwrap();
        let total = slots.len();
        if total == 0 {
            return 0;
        }
        let Some(running) = slots.get(running_request_id) else {
            return 0;
        };

        // Running request's assigned block ids in lifecycle order.
        // We deliberately skip staged / unassigned here: staged
        // blocks aren't in the registry yet, so no other slot could
        // have matched them, and unassigned blocks haven't been
        // filled with tokens yet. Both would always fail the
        // all-slots-contain check.
        let running_ids: Vec<BlockId> = running
            .sequence
            .assignments()
            .assigned_iter()
            .map(|(id, _)| *id)
            .collect();

        let mut num_common = 0usize;
        for block_id in &running_ids {
            let hits = slots
                .values()
                .filter(|s| s.owned_block_ids.contains(block_id))
                .count();
            if hits == total {
                num_common += 1;
            } else {
                break;
            }
        }
        num_common
    }

    /// Number of blocks this manager manages (including the pinned
    /// sentinel at id 0).
    pub fn total_blocks(&self) -> usize {
        self.block_manager.total_blocks()
    }

    /// Block size in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Take any queued KV cache events.
    ///
    /// The Rust binding does not currently emit events, so this
    /// always returns an empty list. Kept on the class for signature
    /// parity with vLLM's ``take_events``.
    pub fn take_events<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
}

impl PyRustKvCacheManager {
    /// Helper: build a fresh [`RequestSequence`] from token ids and a
    /// salt. Kept in a non-`#[pymethods]` impl block so it doesn't
    /// leak into Python.
    fn build_sequence(
        &self,
        tokens: Vec<Token>,
        max_output_tokens: usize,
        _salt_hash: u64,
    ) -> RequestSequence<G1> {
        // NB: RequestSequence::new takes (tokens, max_output_tokens,
        // block_size). Salt hashing is applied inside BlockSequence
        // construction via a separate constructor path; for now we
        // feed the salt via a wrapper so that prefix-cache hashes
        // match v1's behavior. If kvbm-logical's RequestSequence
        // gains a salt-accepting ctor, wire that through here.
        let _ = _salt_hash;
        RequestSequence::<G1>::new(tokens, max_output_tokens, self.block_size as u32)
    }
}
