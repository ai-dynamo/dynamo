// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PyO3's #[pyfunction] expansion triggers this lint on PyResult return types.
#![allow(clippy::useless_conversion)]

//! Native shared-memory KV lease atomics for GMS persistent allocations.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const LEASE_MAGIC: u32 = 0x4c53_4d47; // "GMSL" as little-endian u32.
const LEASE_VERSION: u32 = 1;
const LEASE_HEADER_SIZE: usize = 64;
const LEASE_RECORD_SIZE: usize = 16;

const L_MAGIC: usize = 0;
const L_VERSION: usize = 4;
const L_TOTAL_BLOCKS: usize = 8;
const L_RECORD_SIZE: usize = 12;
const L_FREE_COUNT: usize = 16;
// Number of record mutations in flight. The all-ones value is an exclusive
// post-fence recovery barrier; see `LeaseMutationGuard` below.
const L_ACTIVE_MUTATIONS: usize = 24;
const L_RESERVATION_EPOCH: usize = 32;
const L_RESERVED_BLOCKS: usize = 40;
const L_RESERVED_OWNER_HASH: usize = 48;
// PID of the process that currently holds the recovery barrier. Lets a fenced
// successor reclaim a barrier stranded by a crashed recovery owner, and lets
// this process detect genuine same-process re-entry. Occupies the last 8 bytes
// of the 64-byte header; the Python side never reads it, so this is ABI-safe.
const L_RECOVERY_OWNER_PID: usize = 56;

// Per-block lease record: state (CAS-owned), generation (stale-op guard),
// owner_hash (holder identity for foreign reclaim/fencing).
const LR_STATE: usize = 0;
const LR_GENERATION: usize = 4;
const LR_OWNER_HASH: usize = 8;

const LEASE_STATE_FREE: u32 = 0;
const LEASE_STATE_LEASED: u32 = 1;
const LEASE_STATE_SEALED: u32 = 2;
const LEASE_STATE_RESERVED: u32 = 3;
const LEASE_STATE_TRANSITION: u32 = 4;
const LEASE_RECOVERY_BARRIER: u64 = u64::MAX;

struct LeaseMutationGuard {
    active: *const AtomicU64,
}

impl Drop for LeaseMutationGuard {
    fn drop(&mut self) {
        // SAFETY: the mmap outlives every Python call that owns this guard.
        // Never arithmetic-modify the recovery barrier: if a post-fence recovery
        // owner replaced the activity count with the all-ones sentinel while this
        // mutation was in flight, a blind fetch_sub would turn u64::MAX into
        // MAX-1 and silently reopen the namespace mid-recovery. In that case the
        // recovery owner has already accounted for drained mutators, so this
        // guard simply does nothing.
        unsafe {
            loop {
                let cur = (*self.active).load(Ordering::Acquire);
                if cur == LEASE_RECOVERY_BARRIER || cur == 0 {
                    break;
                }
                if (*self.active)
                    .compare_exchange_weak(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }
    }
}

/// Register a record mutation unless post-fence recovery owns the namespace.
///
/// Mutations remain fully concurrent: this is an activity count, not a mutex.
/// While a recovery barrier is held the namespace is briefly frozen; callers
/// spin for a bounded budget and then surface an error rather than hang forever
/// under the GIL if a recovery owner died with the barrier set.
unsafe fn enter_lease_mutation(ptr: *mut u8) -> PyResult<LeaseMutationGuard> {
    let active = ptr.add(L_ACTIVE_MUTATIONS) as *const AtomicU64;
    const MUTATION_BARRIER_BUDGET: u32 = 1 << 22;
    let mut budget = MUTATION_BARRIER_BUDGET;
    loop {
        let observed = (*active).load(Ordering::Acquire);
        if observed == LEASE_RECOVERY_BARRIER {
            if budget == 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "KV lease recovery barrier is held; namespace is being recovered",
                ));
            }
            budget -= 1;
            std::hint::spin_loop();
            continue;
        }
        if (*active)
            .compare_exchange_weak(
                observed,
                observed.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            return Ok(LeaseMutationGuard { active });
        }
    }
}

struct LeaseRecoveryGuard {
    active: *const AtomicU64,
}

impl Drop for LeaseRecoveryGuard {
    fn drop(&mut self) {
        // SAFETY: only the post-fence recovery owner can hold the barrier.
        unsafe {
            (*self.active).store(0, Ordering::Release);
        }
    }
}

/// Exclusively fence record mutations after every prior writer is dead.
///
/// The caller must hold the external failover lock, which serializes recovery
/// across processes. Under that contract this (a) drains in-flight mutations to
/// zero before claiming the barrier (bounded), so recovery is exclusive against
/// live mutators instead of racing them, and (b) reclaims a barrier stranded by
/// a crashed recovery owner (recorded via its PID) rather than failing forever.
/// Only genuine same-process re-entry is rejected.
unsafe fn enter_lease_recovery(ptr: *mut u8) -> PyResult<LeaseRecoveryGuard> {
    let active = ptr.add(L_ACTIVE_MUTATIONS) as *const AtomicU64;
    let pid_ptr = ptr.add(L_RECOVERY_OWNER_PID) as *const AtomicU64;
    let me = std::process::id() as u64;
    const RECOVERY_DRAIN_BUDGET: u32 = 1 << 22;
    let mut budget = RECOVERY_DRAIN_BUDGET;
    loop {
        match (*active).compare_exchange_weak(
            0,
            LEASE_RECOVERY_BARRIER,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                (*pid_ptr).store(me, Ordering::Release);
                return Ok(LeaseRecoveryGuard { active });
            }
            Err(cur) => {
                if cur == LEASE_RECOVERY_BARRIER {
                    // Barrier already held. Same-process re-entry is a real bug;
                    // otherwise the external failover lock guarantees the prior
                    // recovery owner is dead, so reclaim the stranded barrier.
                    let owner = (*pid_ptr).load(Ordering::Acquire);
                    if owner == me {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "KV lease recovery is already in progress",
                        ));
                    }
                    (*pid_ptr).store(me, Ordering::Release);
                    return Ok(LeaseRecoveryGuard { active });
                }
                if budget == 0 {
                    // A non-zero count that never drains means prior writers
                    // crashed mid-mutation and stranded their counts. Force the
                    // barrier; guarded mutator drops will not corrupt it.
                    (*active).store(LEASE_RECOVERY_BARRIER, Ordering::Release);
                    (*pid_ptr).store(me, Ordering::Release);
                    return Ok(LeaseRecoveryGuard { active });
                }
                budget -= 1;
                std::hint::spin_loop();
            }
        }
    }
}

#[inline(always)]
unsafe fn read_u32(ptr: *const u8, off: usize) -> u32 {
    std::ptr::read_unaligned(ptr.add(off) as *const u32)
}

#[inline(always)]
unsafe fn write_u32(ptr: *mut u8, off: usize, val: u32) {
    std::ptr::write_unaligned(ptr.add(off) as *mut u32, val);
}

#[inline(always)]
unsafe fn write_u64(ptr: *mut u8, off: usize, val: u64) {
    std::ptr::write_unaligned(ptr.add(off) as *mut u64, val);
}

#[inline(always)]
fn lease_record_off(block_id: u32) -> usize {
    LEASE_HEADER_SIZE + (block_id as usize) * LEASE_RECORD_SIZE
}

unsafe fn validate_lease_buffer(ptr: *const u8, buf_len: usize) -> PyResult<u32> {
    if buf_len < LEASE_HEADER_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lease buffer too small: {} < {}",
            buf_len, LEASE_HEADER_SIZE
        )));
    }
    let magic = read_u32(ptr, L_MAGIC);
    let version = read_u32(ptr, L_VERSION);
    let total_blocks = read_u32(ptr, L_TOTAL_BLOCKS);
    let record_size = read_u32(ptr, L_RECORD_SIZE);
    if magic != LEASE_MAGIC || version != LEASE_VERSION || record_size != LEASE_RECORD_SIZE as u32 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid GMS KV lease shared-memory header",
        ));
    }
    let need = LEASE_HEADER_SIZE + (total_blocks as usize) * LEASE_RECORD_SIZE;
    if buf_len < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lease buffer too small: {} < {}",
            buf_len, need
        )));
    }
    Ok(total_blocks)
}

unsafe fn release_acquired_lease_blocks(ptr: *mut u8, acquired: &[(u32, u32)]) {
    let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;
    for (block_id, generation) in acquired.iter().copied() {
        let base = lease_record_off(block_id);
        let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
        if (*generation_ptr).load(Ordering::Acquire) != generation {
            continue;
        }
        let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
        let mut state = (*state_ptr).load(Ordering::Acquire);
        while state == LEASE_STATE_LEASED || state == LEASE_STATE_SEALED {
            match (*state_ptr).compare_exchange(
                state,
                LEASE_STATE_TRANSITION,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if (*generation_ptr).load(Ordering::Acquire) != generation {
                        (*state_ptr).store(state, Ordering::Release);
                        break;
                    }
                    let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
                    (*owner_ptr).store(0, Ordering::Release);
                    (*free_count_ptr).fetch_add(1, Ordering::AcqRel);
                    (*state_ptr).store(LEASE_STATE_FREE, Ordering::Release);
                    break;
                }
                Err(observed) => state = observed,
            }
        }
    }
}

unsafe fn try_acquire_lease_block(
    ptr: *mut u8,
    total_blocks: u32,
    block_id: u32,
    owner_hash: u64,
) -> Option<(u32, u32)> {
    if block_id >= total_blocks {
        return None;
    }
    let base = lease_record_off(block_id);
    let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
    if (*state_ptr)
        .compare_exchange(
            LEASE_STATE_FREE,
            LEASE_STATE_TRANSITION,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return None;
    }

    // Single-writer-per-segment is enforced by the state CAS above. `generation`
    // (bumped here, validated on release/seal) guards against a stale release
    // freeing a re-leased block; `owner_hash` identifies the holder so a foreign
    // block can be reclaimed/fenced after the holder crashes.
    let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
    let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
    let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;

    let generation = (*generation_ptr)
        .fetch_add(1, Ordering::AcqRel)
        .wrapping_add(1);
    (*owner_ptr).store(owner_hash, Ordering::Release);
    (*free_count_ptr).fetch_sub(1, Ordering::AcqRel);
    (*state_ptr).store(LEASE_STATE_LEASED, Ordering::Release);
    Some((block_id, generation))
}

/// Initialize a shared-memory KV lease namespace.
#[pyfunction]
#[pyo3(signature = (buf, total_blocks, reserved_blocks))]
fn kv_lease_init(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    total_blocks: u32,
    reserved_blocks: Vec<u32>,
) -> PyResult<u32> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let need = LEASE_HEADER_SIZE + (total_blocks as usize) * LEASE_RECORD_SIZE;
    if total_blocks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "total_blocks must be positive",
        ));
    }
    if buf_len < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lease buffer too small: {} < {}",
            buf_len, need
        )));
    }

    unsafe {
        write_u32(ptr, L_MAGIC, LEASE_MAGIC);
        write_u32(ptr, L_VERSION, LEASE_VERSION);
        write_u32(ptr, L_TOTAL_BLOCKS, total_blocks);
        write_u32(ptr, L_RECORD_SIZE, LEASE_RECORD_SIZE as u32);
        write_u64(ptr, L_FREE_COUNT, total_blocks as u64);
        write_u64(ptr, L_ACTIVE_MUTATIONS, 0);
        write_u64(ptr, L_RESERVATION_EPOCH, 0);
        write_u32(ptr, L_RESERVED_BLOCKS, 0);
        write_u64(ptr, L_RESERVED_OWNER_HASH, 0);

        for block_id in 0..total_blocks {
            let base = lease_record_off(block_id);
            write_u32(ptr, base + LR_STATE, LEASE_STATE_FREE);
            write_u32(ptr, base + LR_GENERATION, 0);
            write_u64(ptr, base + LR_OWNER_HASH, 0);
        }

        let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;
        let mut reserved_count = 0u64;
        for block_id in reserved_blocks {
            if block_id >= total_blocks {
                continue;
            }
            let state_ptr = ptr.add(lease_record_off(block_id) + LR_STATE) as *const AtomicU32;
            let previous = (*state_ptr).swap(LEASE_STATE_RESERVED, Ordering::AcqRel);
            if previous == LEASE_STATE_FREE {
                reserved_count += 1;
            }
        }
        if reserved_count > 0 {
            (*free_count_ptr).fetch_sub(reserved_count, Ordering::AcqRel);
        }
        Ok((*free_count_ptr).load(Ordering::Acquire) as u32)
    }
}

/// Return the current free KV lease count from shared memory.
#[pyfunction]
#[pyo3(signature = (buf))]
fn kv_lease_free_count(py: Python<'_>, buf: PyBuffer<u8>) -> PyResult<u32> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    unsafe {
        validate_lease_buffer(ptr, buf_len)?;
        let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;
        Ok((*free_count_ptr).load(Ordering::Acquire) as u32)
    }
}

unsafe fn acquire_lease_blocks(
    ptr: *mut u8,
    total_blocks: u32,
    preferred_blocks: &[u32],
    count: u32,
    allow_partial: bool,
    strict_preferred: bool,
    owner_hash: u64,
) -> PyResult<Vec<(u32, u32)>> {
    let mut acquired: Vec<(u32, u32)> = Vec::with_capacity(count as usize);

    for block_id in preferred_blocks.iter().copied() {
        if acquired.len() >= count as usize {
            break;
        }
        if acquired.iter().any(|(existing, _)| *existing == block_id) {
            continue;
        }
        if let Some(lease) = try_acquire_lease_block(ptr, total_blocks, block_id, owner_hash) {
            acquired.push(lease);
        }
    }

    if !strict_preferred && acquired.len() < count as usize {
        for block_id in 0..total_blocks {
            if acquired.len() >= count as usize {
                break;
            }
            if preferred_blocks.contains(&block_id) {
                continue;
            }
            if let Some(lease) = try_acquire_lease_block(ptr, total_blocks, block_id, owner_hash) {
                acquired.push(lease);
            }
        }
    }

    if acquired.len() < count as usize && !allow_partial {
        release_acquired_lease_blocks(ptr, &acquired);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "could not acquire {} KV leases from shared memory, acquired {}",
            count,
            acquired.len()
        )));
    }
    Ok(acquired)
}

/// Atomically acquire KV block leases from shared memory.
#[pyfunction]
#[pyo3(signature = (
    buf,
    preferred_blocks,
    count,
    allow_partial = false,
    strict_preferred = false,
    owner_hash = 0,
))]
fn kv_lease_acquire(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    preferred_blocks: Vec<u32>,
    count: u32,
    allow_partial: bool,
    strict_preferred: bool,
    owner_hash: u64,
) -> PyResult<Vec<(u32, u32)>> {
    let _ = py;
    if count == 0 {
        return Ok(Vec::new());
    }
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();

    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let _mutation = enter_lease_mutation(ptr)?;
        acquire_lease_blocks(
            ptr,
            total_blocks,
            &preferred_blocks,
            count,
            allow_partial,
            strict_preferred,
            owner_hash,
        )
    }
}

/// Mark leased KV blocks as sealed.
#[pyfunction]
#[pyo3(signature = (buf, block_ids, generations))]
fn kv_lease_seal(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    block_ids: Vec<u32>,
    generations: Vec<u32>,
) -> PyResult<u32> {
    let _ = py;
    if block_ids.len() != generations.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_ids and generations length mismatch",
        ));
    }
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let mut sealed = 0u32;
    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let _mutation = enter_lease_mutation(ptr)?;
        for (block_id, generation) in block_ids.into_iter().zip(generations.into_iter()) {
            if block_id >= total_blocks {
                continue;
            }
            let base = lease_record_off(block_id);
            let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
            if (*generation_ptr).load(Ordering::Acquire) != generation {
                continue;
            }
            let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
            // Lock the record in TRANSITION, then re-validate the generation
            // before publishing SEALED. A fenced (stale) owner must not seal a
            // block that changed hands between the pre-CAS generation check and
            // the state transition: e.g. the new owner adopted it (bumping the
            // generation) and the stale seal would otherwise mark unwritten
            // bytes complete. Mirrors release_acquired_lease_blocks' post-CAS
            // generation recheck under the TRANSITION lock.
            if (*state_ptr)
                .compare_exchange(
                    LEASE_STATE_LEASED,
                    LEASE_STATE_TRANSITION,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                if (*generation_ptr).load(Ordering::Acquire) != generation {
                    (*state_ptr).store(LEASE_STATE_LEASED, Ordering::Release);
                } else {
                    (*state_ptr).store(LEASE_STATE_SEALED, Ordering::Release);
                    sealed = sealed.wrapping_add(1);
                }
            }
        }
    }
    Ok(sealed)
}

/// Atomically transfer sealed or leased blocks to a new owner.
///
/// Every requested record is first locked in TRANSITION with its old generation
/// intact. If any lock or generation check fails, all records are restored. The
/// successful path changes owner and generation without ever making bytes FREE.
#[pyfunction]
#[pyo3(signature = (buf, block_ids, generations, owner_hash = 0))]
fn kv_lease_adopt(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    block_ids: Vec<u32>,
    generations: Vec<u32>,
    owner_hash: u64,
) -> PyResult<Option<Vec<(u32, u32)>>> {
    let _ = py;
    if block_ids.len() != generations.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_ids and generations length mismatch",
        ));
    }
    if block_ids
        .iter()
        .enumerate()
        .any(|(index, block_id)| block_ids[..index].contains(block_id))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "duplicate block_ids are not allowed",
        ));
    }
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let _mutation = enter_lease_mutation(ptr)?;
        let mut locked: Vec<(u32, u32, u64, u32)> = Vec::with_capacity(block_ids.len());
        for (block_id, generation) in block_ids.iter().copied().zip(generations.iter().copied()) {
            if block_id >= total_blocks {
                for (id, _generation, _owner, state) in locked.iter().copied() {
                    let state_ptr = ptr.add(lease_record_off(id) + LR_STATE) as *const AtomicU32;
                    (*state_ptr).store(state, Ordering::Release);
                }
                return Ok(None);
            }
            let base = lease_record_off(block_id);
            let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
            if (*generation_ptr).load(Ordering::Acquire) != generation {
                for (id, _generation, _owner, state) in locked.iter().copied() {
                    let state_ptr = ptr.add(lease_record_off(id) + LR_STATE) as *const AtomicU32;
                    (*state_ptr).store(state, Ordering::Release);
                }
                return Ok(None);
            }
            let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
            let mut state = (*state_ptr).load(Ordering::Acquire);
            let mut acquired = false;
            while state == LEASE_STATE_LEASED || state == LEASE_STATE_SEALED {
                match (*state_ptr).compare_exchange(
                    state,
                    LEASE_STATE_TRANSITION,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        acquired = true;
                        break;
                    }
                    Err(observed) => state = observed,
                }
            }
            if !acquired || (*generation_ptr).load(Ordering::Acquire) != generation {
                if acquired {
                    (*state_ptr).store(state, Ordering::Release);
                }
                for (id, _generation, _owner, old_state) in locked.iter().copied() {
                    let old_state_ptr =
                        ptr.add(lease_record_off(id) + LR_STATE) as *const AtomicU32;
                    (*old_state_ptr).store(old_state, Ordering::Release);
                }
                return Ok(None);
            }
            let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
            locked.push((
                block_id,
                generation,
                (*owner_ptr).load(Ordering::Acquire),
                state,
            ));
        }

        let mut adopted = Vec::with_capacity(locked.len());
        for (block_id, generation, _old_owner, _old_state) in locked {
            let base = lease_record_off(block_id);
            let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
            let new_generation = generation.wrapping_add(1);
            (*generation_ptr).store(new_generation, Ordering::Release);
            let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
            (*owner_ptr).store(owner_hash, Ordering::Release);
            let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
            (*state_ptr).store(LEASE_STATE_LEASED, Ordering::Release);
            adopted.push((block_id, new_generation));
        }
        Ok(Some(adopted))
    }
}

/// Release KV block leases back to the shared-memory free pool.
#[pyfunction]
#[pyo3(signature = (buf, block_ids, generations))]
fn kv_lease_release(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    block_ids: Vec<u32>,
    generations: Vec<u32>,
) -> PyResult<u32> {
    let _ = py;
    if block_ids.len() != generations.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_ids and generations length mismatch",
        ));
    }
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let mut released = 0u32;
    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let _mutation = enter_lease_mutation(ptr)?;
        let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;
        for (block_id, generation) in block_ids.into_iter().zip(generations.into_iter()) {
            if block_id >= total_blocks {
                continue;
            }
            let base = lease_record_off(block_id);
            let generation_ptr = ptr.add(base + LR_GENERATION) as *const AtomicU32;
            if (*generation_ptr).load(Ordering::Acquire) != generation {
                continue;
            }
            let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
            let mut state = (*state_ptr).load(Ordering::Acquire);
            while state == LEASE_STATE_LEASED || state == LEASE_STATE_SEALED {
                match (*state_ptr).compare_exchange(
                    state,
                    LEASE_STATE_TRANSITION,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        if (*generation_ptr).load(Ordering::Acquire) != generation {
                            (*state_ptr).store(state, Ordering::Release);
                            break;
                        }
                        let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
                        (*owner_ptr).store(0, Ordering::Release);
                        (*free_count_ptr).fetch_add(1, Ordering::AcqRel);
                        (*state_ptr).store(LEASE_STATE_FREE, Ordering::Release);
                        released = released.wrapping_add(1);
                        break;
                    }
                    Err(observed) => state = observed,
                }
            }
        }
    }
    Ok(released)
}

unsafe fn reclaim_foreign_lease_blocks(
    ptr: *mut u8,
    buf_len: usize,
    owner_hash: u64,
    max_blocks: u32,
    protected_blocks: &[u32],
) -> PyResult<u32> {
    let mut released = 0u32;
    let total_blocks = validate_lease_buffer(ptr, buf_len)?;
    let _recovery = enter_lease_recovery(ptr)?;
    let free_count_ptr = ptr.add(L_FREE_COUNT) as *const AtomicU64;
    for block_id in 0..total_blocks {
        let base = lease_record_off(block_id);
        let owner_ptr = ptr.add(base + LR_OWNER_HASH) as *const AtomicU64;
        let observed_owner = (*owner_ptr).load(Ordering::Acquire);
        let state_ptr = ptr.add(base + LR_STATE) as *const AtomicU32;
        let mut state = (*state_ptr).load(Ordering::Acquire);
        // A transition belongs to a writer stopped at an arbitrary point
        // around the free-count update. Once that writer is fenced,
        // rolling the record forward to FREE is safe; the final full scan
        // reconstructs the exact count without guessing whether the
        // interrupted update happened.
        if state == LEASE_STATE_TRANSITION {
            // A protected (directory-advertised, recoverable) block that was
            // mid-transition when its writer died must NOT be rolled forward to
            // FREE -- doing so lets a later acquire reuse HBM the directory
            // still believes holds valid KV. Restore it to SEALED (the safe,
            // adoptable state) so lazy adoption can recover it. Only unprotected
            // interrupted transitions are freed.
            if protected_blocks.binary_search(&block_id).is_ok() {
                (*state_ptr).store(LEASE_STATE_SEALED, Ordering::Release);
                continue;
            }
            (*owner_ptr).store(0, Ordering::Release);
            (*state_ptr).store(LEASE_STATE_FREE, Ordering::Release);
            released = released.wrapping_add(1);
            continue;
        }
        if max_blocks > 0 && released >= max_blocks {
            continue;
        }
        if observed_owner == 0
            || observed_owner == owner_hash
            || protected_blocks.binary_search(&block_id).is_ok()
        {
            continue;
        }
        while state == LEASE_STATE_LEASED || state == LEASE_STATE_SEALED {
            match (*state_ptr).compare_exchange(
                state,
                LEASE_STATE_TRANSITION,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if (*owner_ptr).load(Ordering::Acquire) != observed_owner {
                        (*state_ptr).store(state, Ordering::Release);
                        break;
                    }
                    (*owner_ptr).store(0, Ordering::Release);
                    (*state_ptr).store(LEASE_STATE_FREE, Ordering::Release);
                    released = released.wrapping_add(1);
                    break;
                }
                Err(observed) => state = observed,
            }
        }
    }
    let mut exact_free = 0u64;
    for block_id in 0..total_blocks {
        let state_ptr = ptr.add(lease_record_off(block_id) + LR_STATE) as *const AtomicU32;
        if (*state_ptr).load(Ordering::Acquire) == LEASE_STATE_FREE {
            exact_free += 1;
        }
    }
    (*free_count_ptr).store(exact_free, Ordering::Release);
    Ok(released)
}

/// Reclaim all leased/sealed KV blocks not owned by `owner_hash`.
#[pyfunction]
#[pyo3(signature = (buf, owner_hash = 0, max_blocks = 0))]
fn kv_lease_reclaim_foreign(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    owner_hash: u64,
    max_blocks: u32,
) -> PyResult<u32> {
    let _ = py;
    unsafe {
        reclaim_foreign_lease_blocks(
            buf.buf_ptr() as *mut u8,
            buf.len_bytes(),
            owner_hash,
            max_blocks,
            &[],
        )
    }
}

/// Reclaim foreign blocks except READY HBM slots protected by the directory.
#[pyfunction]
#[pyo3(signature = (buf, protected_blocks, owner_hash = 0, max_blocks = 0))]
fn kv_lease_reclaim_foreign_except(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    mut protected_blocks: Vec<u32>,
    owner_hash: u64,
    max_blocks: u32,
) -> PyResult<u32> {
    let _ = py;
    protected_blocks.sort_unstable();
    protected_blocks.dedup();
    unsafe {
        reclaim_foreign_lease_blocks(
            buf.buf_ptr() as *mut u8,
            buf.len_bytes(),
            owner_hash,
            max_blocks,
            &protected_blocks,
        )
    }
}

// IEEE CRC32 polynomial in reversed representation, matching zlib.
#[pymodule]
fn gms_rust_ring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kv_lease_init, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_free_count, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_acquire, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_seal, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_adopt, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_release, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_reclaim_foreign, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_reclaim_foreign_except, m)?)?;
    m.add("KV_LEASE_HEADER_SIZE", LEASE_HEADER_SIZE)?;
    m.add("KV_LEASE_RECORD_SIZE", LEASE_RECORD_SIZE)?;
    m.add("KV_LEASE_MAGIC", LEASE_MAGIC)?;
    m.add("KV_LEASE_VERSION", LEASE_VERSION)?;
    Ok(())
}
