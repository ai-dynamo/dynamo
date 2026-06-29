// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native hot-path SPSC ring push/pop for GMS RestoreRing.
//!
//! Mirrors the on-disk layout defined in `common/restore_ring.py` exactly.
//! Engine/daemon attach the mmap'd file via the Python wrapper, then
//! pass the buffer pointer + capacity into these native push/pop calls
//! to avoid the ~9 µs/op Python overhead (struct.pack_into × 7 +
//! function-call overhead).
//!
//! Layout (little-endian, must match common/restore_ring.py):
//!
//!   Header (64B):
//!     magic        : u32   = 0x4752_5347  ("GMSR" little-endian read)
//!     version      : u32   = 1
//!     capacity     : u32   record count (power of 2)
//!     record_size  : u32   = 512
//!     head_seq     : u64   producer-only
//!     tail_seq     : u64   consumer-only
//!     drops        : u64   producer-only
//!     padding      : 16B
//!
//!   Record (512B):
//!     seq            : u64    (non-zero ⇒ ready for consumer)
//!     op_kind        : u8
//!     flags          : u8
//!     _pad           : u16
//!     counter_slot   : u32
//!     counter_target : u32
//!     n_blocks       : u32
//!     _pad2          : u64
//!     src_engine_id  : char[48]   (NUL-padded)
//!     block_pairs[54]: each (u32 src_blk, u32 dest_blk)

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const HEADER_SIZE: usize = 64;
const RECORD_SIZE: usize = 512;
const ENGINE_ID_MAX_LEN: usize = 48;
const MAX_BLOCK_PAIRS_PER_RECORD: usize = 54;

const OFF_HEAD_SEQ: usize = 16;
const OFF_TAIL_SEQ: usize = 24;
const OFF_DROPS: usize = 32;

const R_SEQ: usize = 0;
const R_OP: usize = 8;
const R_FLAGS: usize = 9;
const R_COUNTER_SLOT: usize = 12;
const R_COUNTER_TARGET: usize = 16;
const R_N_BLOCKS: usize = 20;
const R_ENGINE_ID: usize = 32;
const R_BLOCK_PAIRS: usize = 80;
const BLOCK_PAIR_STRIDE: usize = 8;

const OP_RESTORE_CHUNK: u8 = 1;

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
        unsafe {
            (*self.active).fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Register a record mutation unless post-fence recovery owns the namespace.
///
/// Mutations remain fully concurrent: this is an activity count, not a mutex.
/// A dead process may strand its count, which is harmless until the fenced
/// successor atomically replaces it with the recovery barrier.
unsafe fn enter_lease_mutation(ptr: *mut u8) -> LeaseMutationGuard {
    let active = ptr.add(L_ACTIVE_MUTATIONS) as *const AtomicU64;
    loop {
        let observed = (*active).load(Ordering::Acquire);
        if observed == LEASE_RECOVERY_BARRIER {
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
            return LeaseMutationGuard { active };
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
/// This deliberately replaces a possibly stranded activity count. Calling it
/// before the old writer is externally fenced would allow an old mutation to
/// resume and is therefore outside the API contract.
unsafe fn enter_lease_recovery(ptr: *mut u8) -> PyResult<LeaseRecoveryGuard> {
    let active = ptr.add(L_ACTIVE_MUTATIONS) as *const AtomicU64;
    let previous = (*active).swap(LEASE_RECOVERY_BARRIER, Ordering::AcqRel);
    if previous == LEASE_RECOVERY_BARRIER {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "KV lease recovery is already in progress",
        ));
    }
    Ok(LeaseRecoveryGuard { active })
}

#[inline(always)]
unsafe fn read_u32(ptr: *const u8, off: usize) -> u32 {
    std::ptr::read_unaligned(ptr.add(off) as *const u32)
}

#[inline(always)]
unsafe fn read_u64(ptr: *const u8, off: usize) -> u64 {
    std::ptr::read_unaligned(ptr.add(off) as *const u64)
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

#[inline(always)]
fn reservation_applies_to_owner(
    reserved_blocks: u32,
    reserved_owner_hash: u64,
    owner_hash: u64,
) -> bool {
    reserved_blocks > 0 && (reserved_owner_hash == 0 || reserved_owner_hash != owner_hash)
}

unsafe fn load_lease_reservation(ptr: *const u8) -> (u32, u64, u64) {
    let epoch_ptr = ptr.add(L_RESERVATION_EPOCH) as *const AtomicU64;
    loop {
        let before = (*epoch_ptr).load(Ordering::Acquire);
        if before & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        let reserved_blocks =
            (*(ptr.add(L_RESERVED_BLOCKS) as *const AtomicU32)).load(Ordering::Acquire);
        let reserved_owner_hash =
            (*(ptr.add(L_RESERVED_OWNER_HASH) as *const AtomicU64)).load(Ordering::Acquire);
        let after = (*epoch_ptr).load(Ordering::Acquire);
        if before == after {
            return (reserved_blocks, reserved_owner_hash, after);
        }
    }
}

/// Push one chunk-restore record into the ring.
///
/// Arguments:
///   - `buf`: mutable bytes-like object backing the mmap. Must be at
///     least HEADER_SIZE + capacity*RECORD_SIZE bytes.
///   - `capacity`: ring capacity (power of 2).
///   - `src_engine_id_bytes`: UTF-8 engine_id, NUL-padded externally
///     OR length must be ≤ 47.
///   - `src_blocks` / `dest_blocks`: same-length lists of u32 ids.
///   - `counter_slot`, `counter_target`, `flags`: scalar fields.
///
/// Returns True on success; False if the ring was full (drops counter
/// auto-incremented).
#[pyfunction]
#[pyo3(signature = (
    buf, capacity, src_engine_id_bytes, src_blocks, dest_blocks,
    counter_slot, counter_target, flags,
))]
fn push_record(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    capacity: u32,
    src_engine_id_bytes: &[u8],
    src_blocks: Vec<u32>,
    dest_blocks: Vec<u32>,
    counter_slot: u32,
    counter_target: u32,
    flags: u8,
) -> PyResult<bool> {
    let _ = py;
    if !buf.readonly() || true {
        // PyBuffer doesn't expose mut directly; we'll use the raw pointer.
    }
    if src_blocks.len() != dest_blocks.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "src_blocks and dest_blocks length mismatch",
        ));
    }
    let n_blocks = src_blocks.len();
    if n_blocks > MAX_BLOCK_PAIRS_PER_RECORD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "too many block pairs: {} > {}",
            n_blocks, MAX_BLOCK_PAIRS_PER_RECORD
        )));
    }
    if src_engine_id_bytes.len() >= ENGINE_ID_MAX_LEN {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "engine_id too long: {} >= {}",
            src_engine_id_bytes.len(),
            ENGINE_ID_MAX_LEN
        )));
    }
    let mask = (capacity - 1) as u64;

    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let need = HEADER_SIZE + (capacity as usize) * RECORD_SIZE;
    if buf_len < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "buffer too small: {} < {}",
            buf_len, need
        )));
    }

    // Single-producer: head bump is naturally atomic under GIL.
    unsafe {
        let head = read_u64(ptr, OFF_HEAD_SEQ);
        let tail = read_u64(ptr, OFF_TAIL_SEQ);
        if head.wrapping_sub(tail) >= capacity as u64 {
            // Full; bump drops + return False.
            let drops = read_u64(ptr, OFF_DROPS);
            write_u64(ptr, OFF_DROPS, drops.wrapping_add(1));
            return Ok(false);
        }

        let slot_idx = (head & mask) as usize;
        let base = HEADER_SIZE + slot_idx * RECORD_SIZE;

        // Zero seq first so consumer never observes half-written.
        write_u64(ptr, base + R_SEQ, 0);

        // Scalar fields
        *(ptr.add(base + R_OP)) = OP_RESTORE_CHUNK;
        *(ptr.add(base + R_FLAGS)) = flags;
        write_u32(ptr, base + R_COUNTER_SLOT, counter_slot);
        write_u32(ptr, base + R_COUNTER_TARGET, counter_target);
        write_u32(ptr, base + R_N_BLOCKS, n_blocks as u32);

        // engine_id (NUL-padded). Zero the field first.
        std::ptr::write_bytes(ptr.add(base + R_ENGINE_ID), 0, ENGINE_ID_MAX_LEN);
        std::ptr::copy_nonoverlapping(
            src_engine_id_bytes.as_ptr(),
            ptr.add(base + R_ENGINE_ID),
            src_engine_id_bytes.len(),
        );

        // Block pairs (zero out unused tail)
        let pairs_base = base + R_BLOCK_PAIRS;
        for i in 0..n_blocks {
            write_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE, src_blocks[i]);
            write_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE + 4, dest_blocks[i]);
        }
        for i in n_blocks..MAX_BLOCK_PAIRS_PER_RECORD {
            write_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE, 0);
            write_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE + 4, 0);
        }

        // Release: publish seq AFTER payload (x86-TSO; under GIL).
        // Use atomic store with Release ordering for memory ordering
        // guarantee.
        let seq_ptr = ptr.add(base + R_SEQ) as *const AtomicU64;
        (*seq_ptr).store(head.wrapping_add(1), Ordering::Release);

        // Publish head LAST.
        let head_ptr = ptr.add(OFF_HEAD_SEQ) as *const AtomicU64;
        (*head_ptr).store(head.wrapping_add(1), Ordering::Release);
    }
    Ok(true)
}

/// Pop one ready record from the ring. Returns None if empty.
///
/// Returns a tuple: (op, flags, counter_slot, counter_target,
/// engine_id_bytes, [(src_blk, dest_blk), ...]).
#[pyfunction]
#[pyo3(signature = (buf, capacity))]
fn try_pop_record(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    capacity: u32,
) -> PyResult<Option<(u8, u8, u32, u32, Py<PyBytes>, Vec<(u32, u32)>)>> {
    let mask = (capacity - 1) as u64;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let need = HEADER_SIZE + (capacity as usize) * RECORD_SIZE;
    if buf_len < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "buffer too small: {} < {}",
            buf_len, need
        )));
    }

    unsafe {
        let tail_ptr = ptr.add(OFF_TAIL_SEQ) as *const AtomicU64;
        let head_ptr = ptr.add(OFF_HEAD_SEQ) as *const AtomicU64;
        let tail = (*tail_ptr).load(Ordering::Acquire);
        let head = (*head_ptr).load(Ordering::Acquire);
        if tail >= head {
            return Ok(None);
        }
        let slot_idx = (tail & mask) as usize;
        let base = HEADER_SIZE + slot_idx * RECORD_SIZE;
        let seq_ptr = ptr.add(base + R_SEQ) as *const AtomicU64;
        let seq = (*seq_ptr).load(Ordering::Acquire);
        if seq != tail.wrapping_add(1) {
            return Ok(None); // producer hasn't released yet
        }

        let op = *(ptr.add(base + R_OP));
        let flags = *(ptr.add(base + R_FLAGS));
        let counter_slot = read_u32(ptr, base + R_COUNTER_SLOT);
        let counter_target = read_u32(ptr, base + R_COUNTER_TARGET);
        let n_blocks_raw = read_u32(ptr, base + R_N_BLOCKS) as usize;
        let n_blocks = n_blocks_raw.min(MAX_BLOCK_PAIRS_PER_RECORD);

        // engine_id: find NUL or end of field
        let eid_slice = std::slice::from_raw_parts(ptr.add(base + R_ENGINE_ID), ENGINE_ID_MAX_LEN);
        let eid_len = eid_slice
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(ENGINE_ID_MAX_LEN);
        let eid_bytes = PyBytes::new_bound(py, &eid_slice[..eid_len]);

        let mut pairs = Vec::with_capacity(n_blocks);
        let pairs_base = base + R_BLOCK_PAIRS;
        for i in 0..n_blocks {
            let src = read_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE);
            let dest = read_u32(ptr, pairs_base + i * BLOCK_PAIR_STRIDE + 4);
            pairs.push((src, dest));
        }

        // Advance tail AFTER reading payload.
        (*tail_ptr).store(tail.wrapping_add(1), Ordering::Release);

        Ok(Some((
            op,
            flags,
            counter_slot,
            counter_target,
            eid_bytes.unbind(),
            pairs,
        )))
    }
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

/// Return free KV leases visible to an owner after active reservation headroom.
#[pyfunction]
#[pyo3(signature = (buf, owner_hash = 0))]
fn kv_lease_free_count_for_owner(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    owner_hash: u64,
) -> PyResult<u32> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    unsafe {
        validate_lease_buffer(ptr, buf_len)?;
        let free = (*(ptr.add(L_FREE_COUNT) as *const AtomicU64)).load(Ordering::Acquire);
        let (reserved_blocks, reserved_owner_hash, _reservation_epoch) =
            load_lease_reservation(ptr);
        if reservation_applies_to_owner(reserved_blocks, reserved_owner_hash, owner_hash) {
            Ok(free.saturating_sub(reserved_blocks as u64) as u32)
        } else {
            Ok(free as u32)
        }
    }
}

/// Store active reservation headroom in the lease shared-memory header.
#[pyfunction]
#[pyo3(signature = (buf, reserved_blocks, reserved_owner_hash = 0))]
fn kv_lease_set_reservation(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    reserved_blocks: u32,
    reserved_owner_hash: u64,
) -> PyResult<(u32, u64, u64)> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let bounded = reserved_blocks.min(total_blocks);
        let epoch_ptr = ptr.add(L_RESERVATION_EPOCH) as *const AtomicU64;
        (*epoch_ptr).fetch_add(1, Ordering::AcqRel);
        (*(ptr.add(L_RESERVED_OWNER_HASH) as *const AtomicU64))
            .store(reserved_owner_hash, Ordering::Release);
        (*(ptr.add(L_RESERVED_BLOCKS) as *const AtomicU32)).store(bounded, Ordering::Release);
        let epoch = (*epoch_ptr).fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        Ok((bounded, reserved_owner_hash, epoch))
    }
}

/// Read active reservation headroom from the lease shared-memory header.
#[pyfunction]
#[pyo3(signature = (buf))]
fn kv_lease_reservation(py: Python<'_>, buf: PyBuffer<u8>) -> PyResult<(u32, u64, u64)> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    unsafe {
        validate_lease_buffer(ptr, buf_len)?;
        Ok(load_lease_reservation(ptr))
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
            if preferred_blocks
                .iter()
                .any(|preferred| *preferred == block_id)
            {
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
        let _mutation = enter_lease_mutation(ptr);
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

/// Acquire KV block leases only if no reservation applies to this owner.
///
/// Returns None when a transition reservation is active or becomes active
/// during acquisition; callers should retry under the reservation lock.
#[pyfunction]
#[pyo3(signature = (
    buf,
    preferred_blocks,
    count,
    allow_partial = false,
    strict_preferred = false,
    owner_hash = 0,
))]
fn kv_lease_acquire_lockless_if_unreserved(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    preferred_blocks: Vec<u32>,
    count: u32,
    allow_partial: bool,
    strict_preferred: bool,
    owner_hash: u64,
) -> PyResult<Option<Vec<(u32, u32)>>> {
    let _ = py;
    if count == 0 {
        return Ok(Some(Vec::new()));
    }
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();

    unsafe {
        let total_blocks = validate_lease_buffer(ptr, buf_len)?;
        let _mutation = enter_lease_mutation(ptr);
        let (before_reserved_blocks, before_reserved_owner_hash, before_epoch) =
            load_lease_reservation(ptr);
        if reservation_applies_to_owner(
            before_reserved_blocks,
            before_reserved_owner_hash,
            owner_hash,
        ) {
            return Ok(None);
        }

        let acquired = acquire_lease_blocks(
            ptr,
            total_blocks,
            &preferred_blocks,
            count,
            allow_partial,
            strict_preferred,
            owner_hash,
        )?;

        let (after_reserved_blocks, after_reserved_owner_hash, after_epoch) =
            load_lease_reservation(ptr);
        if after_epoch != before_epoch
            && reservation_applies_to_owner(
                after_reserved_blocks,
                after_reserved_owner_hash,
                owner_hash,
            )
        {
            release_acquired_lease_blocks(ptr, &acquired);
            return Ok(None);
        }

        Ok(Some(acquired))
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
        let _mutation = enter_lease_mutation(ptr);
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
            if (*state_ptr)
                .compare_exchange(
                    LEASE_STATE_LEASED,
                    LEASE_STATE_SEALED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                sealed = sealed.wrapping_add(1);
            }
        }
    }
    Ok(sealed)
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
        let _mutation = enter_lease_mutation(ptr);
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

/// Reclaim all leased/sealed KV blocks not owned by `owner_hash`.
///
/// This is a post-fence failover primitive. Callers must only invoke it after
/// the old writer has been fenced, for example after a shadow acquires the
/// failover lock. It deliberately ignores generation checks because the old
/// owner is dead and any remaining leased/sealed records are orphaned.
#[pyfunction]
#[pyo3(signature = (buf, owner_hash = 0, max_blocks = 0))]
fn kv_lease_reclaim_foreign(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    owner_hash: u64,
    max_blocks: u32,
) -> PyResult<u32> {
    let _ = py;
    let ptr = buf.buf_ptr() as *mut u8;
    let buf_len = buf.len_bytes();
    let mut released = 0u32;
    unsafe {
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
                (*owner_ptr).store(0, Ordering::Release);
                (*state_ptr).store(LEASE_STATE_FREE, Ordering::Release);
                released = released.wrapping_add(1);
                continue;
            }
            if max_blocks > 0 && released >= max_blocks {
                continue;
            }
            if observed_owner == 0 || observed_owner == owner_hash {
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
    }
    Ok(released)
}

// IEEE CRC32 polynomial in reversed representation, matching zlib.
const CRC32_POLY: u32 = 0xEDB8_8320;

#[inline]
fn gf2_mat_mul(mat: &[u32; 32], mut vec: u32) -> u32 {
    let mut out = 0;
    let mut i = 0;
    while vec != 0 {
        if vec & 1 != 0 {
            out ^= mat[i];
        }
        vec >>= 1;
        i += 1;
    }
    out
}

#[inline]
fn gf2_mat_square(square: &mut [u32; 32], mat: &[u32; 32]) {
    for i in 0..32 {
        square[i] = gf2_mat_mul(mat, mat[i]);
    }
}

/// Return crc32(A || B) from crc32(A), crc32(B), and len(B).
fn crc32_combine_one(crc1: u32, crc2: u32, len2: u64) -> u32 {
    if len2 == 0 {
        return crc1;
    }
    let mut odd = [0u32; 32];
    let mut even = [0u32; 32];
    odd[0] = CRC32_POLY;
    for (n, value) in odd.iter_mut().enumerate().skip(1) {
        *value = 1u32 << (n - 1);
    }
    gf2_mat_square(&mut even, &odd);
    gf2_mat_square(&mut odd, &even);
    let mut crc = crc1;
    let mut len = len2;
    loop {
        gf2_mat_square(&mut even, &odd);
        if len & 1 != 0 {
            crc = gf2_mat_mul(&even, crc);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
        gf2_mat_square(&mut odd, &even);
        if len & 1 != 0 {
            crc = gf2_mat_mul(&odd, crc);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
    }
    crc ^ crc2
}

/// Combine ordered GPU-computed chunk CRCs without copying payload bytes.
#[pyfunction]
#[pyo3(signature = (crcs, chunk_bytes, total_bytes))]
fn crc32_combine_chunks(crcs: Vec<u32>, chunk_bytes: u64, total_bytes: u64) -> u32 {
    if total_bytes == 0 || crcs.is_empty() {
        return 0;
    }
    let mut combined = 0;
    let mut remaining = total_bytes;
    for crc in crcs {
        let this_len = remaining.min(chunk_bytes);
        if this_len == 0 {
            break;
        }
        combined = crc32_combine_one(combined, crc, this_len);
        remaining -= this_len;
    }
    combined
}

#[pymodule]
fn gms_rust_ring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(push_record, m)?)?;
    m.add_function(wrap_pyfunction!(try_pop_record, m)?)?;
    m.add_function(wrap_pyfunction!(crc32_combine_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_init, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_free_count, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_free_count_for_owner, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_set_reservation, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_reservation, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_acquire, m)?)?;
    m.add_function(wrap_pyfunction!(
        kv_lease_acquire_lockless_if_unreserved,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kv_lease_seal, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_release, m)?)?;
    m.add_function(wrap_pyfunction!(kv_lease_reclaim_foreign, m)?)?;
    m.add("RECORD_SIZE", RECORD_SIZE)?;
    m.add("HEADER_SIZE", HEADER_SIZE)?;
    m.add("MAX_BLOCK_PAIRS", MAX_BLOCK_PAIRS_PER_RECORD)?;
    m.add("MAX_BLOCK_PAIRS_PER_RECORD", MAX_BLOCK_PAIRS_PER_RECORD)?;
    m.add("ENGINE_ID_MAX_LEN", ENGINE_ID_MAX_LEN)?;
    m.add("KV_LEASE_HEADER_SIZE", LEASE_HEADER_SIZE)?;
    m.add("KV_LEASE_RECORD_SIZE", LEASE_RECORD_SIZE)?;
    m.add("KV_LEASE_MAGIC", LEASE_MAGIC)?;
    m.add("KV_LEASE_VERSION", LEASE_VERSION)?;
    Ok(())
}
