// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native hot-path SPSC ring push/pop for gms_kv_ring restore ring.
//!
//! Mirrors the on-disk layout defined in `common/restore_ring.py`
//! exactly. Engine and daemon both attach the mmap'd file via the
//! Python wrapper, then pass the buffer pointer + capacity into these
//! native push/pop calls — bypasses the Python `struct.pack_into ×7`
//! overhead (~9 µs/op) in favor of unaligned writes (~1 µs/op).
//!
//! Layout (must match common/restore_ring.py bit-for-bit):
//!
//!   Header (64B):
//!     magic        : u32   = 0x5253_4D47  ("GMSR" little-endian read)
//!     version      : u32   = 1
//!     capacity     : u32   record count (power of 2)
//!     record_size  : u32   = 512
//!     head_seq     : u64
//!     tail_seq     : u64
//!     drops        : u64
//!     padding      : 16B
//!
//!   Record (512B):
//!     seq            : u64
//!     op_kind        : u8     = 1 (RESTORE_CHUNK)
//!     flags          : u8
//!     _pad           : u16
//!     counter_slot   : u32
//!     counter_target : u32
//!     n_pairs        : u32
//!     _pad2          : u64
//!     src_engine_id  : char[48]  NUL-padded
//!     pairs[54]      : each (u32 src_block, u32 dest_block)

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::{AtomicU64, Ordering};

const HEADER_SIZE: usize = 64;
const RECORD_SIZE: usize = 512;
const ENGINE_ID_MAX_LEN: usize = 48;
const MAX_BLOCK_PAIRS: usize = 54;

const OFF_HEAD_SEQ: usize = 16;
const OFF_TAIL_SEQ: usize = 24;
const OFF_DROPS: usize = 32;

const R_SEQ: usize = 0;
const R_OP: usize = 8;
const R_FLAGS: usize = 9;
const R_COUNTER_SLOT: usize = 12;
const R_COUNTER_TARGET: usize = 16;
const R_N_PAIRS: usize = 20;
const R_ENGINE_ID: usize = 32;
const R_PAIRS: usize = 80;
const PAIR_STRIDE: usize = 8;

const OP_RESTORE_CHUNK: u8 = 1;

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

/// Push one chunk-restore record into the ring.
///
/// Args:
///   buf: mutable bytes-like backing the mmap'd ring file.
///   capacity: ring capacity (power of 2).
///   src_engine_id_bytes: UTF-8 engine_id, length < 48.
///   src_blocks / dest_blocks: same-length lists of u32.
///   counter_slot, counter_target, flags: scalar fields.
///
/// Returns True on success; False if ring was full (drops bumped).
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
    if src_blocks.len() != dest_blocks.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "src_blocks and dest_blocks length mismatch",
        ));
    }
    let n_pairs = src_blocks.len();
    if n_pairs > MAX_BLOCK_PAIRS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "too many block pairs: {} > {}",
            n_pairs, MAX_BLOCK_PAIRS
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
    let need = HEADER_SIZE + (capacity as usize) * RECORD_SIZE;
    if buf.len_bytes() < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "buffer too small: {} < {}",
            buf.len_bytes(),
            need
        )));
    }

    unsafe {
        let head = read_u64(ptr, OFF_HEAD_SEQ);
        let tail = read_u64(ptr, OFF_TAIL_SEQ);
        if head.wrapping_sub(tail) >= capacity as u64 {
            let drops = read_u64(ptr, OFF_DROPS);
            write_u64(ptr, OFF_DROPS, drops.wrapping_add(1));
            return Ok(false);
        }

        let slot_idx = (head & mask) as usize;
        let base = HEADER_SIZE + slot_idx * RECORD_SIZE;

        // Zero seq first so a consumer never observes half-written record.
        write_u64(ptr, base + R_SEQ, 0);

        *(ptr.add(base + R_OP)) = OP_RESTORE_CHUNK;
        *(ptr.add(base + R_FLAGS)) = flags;
        write_u32(ptr, base + R_COUNTER_SLOT, counter_slot);
        write_u32(ptr, base + R_COUNTER_TARGET, counter_target);
        write_u32(ptr, base + R_N_PAIRS, n_pairs as u32);

        // engine_id (NUL-padded)
        std::ptr::write_bytes(ptr.add(base + R_ENGINE_ID), 0, ENGINE_ID_MAX_LEN);
        std::ptr::copy_nonoverlapping(
            src_engine_id_bytes.as_ptr(),
            ptr.add(base + R_ENGINE_ID),
            src_engine_id_bytes.len(),
        );

        // Block pairs (zero out unused tail)
        let pairs_base = base + R_PAIRS;
        for i in 0..n_pairs {
            write_u32(ptr, pairs_base + i * PAIR_STRIDE, src_blocks[i]);
            write_u32(ptr, pairs_base + i * PAIR_STRIDE + 4, dest_blocks[i]);
        }
        for i in n_pairs..MAX_BLOCK_PAIRS {
            write_u32(ptr, pairs_base + i * PAIR_STRIDE, 0);
            write_u32(ptr, pairs_base + i * PAIR_STRIDE + 4, 0);
        }

        // Release: seq AFTER payload. Atomic store on x86 is naturally
        // ordered, but we use Release explicitly to be safe across archs.
        let seq_ptr = ptr.add(base + R_SEQ) as *const AtomicU64;
        (*seq_ptr).store(head.wrapping_add(1), Ordering::Release);

        // Publish head LAST.
        let head_ptr = ptr.add(OFF_HEAD_SEQ) as *const AtomicU64;
        (*head_ptr).store(head.wrapping_add(1), Ordering::Release);
    }
    Ok(true)
}

/// Pop one ready record. Returns None if ring is empty.
///
/// Returns a tuple: (op, flags, counter_slot, counter_target,
///                   engine_id_bytes, [(src_blk, dest_blk), ...]).
#[pyfunction]
#[pyo3(signature = (buf, capacity))]
fn try_pop_record(
    py: Python<'_>,
    buf: PyBuffer<u8>,
    capacity: u32,
) -> PyResult<Option<(u8, u8, u32, u32, Py<PyBytes>, Vec<(u32, u32)>)>> {
    let mask = (capacity - 1) as u64;
    let ptr = buf.buf_ptr() as *mut u8;
    let need = HEADER_SIZE + (capacity as usize) * RECORD_SIZE;
    if buf.len_bytes() < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "buffer too small: {} < {}",
            buf.len_bytes(),
            need
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
            return Ok(None);
        }

        let op = *(ptr.add(base + R_OP));
        let flags = *(ptr.add(base + R_FLAGS));
        let counter_slot = read_u32(ptr, base + R_COUNTER_SLOT);
        let counter_target = read_u32(ptr, base + R_COUNTER_TARGET);
        let n_pairs_raw = read_u32(ptr, base + R_N_PAIRS) as usize;
        let n_pairs = n_pairs_raw.min(MAX_BLOCK_PAIRS);

        let eid_slice = std::slice::from_raw_parts(
            ptr.add(base + R_ENGINE_ID),
            ENGINE_ID_MAX_LEN,
        );
        let eid_len = eid_slice
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(ENGINE_ID_MAX_LEN);
        let eid_bytes = PyBytes::new_bound(py, &eid_slice[..eid_len]);

        let mut pairs = Vec::with_capacity(n_pairs);
        let pairs_base = base + R_PAIRS;
        for i in 0..n_pairs {
            let src = read_u32(ptr, pairs_base + i * PAIR_STRIDE);
            let dest = read_u32(ptr, pairs_base + i * PAIR_STRIDE + 4);
            pairs.push((src, dest));
        }

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

// CRC32 polynomial (IEEE 802.3, reversed). Same as zlib / crc32fast.
const CRC32_POLY: u32 = 0xEDB88320;

// GF(2) polynomial matrix-vector multiply over u32. Each `mat` row
// is the image of basis polynomial x^i under the linear map being
// applied. Standard zlib crc32_combine helper.
#[inline]
fn gf2_mat_mul(mat: &[u32; 32], mut vec: u32) -> u32 {
    let mut out = 0u32;
    let mut i = 0usize;
    while vec != 0 && i < 32 {
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

/// Combine two CRC32 values (zlib-compatible): given
/// crc1 = crc32(buf_A) and crc2 = crc32(buf_B), return crc32(A ++ B).
/// `len2` is the byte length of B. Implementation mirrors zlib's
/// crc32_combine — repeated squaring of the polynomial "advance"
/// matrix.
fn crc32_combine_one(crc1: u32, crc2: u32, len2: u64) -> u32 {
    if len2 == 0 {
        return crc1;
    }
    let mut odd = [0u32; 32];
    let mut even = [0u32; 32];
    // odd-power matrix: shift right by 1 in CRC space (x * poly).
    odd[0] = CRC32_POLY;
    for n in 1..32 {
        odd[n] = 1u32 << (n - 1);
    }
    // even = odd^2 — the "shift by 2 bits" operator.
    gf2_mat_square(&mut even, &odd);
    // odd = even^2 — the "shift by 4 bits" operator.
    gf2_mat_square(&mut odd, &even);
    let mut crc = crc1;
    let mut len = len2;
    loop {
        // even is the "shift by 2^(k+1) bits" operator.
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

/// Combine a chain of chunk CRCs into the CRC of the concatenated
/// buffer. Used by the GPU CRC kernel's host-side reduction: kernel
/// produces one CRC per thread chunk; this combines them into the
/// final CRC32 of the whole buffer.
///
/// Args:
///   crcs: per-chunk CRCs, in concatenation order
///   chunk_bytes: byte length of each chunk except possibly the last
///   total_bytes: total byte length of the buffer (last chunk = total - chunk*N)
///
/// Returns the combined CRC32 (zlib-compatible).
#[pyfunction]
#[pyo3(signature = (crcs, chunk_bytes, total_bytes))]
fn crc32_combine_chunks(
    crcs: Vec<u32>,
    chunk_bytes: u64,
    total_bytes: u64,
) -> u32 {
    if total_bytes == 0 || crcs.is_empty() {
        return 0;
    }
    let mut combined = 0u32;
    let mut remaining = total_bytes;
    for &crc in &crcs {
        let this_len = remaining.min(chunk_bytes);
        if this_len == 0 {
            break;
        }
        combined = crc32_combine_one(combined, crc, this_len);
        remaining -= this_len;
    }
    combined
}

/// Compute CRC32 (IEEE polynomial) over `len` bytes starting at the
/// given raw pointer. Used by the daemon to checksum host-tier slots
/// after D2H sync (offload integrity) and before H2D (restore
/// integrity). Uses SIMD when the CPU advertises it (crc32fast picks
/// at runtime).
///
/// Caller MUST guarantee the memory range is valid + readable for
/// `len` bytes and stable for the duration of the call. In our use,
/// the memory is pinned via cudaHostAlloc, so it's always backed.
///
/// Returns the 32-bit CRC as a Python int.
#[pyfunction]
#[pyo3(signature = (ptr, len))]
fn crc32_at_ptr(ptr: usize, len: usize) -> u32 {
    if len == 0 {
        return 0;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let mut h = crc32fast::Hasher::new();
    h.update(slice);
    h.finalize()
}

#[pymodule]
fn gms_rust_ring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(push_record, m)?)?;
    m.add_function(wrap_pyfunction!(try_pop_record, m)?)?;
    m.add_function(wrap_pyfunction!(crc32_at_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(crc32_combine_chunks, m)?)?;
    m.add("RECORD_SIZE", RECORD_SIZE)?;
    m.add("HEADER_SIZE", HEADER_SIZE)?;
    m.add("MAX_BLOCK_PAIRS", MAX_BLOCK_PAIRS)?;
    m.add("ENGINE_ID_MAX_LEN", ENGINE_ID_MAX_LEN)?;
    Ok(())
}
