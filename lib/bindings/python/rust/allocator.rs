// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The extension's Rust global allocator: a jemalloc linked into the extension.
//!
//! Its symbols carry the `_rjem_` prefix and stay local to the extension, so it serves only
//! the extension's Rust allocations. Python and C/C++ libraries in the process keep their
//! allocator: glibc, or a preloaded jemalloc, which is then a second, independent instance.
//! Its options are built in (see the `jemalloc` feature) and `_RJEM_MALLOC_CONF` overrides them.

use std::ptr;

use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// Returns the unused pages of every arena of the extension's jemalloc to the OS.
pub fn purge() {
    // `arena.<MALLCTL_ARENAS_ALL>.purge`; MALLCTL_ARENAS_ALL is 4096.
    const PURGE_ALL_ARENAS: &std::ffi::CStr = c"arena.4096.purge";
    // SAFETY: the name is NUL-terminated, and this mallctl neither reads nor writes a value,
    // so it requires null value pointers and a zero length.
    let status = unsafe {
        tikv_jemalloc_sys::mallctl(
            PURGE_ALL_ARENAS.as_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        tracing::debug!(status, "jemalloc purge failed");
    }
}
