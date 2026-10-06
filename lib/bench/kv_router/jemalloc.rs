// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router benches' global allocator: jemalloc, built like the Python extension's that runs
//! the router in production (`lib/bindings/python/rust/allocator.rs`).

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Returns the unused pages of every jemalloc arena to the OS.
#[allow(dead_code, reason = "only the benches that quiesce their heap call it")]
pub fn purge() {
    // SAFETY: the name (`arena.<MALLCTL_ARENAS_ALL>.purge`) is NUL-terminated, and this
    // mallctl requires null value pointers and a zero length.
    unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arena.4096.purge".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
    }
}
