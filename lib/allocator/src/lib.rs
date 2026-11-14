// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! jemalloc as the Rust global allocator, with one set of options for every Dynamo binary.
//!
//! jemalloc's symbols are prefixed (`_rjem_`), so CPython and C code in the same process keep the
//! system allocator. musl and MSVC targets keep the system allocator for Rust as well.

#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[doc(hidden)]
pub use tikv_jemallocator;

/// Makes jemalloc the global allocator of the binary, bench or `cdylib` that invokes it.
///
/// Invoke it once, in the root file of the final artifact and never in a library: two global
/// allocators, or two `_rjem_malloc_conf` symbols, do not link.
///
/// The options are built in, and `_RJEM_MALLOC_CONF` overrides them one by one. The background
/// thread returns freed pages while idle, the decay values pin jemalloc 5.3's defaults, and
/// `narenas` caps the default of four arenas per CPU. macOS jemalloc has no background threads;
/// requesting one leaves every allocation on a lock.
#[macro_export]
macro_rules! jemalloc_global_allocator {
    () => {
        #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
        #[global_allocator]
        static GLOBAL: $crate::tikv_jemallocator::Jemalloc = $crate::tikv_jemallocator::Jemalloc;

        #[cfg(all(
            not(target_os = "macos"),
            not(target_env = "msvc"),
            not(target_env = "musl")
        ))]
        #[unsafe(export_name = "_rjem_malloc_conf")]
        static MALLOC_CONF: &[u8; 72] =
            b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";

        #[cfg(target_os = "macos")]
        #[unsafe(export_name = "_rjem_malloc_conf")]
        static MALLOC_CONF: &[u8; 49] = b"dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";
    };
}

/// Bytes jemalloc has handed out and not freed (`stats.allocated`).
///
/// `None` unless jemalloc was built with statistics, which the `stats` feature turns on and
/// production builds leave off, or where jemalloc is not compiled in.
pub fn allocated() -> Option<u64> {
    #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
    {
        // Writing `epoch` refreshes jemalloc's cached statistics.
        let mut epoch = 1u64;
        let mut epoch_len = size_of::<u64>();
        // SAFETY: both pointers name `epoch`, a live u64 whose size is passed alongside.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                c"epoch".as_ptr(),
                (&raw mut epoch).cast(),
                &mut epoch_len,
                (&raw mut epoch).cast(),
                epoch_len,
            )
        };
        if rc != 0 {
            return None;
        }
        let mut allocated = 0usize;
        let mut allocated_len = size_of::<usize>();
        // SAFETY: reads one size_t into `allocated`, whose size is passed alongside.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                c"stats.allocated".as_ptr(),
                (&raw mut allocated).cast(),
                &mut allocated_len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(allocated as u64)
    }
    #[cfg(any(target_env = "msvc", target_env = "musl"))]
    None
}

/// Returns the unused dirty pages of every arena to the operating system now, without waiting
/// for them to decay. Returns whether jemalloc purged; `false` where it is not compiled in.
pub fn purge() -> bool {
    #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
    {
        // 4096 is jemalloc's `MALLCTL_ARENAS_ALL`.
        // SAFETY: the purge control reads and writes nothing through its pointers.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                c"arena.4096.purge".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        rc == 0
    }
    #[cfg(any(target_env = "msvc", target_env = "musl"))]
    false
}
