// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! jemalloc as the Rust global allocator. Its symbols are prefixed (`_rjem_`), so CPython and C
//! code keep the system allocator.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Built-in options, overridden one by one by `_RJEM_MALLOC_CONF`. The background thread returns
/// freed pages while idle, the decay values pin jemalloc 5.3's defaults, and `narenas` caps the
/// default of four arenas per CPU.
#[cfg(not(target_os = "macos"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8; 72] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";

/// macOS jemalloc has no background threads; requesting one leaves every allocation on a lock.
#[cfg(target_os = "macos")]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8; 49] = b"dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";

#[cfg(test)]
mod tests {
    use tikv_jemalloc_ctl::{opt, raw};

    #[test]
    fn exported_malloc_conf_is_applied() {
        #[cfg(not(target_os = "macos"))]
        assert!(opt::background_thread::read().unwrap());
        assert_eq!(opt::narenas::read().unwrap(), 16);
        // SAFETY: both options are `ssize_t` in jemalloc.
        let dirty = unsafe { raw::read::<isize>(b"opt.dirty_decay_ms\0") }.unwrap();
        let muzzy = unsafe { raw::read::<isize>(b"opt.muzzy_decay_ms\0") }.unwrap();
        assert_eq!((dirty, muzzy), (10_000, 0));
    }
}
