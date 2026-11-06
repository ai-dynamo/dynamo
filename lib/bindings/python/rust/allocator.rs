// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! jemalloc as the global allocator for every Rust allocation in the extension.
//!
//! The jemalloc symbols are prefixed (`_rjem_`), so CPython and other C code in the process
//! keep the system allocator.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's application-provided options. `_RJEM_MALLOC_CONF` is read after this string and
/// overrides it option by option.
///
/// - `background_thread:true`: purge freed pages on a schedule. Without it, an arena returns
///   dirty pages only when one of its threads allocates again, so an idle process keeps the
///   resident size of its last traffic burst.
/// - `dirty_decay_ms:10000,muzzy_decay_ms:0`: return dirty pages after 10 s and skip the muzzy
///   state. These are jemalloc 5.3's defaults, pinned so the purge schedule does not move with a
///   jemalloc upgrade.
/// - `narenas:16`: the default is four arenas per CPU in the affinity mask, each holding
///   partly used slabs for every size class.
#[cfg(not(target_os = "macos"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8; 72] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";

/// jemalloc has no background-thread support on macOS, and `background_thread:true` there stops
/// initialization halfway, which leaves every allocation on a global lock.
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
