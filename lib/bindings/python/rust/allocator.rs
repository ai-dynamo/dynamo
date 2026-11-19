// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! jemalloc as the Rust global allocator. Its symbols are prefixed (`_rjem_`), so CPython and C
//! code keep the system allocator. The options are in `lib/allocator/src/lib.rs`.

dynamo_allocator::jemalloc_global_allocator!();

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
