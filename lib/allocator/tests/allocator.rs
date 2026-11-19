// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]

use tikv_jemalloc_ctl::{opt, raw};

dynamo_allocator::jemalloc_global_allocator!();

#[test]
fn built_in_options_are_applied() {
    #[cfg(not(target_os = "macos"))]
    assert!(opt::background_thread::read().unwrap());
    assert_eq!(opt::narenas::read().unwrap(), 16);
    // SAFETY: both options are `ssize_t` in jemalloc.
    let dirty = unsafe { raw::read::<isize>(b"opt.dirty_decay_ms\0") }.unwrap();
    let muzzy = unsafe { raw::read::<isize>(b"opt.muzzy_decay_ms\0") }.unwrap();
    assert_eq!((dirty, muzzy), (10_000, 0));
}

#[test]
fn allocated_follows_the_stats_feature() {
    let before = dynamo_allocator::allocated();
    let held = std::hint::black_box(vec![0u8; 1 << 20]);
    let after = dynamo_allocator::allocated();
    drop(held);
    if cfg!(feature = "stats") {
        assert!(after.unwrap() > before.unwrap());
    } else {
        assert_eq!((before, after), (None, None));
    }
    assert!(dynamo_allocator::purge());
}
