// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
pub use tikv_jemallocator;

/// Makes jemalloc the global allocator, with the options of `lib/bindings/python/rust/allocator.rs`.
/// Invoke it only in a binary's `main.rs`: the Python extension links the sidecar libraries and
/// declares its own allocator.
#[macro_export]
macro_rules! jemalloc_global_allocator {
    () => {
        #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
        #[global_allocator]
        static GLOBAL: $crate::allocator::tikv_jemallocator::Jemalloc =
            $crate::allocator::tikv_jemallocator::Jemalloc;

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
