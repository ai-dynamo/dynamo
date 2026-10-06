// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The extension's Rust global allocator: mimalloc, or the system allocator when jemalloc is
//! preloaded, so that a jemalloc preload (`DYN_FRONTEND_JEMALLOC` or a bare `LD_PRELOAD`)
//! serves the Rust heap too.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU8, Ordering};

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: Dispatch = Dispatch;

const UNDECIDED: u8 = 0;
const MIMALLOC: u8 = 1;
const SYSTEM: u8 = 2;

static CHOICE: AtomicU8 = AtomicU8::new(UNDECIDED);

/// Forwards every call to the allocator chosen by the first call.
///
/// No block can cross allocators: none exists before the choice, the choice never changes,
/// and `dealloc` and `realloc` dispatch on it exactly as `alloc` does.
struct Dispatch;

// SAFETY: each method forwards to one of two sound allocators, and a block is always
// returned to the allocator that produced it (see `Dispatch`).
unsafe impl GlobalAlloc for Dispatch {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe {
            if uses_mimalloc() {
                MiMalloc.alloc(layout)
            } else {
                System.alloc(layout)
            }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe {
            if uses_mimalloc() {
                MiMalloc.alloc_zeroed(layout)
            } else {
                System.alloc_zeroed(layout)
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, which made the same choice for it.
        unsafe {
            if uses_mimalloc() {
                MiMalloc.dealloc(ptr, layout)
            } else {
                System.dealloc(ptr, layout)
            }
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from this allocator, which made the same choice for it.
        unsafe {
            if uses_mimalloc() {
                MiMalloc.realloc(ptr, layout, new_size)
            } else {
                System.realloc(ptr, layout, new_size)
            }
        }
    }
}

/// Whether mimalloc serves the extension's Rust allocations, choosing on first use.
#[inline]
pub fn uses_mimalloc() -> bool {
    match CHOICE.load(Ordering::Relaxed) {
        MIMALLOC => true,
        SYSTEM => false,
        _ => choose() == MIMALLOC,
    }
}

#[cold]
#[inline(never)]
fn choose() -> u8 {
    let choice = if jemalloc_preloaded() {
        SYSTEM
    } else {
        MIMALLOC
    };
    // Racing first calls read the same environment, and the first store wins regardless.
    match CHOICE.compare_exchange(UNDECIDED, choice, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => choice,
        Err(chosen) => chosen,
    }
}

/// Whether `LD_PRELOAD` has an entry, split on `:` or space as the loader splits them, whose
/// file name starts with `libjemalloc`; `dynamo.frontend` applies the same test before it
/// re-executes with the preload. Reads it through `getenv`, because `std::env` allocates and
/// the allocator is the caller.
fn jemalloc_preloaded() -> bool {
    unsafe extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }
    // SAFETY: the name is NUL-terminated; getenv neither allocates nor keeps it.
    let value = unsafe { getenv(c"LD_PRELOAD".as_ptr()) };
    if value.is_null() {
        return false;
    }
    // SAFETY: a non-null getenv result is a NUL-terminated string owned by the environment.
    let value = unsafe { CStr::from_ptr(value) }.to_bytes();
    value
        .split(|&b| b == b':' || b == b' ')
        .filter_map(|entry| entry.rsplit(|&b| b == b'/').next())
        .any(|name| name.starts_with(b"libjemalloc"))
}

/// Keeps mimalloc's arenas out of transparent huge pages unless `MIMALLOC_ALLOW_THP` says
/// otherwise; glibc, which held these allocations before, never asked for them. Does nothing
/// when the system allocator serves the extension, which then never reserves a mimalloc arena;
/// mimalloc's load-time constructor still runs and reads `MIMALLOC_*`.
///
/// mimalloc advises each arena it reserves for huge pages. In a many-threaded process each
/// thread's sparse pages then fault in whole 2 MiB pages, and khugepaged refills partly freed
/// ones while idle.
///
/// Must run before the extension's first Rust allocation, which reserves the first arena.
pub fn configure() {
    if !uses_mimalloc() {
        return;
    }
    // libmimalloc-sys does not export this option; pin the mimalloc 3.3 enum it indexes.
    const _: () = assert!(libmimalloc_sys::_mi_option_last == 47);
    const MI_OPTION_ALLOW_THP: libmimalloc_sys::mi_option_t = 43;
    // SAFETY: both calls only touch mimalloc's option state, and module init runs before
    // this extension starts any thread. Initializing first keeps the environment
    // authoritative and means mimalloc saw THP allowed at startup, so it skips the
    // process-wide PR_SET_THP_DISABLE and leaves other libraries' memory to the host's
    // THP policy.
    unsafe {
        libmimalloc_sys::mi_process_init();
        libmimalloc_sys::mi_option_set_default(MI_OPTION_ALLOW_THP, 0);
    }
}

/// Returns mimalloc's freed memory to the OS.
pub fn collect() {
    // SAFETY: mi_collect only releases memory mimalloc already considers free.
    unsafe { libmimalloc_sys::mi_collect(true) }
}
