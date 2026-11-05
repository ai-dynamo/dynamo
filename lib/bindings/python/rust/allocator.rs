// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! jemalloc as the global allocator for every Rust allocation in the extension, unless
//! `DYN_DISABLE_JEMALLOC` is set when the process first allocates.
//!
//! The jemalloc symbols are prefixed (`_rjem_`), so CPython and other C code in the process
//! keep the system allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU8, Ordering};

use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Allocator = Allocator;

const UNSET: u8 = 0;
const JEMALLOC: u8 = 1;
const SYSTEM: u8 = 2;

/// Chosen at the first allocation and never changed, because memory must be freed by the
/// allocator that returned it. Relaxed is enough: every thread derives the same value from the
/// environment, and a pointer reaches another thread only through synchronization that also
/// publishes this store.
static CHOICE: AtomicU8 = AtomicU8::new(UNSET);

unsafe extern "C" {
    // libc's, because `std::env::var` allocates and would re-enter this allocator.
    fn getenv(name: *const c_char) -> *const c_char;
}

#[cold]
fn choose() -> u8 {
    // SAFETY: the name is NUL-terminated, and getenv returns null or a NUL-terminated string.
    let value = unsafe { getenv(c"DYN_DISABLE_JEMALLOC".as_ptr()) };
    let disabled = !value.is_null() && {
        // SAFETY: non-null, NUL-terminated, and not modified while we read it.
        let value = unsafe { CStr::from_ptr(value) }.to_bytes();
        value == b"1" || value.eq_ignore_ascii_case(b"true") || value.eq_ignore_ascii_case(b"yes")
    };
    let choice = if disabled { SYSTEM } else { JEMALLOC };
    match CHOICE.compare_exchange(UNSET, choice, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) if choice == SYSTEM => {
            stop_background_thread();
            choice
        }
        Ok(_) => choice,
        Err(existing) => existing,
    }
}

/// jemalloc initializes from an ELF constructor when the extension loads, which starts the
/// background thread before the first Rust allocation chooses an allocator. Unused, it has
/// nothing to purge, so stop it.
fn stop_background_thread() {
    let mut off = false;
    // SAFETY: `background_thread` is a writable bool. jemalloc copies the new value and does not
    // call back into the Rust allocator. A failure leaves an idle thread, so it is ignored.
    unsafe {
        tikv_jemalloc_sys::mallctl(
            c"background_thread".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            (&raw mut off).cast(),
            size_of::<bool>(),
        );
    }
}

#[inline(always)]
fn use_system() -> bool {
    match CHOICE.load(Ordering::Relaxed) {
        UNSET => choose() == SYSTEM,
        choice => choice == SYSTEM,
    }
}

/// Routes each call to jemalloc or the system allocator, as chosen once by [`choose`].
pub struct Allocator;

// SAFETY: every method forwards its arguments unchanged to one of two `GlobalAlloc`
// implementations, and the choice between them never changes once made, so each pointer is
// freed or resized by the allocator that returned it.
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe {
            if use_system() {
                System.alloc(layout)
            } else {
                Jemalloc.alloc(layout)
            }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe {
            if use_system() {
                System.alloc_zeroed(layout)
            } else {
                Jemalloc.alloc_zeroed(layout)
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract, and `ptr` came from the
        // same allocator because the choice is fixed.
        unsafe {
            if use_system() {
                System.dealloc(ptr, layout)
            } else {
                Jemalloc.dealloc(ptr, layout)
            }
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract, and `ptr` came from the
        // same allocator because the choice is fixed.
        unsafe {
            if use_system() {
                System.realloc(ptr, layout, new_size)
            } else {
                Jemalloc.realloc(ptr, layout, new_size)
            }
        }
    }
}

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
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8; 72] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16\0";

#[cfg(test)]
mod tests {
    use tikv_jemalloc_ctl::{opt, raw};

    #[test]
    fn exported_malloc_conf_is_applied() {
        assert!(opt::background_thread::read().unwrap());
        assert_eq!(opt::narenas::read().unwrap(), 16);
        // SAFETY: both options are `ssize_t` in jemalloc.
        let dirty = unsafe { raw::read::<isize>(b"opt.dirty_decay_ms\0") }.unwrap();
        let muzzy = unsafe { raw::read::<isize>(b"opt.muzzy_decay_ms\0") }.unwrap();
        assert_eq!((dirty, muzzy), (10_000, 0));
    }
}
