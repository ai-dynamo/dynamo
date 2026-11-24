//! Recycling pools for the large per-image vision buffers.
//!
//! The preprocess pipeline allocates tens of MB per image (the [C, H, W] f32
//! tensor and the batched patch buffer are each large).
//! Freshly-allocated buffers of this size bypass the allocator's reuse paths
//! (glibc caps non-main-arena chunks at 64 MB and mmaps anything larger or
//! colder), so every image pays tens of thousands of minor page faults; the
//! fault path serializes process-wide and caps the data plane's effective
//! parallelism. Recycling keeps the pages mapped and hot.
//!
//! A lock-free thread-local pool serves same-thread take/give (preprocess
//! internals run on blocking-pool threads). The pool is capped to bound
//! residency; buffers beyond the cap are dropped.

use std::cell::RefCell;

/// Max recycled buffers kept per thread per class; excess is dropped. The
/// vision path holds at most a couple of live tensors per request, so a small
/// cap captures same-thread reuse.
const MAX_THREAD_POOLED: usize = 2;

thread_local! {
    static F32_LOCAL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

macro_rules! pool_impl {
    ($take_cap:ident, $give:ident, $ty:ty, $local:ident) => {
        /// Take an empty `Vec` with at least `cap` capacity, reusing pooled storage.
        pub fn $take_cap(cap: usize) -> Vec<$ty> {
            let mut v = $local.with(|p| p.borrow_mut().pop()).unwrap_or_default();
            v.clear();
            v.reserve(cap);
            v
        }

        /// Return a buffer for reuse by a later same-thread take.
        pub fn $give(v: Vec<$ty>) {
            if v.capacity() == 0 {
                return;
            }
            $local.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < MAX_THREAD_POOLED {
                    p.push(v);
                }
            });
        }
    };
}

pool_impl!(take_f32_cap, give_f32, f32, F32_LOCAL);

/// Take a zero-filled `Vec<f32>` of exactly `len`, reusing pooled storage.
pub fn take_f32(len: usize) -> Vec<f32> {
    let mut v = take_f32_cap(len);
    v.resize(len, 0.0);
    v
}
