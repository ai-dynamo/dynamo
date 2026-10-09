// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronization facade for `protocol.rs`: std atomics and parking_lot here, loom's
//! types in the `arena-loom` model crate, which includes `protocol.rs` by path.

pub(crate) use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering, fence};

/// A mutex with the guard-returning `lock` both facades share.
#[derive(Default)]
pub(crate) struct Mutex<T>(parking_lot::Mutex<T>);

impl<T> Mutex<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(parking_lot::Mutex::new(value))
    }

    #[inline]
    pub(crate) fn lock(&self) -> parking_lot::MutexGuard<'_, T> {
        self.0.lock()
    }
}

#[inline]
pub(crate) fn spin_hint() {
    std::hint::spin_loop();
}
