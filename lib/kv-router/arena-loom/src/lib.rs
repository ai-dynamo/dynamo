// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom models for `lib/kv-router/src/indexer/arena_b/protocol.rs` (spec B 14). Each model
//! runs the production primitive and a negative control with one fix removed; the control
//! must fail, which shows the model can see the bug the fix prevents.
//!
//! Built only with `--cfg loom`; without it the crate is empty.

#![cfg(loom)]
#![allow(dead_code)]

/// The loom side of the `sync` facade `protocol.rs` imports.
mod sync {
    pub(crate) use loom::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering, fence};

    pub(crate) struct Mutex<T>(loom::sync::Mutex<T>);

    impl<T> Mutex<T> {
        pub(crate) fn new(value: T) -> Self {
            Self(loom::sync::Mutex::new(value))
        }

        pub(crate) fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            self.0.lock().unwrap()
        }
    }

    pub(crate) fn spin_hint() {
        loom::thread::yield_now();
    }
}

#[path = "../../src/indexer/arena_b/protocol.rs"]
mod protocol;

#[cfg(test)]
mod models;
