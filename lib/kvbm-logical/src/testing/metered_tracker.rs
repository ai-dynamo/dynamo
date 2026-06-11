// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only [`FrequencyTracker`] decorator that counts every `touch()` and
//! `count()` call.
//!
//! Wraps any [`FrequencyTracker<u128>`] (typically [`TinyLFUTracker`])
//! and forwards calls unchanged, while incrementing two `AtomicU64`
//! audit counters. Tests assert exact equality on the counters to catch
//! double-counting bugs that probabilistic sketches like TinyLFU would
//! otherwise hide.
//!
//! Bind this into the [`BlockRegistry`] via
//! `BlockRegistry::builder().frequency_tracker(metered.clone())`. The
//! caller keeps a clone of the `Arc<MeteredFrequencyTracker>` to read
//! the counters from tests.
//!
//! Production code never depends on this — it lives behind the `testing`
//! feature.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tinylfu::{FrequencyTracker, TinyLFUTracker};

/// Test-only wrapper that counts `touch`/`count` calls into atomics.
pub struct MeteredFrequencyTracker {
    inner: Arc<dyn FrequencyTracker<u128>>,
    touches: AtomicU64,
    count_calls: AtomicU64,
}

impl MeteredFrequencyTracker {
    /// Wrap an existing tracker.
    pub fn new(inner: Arc<dyn FrequencyTracker<u128>>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            touches: AtomicU64::new(0),
            count_calls: AtomicU64::new(0),
        })
    }

    /// Wrap a fresh [`TinyLFUTracker`] with the given capacity.
    pub fn with_tinylfu(capacity: usize) -> Arc<Self> {
        Self::new(Arc::new(TinyLFUTracker::new(capacity)))
    }

    /// Total `touch()` calls since construction.
    pub fn touches(&self) -> u64 {
        self.touches.load(Ordering::Relaxed)
    }

    /// Total `count()` calls since construction.
    pub fn count_calls(&self) -> u64 {
        self.count_calls.load(Ordering::Relaxed)
    }

    /// Reset both counters to zero.
    pub fn reset(&self) {
        self.touches.store(0, Ordering::Relaxed);
        self.count_calls.store(0, Ordering::Relaxed);
    }
}

impl FrequencyTracker<u128> for MeteredFrequencyTracker {
    fn touch(&self, key: u128) {
        self.touches.fetch_add(1, Ordering::Relaxed);
        self.inner.touch(key);
    }

    fn count(&self, key: u128) -> u32 {
        self.count_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.count(key)
    }
}
