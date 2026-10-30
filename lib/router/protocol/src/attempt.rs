// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attempt identities for admitted requests.
//!
//! An [`AttemptId`] has no public constructor. The only way to obtain one is
//! [`AttemptIdMinter::mint`], which yields fresh monotonic values and never a
//! chosen one, so no crate can forge the identity of another request's attempt.
//! The slot tracker holds the one minter instance and mints when a booking is born.

use std::sync::atomic::{AtomicU64, Ordering};

/// Identity of one admitted request attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttemptId(u64);

impl AttemptId {
    /// Construct a chosen value. Tests only: a forged attempt id defeats the
    /// `(request_id, attempt_id)` fence in production code.
    #[cfg(feature = "testing")]
    pub fn for_testing(value: u64) -> Self {
        Self(value)
    }

    /// Construct a chosen value for benchmarks.
    #[cfg(feature = "bench")]
    pub fn for_benchmark(value: u64) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for AttemptId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// The only source of [`AttemptId`] values. The first id minted is 1.
#[derive(Debug, Default)]
pub struct AttemptIdMinter {
    next: AtomicU64,
}

impl AttemptIdMinter {
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
        }
    }

    /// Mint a fresh attempt id. Monotonic per minter; never returns a value twice.
    pub fn mint(&self) -> AttemptId {
        AttemptId(self.next.fetch_add(1, Ordering::Relaxed) + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minter_yields_fresh_monotonic_ids_starting_at_one() {
        let minter = AttemptIdMinter::new();
        let first = minter.mint();
        let second = minter.mint();
        assert_eq!(first.to_string(), "1");
        assert_eq!(second.to_string(), "2");
        assert_ne!(first, second);
    }
}
