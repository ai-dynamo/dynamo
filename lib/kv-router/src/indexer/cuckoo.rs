// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod addressing;
mod bucket;
mod event_indexer;
mod mutator;
mod search;

#[cfg(test)]
mod tests;

pub use event_indexer::EventTransposedCkfIndexer;

pub(crate) const DC_COUNT: usize = 16;
pub(crate) const MAX_KICKS: usize = 4096;
pub(crate) const MAX_VERIFICATION_WINDOW: usize = 8;

const DEFAULT_SEED: u64 = 0x5DEE_CE66_D1B5_4A33;
const DEFAULT_MAX_KICKS: usize = 500;
const DEFAULT_EXPECTED_BLOCKS_PER_DC: usize = 1;
const DEFAULT_VERIFICATION_WINDOW: usize = 2;

/// Search behavior for CKF prefix lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixSearchConfig {
    /// Number of positions immediately before the tentative depth to verify linearly.
    ///
    /// If the first miss is the window's left edge, search may also scan the
    /// previously discarded gap after the predecessor of the terminal lower bound.
    /// Stable snapshots make that contradiction evidence of a false terminal branch;
    /// concurrent mutation can instead expose temporary false negatives.
    pub verification_window: usize,
}

impl Default for PrefixSearchConfig {
    fn default() -> Self {
        Self {
            verification_window: DEFAULT_VERIFICATION_WINDOW,
        }
    }
}

/// Capacity, addressing, and search configuration for the D=16 CKF indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CkfConfig {
    /// Logical blocks expected in each DC lane.
    pub expected_blocks_per_dc: usize,
    /// Shared deterministic addressing seed.
    pub seed: u64,
    /// Maximum relocation steps before one block insertion is rolled back.
    pub max_kicks: usize,
    /// Prefix-search behavior.
    pub search: PrefixSearchConfig,
}

impl CkfConfig {
    /// Create a configuration with the requested per-DC capacity and standard defaults.
    pub fn new(expected_blocks_per_dc: usize) -> Self {
        Self {
            expected_blocks_per_dc,
            ..Self::default()
        }
    }
}

impl Default for CkfConfig {
    fn default() -> Self {
        Self {
            expected_blocks_per_dc: DEFAULT_EXPECTED_BLOCKS_PER_DC,
            seed: DEFAULT_SEED,
            max_kicks: DEFAULT_MAX_KICKS,
            search: PrefixSearchConfig::default(),
        }
    }
}

/// Construction failures for [`EventTransposedCkfIndexer`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CkfBuildError {
    #[error("duplicate CKF worker identity: {worker:?}")]
    DuplicateWorker {
        worker: crate::protocols::WorkerWithDpRank,
    },

    #[error("expected_blocks_per_dc must be greater than zero")]
    ExpectedCapacityZero,

    #[error("max_kicks {value} is outside the supported range 1..={maximum}")]
    InvalidMaxKicks { value: usize, maximum: usize },

    #[error(
        "verification_window {value} is outside the supported range 1..={MAX_VERIFICATION_WINDOW}"
    )]
    InvalidVerificationWindow { value: usize },

    #[error("CKF capacity arithmetic overflowed")]
    CapacityOverflow,

    #[error("failed to allocate CKF storage")]
    AllocationFailed,
}
