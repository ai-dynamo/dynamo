// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod addressing;
mod bucket;
mod event_indexer;
mod mutator;
mod relay;
mod search;

#[cfg(test)]
mod tests;

pub use event_indexer::EventTransposedCkfIndexer;
pub use relay::{
    CkfBucketImage, CkfDeltaBatch, CkfEventOutcome, CkfFormatIdentity, CkfMemorySnapshot,
    CkfRelayAggregator, RelayLaneConfig, RelayManifest, RouterLocalCkfPipeline,
    TransposedCkfReplica,
};

pub(crate) const DC_COUNT: usize = 16;
pub(crate) const MAX_KICKS: usize = 4096;
pub(crate) const MAX_VERIFICATION_WINDOW: usize = 8;

const DEFAULT_SEED: u64 = 0x5DEE_CE66_D1B5_4A33;
const DEFAULT_MAX_KICKS: usize = 500;
const DEFAULT_EXPECTED_BLOCKS_PER_DC: usize = 1;
const DEFAULT_VERIFICATION_WINDOW: usize = 2;
const DEFAULT_PUBLISH_EVERY_N_EVENTS: usize = 1;

/// Output shape for CKF prefix lookups through [`EventTransposedCkfIndexer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CkfMatchMode {
    /// Return every positive DC-lane depth.
    #[default]
    FullMap,
    /// Return every positive lane tied at the greatest verified depth.
    MaxDepthMatches,
}

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
    /// Number of normalized mutation events accumulated per DC lane before publication.
    pub publish_every_n_events: usize,
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
            publish_every_n_events: DEFAULT_PUBLISH_EVERY_N_EVENTS,
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

    #[error("publish_every_n_events must be greater than zero")]
    InvalidPublishEveryNEvents,

    #[error("CKF capacity arithmetic overflowed")]
    CapacityOverflow,

    #[error("failed to allocate CKF storage")]
    AllocationFailed,

    #[error("lane {lane} expected_contributions {value} is below the required minimum {minimum}")]
    InvalidContributionCapacity {
        lane: usize,
        value: usize,
        minimum: usize,
    },

    #[error("empty lane {lane} must have zero expected_contributions, got {value}")]
    InvalidEmptyLaneContributionCapacity { lane: usize, value: usize },
}

pub(super) fn validate_config(config: CkfConfig) -> Result<(), CkfBuildError> {
    if config.expected_blocks_per_dc == 0 {
        return Err(CkfBuildError::ExpectedCapacityZero);
    }
    if !(1..=MAX_KICKS).contains(&config.max_kicks) {
        return Err(CkfBuildError::InvalidMaxKicks {
            value: config.max_kicks,
            maximum: MAX_KICKS,
        });
    }
    if !(1..=MAX_VERIFICATION_WINDOW).contains(&config.search.verification_window) {
        return Err(CkfBuildError::InvalidVerificationWindow {
            value: config.search.verification_window,
        });
    }
    if config.publish_every_n_events == 0 {
        return Err(CkfBuildError::InvalidPublishEveryNEvents);
    }
    Ok(())
}

pub(super) fn bucket_count(expected_blocks_per_dc: usize) -> Result<usize, CkfBuildError> {
    let numerator = expected_blocks_per_dc
        .checked_mul(5)
        .and_then(|value| value.checked_add(15))
        .ok_or(CkfBuildError::CapacityOverflow)?;
    let required = (numerator / 16).max(2);
    required
        .checked_next_power_of_two()
        .ok_or(CkfBuildError::CapacityOverflow)
}
