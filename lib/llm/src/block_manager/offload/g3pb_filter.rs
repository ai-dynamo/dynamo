// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::block_manager::config::{G3pbAdmissionConfig, G3pbAdmissionPolicy};
use crate::tokens::SequenceHash;

use super::OffloadFilter;

/// A filter that admits blocks to G3PB based on reuse frequency.
///
/// The default policy is to admit blocks that have been reused at least once.
/// This can be overridden with the `G3PB_OFFLOAD_ALL` environment variable to
/// eagerly admit every block.
///
/// The semantics are copy/replication, not ownership transfer - blocks remain
/// available locally after being admitted to G3PB.
#[derive(Debug, Clone)]
pub struct G3pbAdmissionFilter {
    /// Track reuse count for each sequence hash
    reuse_map: Arc<Mutex<HashMap<SequenceHash, usize>>>,
    /// If true, admit all blocks regardless of reuse count
    offload_all: bool,
}

impl G3pbAdmissionFilter {
    /// Create a new G3PB admission filter.
    ///
    /// By default, blocks are admitted after being reused at least once.
    /// Set the `G3PB_OFFLOAD_ALL` environment variable to "1" or "true" to
    /// eagerly admit every block.
    pub fn new() -> Self {
        let config = G3pbAdmissionConfig::from_legacy_env().unwrap_or_default();
        Self::from_config(config)
    }

    pub fn from_config(config: G3pbAdmissionConfig) -> Self {
        Self {
            reuse_map: Arc::new(Mutex::new(HashMap::new())),
            offload_all: matches!(config.policy, G3pbAdmissionPolicy::Eager),
        }
    }

    /// Check if a block should be admitted to G3PB.
    ///
    /// Returns true if:
    /// - `G3PB_OFFLOAD_ALL` is set, OR
    /// - The block has been reused at least once (i.e., this is the second or later access)
    pub fn should_admit(&self, sequence_hash: SequenceHash) -> bool {
        if self.offload_all {
            return true;
        }

        let mut reuse_map = self.reuse_map.lock().unwrap();
        let count = reuse_map.entry(sequence_hash).or_insert(0);
        *count += 1;

        // Admit if this is the second or later access (count >= 2)
        *count >= 2
    }
}

impl OffloadFilter for G3pbAdmissionFilter {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool {
        self.should_admit(sequence_hash)
    }
}

impl Default for G3pbAdmissionFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(x: u64) -> SequenceHash {
        SequenceHash::from(x)
    }

    #[test]
    fn test_default_admission_requires_reuse() {
        let filter = G3pbAdmissionFilter::from_config(G3pbAdmissionConfig::after_first_reuse());

        // First access should not be admitted
        assert!(!filter.should_offload(hash(1)));

        // Second access should be admitted
        assert!(filter.should_offload(hash(1)));

        // Third access should also be admitted
        assert!(filter.should_offload(hash(1)));
    }

    #[test]
    fn test_offload_all_admits_immediately() {
        let filter = G3pbAdmissionFilter::from_config(G3pbAdmissionConfig::eager());

        // First access should be admitted when G3PB_OFFLOAD_ALL is set
        assert!(filter.should_offload(hash(1)));
        assert!(filter.should_offload(hash(2)));
        assert!(filter.should_offload(hash(3)));
    }

    #[test]
    fn test_legacy_env_true_string() {
        unsafe {
            std::env::set_var("G3PB_OFFLOAD_ALL", "true");
        }
        let filter = G3pbAdmissionFilter::new();

        // First access should be admitted when G3PB_OFFLOAD_ALL is "true"
        assert!(filter.should_offload(hash(1)));

        unsafe {
            std::env::remove_var("G3PB_OFFLOAD_ALL");
        }
    }

    #[test]
    fn test_legacy_env_false_string() {
        unsafe {
            std::env::set_var("G3PB_OFFLOAD_ALL", "false");
        }
        let filter = G3pbAdmissionFilter::new();

        // First access should not be admitted when G3PB_OFFLOAD_ALL is "false"
        assert!(!filter.should_offload(hash(1)));

        unsafe {
            std::env::remove_var("G3PB_OFFLOAD_ALL");
        }
    }

    #[test]
    fn test_multiple_hashes_tracked_separately() {
        let filter = G3pbAdmissionFilter::from_config(G3pbAdmissionConfig::after_first_reuse());

        // First access to each hash should not be admitted
        assert!(!filter.should_offload(hash(1)));
        assert!(!filter.should_offload(hash(2)));
        assert!(!filter.should_offload(hash(3)));

        // Second access to hash 1 should be admitted
        assert!(filter.should_offload(hash(1)));

        // First access to hash 4 should not be admitted
        assert!(!filter.should_offload(hash(4)));

        // Second access to hash 2 should be admitted
        assert!(filter.should_offload(hash(2)));
    }

    #[test]
    fn test_default_trait() {
        unsafe {
            std::env::remove_var("G3PB_OFFLOAD_ALL");
        }
        let filter = G3pbAdmissionFilter::default();
        // Should behave the same as new()
        assert!(!filter.should_offload(hash(1)));
        assert!(filter.should_offload(hash(1)));
    }
}
