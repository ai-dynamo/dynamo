// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event emission policies for filtering which blocks trigger events.
//!
//! Policies determine whether a block registration should emit a KvCacheEvent
//! based on characteristics like position, frequency, or other criteria.

use crate::SequenceHash;

/// Trait for policies that determine whether to emit events for a block.
pub trait EventEmissionPolicy: Send + Sync {
    /// Returns true if an event should be emitted for the given sequence hash.
    fn should_emit(&self, seq_hash: SequenceHash) -> bool;
}

/// Policy that emits events only for blocks at power-of-2 positions within a range.
///
/// This creates a sparse sampling of the sequence space, emitting events at
/// positions 2^4 (16), 2^5 (32), 2^6 (64), ..., 2^16 (65536).
///
/// This sparse radix approach allows the hub to efficiently narrow down search
/// space when locating blocks across the fleet without tracking every block.
#[derive(Debug, Clone)]
pub struct PowerOfTwoPolicy {
    min_position: u64,
    max_position: u64,
}

impl PowerOfTwoPolicy {
    /// Creates a new power-of-2 policy with default range [2^4, 2^16].
    pub fn new() -> Self {
        Self {
            min_position: 16,    // 2^4
            max_position: 65536, // 2^16
        }
    }

    /// Creates a new power-of-2 policy with custom range.
    ///
    /// # Arguments
    /// * `min_position` - Minimum position (inclusive), should be a power of 2
    /// * `max_position` - Maximum position (inclusive), should be a power of 2
    pub fn with_range(min_position: u64, max_position: u64) -> Self {
        Self {
            min_position,
            max_position,
        }
    }
}

impl Default for PowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEmissionPolicy for PowerOfTwoPolicy {
    fn should_emit(&self, seq_hash: SequenceHash) -> bool {
        let position = seq_hash.position();

        // Check if position is within range
        if position < self.min_position || position > self.max_position {
            return false;
        }

        // Check if position is a power of 2
        position.is_power_of_two()
    }
}

/// Policy that emits events for all blocks.
///
/// This is primarily useful for testing, where you want to emit events for
/// every block registration without filtering.
#[derive(Debug, Clone, Default)]
pub struct AllEventsPolicy;

impl AllEventsPolicy {
    /// Creates a new all-events policy.
    pub fn new() -> Self {
        Self
    }
}

impl EventEmissionPolicy for AllEventsPolicy {
    fn should_emit(&self, _seq_hash: SequenceHash) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KvbmSequenceHashProvider, SequenceHash};
    use dynamo_tokens::TokenBlockSequence;

    fn create_seq_hash_at_position(position: usize) -> SequenceHash {
        // Create a sequence with enough blocks to reach the desired position
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();

        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    #[test]
    fn test_power_of_two_policy_accepts_valid_positions() {
        let policy = PowerOfTwoPolicy::new();

        // Test power-of-2 positions within range
        let valid_positions = vec![
            16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        ];

        for pos in valid_positions {
            let seq_hash = create_seq_hash_at_position(pos);
            assert!(
                policy.should_emit(seq_hash),
                "Should emit for position {}",
                pos
            );
        }
    }

    #[test]
    fn test_power_of_two_policy_rejects_non_power_of_two() {
        let policy = PowerOfTwoPolicy::new();

        // Test non-power-of-2 positions
        let invalid_positions = vec![15, 17, 31, 33, 63, 65, 100, 1000, 10000];

        for pos in invalid_positions {
            let seq_hash = create_seq_hash_at_position(pos);
            assert!(
                !policy.should_emit(seq_hash),
                "Should not emit for position {}",
                pos
            );
        }
    }

    #[test]
    fn test_power_of_two_policy_rejects_out_of_range() {
        let policy = PowerOfTwoPolicy::new();

        // Test power-of-2 positions outside range
        let out_of_range = vec![1, 2, 4, 8]; // Below min (16)

        for pos in out_of_range {
            let seq_hash = create_seq_hash_at_position(pos);
            assert!(
                !policy.should_emit(seq_hash),
                "Should not emit for position {} (below min)",
                pos
            );
        }

        // Position above max would require creating very large sequences
        // Skip testing 2^17 and above for practical reasons
    }

    #[test]
    fn test_power_of_two_policy_custom_range() {
        let policy = PowerOfTwoPolicy::with_range(32, 128);

        // Should accept 32, 64, 128
        assert!(policy.should_emit(create_seq_hash_at_position(32)));
        assert!(policy.should_emit(create_seq_hash_at_position(64)));
        assert!(policy.should_emit(create_seq_hash_at_position(128)));

        // Should reject 16 (below min) and 256 (above max)
        assert!(!policy.should_emit(create_seq_hash_at_position(16)));
        assert!(!policy.should_emit(create_seq_hash_at_position(256)));
    }
}
