// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Centralized test configuration for v2 block manager.
//!
//! This module provides constants and utilities for consistent testing across
//! all v2 block manager components.

/// Default block size for testing (4 tokens)
///
/// Most tests use 4-token sequences like [100, 101, 102, 103].
/// This size is small enough for easy test data generation while being
/// a valid power-of-2 block size.
pub const DEFAULT_TEST_BLOCK_SIZE: usize = 4;

/// Default number of blocks for testing pools and managers
pub const DEFAULT_TEST_BLOCK_COUNT: usize = 10;

/// Common test block sizes for parameterized testing
///
/// These cover the full range of valid block sizes from 1 to 1024,
/// all being powers of 2 as required by the validation.
pub const TEST_BLOCK_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

/// Small set of block sizes for focused testing
pub const COMMON_TEST_BLOCK_SIZES: &[usize] = &[1, 4, 16, 64];

/// Test configuration constants
pub mod constants {

    /// Very small block size for edge case testing
    pub const TINY: usize = 1;

    /// Small block size (our standard)
    pub const SMALL: usize = 4;

    /// Medium block size
    pub const MEDIUM: usize = 16;

    /// Large block size
    pub const LARGE: usize = 64;

    /// Maximum block size
    pub const MAX: usize = 1024;
}

/// Validate that a block size is suitable for testing
///
/// # Arguments
/// * `size` - The block size to validate
///
/// # Returns
/// `true` if the size is a power of 2 between 1 and 1024 (inclusive)
pub fn validate_test_block_size(size: usize) -> bool {
    size.is_power_of_two() && (1..=1024).contains(&size)
}

/// Generate a sequence of tokens for testing with the given block size
///
/// # Arguments
/// * `base` - Starting token value
/// * `block_size` - Number of tokens to generate
///
/// # Returns
/// Vector of tokens: [base, base+1, base+2, ..., base+block_size-1]
///
/// # Example
/// ```
/// use dynamo_kvbm_logical::testing::config::generate_test_tokens;
///
/// let tokens = generate_test_tokens(100, 4);
/// assert_eq!(tokens, vec![100, 101, 102, 103]);
/// ```
pub fn generate_test_tokens(base: u32, block_size: usize) -> Vec<u32> {
    (base..base + block_size as u32).collect()
}

/// Generate unique token sequences for multiple blocks
///
/// # Arguments
/// * `count` - Number of token sequences to generate
/// * `block_size` - Size of each token sequence
///
/// # Returns
/// Vector of token vectors, each with unique values
///
/// # Example
/// ```
/// use dynamo_kvbm_logical::testing::config::generate_unique_token_sequences;
///
/// let sequences = generate_unique_token_sequences(2, 4);
/// assert_eq!(sequences.len(), 2);
/// assert_eq!(sequences[0], vec![0, 1, 2, 3]);
/// assert_eq!(sequences[1], vec![1000, 1001, 1002, 1003]);
/// ```
pub fn generate_unique_token_sequences(count: usize, block_size: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|i| generate_test_tokens(i as u32 * 1000, block_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_test_block_size() {
        // Valid sizes
        assert!(validate_test_block_size(1));
        assert!(validate_test_block_size(4));
        assert!(validate_test_block_size(16));
        assert!(validate_test_block_size(1024));

        // Invalid sizes
        assert!(!validate_test_block_size(0));
        assert!(!validate_test_block_size(3)); // Not power of 2
        assert!(!validate_test_block_size(5)); // Not power of 2
        assert!(!validate_test_block_size(2048)); // Too large
    }

    #[test]
    fn test_generate_test_tokens() {
        let tokens = generate_test_tokens(100, 4);
        assert_eq!(tokens, vec![100, 101, 102, 103]);

        let tokens = generate_test_tokens(0, 1);
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_generate_unique_token_sequences() {
        let sequences = generate_unique_token_sequences(3, 2);
        assert_eq!(sequences.len(), 3);
        assert_eq!(sequences[0], vec![0, 1]);
        assert_eq!(sequences[1], vec![1000, 1001]);
        assert_eq!(sequences[2], vec![2000, 2001]);

        // Ensure all sequences are unique
        for i in 0..sequences.len() {
            for j in (i + 1)..sequences.len() {
                assert_ne!(sequences[i], sequences[j]);
            }
        }
    }

    #[test]
    fn test_constants_are_valid() {
        assert!(validate_test_block_size(constants::TINY));
        assert!(validate_test_block_size(constants::SMALL));
        assert!(validate_test_block_size(constants::MEDIUM));
        assert!(validate_test_block_size(constants::LARGE));
        assert!(validate_test_block_size(constants::MAX));
    }

    #[test]
    fn test_block_sizes_arrays_are_valid() {
        for &size in TEST_BLOCK_SIZES {
            assert!(
                validate_test_block_size(size),
                "Invalid test block size: {}",
                size
            );
        }

        for &size in COMMON_TEST_BLOCK_SIZES {
            assert!(
                validate_test_block_size(size),
                "Invalid common test block size: {}",
                size
            );
        }
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_TEST_BLOCK_SIZE, 4);
        assert_eq!(DEFAULT_TEST_BLOCK_COUNT, 10);
        assert!(validate_test_block_size(DEFAULT_TEST_BLOCK_SIZE));
    }
}
