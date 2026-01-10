// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock model runner for deterministic token generation.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Deterministic model output generator.
///
/// Uses ChaCha8Rng for reproducible random number generation.
/// Same seed always produces the same sequence of tokens.
pub struct MockModelRunner {
    rng: ChaCha8Rng,
    vocab_size: u32,
}

impl MockModelRunner {
    /// Create a new mock model runner.
    ///
    /// # Arguments
    /// * `seed` - Random seed for reproducibility
    /// * `vocab_size` - Vocabulary size for token generation
    pub fn new(seed: u64, vocab_size: u32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            vocab_size,
        }
    }

    /// Generate one token for each scheduled request.
    ///
    /// # Arguments
    /// * `request_ids` - IDs of scheduled requests
    ///
    /// # Returns
    /// Map from request ID to generated tokens (one token per request)
    pub fn generate(&mut self, request_ids: &[String]) -> HashMap<String, Vec<u32>> {
        request_ids
            .iter()
            .map(|id| {
                let token = self.rng.random_range(0..self.vocab_size);
                (id.clone(), vec![token])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_output() {
        let request_ids: Vec<String> = vec!["req-1".into(), "req-2".into()];

        // Generate with seed 42
        let mut runner1 = MockModelRunner::new(42, 50257);
        let output1 = runner1.generate(&request_ids);

        // Generate again with same seed
        let mut runner2 = MockModelRunner::new(42, 50257);
        let output2 = runner2.generate(&request_ids);

        assert_eq!(output1, output2);
    }

    #[test]
    fn test_different_seeds_different_output() {
        let request_ids: Vec<String> = vec!["req-1".into()];

        let mut runner1 = MockModelRunner::new(42, 50257);
        let output1 = runner1.generate(&request_ids);

        let mut runner2 = MockModelRunner::new(99, 50257);
        let output2 = runner2.generate(&request_ids);

        // Very unlikely to be the same
        assert_ne!(output1, output2);
    }

    #[test]
    fn test_tokens_in_vocab_range() {
        let request_ids: Vec<String> = (0..100).map(|i| format!("req-{i}")).collect();
        let vocab_size = 1000u32;

        let mut runner = MockModelRunner::new(42, vocab_size);
        let output = runner.generate(&request_ids);

        for tokens in output.values() {
            for &token in tokens {
                assert!(token < vocab_size);
            }
        }
    }
}
