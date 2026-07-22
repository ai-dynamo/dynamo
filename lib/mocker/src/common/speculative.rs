// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_CONDITIONAL_ACCEPT_RATES: [f64; 5] = [0.85, 0.3, 0.0, 0.0, 0.0];

pub(crate) fn normalize_conditional_accept_rates(
    nextn: usize,
    rates: Option<&str>,
) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(
        (1..=5).contains(&nextn),
        "aic_nextn must be in 1..=5, got {nextn}"
    );
    let mut parsed = match rates.map(str::trim).filter(|rates| !rates.is_empty()) {
        Some(rates) => rates
            .split(',')
            .map(|rate| {
                rate.trim().parse::<f64>().map_err(|error| {
                    anyhow::anyhow!("invalid aic_nextn_accept_rates value {rate:?}: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        None => DEFAULT_CONDITIONAL_ACCEPT_RATES.to_vec(),
    };

    for (index, rate) in parsed.iter().copied().enumerate() {
        anyhow::ensure!(
            rate.is_finite() && (0.0..=1.0).contains(&rate),
            "aic_nextn_accept_rates[{index}] must be finite and in [0, 1], got {rate}"
        );
    }

    parsed.resize(nextn, 0.0);
    Ok(parsed)
}

pub(crate) fn format_accept_rates(rates: &[f64]) -> String {
    rates
        .iter()
        .map(|rate| rate.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn undiscounted_aic_accept_rates(nextn: Option<usize>) -> Option<String> {
    nextn.map(|nextn| vec!["0"; nextn].join(","))
}

pub(crate) struct SpeculativeDecodeSampler {
    conditional_accept_rates: Vec<f64>,
    rng: StdRng,
}

impl SpeculativeDecodeSampler {
    pub(crate) fn new(conditional_accept_rates: Vec<f64>, seed: u64) -> Self {
        Self {
            conditional_accept_rates,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub(crate) fn sample_output_tokens(&mut self, remaining_output_tokens: usize) -> usize {
        if remaining_output_tokens == 0 {
            return 0;
        }

        let mut output_tokens = 1;
        for rate in &self.conditional_accept_rates {
            if !self.rng.random_bool(*rate) {
                break;
            }
            output_tokens += 1;
        }
        output_tokens.min(remaining_output_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_rates_are_exact() {
        let mut zero = SpeculativeDecodeSampler::new(vec![0.0, 1.0], 42);
        assert_eq!(zero.sample_output_tokens(10), 1);

        let mut one = SpeculativeDecodeSampler::new(vec![1.0, 1.0], 42);
        assert_eq!(one.sample_output_tokens(10), 3);
    }

    #[test]
    fn sampling_stops_at_first_rejection() {
        let mut sampler = SpeculativeDecodeSampler::new(vec![1.0, 0.0, 1.0], 42);
        assert_eq!(sampler.sample_output_tokens(10), 2);
    }

    #[test]
    fn sampling_clamps_to_remaining_output() {
        let mut sampler = SpeculativeDecodeSampler::new(vec![1.0, 1.0], 42);
        assert_eq!(sampler.sample_output_tokens(2), 2);
        assert_eq!(sampler.sample_output_tokens(0), 0);
    }

    #[test]
    fn equal_seeds_produce_equal_streams() {
        let rates = vec![0.8, 0.6, 0.4];
        let mut left = SpeculativeDecodeSampler::new(rates.clone(), 123);
        let mut right = SpeculativeDecodeSampler::new(rates, 123);
        let left_samples: Vec<_> = (0..100).map(|_| left.sample_output_tokens(10)).collect();
        let right_samples: Vec<_> = (0..100).map(|_| right.sample_output_tokens(10)).collect();
        assert_eq!(left_samples, right_samples);
    }

    #[test]
    fn worker_seed_offsets_produce_distinct_streams() {
        let rates = vec![0.8, 0.6, 0.4];
        let mut left = SpeculativeDecodeSampler::new(rates.clone(), 42);
        let mut right = SpeculativeDecodeSampler::new(rates, 43);
        let left_samples: Vec<_> = (0..100).map(|_| left.sample_output_tokens(10)).collect();
        let right_samples: Vec<_> = (0..100).map(|_| right.sample_output_tokens(10)).collect();
        assert_ne!(left_samples, right_samples);
    }

    #[test]
    fn empirical_mean_matches_conditional_expectation() {
        let rates = vec![0.7, 0.5, 0.2];
        let expected = 1.0 + 0.7 + 0.7 * 0.5 + 0.7 * 0.5 * 0.2;
        let mut sampler = SpeculativeDecodeSampler::new(rates, 42);
        let samples = 200_000;
        let mean = (0..samples)
            .map(|_| sampler.sample_output_tokens(10) as f64)
            .sum::<f64>()
            / samples as f64;
        assert!(
            (mean - expected).abs() < 0.01,
            "mean={mean}, expected={expected}"
        );
    }

    #[test]
    fn rates_are_validated_then_padded_or_truncated() {
        assert_eq!(
            normalize_conditional_accept_rates(3, Some("1, 0.5")).unwrap(),
            vec![1.0, 0.5, 0.0]
        );
        assert_eq!(
            normalize_conditional_accept_rates(2, Some("1,0.5,0.25")).unwrap(),
            vec![1.0, 0.5]
        );
        for rates in ["nan", "inf", "-0.1", "1.1", "not-a-rate"] {
            assert!(normalize_conditional_accept_rates(1, Some(rates)).is_err());
        }
    }
}
