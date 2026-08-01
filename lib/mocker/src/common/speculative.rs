// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use aisimulate_engine::normalize_conditional_accept_rates;

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

#[cfg(test)]
mod tests {
    use super::*;

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
