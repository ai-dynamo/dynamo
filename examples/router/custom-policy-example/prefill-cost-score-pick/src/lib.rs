// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `prefill-cost-score-pick` policy.
//!
//! The policy prices prefill work with a quadratic cost model,
//! `cost(q, c) = uncached_squared * q * q + uncached_times_cached * q * c + uncached * q`,
//! where `q` is a request's uncached prompt tokens and `c` its cached prompt tokens. A worker's
//! score is the cost of the prefills it is running plus the incoming request's cost on it.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;
use picker::LowestCostPicker;
use scorer::{PrefillCostModel, PrefillCostScorer};

/// Coefficients of the cost model, fitted for one model and GPU type.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    /// Cost per `q * q`.
    uncached_squared: f64,
    /// Cost per `q * c`.
    uncached_times_cached: f64,
    /// Cost per `q`.
    uncached: f64,
}

fn cost_model(
    parameters: Parameters,
) -> Result<PrefillCostModel, WorkerSelectionPolicyProviderError> {
    let Parameters {
        uncached_squared,
        uncached_times_cached,
        uncached,
    } = parameters;
    for (name, value) in [
        ("uncached_squared", uncached_squared),
        ("uncached_times_cached", uncached_times_cached),
        ("uncached", uncached),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(format!(
                "{name} must be a finite non-negative number"
            )));
        }
    }
    Ok(PrefillCostModel {
        uncached_squared,
        uncached_times_cached,
        uncached,
    })
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let model = cost_model(parameters.deserialize()?)?;

    Ok(Arc::new(
        move |config: &SelectionConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                vec![Box::new(PrefillCostScorer { model })],
                Box::new(LowestCostPicker),
            )
        },
    ))
}

/// Register the `prefill-cost-score-pick` policy type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("prefill-cost-score-pick", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(uncached_squared: f64) -> Parameters {
        Parameters {
            uncached_squared,
            uncached_times_cached: 2e-7,
            uncached: 0.02,
        }
    }

    #[test]
    fn validates_coefficients() {
        assert!(cost_model(parameters(1e-7)).is_ok());
        assert!(cost_model(parameters(0.0)).is_ok());
        assert!(cost_model(parameters(-1e-7)).is_err());
        assert!(cost_model(parameters(f64::NAN)).is_err());
        assert!(cost_model(parameters(f64::INFINITY)).is_err());
    }
}
