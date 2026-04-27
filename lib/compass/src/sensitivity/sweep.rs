// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::{PerturbationSpec, SensitivityMatrix, SweepConfig, SweepResult};

pub fn run_sensitivity_sweep(config: &SweepConfig) -> SensitivityMatrix {
    let mut results = Vec::new();

    for perturbation in &config.perturbations {
        for &multiplier in &perturbation.multipliers {
            for &concurrency in &config.concurrency_levels {
                let result = simulate_perturbation(&perturbation.component, multiplier, concurrency);
                results.push(result);
            }
        }
    }

    SensitivityMatrix {
        sweep_config: config.clone(),
        results,
    }
}

fn simulate_perturbation(component: &str, multiplier: f64, concurrency: usize) -> SweepResult {
    let base_ttft_p99 = 620.0;
    let base_throughput = 50.0;

    let component_contribution = match component {
        c if c.contains("kvbm") => 0.47,
        c if c.contains("nixl") => 0.25,
        c if c.contains("prefill") => 0.20,
        c if c.contains("router") => 0.08,
        _ => 0.10,
    };

    let component_latency_change = (multiplier - 1.0) * component_contribution;
    let predicted_ttft = base_ttft_p99 * (1.0 + component_latency_change);

    let concurrency_factor = 1.0 + (concurrency as f64 / 64.0 - 1.0).max(0.0) * 0.15;
    let predicted_ttft = predicted_ttft * concurrency_factor;

    let throughput_factor = if predicted_ttft > 0.0 {
        base_ttft_p99 / predicted_ttft
    } else {
        1.0
    };
    let predicted_throughput = base_throughput * throughput_factor * (concurrency as f64 / 32.0).min(2.0);

    SweepResult {
        perturbation: component.to_string(),
        multiplier,
        concurrency,
        predicted_ttft_p99_ms: predicted_ttft.max(1.0),
        predicted_throughput_rps: predicted_throughput.max(0.1),
    }
}

pub fn parse_sweep_spec(spec: &str) -> anyhow::Result<PerturbationSpec> {
    let parts: Vec<&str> = spec.splitn(2, '=').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid sweep spec '{}'. Expected format: component=val1,val2,...", spec);
    }

    let component = parts[0].to_string();
    let multipliers: Vec<f64> = parts[1]
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Invalid multiplier in '{}': {}", spec, e))?;

    Ok(PerturbationSpec {
        component,
        multipliers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_produces_results() {
        let config = SweepConfig {
            trace_source: "test.jsonl".to_string(),
            perturbations: vec![PerturbationSpec {
                component: "kvbm-allocate-ms".to_string(),
                multipliers: vec![0.5, 1.0, 2.0],
            }],
            concurrency_levels: vec![16, 32],
        };
        let matrix = run_sensitivity_sweep(&config);
        assert_eq!(matrix.results.len(), 6); // 3 multipliers * 2 concurrency levels
    }

    #[test]
    fn test_lower_latency_improves_ttft() {
        let config = SweepConfig {
            trace_source: "test.jsonl".to_string(),
            perturbations: vec![PerturbationSpec {
                component: "kvbm-allocate-ms".to_string(),
                multipliers: vec![0.5, 1.0, 2.0],
            }],
            concurrency_levels: vec![32],
        };
        let matrix = run_sensitivity_sweep(&config);
        let ttft_at_half: f64 = matrix.results[0].predicted_ttft_p99_ms;
        let ttft_at_one: f64 = matrix.results[1].predicted_ttft_p99_ms;
        let ttft_at_two: f64 = matrix.results[2].predicted_ttft_p99_ms;
        assert!(ttft_at_half < ttft_at_one);
        assert!(ttft_at_one < ttft_at_two);
    }

    #[test]
    fn test_parse_sweep_spec() {
        let spec = parse_sweep_spec("kvbm-allocate-ms=0.5,1.0,2.0,4.0").unwrap();
        assert_eq!(spec.component, "kvbm-allocate-ms");
        assert_eq!(spec.multipliers.len(), 4);
        assert!((spec.multipliers[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_parse_sweep_spec_invalid() {
        assert!(parse_sweep_spec("no-equals-sign").is_err());
        assert!(parse_sweep_spec("comp=abc").is_err());
    }
}
