// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::FloorParams;

#[derive(Debug, Clone)]
pub struct FloorResult {
    pub component: String,
    pub observed_ms: f64,
    pub theoretical_floor_ms: f64,
    pub floor_ratio: f64,
    pub is_optimization_candidate: bool,
}

#[derive(Debug, Clone)]
pub struct FloorInput {
    pub component: String,
    pub observed_p99_ms: f64,
    pub formula: FloorFormula,
}

#[derive(Debug, Clone)]
pub enum FloorFormula {
    KvbmAllocate {
        hash_time_ms: f64,
        radix_depth: u32,
    },
    PrefillCompute {
        isl_tokens: u64,
        model_tflops: f64,
        flops_per_token: f64,
    },
    NixlTransfer {
        block_size_bytes: u64,
        bandwidth_gbps: f64,
    },
    Custom {
        floor_ms: f64,
    },
}

pub fn compute_floors(inputs: &[FloorInput], params: &FloorParams) -> Vec<FloorResult> {
    inputs
        .iter()
        .map(|input| {
            let floor_ms = match &input.formula {
                FloorFormula::KvbmAllocate {
                    hash_time_ms,
                    radix_depth,
                } => {
                    hash_time_ms + (*radix_depth as f64) * params.pointer_chase_ns / 1_000_000.0
                }
                FloorFormula::PrefillCompute {
                    isl_tokens,
                    model_tflops,
                    flops_per_token,
                } => {
                    if *model_tflops > 0.0 {
                        (*isl_tokens as f64 * flops_per_token) / (model_tflops * 1e9)
                    } else {
                        0.0
                    }
                }
                FloorFormula::NixlTransfer {
                    block_size_bytes,
                    bandwidth_gbps,
                } => {
                    if *bandwidth_gbps > 0.0 {
                        (*block_size_bytes as f64 * 8.0) / (bandwidth_gbps * 1e6)
                    } else {
                        0.0
                    }
                }
                FloorFormula::Custom { floor_ms } => *floor_ms,
            };

            let floor_ratio = if floor_ms > 0.0 {
                input.observed_p99_ms / floor_ms
            } else {
                1.0
            };

            FloorResult {
                component: input.component.clone(),
                observed_ms: input.observed_p99_ms,
                theoretical_floor_ms: floor_ms,
                floor_ratio,
                is_optimization_candidate: floor_ratio > params.optimization_candidate_threshold,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kvbm_floor() {
        let params = FloorParams::default();
        let inputs = vec![FloorInput {
            component: "kvbm.allocate".to_string(),
            observed_p99_ms: 3.51,
            formula: FloorFormula::KvbmAllocate {
                hash_time_ms: 0.1,
                radix_depth: 10,
            },
        }];
        let results = compute_floors(&inputs, &params);
        assert_eq!(results.len(), 1);
        assert!(results[0].floor_ratio > 2.0);
        assert!(results[0].is_optimization_candidate);
    }

    #[test]
    fn test_prefill_at_floor() {
        let params = FloorParams::default();
        let inputs = vec![FloorInput {
            component: "prefill.compute".to_string(),
            observed_p99_ms: 12.0,
            formula: FloorFormula::Custom { floor_ms: 11.9 },
        }];
        let results = compute_floors(&inputs, &params);
        assert!((results[0].floor_ratio - 1.008).abs() < 0.01);
        assert!(!results[0].is_optimization_candidate);
    }

    #[test]
    fn test_nixl_floor() {
        let params = FloorParams::default();
        let inputs = vec![FloorInput {
            component: "nixl.transfer.h2d".to_string(),
            observed_p99_ms: 2.4,
            formula: FloorFormula::NixlTransfer {
                block_size_bytes: 1024 * 1024,
                bandwidth_gbps: 100.0,
            },
        }];
        let results = compute_floors(&inputs, &params);
        assert!(results[0].theoretical_floor_ms > 0.0);
        assert!(results[0].floor_ratio > 1.0);
    }
}
