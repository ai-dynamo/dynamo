// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scoring functions for the KV router.

use super::protocols::LoadMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// [gluo FIXME] exactly the same as EndpointInfo except that 'data'
/// is cleaned (not optional)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Endpoint {
    pub name: String,
    pub subject: String,
    pub data: LoadMetrics,
}

impl Endpoint {
    pub fn worker_id(&self) -> u64 {
        u64::from_str_radix(
            self.subject
                .split("-")
                .last()
                .expect("invalid subject")
                .to_string()
                .as_str(),
            16,
        )
        .expect("invalid worker id")
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct ProcessedEndpoints {
    pub endpoints: HashMap<u64, Endpoint>,
    pub load_avg: f64,
    pub load_std: f64,
}

impl ProcessedEndpoints {
    pub fn new(endpoints: Vec<Endpoint>) -> Self {
        // compute some basic statistics
        let load_values: Vec<f64> = endpoints
            .iter()
            .map(|endpoint| endpoint.data.kv_active_blocks() as f64)
            .collect();
        let load_avg = load_values.iter().copied().sum::<f64>() / load_values.len() as f64;
        let variance = load_values
            .iter()
            .map(|&x| (x - load_avg).powi(2))
            .sum::<f64>()
            / load_values.len() as f64;
        let load_std = variance.sqrt();

        let endpoints = endpoints.into_iter().map(|e| (e.worker_id(), e)).collect();

        ProcessedEndpoints {
            endpoints,
            load_avg,
            load_std,
        }
    }

    pub fn worker_ids(&self) -> Vec<u64> {
        self.endpoints.keys().copied().collect()
    }

    pub fn active_blocks(&self) -> HashMap<u64, usize> {
        self.endpoints
            .iter()
            .map(|(&worker_id, endpoint)| (worker_id, endpoint.data.kv_active_blocks() as usize))
            .collect()
    }
}
