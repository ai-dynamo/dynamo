// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Scoring functions for the KV router.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::kv_router::protocols::LoadMetrics;
use crate::kv_router::scheduler::Endpoint;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ProcessedEndpoints {
    pub endpoints: HashMap<i64, Endpoint>,
    pub load_avg: f64,
    pub load_std: f64,
}

impl ProcessedEndpoints {
    pub fn new(endpoints: Vec<Endpoint>) -> Self {
        // compute some basic statistics
        let load_values: Vec<f64> = endpoints
            .iter()
            .map(|x| x.data.kv_active_blocks() as f64)
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

    pub fn worker_ids(&self) -> Vec<i64> {
        self.endpoints.keys().copied().collect()
    }

    pub fn active_blocks(&self) -> HashMap<i64, usize> {
        self.endpoints
            .iter()
            .map(|(&worker_id, endpoint)| (worker_id, endpoint.data.kv_active_blocks() as usize))
            .collect()
    }

    pub fn update_active_blocks(&mut self, worker_id: i64, active_blocks: usize) {
        if let Some(endpoint) = self.endpoints.get_mut(&worker_id) {
            match &mut endpoint.data {
                LoadMetrics::ForwardPassMetrics(metrics) => {
                    metrics.kv_active_blocks = active_blocks as u64;
                }
                LoadMetrics::PredictiveLoadMetrics(metrics) => {
                    metrics.kv_active_blocks = active_blocks as u64;
                }
            }
        }
    }

    pub fn update_active_blocks_all(&mut self, worker_blocks: HashMap<i64, usize>) {
        for (worker_id, blocks) in worker_blocks {
            self.update_active_blocks(worker_id, blocks);
        }
    }
}
