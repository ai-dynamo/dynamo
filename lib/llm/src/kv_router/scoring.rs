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

use crate::kv_router::protocols::{DpRank, ForwardPassMetrics, WorkerDp};

/// [gluo FIXME] exactly the same as EndpointInfo except that 'data'
/// is cleaned (not optional)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub name: String,
    // contains dp
    pub subject: String,
    // one set of metrics for each dp worker
    pub data: ForwardPassMetrics,
}

impl Endpoint {
    pub fn worker_id(&self) -> i64 {
        i64::from_str_radix(
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

    pub fn dp_rank(&self) -> Option<DpRank> {
        let parts: Vec<&str> = self.subject.split("-").collect();
        if parts.len() < 3 {
            return None;
        }
        let second_to_last = parts[parts.len() - 2];
        second_to_last.parse::<DpRank>().ok()
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ProcessedEndpoints {
    pub endpoints: HashMap<WorkerDp, Endpoint>,
    pub load_avg: f64,
    pub load_std: f64,
}

impl ProcessedEndpoints {
    pub fn new(endpoints: Vec<Endpoint>) -> Self {
        // compute some basic statistics
        let load_values: Vec<f64> = endpoints
            .iter()
            .map(|endpoint| endpoint.data.kv_active_blocks as f64)
            .collect();
        if load_values.is_empty() {
            // TODO we hit this panic while vLLM is starting the ranks up. Need to avoid this
            panic!("No endpoints to process!")
        };
        let load_avg = load_values.iter().copied().sum::<f64>() / load_values.len() as f64;
        let variance = load_values
            .iter()
            .map(|&x| (x - load_avg).powi(2))
            .sum::<f64>()
            / load_values.len() as f64;
        let load_std = variance.sqrt();

        // pass in (worker_id, dp_rank) tuple
        let endpoints = endpoints
            .into_iter()
            .map(|e| {
                (
                    WorkerDp {
                        worker_id: e.worker_id(),
                        dp_rank: e.dp_rank(),
                    },
                    e,
                )
            })
            .collect();

        ProcessedEndpoints {
            endpoints,
            load_avg,
            load_std,
        }
    }
}
