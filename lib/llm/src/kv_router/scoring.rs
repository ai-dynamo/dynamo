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
use std::collections::HashSet;

use crate::kv_router::scheduler::Endpoint;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ProcessedEndpoints {
    pub endpoints: Vec<Endpoint>,
    pub worker_ids: Vec<i64>,
    pub load_avg: f64,
    pub load_std: f64,
}

impl ProcessedEndpoints {
    pub fn new(endpoints: Vec<Endpoint>) -> Self {
        // compute some basic statistics
        let load_values: Vec<f64> = endpoints
            .iter()
            .map(|x| x.data.kv_active_blocks as f64)
            .collect();
        let load_avg = load_values.iter().copied().sum::<f64>() / load_values.len() as f64;
        let variance = load_values
            .iter()
            .map(|&x| (x - load_avg).powi(2))
            .sum::<f64>()
            / load_values.len() as f64;
        let load_std = variance.sqrt();

        let worker_ids: HashSet<i64> = endpoints.iter().map(|x| x.worker_id()).collect();
        let worker_ids: Vec<i64> = worker_ids.into_iter().collect();

        ProcessedEndpoints {
            endpoints,
            worker_ids,
            load_avg,
            load_std,
        }
    }
}
