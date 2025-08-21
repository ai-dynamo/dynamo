// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use dynamo_runtime::metrics::MetricsRegistry;
use prometheus::IntCounter;

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    pub offload_requests: IntCounter,
    pub save_kv_layer_requests: IntCounter,
}

impl KvbmMetrics {
    pub fn new(mr: &dyn MetricsRegistry) -> Self {
        let offload_requests = mr
            .create_intcounter("offload_requests", "The number of offload requests", &[])
            .unwrap();
        let save_kv_layer_requests = mr
            .create_intcounter(
                "save_kv_layer_requests",
                "The number of save kv layer requests",
                &[],
            )
            .unwrap();
        Self {
            offload_requests,
            save_kv_layer_requests,
        }
    }
}
