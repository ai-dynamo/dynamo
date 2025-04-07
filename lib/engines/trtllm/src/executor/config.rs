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

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default, Builder)]
pub struct ExecutorConfig {
    model_path: String,

    #[builder(default = "LogLevel::Error")]
    log_level: LogLevel,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    enable_chunked_context: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    normalize_log_probs: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    iter_stats_max_iterations: Option<u32>,

    /// The number of processes for tensor parallelism. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    tensor_parallel_size: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum LogLevel {
    #[default]
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<&str> for LogLevel {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "error" => LogLevel::Error,
            "warn" => LogLevel::Warn,
            "info" => LogLevel::Info,
            "debug" => LogLevel::Debug,
            "trace" => LogLevel::Trace,
            _ => LogLevel::default(), // Default to Error if no match
        }
    }
}

impl ExecutorConfig {
    pub fn builder() -> ExecutorConfigBuilder {
        ExecutorConfigBuilder::default()
    }

    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            log_level: LogLevel::Error,
            enable_chunked_context: None,
            normalize_log_probs: None,
            iter_stats_max_iterations: None,
            tensor_parallel_size: None,
        }
    }
}
