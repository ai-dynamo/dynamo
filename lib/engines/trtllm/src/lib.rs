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

use std::sync::Arc;

use dynamo_llm::backend::ExecutionContext;
use dynamo_runtime::pipeline::error as pipeline_error;

pub mod executor;

/// Create a TRT-LLM engine.
pub fn make_engine<P: ToString>(
    // A full repo with .engine files, config.json,
    model_path: P,
    // How many GPUs to use
    tensor_parallel_size: u32,
) -> pipeline_error::Result<ExecutionContext> {
    let config = executor::config::ExecutorConfig::builder()
        .model_path(model_path.to_string())
        .tensor_parallel_size(Some(tensor_parallel_size))
        .build()?;
    let engine = executor::Executor::new(config)?;
    engine.start_response_processor();
    engine.start_kv_event_processor();
    engine.start_iteration_metrics_processor();
    let engine: ExecutionContext = Arc::new(engine);
    Ok(engine)
}
