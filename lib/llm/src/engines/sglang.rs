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

use std::path::Path;
use std::sync::Arc;

use crate::backend::ExecutionContext;
use dynamo_runtime::pipeline::error as pipeline_error;
use dynamo_runtime::CancellationToken;

mod worker;

mod engine;
use engine::SgLangEngine;

mod subprocess;
pub use subprocess::run_subprocess;

pub async fn make_engine(
    cancel_token: CancellationToken,
    // Full path to the model directory
    model_path: &Path,
    // Unique string to name zmq sockets
    sock_code: &str,
    // Multi node settings
    node_conf: super::MultiNodeConfig,
    // How many GPUs to use
    tensor_parallel_size: u32,
    // The base GPU ID to start allocating GPUs from
    base_gpu_id: u32,
) -> pipeline_error::Result<(ExecutionContext, tokio::task::JoinHandle<()>)> {
    let mut engine = SgLangEngine::new(
        cancel_token,
        sock_code,
        model_path,
        node_conf,
        tensor_parallel_size,
        base_gpu_id,
    )
    .await?;
    let sglang_process = engine.take_sglang_worker_handle();
    let engine: ExecutionContext = Arc::new(engine);
    Ok((engine, sglang_process))
}

#[derive(Debug, Clone, Copy)]
pub struct MultiGPUConfig {
    /// How many GPUs we are using / how many processes
    pub tp_size: u32,
    /// Tensor Parallel Rank. Must be unique across all nodes and GPUs.
    pub tp_rank: u32,
    /// GPU ID. Which GPU to run on. In single-node setup this is the same as tp_rank.
    pub gpu_id: u32,
}

impl Default for MultiGPUConfig {
    fn default() -> Self {
        MultiGPUConfig {
            tp_size: 1,
            tp_rank: 0,
            gpu_id: 0,
        }
    }
}
