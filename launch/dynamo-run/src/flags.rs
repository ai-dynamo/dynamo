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

use std::path::PathBuf;
use std::str::FromStr;

/// Required options depend on the in and out choices
#[derive(clap::Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Flags {
    /// The model. The options depend on the engine.
    ///
    /// The full list - only mistralrs supports all three currently:
    /// - Full path to a GGUF file
    /// - Full path of a checked out Hugging Face repository containing safetensor files
    /// - Name of a Hugging Face repository, e.g 'google/flan-t5-small'. The model will be
    /// dowloaded and cached.
    /// For the `echo_full` engine omit the flag.
    #[arg(index = 1)]
    pub model_path_pos: Option<PathBuf>,

    // `--model-path`. The one above is `dynamo-run <positional-model-path>`
    #[arg(long = "model-path")]
    pub model_path_flag: Option<PathBuf>,

    /// HTTP port. `in=http` only
    #[arg(long, default_value = "8080")]
    pub http_port: u16,

    /// The name of the model we are serving
    #[arg(long)]
    pub model_name: Option<String>,

    /// llamacpp only
    ///
    /// The path to the tokenizer and model config because:
    /// - llama_cpp only runs GGUF files
    /// - our engine is a 'core' engine in that we do the tokenization, so we need the vocab
    /// - TODO: we don't yet extract that from the GGUF. Once we do we can remove this flag.
    #[arg(long)]
    pub model_config: Option<PathBuf>,

    /// sglang, vllm, trtllm
    ///
    /// How many GPUs to use at once, total across all nodes.
    /// This must divide by num_nodes, and each node must use the same number of GPUs.
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u32).range(1..256))]
    pub tensor_parallel_size: u32,

    /// sglang only
    /// vllm uses CUDA_VISIBLE_DEVICES env var
    ///
    /// Use GPUs from this ID upwards.
    /// If your machine has four GPUs but the first two (0 and 1) are in use,
    /// pass --base-gpu-id 2 to use the third GPU (and up, if tensor_parallel_size > 1)
    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u32).range(0..256))]
    pub base_gpu_id: u32,

    /// vllm and sglang only
    ///
    /// How many nodes/hosts to use
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u32).range(1..256))]
    pub num_nodes: u32,

    /// vllm and sglang only
    ///
    /// This nodes' unique ID, running from 0 to num_nodes.
    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u32).range(0..255))]
    pub node_rank: u32,

    /// For multi-node / pipeline parallel this is the <host>:<port> of the first node.
    ///
    /// - vllm: The address/port of the Ray head node.
    ///
    /// - sglang: The Torch Distributed init method address, in format <host>:<port>.
    ///   It becomes "tcp://<host>:<port>" when given to torch.distributed.init_process_group.
    ///   This expects to use the nccl backend (transparently to us here).
    ///   All nodes must use the same address here, which is node_rank == 0's address.
    ///
    #[arg(long)]
    pub leader_addr: Option<String>,

    /// Internal use only.
    // Start the python vllm engine sub-process.
    #[arg(long)]
    #[clap(hide = true, default_value = "false")]
    pub internal_vllm_process: bool,

    /// Internal use only.
    /// Start the sglang Python sub-process.
    /// The params in the tuple are:
    /// - the fd of the write end of a pipe where sglang will signal that it's ready.
    /// - the node rank (0 for first host, 1 for second host, etc)
    /// - the workers' rank (globally unique)
    /// - the GPU to use (locally unique)
    #[arg(long)]
    #[clap(hide = true, value_parser = parse_sglang_flags)]
    pub internal_sglang_process: Option<SgLangFlags>,
}

#[derive(Debug, Clone, Copy)]
pub struct SgLangFlags {
    pub pipe_fd: u32,
    pub tp_rank: u32,
    pub gpu_id: u32,
}
fn parse_sglang_flags(s: &str) -> Result<SgLangFlags, String> {
    let nums: Vec<u32> = s
        .split(',')
        .map(u32::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if nums.len() != 3 {
        return Err("Need exactly 3 numbers".into());
    }

    Ok(SgLangFlags {
        pipe_fd: nums[0],
        tp_rank: nums[1],
        gpu_id: nums[2],
    })
}
