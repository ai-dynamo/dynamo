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

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use clap::ValueEnum;
use dynamo_runtime::pipeline::RouterMode as RuntimeRouterMode;

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
    ///   downloaded and cached.
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

    /// Verbose output (-v for debug, -vv for trace)
    #[arg(short = 'v', action = clap::ArgAction::Count, default_value_t = 0)]
    pub verbosity: u8,

    /// llamacpp only
    ///
    /// The path to the tokenizer and model config because:
    /// - llama_cpp only runs GGUF files
    /// - our engine is a 'core' engine in that we do the tokenization, so we need the vocab
    /// - TODO: we don't yet extract that from the GGUF. Once we do we can remove this flag.
    #[arg(long)]
    pub model_config: Option<PathBuf>,

    /// sglang, vllm
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

    /// If using `out=dyn://..` with multiple backends, this says how to route the requests.
    ///
    /// Mostly interesting for KV-aware routing.
    /// Defaults to RouterMode::RoundRobin
    #[arg(long, default_value = "round-robin")]
    pub router_mode: RouterMode,

    /// Internal use only.
    // Start the python vllm engine sub-process.
    #[arg(long, hide = true, default_value = "false")]
    pub internal_vllm_process: bool,

    /// Internal use only.
    /// Start the sglang Python sub-process.
    /// The params in the tuple are:
    /// - the fd of the write end of a pipe where sglang will signal that it's ready.
    /// - the node rank (0 for first host, 1 for second host, etc)
    /// - the workers' rank (globally unique)
    /// - the GPU to use (locally unique)
    #[arg(long, hide = true, value_parser = parse_sglang_flags)]
    pub internal_sglang_process: Option<SgLangFlags>,

    /// Additional engine-specific arguments from a JSON file.
    /// Contains a mapping of parameter names to values.
    #[arg(long)]
    pub extra_engine_args: Option<PathBuf>,

    /// Path to a JSON file containing default request fields.
    /// These fields will be merged with each request, but can be overridden by the request.
    /// Example file contents:
    /// {
    ///     "model": "Qwen2.5-3B-Instruct",
    ///     "temperature": 0.7,
    ///     "max_completion_tokens": 4096
    /// }
    #[arg(long)]
    pub request_template: Option<PathBuf>,

    /// Everything after a `--`.
    /// These are the command line arguments to the python engine when using `pystr` or `pytok`.
    #[arg(index = 2, last = true, hide = true, allow_hyphen_values = true)]
    pub last: Vec<String>,
}

impl Flags {
    /// Convert the flags back to a command line. Including only the non-null values, but
    /// include the defaults. Includes the canonicalized model path and normalized model name.
    ///
    /// Used to pass arguments to python engines via `pystr` and `pytok`.
    pub fn as_vec(&self, path: &str, name: &str) -> Vec<String> {
        let mut out = vec![
            "--model-path".to_string(),
            path.to_string(),
            "--model-name".to_string(),
            name.to_string(),
            "--http-port".to_string(),
            self.http_port.to_string(),
            // Default 1
            "--tensor-parallel-size".to_string(),
            self.tensor_parallel_size.to_string(),
            // Default 0
            "--base-gpu-id".to_string(),
            self.base_gpu_id.to_string(),
            // Default 1
            "--num-nodes".to_string(),
            self.num_nodes.to_string(),
            // Default 0
            "--node-rank".to_string(),
            self.node_rank.to_string(),
        ];
        if let Some(model_config_path) = self.model_config.as_ref() {
            out.push("--model-config".to_string());
            out.push(model_config_path.display().to_string());
        }
        if let Some(leader) = self.leader_addr.as_ref() {
            out.push("--leader-addr".to_string());
            out.push(leader.to_string());
        }
        if let Some(extra_engine_args) = self.extra_engine_args.as_ref() {
            out.push("--extra-engine-args".to_string());
            out.push(extra_engine_args.display().to_string());
        }
        out.extend(self.last.clone());
        out
    }

    /// Load extra engine arguments from a JSON file
    /// Returns a HashMap of parameter names to values
    pub fn load_extra_engine_args(
        &self,
    ) -> anyhow::Result<Option<HashMap<String, serde_json::Value>>> {
        if let Some(path) = &self.extra_engine_args {
            let file_content = std::fs::read_to_string(path)?;
            let args: HashMap<String, serde_json::Value> = serde_json::from_str(&file_content)?;
            Ok(Some(args))
        } else {
            Ok(None)
        }
    }
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

#[derive(Default, PartialEq, Eq, ValueEnum, Clone, Debug)]
pub enum RouterMode {
    #[default]
    Random,
    #[value(name = "round-robin")]
    RoundRobin,
    #[value(name = "kv")]
    KV,
}

impl RouterMode {
    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }
}

impl From<RouterMode> for RuntimeRouterMode {
    fn from(r: RouterMode) -> RuntimeRouterMode {
        match r {
            RouterMode::RoundRobin => RuntimeRouterMode::RoundRobin,
            RouterMode::KV => todo!("KV not implemented yet"),
            _ => RuntimeRouterMode::Random,
        }
    }
}
