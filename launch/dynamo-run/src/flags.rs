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

use clap::ValueEnum;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::kv_router::KvRouterConfig;
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::mocker::protocols::MockEngineArgs;
use dynamo_runtime::pipeline::RouterMode as RuntimeRouterMode;

use crate::Output;

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
    /// If tls_cert_path and tls_key_path are provided, this will be TLS/HTTPS.
    #[arg(long, default_value = "8080")]
    pub http_port: u16,

    /// TLS certificate file
    #[arg(long, requires = "tls_key_path")]
    pub tls_cert_path: Option<PathBuf>,

    /// TLS certificate key file
    #[arg(long, requires = "tls_cert_path")]
    pub tls_key_path: Option<PathBuf>,

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

    /// If using `out=dyn` with multiple instances, this says how to route the requests.
    ///
    /// Mostly interesting for KV-aware routing.
    /// Defaults to RouterMode::RoundRobin
    #[arg(long, default_value = "round-robin")]
    pub router_mode: RouterMode,

    /// Maximum number of batched tokens for KV routing
    /// Needed for informing the KV router
    /// NOTE: this is not actually used for now
    #[arg(long, default_value = "8192")]
    pub max_num_batched_tokens: Option<u32>,

    /// KV Router: Weight for overlap score in worker selection.
    /// Higher values prioritize KV cache reuse. Default: 1.0
    #[arg(long)]
    pub kv_overlap_score_weight: Option<f64>,

    /// KV Router: Temperature for worker sampling via softmax.
    /// Higher values promote more randomness, and 0 fallbacks to deterministic.
    /// Default: 0.0
    #[arg(long)]
    pub router_temperature: Option<f64>,

    /// KV Router: Whether to use KV events to maintain the view of cached blocks
    /// If false, would use ApproxKvRouter for predicting block creation / deletion
    /// based only on incoming requests at a timer.
    /// Default: true
    #[arg(long)]
    pub use_kv_events: Option<bool>,

    /// KV Router: Whether to enable replica synchronization across multiple router instances.
    /// When true, routers will publish and subscribe to events to maintain consistent state.
    /// Default: false
    #[arg(long)]
    pub router_replica_sync: Option<bool>,

    /// Max model context length. Reduce this if you don't have enough VRAM for the full model
    /// context length (e.g. Llama 4).
    /// Defaults to the model's max, which is usually model_max_length in tokenizer_config.json.
    #[arg(long)]
    pub context_length: Option<u32>,

    /// KV cache block size (is this used? Maybe by Python vllm worker?)
    #[arg(long)]
    pub kv_cache_block_size: Option<u32>,

    /// Mocker engine only.
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

    /// How many times a request can be migrated to another worker if the HTTP server lost
    /// connection to the current worker.
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..1024))]
    pub migration_limit: Option<u32>,

    /// Make this a static worker.
    /// Do not connect to or advertise self on etcd.
    /// in=dyn://x.y.z only
    #[arg(long, default_value = "false")]
    pub static_worker: bool,

    /// Everything after a `--`.
    /// These are the command line arguments to the python engine when using `pystr` or `pytok`.
    #[arg(index = 2, last = true, hide = true, allow_hyphen_values = true)]
    pub last: Vec<String>,
}

impl Flags {
    /// For each Output variant, check if it would be able to run.
    /// This takes validation out of the main engine creation path.
    pub fn validate(
        &self,
        local_model: &LocalModel,
        in_opt: &Input,
        out_opt: &Output,
    ) -> anyhow::Result<()> {
        match in_opt {
            Input::Endpoint(_) => {}
            _ => {
                if self.static_worker {
                    anyhow::bail!("'--static-worker true' only applies to in=dyn://x.y.z");
                }
            }
        }

        match out_opt {
            Output::Auto => {
                if self.context_length.is_some() {
                    anyhow::bail!(
                        "'--context-length' flag should only be used on the worker node, not on the ingress"
                    );
                }
                if self.kv_cache_block_size.is_some() {
                    anyhow::bail!(
                        "'--kv-cache-block-size' flag should only be used on the worker node, not on the ingress"
                    );
                }
                if self.migration_limit.is_some() {
                    anyhow::bail!(
                        "'--migration-limit' flag should only be used on the worker node, not on the ingress"
                    );
                }
            }
            Output::Static(_) => {
                if self.model_name.is_none()
                    || self
                        .model_path_pos
                        .as_ref()
                        .or(self.model_path_flag.as_ref())
                        .is_none()
                {
                    anyhow::bail!(
                        "out=dyn://<path> requires --model-name and --model-path, which are the name and path on disk of the model we expect to serve."
                    );
                }
            }
            Output::Echo => {}
            #[cfg(feature = "mistralrs")]
            Output::MistralRs => {}
            #[cfg(feature = "llamacpp")]
            Output::LlamaCpp => {
                if !local_model.path().is_file() {
                    anyhow::bail!(
                        "--model-path should refer to a GGUF file. llama_cpp does not support safetensors."
                    );
                }
            }
            Output::Mocker => {
                // nothing to check here
            }
        }

        match out_opt {
            Output::Mocker => {}
            _ => {
                if self.extra_engine_args.is_some() {
                    anyhow::bail!("`--extra-engine-args` is only for the mocker engine");
                }
            }
        }

        Ok(())
    }

    pub fn router_config(&self) -> RouterConfig {
        RouterConfig::new(
            self.router_mode.into(),
            KvRouterConfig::new(
                self.kv_overlap_score_weight,
                self.router_temperature,
                self.use_kv_events,
                self.router_replica_sync,
                self.max_num_batched_tokens,
                // defaulting below args (no longer maintaining new flags for dynamo-run)
                None,
                None,
            ),
        )
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

    pub fn mocker_config(&self) -> MockEngineArgs {
        let Some(path) = &self.extra_engine_args else {
            tracing::warn!("Did not specify extra engine args. Using default mocker args.");
            return MockEngineArgs::default();
        };
        MockEngineArgs::from_json_file(path)
            .unwrap_or_else(|e| panic!("Failed to build mocker engine args from {path:?}: {e}"))
    }
}

#[derive(Default, PartialEq, Eq, ValueEnum, Clone, Debug, Copy)]
pub enum RouterMode {
    #[default]
    #[value(name = "round-robin")]
    RoundRobin,
    Random,
    #[value(name = "kv")]
    KV,
}

impl From<RouterMode> for RuntimeRouterMode {
    fn from(r: RouterMode) -> RuntimeRouterMode {
        match r {
            RouterMode::RoundRobin => RuntimeRouterMode::RoundRobin,
            RouterMode::Random => RuntimeRouterMode::Random,
            RouterMode::KV => RuntimeRouterMode::KV,
        }
    }
}
