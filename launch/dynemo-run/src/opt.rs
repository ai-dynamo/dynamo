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

use std::fmt;

use crate::ENDPOINT_SCHEME;

pub enum Input {
    /// Run an OpenAI compatible HTTP server
    Http,

    /// Read prompt from stdin
    Text,

    /// Pull requests from a namespace/component/endpoint path.
    Endpoint(String),

    /// Start the engine but don't provide any way to talk to it.
    /// For multi-node sglang, where the engine connects directly
    /// to the co-ordinator via torch distributed / nccl.
    None,
}

impl TryFrom<&str> for Input {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            "http" => Ok(Input::Http),
            "text" => Ok(Input::Text),
            "none" => Ok(Input::None),
            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Input::Endpoint(path.to_string()))
            }
            e => Err(anyhow::anyhow!("Invalid in= option '{e}'")),
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Input::Http => "http",
            Input::Text => "text",
            Input::Endpoint(path) => path,
            Input::None => "none",
        };
        write!(f, "{s}")
    }
}

pub enum Output {
    /// Accept un-preprocessed requests, echo the prompt back as the response
    EchoFull,

    /// Accept preprocessed requests, echo the tokens back as the response
    EchoCore,

    /// Publish requests to a namespace/component/endpoint path.
    Endpoint(String),

    #[cfg(feature = "mistralrs")]
    /// Run inference on a model in a GGUF file using mistralrs w/ candle
    MistralRs,

    #[cfg(feature = "sglang")]
    /// Run inference using sglang
    SgLang,

    #[cfg(feature = "llamacpp")]
    /// Run inference using llama.cpp
    LlamaCpp,

    #[cfg(feature = "vllm")]
    /// Run inference using vllm's engine
    Vllm,

    #[cfg(feature = "trtllm")]
    /// Run inference using trtllm
    TrtLLM,
}

impl TryFrom<&str> for Output {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            #[cfg(feature = "mistralrs")]
            "mistralrs" => Ok(Output::MistralRs),

            #[cfg(feature = "sglang")]
            "sglang" => Ok(Output::SgLang),

            #[cfg(feature = "llamacpp")]
            "llamacpp" | "llama_cpp" => Ok(Output::LlamaCpp),

            #[cfg(feature = "vllm")]
            "vllm" => Ok(Output::Vllm),

            #[cfg(feature = "trtllm")]
            "trtllm" => Ok(Output::TrtLLM),

            "echo_full" => Ok(Output::EchoFull),
            "echo_core" => Ok(Output::EchoCore),

            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Output::Endpoint(path.to_string()))
            }

            e => Err(anyhow::anyhow!("Invalid out= option '{e}'")),
        }
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            #[cfg(feature = "mistralrs")]
            Output::MistralRs => "mistralrs",

            #[cfg(feature = "sglang")]
            Output::SgLang => "sglang",

            #[cfg(feature = "llamacpp")]
            Output::LlamaCpp => "llamacpp",

            #[cfg(feature = "vllm")]
            Output::Vllm => "vllm",

            #[cfg(feature = "trtllm")]
            Output::TrtLLM => "trtllm",

            Output::EchoFull => "echo_full",
            Output::EchoCore => "echo_core",

            Output::Endpoint(path) => path,
        };
        write!(f, "{s}")
    }
}
