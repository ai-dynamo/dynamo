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

use std::{fmt, io::IsTerminal as _, path::PathBuf};

use crate::ENDPOINT_SCHEME;

const BATCH_PREFIX: &str = "batch:";

#[derive(PartialEq)]
pub enum Input {
    /// Run an OpenAI compatible HTTP server
    Http,

    /// Single prompt on stdin
    Stdin,

    /// Interactive chat
    Text,

    /// Pull requests from a namespace/component/endpoint path.
    Endpoint(String),

    /// Batch mode. Run all the prompts, write the outputs, exit.
    Batch(PathBuf),

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
            "stdin" => Ok(Input::Stdin),
            "none" => Ok(Input::None),
            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Input::Endpoint(path.to_string()))
            }
            batch_patch if batch_patch.starts_with(BATCH_PREFIX) => {
                let path = batch_patch.strip_prefix(BATCH_PREFIX).unwrap();
                Ok(Input::Batch(PathBuf::from(path)))
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
            Input::Stdin => "stdin",
            Input::Endpoint(path) => path,
            Input::Batch(path) => &path.display().to_string(),
            Input::None => "none",
        };
        write!(f, "{s}")
    }
}

impl Default for Input {
    fn default() -> Self {
        if std::io::stdin().is_terminal() {
            Input::Text
        } else {
            Input::Stdin
        }
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
    /// Alias for vllm0_8
    Vllm,

    #[cfg(feature = "vllm")]
    /// Run inference using vllm 0.8.X+
    Vllm0_8,

    #[cfg(feature = "vllm")]
    /// Run inference using vllm 0.7.X
    Vllm0_7,

    /// Run inference using a user supplied python file that accepts and returns
    /// strings. It does it's own pre-processing.
    #[cfg(feature = "python")]
    PythonStr(String),

    /// Run inference using a user supplied python file that accepts and returns
    /// tokens. We do the pre-processing.
    #[cfg(feature = "python")]
    PythonTok(String),
    //
    // DEVELOPER NOTE
    // If you add an engine add it to `available_engines` below, and to Default if it makes sense
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
            #[cfg(feature = "vllm")]
            "vllm0_8" => Ok(Output::Vllm0_8),
            #[cfg(feature = "vllm")]
            "vllm0_7" => Ok(Output::Vllm0_7),

            "echo_full" => Ok(Output::EchoFull),
            "echo_core" => Ok(Output::EchoCore),

            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Output::Endpoint(path.to_string()))
            }

            #[cfg(feature = "python")]
            python_str_gen if python_str_gen.starts_with(crate::PYTHON_STR_SCHEME) => {
                let path = python_str_gen
                    .strip_prefix(crate::PYTHON_STR_SCHEME)
                    .unwrap();
                Ok(Output::PythonStr(path.to_string()))
            }

            #[cfg(feature = "python")]
            python_tok_gen if python_tok_gen.starts_with(crate::PYTHON_TOK_SCHEME) => {
                let path = python_tok_gen
                    .strip_prefix(crate::PYTHON_TOK_SCHEME)
                    .unwrap();
                Ok(Output::PythonTok(path.to_string()))
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
            #[cfg(feature = "vllm")]
            Output::Vllm0_8 => "vllm0_8",
            #[cfg(feature = "vllm")]
            Output::Vllm0_7 => "vllm0_7",

            Output::EchoFull => "echo_full",
            Output::EchoCore => "echo_core",

            Output::Endpoint(path) => path,

            #[cfg(feature = "python")]
            Output::PythonStr(_) => "pystr",

            #[cfg(feature = "python")]
            Output::PythonTok(_) => "pytok",
        };
        write!(f, "{s}")
    }
}

/// Returns the engine to use if user did not say on cmd line.
/// Nearly always defaults to mistralrs which has no dependencies and we include by default.
/// If built with --no-default-features and a specific engine, default to that.
#[allow(unused_assignments, unused_mut)]
impl Default for Output {
    fn default() -> Self {
        // Default if no engines
        let mut out = Output::EchoFull;

        #[cfg(feature = "llamacpp")]
        {
            out = Output::LlamaCpp;
        }

        #[cfg(feature = "sglang")]
        {
            out = Output::SgLang;
        }

        #[cfg(feature = "vllm")]
        {
            out = Output::Vllm;
        }

        #[cfg(feature = "mistralrs")]
        {
            out = Output::MistralRs;
        }

        out
    }
}

impl Output {
    #[allow(unused_mut)]
    pub fn available_engines() -> Vec<String> {
        let mut out = vec!["echo_core".to_string(), "echo_full".to_string()];
        #[cfg(feature = "mistralrs")]
        {
            out.push(Output::MistralRs.to_string());
        }

        #[cfg(feature = "llamacpp")]
        {
            out.push(Output::LlamaCpp.to_string());
        }

        #[cfg(feature = "sglang")]
        {
            out.push(Output::SgLang.to_string());
        }

        #[cfg(feature = "vllm")]
        {
            out.push(Output::Vllm.to_string());
            out.push(Output::Vllm0_7.to_string());
            out.push(Output::Vllm0_8.to_string());
        }

        #[cfg(feature = "python")]
        {
            out.push(Output::PythonStr("file.py".to_string()).to_string());
            out.push(Output::PythonTok("file.py".to_string()).to_string());
        }

        out
    }
}
