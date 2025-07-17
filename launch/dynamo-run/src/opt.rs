// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::ENDPOINT_SCHEME;
use std::fmt;

pub enum Output {
    /// Accept un-preprocessed requests, echo the prompt back as the response
    EchoFull,

    /// Accept preprocessed requests, echo the tokens back as the response
    EchoCore,

    /// Listen for models on nats/etcd, add/remove dynamically
    Dynamic,

    #[cfg(feature = "mistralrs")]
    /// Run inference on a model in a GGUF file using mistralrs w/ candle
    MistralRs,

    #[cfg(feature = "llamacpp")]
    /// Run inference using llama.cpp
    LlamaCpp,

    /// Run inference using sglang
    SgLang,

    /// Run inference using trtllm
    Trtllm,

    // Start vllm in a sub-process connecting via nats
    // Sugar for `python vllm_inc.py --endpoint <thing> --model <thing>`
    Vllm,

    Mocker,
}

impl TryFrom<&str> for Output {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            #[cfg(feature = "mistralrs")]
            "mistralrs" => Ok(Output::MistralRs),

            #[cfg(feature = "llamacpp")]
            "llamacpp" | "llama_cpp" => Ok(Output::LlamaCpp),

            "sglang" => Ok(Output::SgLang),
            "trtllm" => Ok(Output::Trtllm),
            "vllm" => Ok(Output::Vllm),
            "mocker" => Ok(Output::Mocker),

            "echo_full" => Ok(Output::EchoFull),
            "echo_core" => Ok(Output::EchoCore),

            "dyn" => Ok(Output::Dynamic),

            // Deprecated, should only use `out=dyn`
            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                tracing::warn!(
                    "out=dyn://<path> is deprecated, the path is not used. Please use 'out=dyn'"
                );
                //let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Output::Dynamic)
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

            #[cfg(feature = "llamacpp")]
            Output::LlamaCpp => "llamacpp",

            Output::SgLang => "sglang",
            Output::Trtllm => "trtllm",
            Output::Vllm => "vllm",
            Output::Mocker => "mocker",

            Output::EchoFull => "echo_full",
            Output::EchoCore => "echo_core",

            Output::Dynamic => "dyn",
        };
        write!(f, "{s}")
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

        out.push(Output::SgLang.to_string());
        out.push(Output::Trtllm.to_string());
        out.push(Output::Vllm.to_string());
        out.push(Output::Mocker.to_string());

        out
    }
}
