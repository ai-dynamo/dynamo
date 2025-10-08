// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::ENDPOINT_SCHEME;
use std::fmt;

pub enum Output {
    /// Echos the prompt back as the response
    Echo,

    /// Listen for models on nats/etcd, add/remove dynamically
    Auto,

    /// Static remote: The dyn://namespace.component.endpoint name of a remote worker we expect to
    /// exists. THIS DISABLES AUTO-DISCOVERY. Only this endpoint will be connected.
    /// `--model-name and `--model-path` must also be set.
    ///
    /// A static remote setup avoids having to run etcd.
    Static(String),

    #[cfg(feature = "mistralrs")]
    MistralRs,

    #[cfg(feature = "llamacpp")]
    /// Run inference using llama.cpp
    LlamaCpp,

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

            "mocker" => Ok(Output::Mocker),
            "echo" | "echo_full" => Ok(Output::Echo),

            "dyn" | "auto" => Ok(Output::Auto),

            endpoint_path if endpoint_path.starts_with(ENDPOINT_SCHEME) => {
                let path = endpoint_path.strip_prefix(ENDPOINT_SCHEME).unwrap();
                Ok(Output::Static(path.to_string()))
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

            Output::Mocker => "mocker",
            Output::Echo => "echo",

            Output::Auto => "auto",
            Output::Static(endpoint) => &format!("{ENDPOINT_SCHEME}{endpoint}"),
        };
        write!(f, "{s}")
    }
}

impl Output {
    #[allow(unused_mut)]
    pub fn available_engines() -> Vec<String> {
        let mut out = vec!["echo".to_string(), Output::Mocker.to_string()];
        #[cfg(feature = "mistralrs")]
        {
            out.push(Output::MistralRs.to_string());
        }

        #[cfg(feature = "llamacpp")]
        {
            out.push(Output::LlamaCpp.to_string());
        }

        out
    }
}
