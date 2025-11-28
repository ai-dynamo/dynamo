// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

pub enum Output {
    /// Echos the prompt back as the response
    Echo,

    /// Listen for models on nats/etcd, add/remove dynamically
    Auto,

    #[cfg(feature = "mistralrs")]
    MistralRs,

    Mocker,
}

impl TryFrom<&str> for Output {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> anyhow::Result<Self> {
        match s {
            #[cfg(feature = "mistralrs")]
            "mistralrs" => Ok(Output::MistralRs),

            "mocker" => Ok(Output::Mocker),
            "echo" | "echo_full" => Ok(Output::Echo),

            "dyn" | "auto" => Ok(Output::Auto),

            e => Err(anyhow::anyhow!("Invalid out= option '{e}'")),
        }
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            #[cfg(feature = "mistralrs")]
            Output::MistralRs => "mistralrs",

            Output::Mocker => "mocker",
            Output::Echo => "echo",

            Output::Auto => "auto",
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
        out
    }
}
