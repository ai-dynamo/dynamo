// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event source tagging.

use std::str::FromStr;

/// Origin of a KV cache event. Drives dedup accounting — a block with both `Vllm` and
/// `Kvbm` sources must survive single-source removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EventSource {
    /// vLLM worker (G1 / GPU).
    Vllm,
    /// TensorRT-LLM worker (G1 / GPU).
    Trtllm,
    /// In-process KVBM (G2 host-pinned or G3 disk).
    Kvbm,
}

impl EventSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventSource::Vllm => "vllm",
            EventSource::Trtllm => "trtllm",
            EventSource::Kvbm => "kvbm",
        }
    }
}

impl FromStr for EventSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vllm" | "VLLM" | "GPU" => Ok(EventSource::Vllm),
            "trtllm" | "TRTLLM" | "TensorRT-LLM" => Ok(EventSource::Trtllm),
            "kvbm" | "KVBM" => Ok(EventSource::Kvbm),
            _ => Err(format!("Unknown event source: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_known_aliases() {
        for (input, expected) in [
            ("vllm", EventSource::Vllm),
            ("VLLM", EventSource::Vllm),
            ("GPU", EventSource::Vllm),
            ("trtllm", EventSource::Trtllm),
            ("TRTLLM", EventSource::Trtllm),
            ("TensorRT-LLM", EventSource::Trtllm),
            ("kvbm", EventSource::Kvbm),
            ("KVBM", EventSource::Kvbm),
        ] {
            assert_eq!(EventSource::from_str(input).unwrap(), expected);
        }
    }

    #[test]
    fn unknown_source_errors() {
        let err = EventSource::from_str("gpu2").unwrap_err();
        assert!(err.contains("Unknown event source"));
    }

    #[test]
    fn as_str_is_stable() {
        assert_eq!(EventSource::Vllm.as_str(), "vllm");
        assert_eq!(EventSource::Trtllm.as_str(), "trtllm");
        assert_eq!(EventSource::Kvbm.as_str(), "kvbm");
    }
}
