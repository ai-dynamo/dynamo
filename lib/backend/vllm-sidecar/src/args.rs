// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// vLLM gRPC endpoint as host:port or an http:// URL.
    #[arg(long, env = "VLLM_GRPC_ENDPOINT")]
    pub vllm_endpoint: String,

    /// Hugging Face model ID or local path used for tokenization and templates.
    #[arg(long)]
    pub model_path: String,
}

pub(crate) fn normalize_endpoint(raw: &str) -> Result<String, String> {
    let endpoint = raw.trim();
    if endpoint.is_empty() {
        return Err("vLLM gRPC endpoint must not be empty".to_string());
    }
    if endpoint.starts_with("http://") {
        Ok(endpoint.to_string())
    } else if endpoint.contains("://") {
        Err(format!(
            "unsupported vLLM gRPC endpoint scheme in `{endpoint}`; use plaintext http://"
        ))
    } else {
        Ok(format!("http://{endpoint}"))
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_endpoint;

    #[test]
    fn normalizes_plaintext_endpoints() {
        assert_eq!(
            normalize_endpoint(" 127.0.0.1:50051 ").unwrap(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            normalize_endpoint("http://vllm:50051").unwrap(),
            "http://vllm:50051"
        );
        assert!(normalize_endpoint("https://vllm:50051").is_err());
        assert!(normalize_endpoint(" ").is_err());
    }
}
