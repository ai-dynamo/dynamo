// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::RuntimeConfig;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs};

fn parse_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-triton-sidecar",
    about = "Run Dynamo tensor workers against NVIDIA Triton Inference Server"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub runtime: RuntimeConfig,

    /// Dynamo namespace used for model discovery.
    #[arg(long, default_value = "dynamo", env = "DYN_NAMESPACE")]
    pub namespace: String,

    /// Dynamo component containing the per-model endpoints.
    #[arg(long, default_value = "triton", env = "DYN_COMPONENT")]
    pub component: String,

    /// Triton gRPC endpoint as `host:port` or a plaintext gRPC/HTTP URL.
    #[arg(
        long,
        env = "DYN_SIDECAR_GRPC_ENDPOINT",
        value_parser = parse_grpc_endpoint
    )]
    pub grpc_endpoint: GrpcEndpoint,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,

    /// Triton model to expose. Repeat to expose multiple models.
    ///
    /// With no explicit model names, the sidecar registers every READY model
    /// returned by Triton's gRPC RepositoryIndex API.
    #[arg(
        long = "model-name",
        env = "DYN_TRITON_MODEL_NAMES",
        value_delimiter = ','
    )]
    pub model_names: Vec<String>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Args;

    #[test]
    fn parses_explicit_models() {
        let args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc-endpoint",
            "localhost:8001",
            "--model-name",
            "a",
            "--model-name",
            "b",
        ])
        .unwrap();
        assert_eq!(args.model_names, ["a", "b"]);
        assert_eq!(args.grpc_endpoint.as_str(), "http://localhost:8001");
    }
}
