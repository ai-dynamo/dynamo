// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::RuntimeConfig;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs, HttpEndpoint};

fn parse_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())
}

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    let raw = raw.trim();
    // Accept `host:port` like `--grpc-endpoint` does.
    let raw = if raw.is_empty() || raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    HttpEndpoint::parse(&raw, "--http-endpoint").map_err(|error| error.to_string())
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-triton-sidecar",
    about = "Run Dynamo tensor workers against NVIDIA Triton Inference Server",
    group(
        clap::ArgGroup::new("triton_endpoint")
            .args(["grpc_endpoint", "http_endpoint"])
            .required(true)
            .multiple(true)
    )
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

    /// Triton KServe v2 gRPC endpoint as `host:port` or a plaintext gRPC/HTTP URL.
    ///
    /// Enables the gRPC transport. At least one of `--grpc-endpoint` and
    /// `--http-endpoint` is required; when both are set, inference requests
    /// alternate between the two protocols.
    #[arg(
        long,
        env = "DYN_SIDECAR_GRPC_ENDPOINT",
        value_parser = parse_grpc_endpoint
    )]
    pub grpc_endpoint: Option<GrpcEndpoint>,

    /// Triton KServe v2 HTTP/REST endpoint as `host:port` or an HTTP(S) URL.
    ///
    /// Enables the HTTP transport. At least one of `--grpc-endpoint` and
    /// `--http-endpoint` is required; when both are set, inference requests
    /// alternate between the two protocols.
    #[arg(
        long,
        env = "DYN_SIDECAR_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub http_endpoint: Option<HttpEndpoint>,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,

    /// Triton model to expose. Repeat to expose multiple models.
    ///
    /// With no explicit model names, the sidecar registers every READY model
    /// returned by Triton's RepositoryIndex API.
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
        assert_eq!(
            args.grpc_endpoint.unwrap().as_str(),
            "http://localhost:8001"
        );
        assert!(args.http_endpoint.is_none());
    }

    #[test]
    fn parses_http_only() {
        let args =
            Args::try_parse_from(["dynamo-triton-sidecar", "--http-endpoint", "localhost:8000"])
                .unwrap();
        assert!(args.grpc_endpoint.is_none());
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "http://localhost:8000/"
        );
    }

    #[test]
    fn parses_both_protocols() {
        let args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc-endpoint",
            "localhost:8001",
            "--http-endpoint",
            "https://triton.example.com/prefix",
        ])
        .unwrap();
        assert!(args.grpc_endpoint.is_some());
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "https://triton.example.com/prefix"
        );
    }

    #[test]
    fn requires_an_endpoint() {
        assert!(Args::try_parse_from(["dynamo-triton-sidecar"]).is_err());
        assert!(Args::try_parse_from(["dynamo-triton-sidecar", "--http-endpoint", ""]).is_err());
        assert!(
            Args::try_parse_from([
                "dynamo-triton-sidecar",
                "--http-endpoint",
                "grpc://localhost:8000"
            ])
            .is_err()
        );
    }
}
