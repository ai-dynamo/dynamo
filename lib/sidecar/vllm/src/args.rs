// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs};

use std::num::NonZeroU32;

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--vllm-http-endpoint").map_err(|error| error.to_string())
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Optional controller-routable vLLM HTTP base URL for RL compatibility operations.
    #[arg(
        long,
        env = "VLLM_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub vllm_http_endpoint: Option<HttpEndpoint>,

    /// Optional total RL process world size for vLLM versions that omit it from gRPC metadata.
    ///
    /// The value includes tensor, pipeline, prefill-context, and data parallelism.
    #[arg(long, env = "DYN_VLLM_RL_WORLD_SIZE")]
    pub vllm_rl_world_size: Option<NonZeroU32>,
}

/// Match the shared media policy's enabled default, rejecting invalid startup configuration.
pub(crate) fn allow_remote_media_urls(
    value: Option<&str>,
) -> Result<bool, dynamo_backend_common::DynamoError> {
    let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
        return Ok(true);
    };
    dynamo_runtime::config::parse_bool(value).map_err(|error| {
        crate::client::invalid_argument(format!("DYN_MM_ALLOW_REMOTE_URLS: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::allow_remote_media_urls;

    #[test]
    fn remote_url_policy_defaults_and_boolean_vocabulary() {
        for value in [
            None,
            Some(""),
            Some("  "),
            Some("1"),
            Some("TRUE"),
            Some(" on "),
            Some("yes"),
        ] {
            assert!(allow_remote_media_urls(value).unwrap(), "{value:?}");
        }
        for value in ["0", "FALSE", " off ", "no"] {
            assert!(!allow_remote_media_urls(Some(value)).unwrap(), "{value}");
        }
        assert!(
            allow_remote_media_urls(Some("invalid"))
                .unwrap_err()
                .to_string()
                .contains("DYN_MM_ALLOW_REMOTE_URLS")
        );
    }
}
