// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{DisaggregationMode, DynamoError};
use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs, invalid_argument, startup_deadline};

fn parse_decode_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    let endpoint =
        HttpEndpoint::parse(raw, "--decode-http-endpoint").map_err(|error| error.to_string())?;
    // Requests replace the path, so a prefix would be dropped without notice.
    if endpoint.as_str() != endpoint.with_path("/").as_str() {
        return Err("`--decode-http-endpoint` must not include a path".to_string());
    }
    Ok(endpoint)
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-tilert-sidecar",
    about = "Run a Dynamo worker against one vLLM prefill (--grpc-endpoint) and one TileRT decode server"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Base URL of the TileRT decode server's HTTP listener.
    #[arg(long, env = "TILERT_DECODE_HTTP_ENDPOINT", value_parser = parse_decode_endpoint)]
    pub decode_http_endpoint: HttpEndpoint,

    /// Host the prefill's TileRT connector pushes KV to. Defaults to the host
    /// in `--decode-http-endpoint`; set it when the prefill reaches the decode
    /// node on a different address than this sidecar does.
    #[arg(long, env = "TILERT_DECODE_KV_HOST")]
    pub decode_kv_host: Option<String>,

    /// Control port of the TileRT decode server (its `--ctrl-port`).
    #[arg(long, env = "TILERT_DECODE_CTRL_PORT", default_value_t = 5556)]
    pub decode_ctrl_port: u16,

    /// Hugging Face model ID or local path used for tokenization and the chat
    /// template.
    #[arg(long)]
    pub model_path: String,

    /// Public model name. Defaults to `--model-path`.
    #[arg(long, env = "DYN_SERVED_MODEL_NAME")]
    pub served_model_name: Option<String>,

    /// Model maximum sequence length (input + output), as the decode server's
    /// `--max-seq-len`. Also the `max_tokens` budget for requests that omit
    /// one.
    #[arg(long, env = "TILERT_CONTEXT_LENGTH", value_parser = clap::value_parser!(u32).range(1..))]
    pub context_length: u32,

    /// How long the decode server waits for a request's KV before giving up.
    #[arg(
        long,
        env = "TILERT_KV_TRANSFER_TIMEOUT_SECS",
        default_value_t = 10,
        value_parser = clap::value_parser!(u64).range(1..=3600)
    )]
    pub kv_transfer_timeout_secs: u64,

    /// Prompt token ids for a short request run through both engines before
    /// the worker registers, so a broken KV handoff fails startup. Empty skips
    /// the check.
    #[arg(long, value_delimiter = ',')]
    pub probe_token_ids: Vec<u32>,

    /// Deployment-level default thinking mode advertised to the Dynamo
    /// frontend's chat formatter.
    #[arg(
        long = "dyn-default-thinking-mode",
        env = "DYN_DEFAULT_THINKING_MODE",
        value_parser = ["enabled", "disabled"]
    )]
    pub default_thinking_mode: Option<String>,
}

impl Args {
    /// Rejects what this sidecar cannot serve before any runtime starts.
    pub(crate) fn validate(&self) -> Result<(), DynamoError> {
        if self.model_path.trim().is_empty() {
            return Err(invalid_argument("model-path must not be empty"));
        }
        let common = &self.sidecar.common;
        if common.disaggregation_mode != DisaggregationMode::Aggregated {
            return Err(invalid_argument(
                "the TileRT sidecar drives prefill and decode itself; run it in aggregated mode",
            ));
        }
        if common.route_to_encoder {
            return Err(invalid_argument(
                "route-to-encoder is not supported by the TileRT sidecar",
            ));
        }
        if common.structural_tag(false).is_some() {
            return Err(invalid_argument(
                "--dyn-structural-tag is not supported by the TileRT sidecar",
            ));
        }
        startup_deadline(self.sidecar.grpc.config().startup_deadline)?;
        Ok(())
    }

    pub(crate) fn decode_kv_host(&self) -> String {
        let explicit = self.decode_kv_host.as_deref().map(str::trim);
        if let Some(host) = explicit.filter(|host| !host.is_empty()) {
            return host.to_string();
        }
        // `HttpEndpoint::parse` rejects an endpoint without a host.
        self.decode_http_endpoint
            .with_path("/")
            .host_str()
            .unwrap_or_default()
            .trim_matches(|c| c == '[' || c == ']')
            .to_string()
    }
}
