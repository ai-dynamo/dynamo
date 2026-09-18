// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_epp_sidecar::{AdapterMode, Config, PdAdapter, UnavailablePdAdapter, VllmNixlAdapter};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let config = Config::from_env()?;
    let adapter: Arc<dyn PdAdapter> = build_adapter(&config)?;
    dynamo_epp_sidecar::run(config, adapter).await
}

/// Construct the configured P/D adapter, failing fast on a bad configuration
/// rather than on the first request that needs it.
fn build_adapter(config: &Config) -> anyhow::Result<Arc<dyn PdAdapter>> {
    match config.adapter_mode {
        AdapterMode::None => {
            tracing::info!(
                "No P/D adapter configured; requests carrying EPP P/D metadata will be rejected"
            );
            Ok(Arc::new(UnavailablePdAdapter))
        }
        AdapterMode::VllmNixl => {
            VllmNixlAdapter::check_protocol_version(&config.protocol_version)
                .map_err(|error| anyhow::anyhow!(error))?;
            tracing::info!(
                protocol_version = VllmNixlAdapter::protocol_version(),
                vllm_version = dynamo_epp_sidecar::vllm_nixl::SUPPORTED_VLLM_VERSION,
                model = %config.model,
                decode_engine = %config.decode_engine_url,
                max_request_bytes = config.max_request_bytes,
                max_prefill_response_bytes = config.max_prefill_response_bytes,
                "Enabled the raw-vLLM NIXL P/D adapter"
            );
            let adapter = VllmNixlAdapter::new(
                config.decode_engine_url.clone(),
                config.connect_timeout,
                config.read_timeout,
                dynamo_epp_sidecar::vllm_nixl::Config {
                    model: config.model.clone(),
                    max_request_bytes: config.max_request_bytes,
                    max_prefill_response_bytes: config.max_prefill_response_bytes,
                },
            )?;
            Ok(adapter)
        }
    }
}
