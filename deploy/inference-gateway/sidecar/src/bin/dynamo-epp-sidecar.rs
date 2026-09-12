// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_epp_sidecar::{Config, PdAdapter, PdBackend, SglangPdAdapter, UnavailablePdAdapter};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let config = Config::from_env()?;
    let adapter: Arc<dyn PdAdapter> = match config.pd_backend {
        PdBackend::Unavailable => Arc::new(UnavailablePdAdapter),
        PdBackend::Sglang => Arc::new(SglangPdAdapter::new(
            config.decode_engine_url.clone(),
            config.connect_timeout,
            config.read_timeout,
        )?),
    };
    dynamo_epp_sidecar::run(config, adapter).await
}
