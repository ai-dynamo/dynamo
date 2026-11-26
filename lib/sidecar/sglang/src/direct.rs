// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang engines a frontend dials directly, with no sidecar beside them.

use std::sync::Arc;

use dynamo_backend_common::LLMEngine;
use dynamo_backend_common::direct::{DirectEngineFactory, DirectEngineOptions, DirectEngineSpec};
use dynamo_sidecar_common::{SidecarStartupError, direct_engine_args};
use futures::FutureExt;

use crate::args::Args;
use crate::client;
use crate::engine::{SglangSidecarEngine, direct_bootstrap_host};

const BINARY: &str = "dynamo-sglang-sidecar";

/// Build each listed engine as the SGLang sidecar builds its own, from what the
/// engine reports: its role, model, parsers, listeners and KV publishers.
///
/// Only a prefill engine's bootstrap host differs. Where the sidecar falls back
/// to its own address, which in direct mode is the frontend's, this falls back
/// to the host the engine is dialed at.
pub fn direct_engine_factory(options: DirectEngineOptions) -> DirectEngineFactory {
    Arc::new(move |spec: &DirectEngineSpec| {
        let options = options.clone();
        let argv = direct_engine_args(BINARY, &options, spec);
        async move {
            let mut args = <Args as clap::Parser>::try_parse_from(argv)
                .map_err(|error| SidecarStartupError::from(error).into_dynamo())?;
            SglangSidecarEngine::validate_args(&args)?;
            let endpoint = &args.sidecar.grpc_endpoint;
            let discovery =
                client::bootstrap_discover(endpoint, &args.sidecar.grpc.config(), false).await?;
            args.bootstrap_host = Some(direct_bootstrap_host(
                args.bootstrap_host.as_deref(),
                endpoint,
                &discovery,
            ));
            let (engine, mut config) = SglangSidecarEngine::from_discovered(args, discovery)?;
            options.apply(&mut config);
            Ok((Arc::new(engine) as Arc<dyn LLMEngine>, config))
        }
        .boxed()
    })
}
