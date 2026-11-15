// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM engines for a frontend that dials vLLM directly: the engine and worker
//! configuration this sidecar would run with, built inside the frontend.

use std::sync::Arc;

use dynamo_backend_common::LLMEngine;
use dynamo_backend_common::direct::{DirectEngineFactory, DirectEngineOptions, DirectEngineSpec};
use futures::FutureExt;

use crate::VllmSidecarEngine;

const BINARY: &str = "dynamo-vllm-sidecar";

/// Builds each listed vLLM server's engine from the arguments its sidecar
/// would be started with, so discovery and worker configuration are the
/// sidecar's own.
pub fn direct_engine_factory(options: DirectEngineOptions) -> DirectEngineFactory {
    Arc::new(move |spec: DirectEngineSpec| {
        let options = options.clone();
        async move {
            let argv = options.sidecar_args(BINARY, &spec);
            let (_, discovered) = VllmSidecarEngine::try_from_args_async(argv)
                .map_err(|error| error.into_dynamo())?;
            let (engine, mut config) = discovered.await?;
            options.apply(&mut config);
            Ok((Arc::new(engine) as Arc<dyn LLMEngine>, config))
        }
        .boxed()
    })
}
