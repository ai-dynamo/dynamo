// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for the LPX engines' OpenEngine (`openengine.v1`) gRPC
//! servers.
//!
//! A hybrid GPU+LPU engine serves two roles behind this sidecar: aggregated,
//! and the decode half of prefill/decode disaggregation, where a vLLM worker
//! runs prefill and the engine pulls its KV cache over NIXL. An LPU-only
//! engine serves aggregated requests. When one engine spans several Pods,
//! only the sidecar on rank 0 serves; the others stand by.

use std::future::Future;
use std::sync::Arc;

use dynamo_backend_common::{RuntimeConfig, Worker};
use dynamo_sidecar_common::{EngineBootstrapResult, SidecarStartupError};

mod args;
mod client;
mod convert;
mod engine;
mod handoff;
mod load;
mod model;
mod standby;

/// The generated `openengine.v1` types, reused from the TensorRT-LLM sidecar.
///
/// Both sidecars speak the same vendored contract, and one set of generated
/// types lets the TensorRT-LLM Mocker server stand in for the LPX engines in
/// tests without conversions. Move to a standalone `openengine` protocol crate
/// once one exists.
pub(crate) use dynamo_trtllm_sidecar::proto as pb;

pub use engine::{Launch, LpxSidecarEngine};
pub use standby::run_standby;

/// Serves the engine until SIGTERM or SIGINT. When the engine's load stays
/// unavailable, the worker withdraws from discovery and drains, and the
/// sidecar exits with an error.
pub fn serve(
    (runtime_config, bootstrap): (
        RuntimeConfig,
        impl Future<Output = EngineBootstrapResult<LpxSidecarEngine>>,
    ),
) -> anyhow::Result<()> {
    dynamo_sidecar_common::run_with_drt(
        runtime_config,
        async { Ok(bootstrap.await.map_err(SidecarStartupError::from)?) },
        |drt, (engine, config), shutdown| async move {
            let engine = Arc::new(engine);
            let worker = Worker::new(engine.clone(), config).run_with_drt(drt, shutdown.clone());
            tokio::pin!(worker);
            tokio::select! {
                result = &mut worker => result.map_err(Into::into),
                error = engine.load_lost() => {
                    // Withdraw from discovery and drain, then exit with the cause.
                    shutdown.cancel();
                    if let Err(stopped) = worker.await {
                        tracing::warn!(error = %stopped, "Worker shutdown failed");
                    }
                    Err(error.into())
                }
            }
        },
    )
}

#[cfg(test)]
mod tests;
