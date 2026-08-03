// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct-gRPC registrar adapter for TensorRT-LLM.
//!
//! [`TrtllmDirectBackend`] implements the engine-agnostic
//! [`dynamo_direct_register::DirectBackend`] contract: it connects to the stock
//! `TrtllmService` gRPC server, resolves the model/context facts, and health-gates
//! its liveness. The `dynamo-direct-register` shim owns the DRT, model-card build,
//! endpoint registration, and shutdown — this crate contributes only the
//! TensorRT-LLM-specific connect / health / cleanup.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use dynamo_backend_common::{DisaggregationMode, DynamoError, LLMEngine, WorkerConfig};
use dynamo_direct_register::{DirectBackend, DirectConfig, DirectRegistration};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use tokio::sync::OnceCell;

use crate::args::Args;
use crate::client::{self, TrtllmClient};
use crate::direct::TRTLLM_BACKEND;
use crate::engine::TrtllmSidecarEngine;

/// A [`DirectBackend`] over TensorRT-LLM's native `TrtllmService` gRPC.
pub struct TrtllmDirectBackend {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    /// HF id or local path used for tokenization / templates on the model card.
    model_path: String,
    /// `--context-length` fallback; overridden by a positive `GetModelInfo` report.
    context_length: Option<u32>,
    client: OnceCell<TrtllmClient>,
}

impl TrtllmDirectBackend {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        model_path: String,
        context_length: Option<u32>,
    ) -> Self {
        Self {
            endpoint,
            transport,
            model_path,
            context_length,
            client: OnceCell::new(),
        }
    }
}

#[async_trait]
impl DirectBackend for TrtllmDirectBackend {
    async fn connect(&self) -> Result<DirectRegistration> {
        if self.client.initialized() {
            anyhow::bail!("TensorRT-LLM direct backend already connected");
        }
        tracing::info!(endpoint = %self.endpoint, "connecting to TensorRT-LLM gRPC (direct)");
        let client = TrtllmClient::connect(&self.endpoint, self.transport).await?;

        // Prefer a server-reported context length; fall back to `--context-length`.
        // GetModelInfo returns zero on current releases, so the argument is
        // currently the only source.
        let mut context_length = self.context_length;
        match client.model_info().await {
            Ok(Some(reported)) => context_length = Some(reported),
            Ok(None) => {}
            Err(error) => tracing::warn!(%error, "GetModelInfo failed; using --context-length"),
        }

        self.client
            .set(client)
            .map_err(|_| anyhow::anyhow!("TensorRT-LLM direct backend already connected"))?;
        tracing::info!(endpoint = %self.endpoint, "TensorRT-LLM gRPC is ready (direct)");

        Ok(DirectRegistration {
            backend: TRTLLM_BACKEND.to_string(),
            grpc_endpoint: self.endpoint.as_str().to_string(),
            model_path: self.model_path.clone(),
            model_name: None,
            context_length,
            tool_call_parser: None,
            reasoning_parser: None,
            // TensorRT-LLM direct is aggregated-only (its convert layer has no
            // disagg mode), so it never publishes a bootstrap endpoint or DP size.
            bootstrap_host: None,
            bootstrap_port: None,
            data_parallel_size: None,
        })
    }

    async fn health_check(&self) -> Result<()> {
        let client = self
            .client
            .get()
            .ok_or_else(|| anyhow::anyhow!("TensorRT-LLM direct backend is not connected"))?;
        client.model_info().await.map(|_| ())?;
        Ok(())
    }

    async fn cleanup(&self) -> Result<()> {
        // The channel pool releases when the backend drops; there is no explicit
        // close RPC.
        Ok(())
    }
}

/// How the sidecar should run, decided from CLI args.
pub enum Launch {
    /// `--direct`: register + health-gate the engine's gRPC via [`run_direct`];
    /// the frontend dispatches inference straight to it.
    ///
    /// [`run_direct`]: dynamo_direct_register::run_direct
    Direct(Arc<dyn DirectBackend>, DirectConfig),
    /// Default: serve the Dynamo request plane through the backend-common `Worker`.
    RequestPlane(Arc<dyn LLMEngine>, WorkerConfig),
}

/// Parse process args and decide the run mode. Called from `main`.
pub fn launch_from_env() -> Result<Launch, DynamoError> {
    let args = <Args as clap::Parser>::parse();
    if args.direct {
        let (backend, config) = direct_from_args(args)?;
        Ok(Launch::Direct(backend, config))
    } else {
        let (engine, config) = TrtllmSidecarEngine::from_parsed(args)?;
        Ok(Launch::RequestPlane(Arc::new(engine), config))
    }
}

/// Build the direct backend + shim config from parsed args.
fn direct_from_args(args: Args) -> Result<(Arc<dyn DirectBackend>, DirectConfig), DynamoError> {
    if args.model_path.trim().is_empty() {
        return Err(client::invalid_argument("model-path must not be empty"));
    }
    if args.context_length == Some(0) {
        return Err(client::invalid_argument(
            "context-length must be greater than zero",
        ));
    }
    if args.sidecar.common.disaggregation_mode != DisaggregationMode::Aggregated {
        return Err(client::invalid_argument(
            "--direct supports aggregated serving only for the TensorRT-LLM sidecar",
        ));
    }

    let endpoint = GrpcEndpoint::parse(&args.trtllm_endpoint, "--trtllm-endpoint")?;
    let backend = Arc::new(TrtllmDirectBackend::new(
        endpoint,
        args.sidecar.grpc.config(),
        args.model_path.clone(),
        args.context_length,
    ));
    let config = DirectConfig {
        namespace: args.sidecar.common.namespace,
        component: args.sidecar.common.component,
        endpoint: args.sidecar.common.endpoint,
        custom_jinja_template: args.sidecar.common.custom_jinja_template,
        tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
        reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
        advertise_grpc_endpoint: args.advertise_grpc_endpoint,
        // Validated to Aggregated above; TensorRT-LLM direct disagg is unsupported.
        disaggregation_mode: DisaggregationMode::Aggregated,
    };
    Ok((backend, config))
}
