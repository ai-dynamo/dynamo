// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct-gRPC registrar adapter for vLLM.
//!
//! [`VllmDirectBackend`] implements the engine-agnostic
//! [`dynamo_direct_register::DirectBackend`] contract: it connects to the stock
//! vLLM gRPC server and health-gates its liveness. The `dynamo-direct-register`
//! shim owns the DRT, model-card build, endpoint registration, and shutdown —
//! this crate contributes only the vLLM-specific connect / health / cleanup.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use dynamo_backend_common::{DisaggregationMode, DynamoError, LLMEngine, WorkerConfig};
use dynamo_direct_register::{DirectBackend, DirectConfig, DirectRegistration};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use tokio::sync::OnceCell;

use crate::args::Args;
use crate::client::{self, VllmClient};
use crate::direct::VLLM_BACKEND;
use crate::engine::VllmSidecarEngine;

/// A [`DirectBackend`] over vLLM's native gRPC service.
pub struct VllmDirectBackend {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    /// HF id or local path used for tokenization / templates on the model card.
    model_path: String,
    client: OnceCell<VllmClient>,
}

impl VllmDirectBackend {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        model_path: String,
    ) -> Self {
        Self {
            endpoint,
            transport,
            model_path,
            client: OnceCell::new(),
        }
    }
}

#[async_trait]
impl DirectBackend for VllmDirectBackend {
    async fn connect(&self) -> Result<DirectRegistration> {
        if self.client.initialized() {
            anyhow::bail!("vLLM direct backend already connected");
        }
        tracing::info!(endpoint = %self.endpoint, "connecting to vLLM gRPC (direct)");
        let client = VllmClient::connect(&self.endpoint, self.transport).await?;
        self.client
            .set(client)
            .map_err(|_| anyhow::anyhow!("vLLM direct backend already connected"))?;
        tracing::info!(endpoint = %self.endpoint, "vLLM gRPC transport connected (direct)");

        Ok(DirectRegistration {
            backend: VLLM_BACKEND.to_string(),
            grpc_endpoint: self.endpoint.as_str().to_string(),
            model_path: self.model_path.clone(),
            model_name: None,
            // vLLM's released gRPC exposes no model-info RPC; the context length
            // is not discoverable here.
            context_length: None,
            tool_call_parser: None,
            reasoning_parser: None,
        })
    }

    async fn health_check(&self) -> Result<()> {
        // vLLM's gRPC exposes no health or model-info RPC (both return
        // UNIMPLEMENTED), so probe with a cheap fresh-channel connect: it
        // succeeds while the engine is listening and fails fast once it dies.
        if !self.client.initialized() {
            anyhow::bail!("vLLM direct backend is not connected");
        }
        client::probe_liveness(&self.endpoint, self.transport).await?;
        Ok(())
    }

    async fn cleanup(&self) -> Result<()> {
        // The channel pool releases when the backend drops.
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
    let args = VllmSidecarEngine::parse(None)?;
    if args.sidecar.common.is_direct {
        let (backend, config) = direct_from_args(args)?;
        Ok(Launch::Direct(backend, config))
    } else {
        let (engine, config) = VllmSidecarEngine::from_parsed(args)?;
        Ok(Launch::RequestPlane(Arc::new(engine), config))
    }
}

/// Build the direct backend + shim config from parsed args.
fn direct_from_args(args: Args) -> Result<(Arc<dyn DirectBackend>, DirectConfig), DynamoError> {
    if args.model_path.trim().is_empty() {
        return Err(client::invalid_argument("model-path must not be empty"));
    }
    if args.sidecar.common.disaggregation_mode != DisaggregationMode::Aggregated {
        return Err(client::invalid_argument(
            "--direct supports aggregated serving only for the vLLM sidecar",
        ));
    }

    let endpoint = GrpcEndpoint::parse(&args.vllm_endpoint, "--vllm-endpoint")?;
    let backend = Arc::new(VllmDirectBackend::new(
        endpoint,
        args.sidecar.grpc.config(),
        args.model_path.clone(),
    ));
    let config = DirectConfig {
        namespace: args.sidecar.common.namespace,
        component: args.sidecar.common.component,
        endpoint: args.sidecar.common.endpoint,
        custom_jinja_template: args.sidecar.common.custom_jinja_template,
        tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
        reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
        advertise_grpc_endpoint: args.sidecar.common.advertise_grpc_endpoint,
    };
    Ok((backend, config))
}
