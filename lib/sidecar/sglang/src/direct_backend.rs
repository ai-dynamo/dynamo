// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct-gRPC registrar adapter for SGLang.
//!
//! [`SglangDirectBackend`] implements the engine-agnostic
//! [`dynamo_direct_register::DirectBackend`] contract: it connects to the stock
//! `sglang.runtime.v1.SglangService` gRPC server, discovers the model /
//! context / parser facts, and health-gates its liveness via the `HealthCheck`
//! RPC. The `dynamo-direct-register` shim owns the DRT, model-card build,
//! endpoint registration, and shutdown — this crate contributes only the
//! SGLang-specific connect / health / cleanup.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use dynamo_backend_common::{DisaggregationMode, DynamoError, LLMEngine, WorkerConfig};
use dynamo_direct_register::{DirectBackend, DirectConfig, DirectRegistration};
use tokio::sync::OnceCell;
use tokio::time::Instant;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Pool};
use crate::direct::SGLANG_BACKEND;
use crate::engine::{
    SglangSidecarEngine, bootstrap_discover, component_for_mode, discovery_bootstrap_port,
    discovery_data_parallel_size, discovery_mode, discovery_string, resolve_bootstrap_host,
};

/// A [`DirectBackend`] over SGLang's native `SglangService` gRPC.
pub struct SglangDirectBackend {
    /// Normalized (`http://…`) SGLang gRPC endpoint the sidecar connects to.
    endpoint: String,
    transport: TransportConfig,
    /// Disaggregation role resolved from SGLang discovery at arg-parse time.
    /// `connect` re-verifies it against the live server.
    mode: DisaggregationMode,
    /// `--bootstrap-host` override (env `SGLANG_DISAGGREGATION_BOOTSTRAP_HOST`),
    /// used only for a prefill worker whose discovery exposes a non-routable host.
    bootstrap_host_arg: Option<String>,
    pool: OnceCell<Pool>,
}

impl SglangDirectBackend {
    pub(crate) fn new(
        endpoint: String,
        transport: TransportConfig,
        mode: DisaggregationMode,
        bootstrap_host_arg: Option<String>,
    ) -> Self {
        Self {
            endpoint,
            transport,
            mode,
            bootstrap_host_arg,
            pool: OnceCell::new(),
        }
    }
}

#[async_trait]
impl DirectBackend for SglangDirectBackend {
    async fn connect(&self) -> Result<DirectRegistration> {
        if self.pool.initialized() {
            anyhow::bail!("SGLang direct backend already connected");
        }
        let deadline = Instant::now() + self.transport.deadline;
        let pool = Pool::connect(
            &self.endpoint,
            &self.transport,
            self.transport.connections,
            deadline,
        )
        .await?;
        let mut control = pool.control_client();
        await_ready(&mut control, &self.transport, deadline).await?;
        let discovery = client::discover(&mut control, deadline).await?;

        // Re-verify the role resolved at arg-parse time against the live server;
        // a mismatch means the component/topology the shim already registered
        // under would be wrong.
        let observed_mode = discovery_mode(&discovery)?;
        if observed_mode != self.mode {
            anyhow::bail!(
                "SGLang role changed since bootstrap: registered as {:?}, now reports {:?}",
                self.mode,
                observed_mode
            );
        }

        // Prefill workers publish the disaggregation bootstrap endpoint the decode
        // hop dials; decode / aggregated workers leave it unset.
        let (bootstrap_host, bootstrap_port) = if self.mode.is_prefill() {
            (
                resolve_bootstrap_host(
                    self.bootstrap_host_arg.as_deref(),
                    &self.endpoint,
                    &discovery,
                )?,
                discovery_bootstrap_port(&discovery)?,
            )
        } else {
            (None, None)
        };

        self.pool
            .set(pool)
            .map_err(|_| anyhow::anyhow!("SGLang direct backend already connected"))?;
        tracing::info!(
            endpoint = %self.endpoint,
            model = %discovery.model_path,
            mode = ?self.mode,
            "sglang gRPC is ready (direct)"
        );

        Ok(DirectRegistration {
            backend: SGLANG_BACKEND.to_string(),
            grpc_endpoint: self.endpoint.clone(),
            model_path: discovery.tokenizer_path.clone(),
            // Match the request-plane served name: discovery's served name, else
            // the model path.
            model_name: discovery
                .served_model_name
                .clone()
                .or_else(|| Some(discovery.model_path.clone())),
            context_length: discovery.max_model_len,
            tool_call_parser: discovery_string(&discovery.server_info, "tool_call_parser"),
            reasoning_parser: discovery_string(&discovery.server_info, "reasoning_parser"),
            bootstrap_host,
            bootstrap_port,
            data_parallel_size: Some(discovery_data_parallel_size(&discovery)),
        })
    }

    async fn health_check(&self) -> Result<()> {
        let mut client = self
            .pool
            .get()
            .map(Pool::control_client)
            .ok_or_else(|| anyhow::anyhow!("SGLang direct backend is not connected"))?;
        let deadline = Instant::now() + self.transport.connect_timeout;
        if client::health_check(&mut client, deadline).await? {
            Ok(())
        } else {
            anyhow::bail!("SGLang reported unhealthy")
        }
    }

    async fn cleanup(&self) -> Result<()> {
        // The connection pool releases when the backend drops.
        Ok(())
    }
}

/// Poll `HealthCheck` until the engine reports healthy or the deadline expires.
async fn await_ready(
    client: &mut Client,
    transport: &TransportConfig,
    deadline: Instant,
) -> Result<(), DynamoError> {
    loop {
        let retry_message = match client::health_check(client, deadline).await {
            Ok(true) => return Ok(()),
            Ok(false) => "SGLang reported unhealthy".to_string(),
            Err(error) => format!("HealthCheck RPC failed: {error}"),
        };
        if Instant::now() >= deadline {
            return Err(client::engine_shutdown(format!(
                "SGLang did not become healthy within {:?}: {retry_message}",
                transport.deadline
            )));
        }
        tokio::time::sleep_until((Instant::now() + transport.poll_interval).min(deadline)).await;
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
    let args = SglangSidecarEngine::parse(None)?;
    if args.direct {
        let (backend, config) = direct_from_args(args)?;
        Ok(Launch::Direct(backend, config))
    } else {
        let (engine, config) = SglangSidecarEngine::from_parsed(args)?;
        Ok(Launch::RequestPlane(Arc::new(engine), config))
    }
}

/// Build the direct backend + shim config from parsed args. Model identity,
/// context, and parsers are resolved from SGLang discovery at connect time; the
/// disaggregation role is resolved here (a throwaway bootstrap discovery, as the
/// request-plane path does) because it decides the discovery component the shim
/// registers under before the engine is connected.
fn direct_from_args(args: Args) -> Result<(Arc<dyn DirectBackend>, DirectConfig), DynamoError> {
    let endpoint = normalize_endpoint(&args.sglang_endpoint).map_err(client::invalid_arg)?;
    let transport = args.transport();
    // Normalize the advertise override to a scheme-qualified URI so the
    // dispatcher's tonic connect accepts it (matches the request-plane path).
    let advertise_grpc_endpoint = match args.advertise_grpc_endpoint {
        Some(addr) => Some(normalize_endpoint(&addr).map_err(client::invalid_arg)?),
        None => None,
    };

    // Resolve the disaggregation role up front: it selects the component
    // (`prefill` vs `backend`) the shim resolves before `connect`.
    let discovery = bootstrap_discover(&endpoint, &transport)?;
    let mode = discovery_mode(&discovery)?;

    let backend = Arc::new(SglangDirectBackend::new(
        endpoint,
        transport,
        mode,
        args.bootstrap_host,
    ));
    let config = DirectConfig {
        namespace: args.namespace,
        component: component_for_mode(mode).to_string(),
        endpoint: args.endpoint,
        custom_jinja_template: args.custom_jinja_template,
        // Resolved from SGLang discovery on the DirectRegistration instead.
        tool_call_parser: None,
        reasoning_parser: None,
        advertise_grpc_endpoint,
        disaggregation_mode: mode,
    };
    Ok((backend, config))
}
