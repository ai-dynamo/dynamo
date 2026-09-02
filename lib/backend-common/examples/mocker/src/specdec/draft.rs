// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    BackendError, DraftTransportDescriptorV1, DynamoError, Endpoint, EndpointId, EngineConfig,
    GenerateContext, LLMEngine, LLMEngineOutput, LlmRegistration, MetricsBindings, MetricsCtx,
    ModelRegistration, PreprocessedRequest, WorkerConfig, WorkerRole, WorkerWithDpRank,
    new_external_speculation_incarnation,
};
use futures::stream::BoxStream;
use tokio::sync::Mutex;

use super::config::{parse_draft, worker_config};
use super::metrics::SpecdecMetrics;
use super::protocol::DraftIdentity;
use super::queue::SchedulerConfig;
use super::transport::{DraftServer, DraftServerConfig};
use super::{DP_RANK, PROTOCOL, backend_error};

pub struct DraftEngine {
    model_name: String,
    endpoint_id: EndpointId,
    advertise_address: String,
    orphan_cleanup_timeout_ms: u32,
    draft_incarnation_id: u64,
    metrics: Arc<SpecdecMetrics>,
    server_config: DraftServerConfig,
    server: Mutex<Option<DraftServer>>,
}

impl DraftEngine {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = parse_draft(argv)?;
        let draft_incarnation_id = new_external_speculation_incarnation().map_err(|error| {
            backend_error(
                BackendError::EngineShutdown,
                format!("create draft incarnation: {error}"),
            )
        })?;
        let endpoint_path = format!(
            "{}/{}/{}",
            args.common.namespace, args.common.component, args.common.endpoint
        );
        let engine = Self {
            model_name: args.model_name.clone(),
            endpoint_id: EndpointId::from(endpoint_path.as_str()),
            advertise_address: args.draft_advertise_address.clone(),
            orphan_cleanup_timeout_ms: args.orphan_cleanup_timeout_ms,
            draft_incarnation_id,
            metrics: Arc::new(SpecdecMetrics::default()),
            server_config: DraftServerConfig {
                bind_address: args.draft_bind_address,
                transport_hwm: args.transport_hwm,
                outbound_capacity: args.transport_queue_capacity,
                prefill_duration: Duration::from_millis(args.draft_prefill_ms),
                token_interval: Duration::from_millis(args.draft_token_interval_ms),
                token_mode: args.draft_token_mode.into(),
                scheduler: SchedulerConfig {
                    queue_capacity: args.inference_queue_capacity,
                    concurrency: args.inference_concurrency,
                    output_capacity: args.inference_output_capacity,
                },
            },
            server: Mutex::new(None),
        };
        let config = worker_config(args.common, args.model_name, String::new());
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for DraftEngine {
    async fn start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError> {
        let identity = DraftIdentity {
            endpoint: self.endpoint_id.clone(),
            worker: WorkerWithDpRank::new(worker_id, DP_RANK),
            draft_incarnation_id: self.draft_incarnation_id,
            protocol: PROTOCOL.to_string(),
            address: self.advertise_address.clone(),
            orphan_cleanup_timeout_ms: self.orphan_cleanup_timeout_ms,
        };
        let server = DraftServer::bind_with_metrics(
            self.server_config.clone(),
            identity,
            self.metrics.clone(),
        )
        .await
        .map_err(|error| {
            backend_error(
                BackendError::CannotConnect,
                format!("start draft protocol server: {error}"),
            )
        })?;
        let mut stored = self.server.lock().await;
        if stored.is_some() {
            drop(stored);
            let _ = server.shutdown().await;
            return Err(backend_error(
                BackendError::EngineShutdown,
                "draft transport already started",
            ));
        }
        *stored = Some(server);
        tracing::info!(
            worker_id,
            dp_rank = DP_RANK,
            draft_incarnation = self.draft_incarnation_id,
            bind_address = %self.server_config.bind_address,
            advertise_address = %self.advertise_address,
            "mock speculative draft started"
        );
        Ok(EngineConfig {
            model: self.model_name.clone(),
            served_model_name: Some(self.model_name.clone()),
            llm: Some(LlmRegistration {
                data_parallel_size: Some(1),
                data_parallel_start_rank: Some(DP_RANK),
                ..LlmRegistration::default()
            }),
            ..EngineConfig::default()
        })
    }

    async fn generate(
        &self,
        _request: PreprocessedRequest,
        _ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        Err(backend_error(
            BackendError::InvalidArgument,
            "speculative draft has no public inference surface",
        ))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        let server = self.server.lock().await.take();
        if let Some(server) = server {
            server.shutdown().await.map_err(|error| {
                backend_error(
                    BackendError::EngineShutdown,
                    format!("shut down draft protocol server: {error}"),
                )
            })
        } else {
            Ok(())
        }
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        self.metrics.register(ctx);
        Ok(MetricsBindings::default())
    }

    async fn model_registration(
        &self,
        _endpoint: &Endpoint,
    ) -> Result<ModelRegistration, DynamoError> {
        let transport = DraftTransportDescriptorV1 {
            protocol: PROTOCOL.to_string(),
            address: self.advertise_address.clone(),
            draft_incarnation_id: self.draft_incarnation_id,
            orphan_cleanup_timeout_ms: self.orphan_cleanup_timeout_ms,
        };
        transport.validate().map_err(|error| {
            backend_error(
                BackendError::InvalidArgument,
                format!("invalid draft transport descriptor: {error}"),
            )
        })?;
        Ok(ModelRegistration {
            worker_role: WorkerRole::SpeculativeDraft,
            external_draft_transports: [(DP_RANK, transport)].into(),
        })
    }
}
