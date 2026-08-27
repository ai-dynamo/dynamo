// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};

pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, status_to_dynamo};

use crate::proto as pb;

pub(crate) const CONTROL_SERVICE: &str = "vllm.Control";
pub(crate) const INFERENCE_SERVICE: &str = "vllm.Inference";
const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        startup_deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let pool = timeout_at(
            startup_deadline,
            GrpcChannelPool::connect("vLLM", endpoint, transport),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(format!(
                "vLLM gRPC connection pool to {endpoint} exceeded the total startup deadline"
            ))
        })??;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    pub(crate) fn control_client(&self) -> pb::control_client::ControlClient<Channel> {
        pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn wait_for_services(
        &self,
        services: &[&str],
        startup_deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        for service in services {
            self.wait_for_service(service, startup_deadline, retry_interval)
                .await?;
        }
        Ok(())
    }

    async fn wait_for_service(
        &self,
        service: &str,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        let started = Instant::now();
        loop {
            let mut client = HealthClient::new(self.pool.next_channel());
            let last_status = match timeout_at(
                deadline,
                client.check(HealthCheckRequest {
                    service: service.to_string(),
                }),
            )
            .await
            {
                Ok(Ok(response)) => {
                    let status = ServingStatus::try_from(response.into_inner().status)
                        .unwrap_or(ServingStatus::Unknown);
                    if status == ServingStatus::Serving {
                        return Ok(());
                    }
                    format!("reported {}", status.as_str_name())
                }
                Ok(Err(status))
                    if matches!(
                        status.code(),
                        tonic::Code::NotFound | tonic::Code::Unimplemented
                    ) =>
                {
                    return Err(protocol_error(format!(
                        "{service} is unavailable through the standard gRPC health API: {status}"
                    )));
                }
                Ok(Err(status)) => format!("health check failed: {status}"),
                Err(_) => "health check exceeded the startup deadline".to_string(),
            };

            let now = Instant::now();
            if now >= deadline {
                return Err(dynamo_sidecar_common::cannot_connect(format!(
                    "{service} did not become SERVING before the total vLLM startup deadline after {:?}: {last_status}",
                    started.elapsed()
                )));
            }
            let retry_at = now.checked_add(retry_interval).unwrap_or(deadline);
            sleep_until(retry_at.min(deadline)).await;
        }
    }

    pub(crate) async fn discover(
        &self,
        startup_deadline: Instant,
    ) -> Result<(pb::ModelInfo, pb::ServerInfo), DynamoError> {
        let mut client = self.control_client();
        let model = timeout_at(
            startup_deadline,
            client.get_model_info(pb::GetModelInfoRequest {}),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(
                "GetModelInfo exceeded the total vLLM startup deadline",
            )
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        let server = timeout_at(
            startup_deadline,
            client.get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(
                "GetServerInfo exceeded the total vLLM startup deadline",
            )
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetServerInfo", status))?;
        Ok((model, server))
    }

    pub(crate) async fn generate_stream(
        &self,
        request: pb::GenerateRequest,
        data_parallel_rank: Option<u32>,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let mut client = pb::inference_client::InferenceClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let mut request = tonic::Request::new(request);
        if let Some(rank) = data_parallel_rank {
            request
                .metadata_mut()
                .insert(DATA_PARALLEL_RANK_METADATA_KEY, MetadataValue::from(rank));
        }
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }

    pub(crate) async fn kv_event_sources(&self) -> Result<Vec<pb::KvEventSource>, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map(|response| response.sources)
            .map_err(|status| status_to_dynamo("GetKvEventSources", status))
    }

    pub(crate) async fn load_lora(
        &self,
        lora_name: String,
        source_path: String,
    ) -> Result<pb::LoadLoraResponse, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc(
            "LoadLora",
            client.load_lora(pb::LoadLoraRequest {
                lora_name,
                source_path,
            }),
        )
        .await
    }

    pub(crate) async fn unload_lora(
        &self,
        lora_name: String,
    ) -> Result<pb::UnloadLoraResponse, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc(
            "UnloadLora",
            client.unload_lora(pb::UnloadLoraRequest { lora_name }),
        )
        .await
    }

    pub(crate) async fn list_loras(&self) -> Result<Vec<pb::LoraAdapter>, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc("ListLoras", client.list_loras(pb::ListLorasRequest {}))
            .await
            .map(|response| response.adapters)
    }
}

pub(crate) fn startup_deadline(duration: Duration) -> Result<Instant, DynamoError> {
    Instant::now().checked_add(duration).ok_or_else(|| {
        invalid_argument(format!(
            "gRPC startup deadline {duration:?} exceeds the supported monotonic clock range"
        ))
    })
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}

/// Bounded deadline applied to every LoRA lifecycle RPC.
///
/// `GrpcTransportConfig` only bounds connection startup, so without this a stalled
/// vLLM server would hold an adapter's lifecycle lock forever.
pub(crate) const LORA_RPC_DEADLINE: Duration = Duration::from_secs(60);

/// A failed LoRA lifecycle RPC, retaining the native gRPC status.
///
/// Dynamo needs the raw code to decide whether the call definitively failed or
/// may have committed on the vLLM side and therefore requires reconciliation.
#[derive(Debug, Clone)]
pub(crate) struct LoraRpcError {
    pub(crate) rpc: &'static str,
    pub(crate) code: tonic::Code,
    pub(crate) message: String,
}

impl LoraRpcError {
    /// True when vLLM gave a definitive answer, so its state is known.
    ///
    /// Anything else (internal errors, timeouts, a dropped connection) may have
    /// committed before failing and must be reconciled against `ListLoras`.
    pub(crate) fn is_definitive(&self) -> bool {
        matches!(
            self.code,
            tonic::Code::InvalidArgument
                | tonic::Code::AlreadyExists
                | tonic::Code::NotFound
                | tonic::Code::FailedPrecondition
        )
    }

    pub(crate) fn into_dynamo(self) -> DynamoError {
        status_to_dynamo(self.rpc, tonic::Status::new(self.code, self.message))
    }
}

impl std::fmt::Display for LoraRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({:?})", self.rpc, self.message, self.code)
    }
}

/// Drive one LoRA lifecycle RPC under [`LORA_RPC_DEADLINE`].
async fn lora_rpc<T>(
    rpc: &'static str,
    call: impl Future<Output = Result<tonic::Response<T>, tonic::Status>>,
) -> Result<T, LoraRpcError> {
    match tokio::time::timeout(LORA_RPC_DEADLINE, call).await {
        Ok(Ok(response)) => Ok(response.into_inner()),
        Ok(Err(status)) => Err(LoraRpcError {
            rpc,
            code: status.code(),
            message: status.message().to_string(),
        }),
        Err(_) => Err(LoraRpcError {
            rpc,
            code: tonic::Code::DeadlineExceeded,
            message: format!("{rpc} exceeded the {LORA_RPC_DEADLINE:?} lifecycle deadline"),
        }),
    }
}
