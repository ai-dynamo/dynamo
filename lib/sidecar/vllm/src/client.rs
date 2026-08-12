// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::metadata::MetadataValue;
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
        let channel = self.pool.next_channel();
        let mut client = pb::control_client::ControlClient::new(channel)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
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
        adapter: pb::LoraAdapter,
    ) -> Result<pb::LoadLoraResponse, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .load_lora(pb::LoadLoraRequest {
                adapter: Some(adapter),
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("LoadLora", status))
    }

    pub(crate) async fn unload_lora(
        &self,
        lora_name: String,
    ) -> Result<pb::UnloadLoraResponse, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .unload_lora(pb::UnloadLoraRequest { lora_name })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("UnloadLora", status))
    }

    pub(crate) async fn list_loras(&self) -> Result<Vec<pb::LoraAdapter>, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .list_loras(pb::ListLorasRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map(|response| response.adapters)
            .map_err(|status| status_to_dynamo("ListLoras", status))
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
