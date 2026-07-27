// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};

pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, status_to_dynamo};

use crate::proto as pb;

pub(crate) const CONTROL_SERVICE: &str = "vllm.Control";
pub(crate) const INFERENCE_SERVICE: &str = "vllm.Inference";

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("vLLM", endpoint, transport).await?;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    pub(crate) async fn wait_for_services(
        &self,
        services: &[&str],
        startup_deadline: Duration,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        let deadline = checked_deadline(startup_deadline)?;
        for service in services {
            self.wait_for_service(service, deadline, startup_deadline, retry_interval)
                .await?;
        }
        Ok(())
    }

    async fn wait_for_service(
        &self,
        service: &str,
        deadline: Instant,
        startup_deadline: Duration,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
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
                Ok(Err(status)) => format!("health check failed: {status}"),
                Err(_) => "health check exceeded the startup deadline".to_string(),
            };

            let now = Instant::now();
            if now >= deadline {
                return Err(dynamo_sidecar_common::cannot_connect(format!(
                    "{service} did not become SERVING within {startup_deadline:?}: {last_status}"
                )));
            }
            let retry_at = now.checked_add(retry_interval).unwrap_or(deadline);
            sleep_until(retry_at.min(deadline)).await;
        }
    }

    pub(crate) async fn discover(
        &self,
        startup_deadline: Duration,
    ) -> Result<(pb::ModelInfo, pb::ServerInfo), DynamoError> {
        let deadline = checked_deadline(startup_deadline)?;
        let channel = self.pool.next_channel();
        let mut client = pb::control_client::ControlClient::new(channel)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let model = timeout_at(deadline, client.get_model_info(pb::GetModelInfoRequest {}))
            .await
            .map_err(|_| {
                dynamo_sidecar_common::connection_timeout(format!(
                    "GetModelInfo exceeded the vLLM startup deadline of {startup_deadline:?}"
                ))
            })?
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        let server = timeout_at(
            deadline,
            client.get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(format!(
                "GetServerInfo exceeded the vLLM startup deadline of {startup_deadline:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetServerInfo", status))?;
        Ok((model, server))
    }

    pub(crate) async fn generate_stream(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let mut client = pb::inference_client::InferenceClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("GenerateStream", status))
    }
}

fn checked_deadline(duration: Duration) -> Result<Instant, DynamoError> {
    Instant::now().checked_add(duration).ok_or_else(|| {
        invalid_argument(format!(
            "gRPC startup deadline {duration:?} exceeds the supported monotonic clock range"
        ))
    })
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}
