// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::future::Future;
use std::time::Duration;

use dynamo_backend_common::{DynamoError, EngineAvailability};
use dynamo_sidecar_common::v14::GrpcChannelPool;
use dynamo_sidecar_common::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig};
use tokio::sync::watch;
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::{HealthCheckRequest, HealthCheckResponse, health_client::HealthClient};

pub(crate) use dynamo_sidecar_common::v14::status_to_dynamo;
pub(crate) use dynamo_sidecar_common::{engine_shutdown, invalid_argument, startup_deadline};

use crate::proto as pb;

pub(crate) const CONTROL_SERVICE: &str = "vllm.Control";
pub(crate) const INFERENCE_SERVICE: &str = "vllm.Inference";
/// Engine-process health on vLLM servers that split it from readiness.
pub(crate) const LIVENESS_SERVICE: &str = "liveness";
/// Whether such a server accepts work; false while paused or asleep.
pub(crate) const READINESS_SERVICE: &str = "readiness";
const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";
// A connection that stops answering fails its open streams after about 40 s.
const HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
const HTTP2_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    /// `bootstrap`: see `GrpcChannelPool::connect`. True for pre-logging
    /// synchronous constructors; false for deferred discovery and engine start.
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        startup_deadline: Instant,
        bootstrap: bool,
    ) -> Result<Self, DynamoError> {
        let pool = timeout_at(
            startup_deadline,
            GrpcChannelPool::connect_with_keepalive(
                "vLLM",
                endpoint,
                transport,
                bootstrap,
                Some((HTTP2_KEEPALIVE_INTERVAL, HTTP2_KEEPALIVE_TIMEOUT)),
            ),
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

    pub(crate) fn health_channel(&self) -> Channel {
        self.pool.next_channel()
    }

    /// Wait until each service reports SERVING. A draining or stalled engine
    /// leaves calls hanging, so each check is retried after `attempt_timeout`.
    /// On a reconnect, an unregistered service is retried too: vLLM removes its
    /// health services just before its listener closes.
    pub(crate) async fn wait_for_services(
        &self,
        services: &[&str],
        startup_deadline: Instant,
        retry_interval: Duration,
        attempt_timeout: Duration,
        is_reconnect: bool,
    ) -> Result<(), DynamoError> {
        for service in services {
            self.wait_for_service(
                service,
                startup_deadline,
                retry_interval,
                attempt_timeout,
                Unregistered::on_connect(is_reconnect),
            )
            .await?;
        }
        Ok(())
    }

    /// Wait until vLLM can serve and return whether it is ready now. A vLLM
    /// with `liveness` keeps `vllm.Inference` NOT_SERVING while it is paused or
    /// asleep, so the sidecar waits for `liveness` there and reads `readiness`.
    pub(crate) async fn wait_for_engine(
        &self,
        startup_deadline: Instant,
        retry_interval: Duration,
        attempt_timeout: Duration,
        is_reconnect: bool,
    ) -> Result<EngineAvailability, DynamoError> {
        let unregistered = Unregistered::on_connect(is_reconnect);
        let wait = |service, unregistered| {
            self.wait_for_service(
                service,
                startup_deadline,
                retry_interval,
                attempt_timeout,
                unregistered,
            )
        };
        wait(CONTROL_SERVICE, unregistered).await?;
        if !wait(LIVENESS_SERVICE, Unregistered::Absent).await? {
            wait(INFERENCE_SERVICE, unregistered).await?;
            return Ok(EngineAvailability::Ready);
        }
        let mut client = HealthClient::new(self.pool.next_channel());
        let readiness = client.check(HealthCheckRequest {
            service: READINESS_SERVICE.to_string(),
        });
        let is_ready = matches!(
            tokio::time::timeout(attempt_timeout, readiness).await,
            Ok(Ok(response)) if response.get_ref().status == ServingStatus::Serving as i32
        );
        // The health monitor corrects a failed check.
        Ok(if is_ready {
            EngineAvailability::Ready
        } else {
            EngineAvailability::NotReady
        })
    }

    /// Wait until `service` is SERVING; false if it is unregistered and
    /// `unregistered` is `Absent`.
    async fn wait_for_service(
        &self,
        service: &str,
        deadline: Instant,
        retry_interval: Duration,
        attempt_timeout: Duration,
        unregistered: Unregistered,
    ) -> Result<bool, DynamoError> {
        let started = Instant::now();
        loop {
            let mut client = HealthClient::new(self.pool.next_channel());
            let attempt_deadline = Instant::now()
                .checked_add(attempt_timeout)
                .map_or(deadline, |attempt| attempt.min(deadline));
            let last_status = match timeout_at(
                attempt_deadline,
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
                        return Ok(true);
                    }
                    format!("reported {}", status.as_str_name())
                }
                Ok(Err(status))
                    if unregistered != Unregistered::Retry
                        && matches!(
                            status.code(),
                            tonic::Code::NotFound | tonic::Code::Unimplemented
                        ) =>
                {
                    if unregistered == Unregistered::Absent {
                        return Ok(false);
                    }
                    return Err(protocol_error(format!(
                        "{service} is unavailable through the standard gRPC health API: {status}"
                    )));
                }
                Ok(Err(status)) => format!("health check failed: {status}"),
                Err(_) => format!("health check did not answer within {attempt_timeout:?}"),
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

    /// Abort by caller request ID. vLLM's GenerateStream producer does not
    /// watch for a closed client stream, so dropping the stream alone leaves a
    /// queued request in the engine until a slot frees up.
    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let mut client = self.control_client();
        tokio::time::timeout(
            ABORT_RPC_DEADLINE,
            client.abort(pb::AbortRequest {
                request_ids: vec![request_id],
            }),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(format!(
                "Abort did not respond within {ABORT_RPC_DEADLINE:?}"
            ))
        })?
        .map(|_| ())
        .map_err(|status| status_to_dynamo("Abort", status))
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

/// What an unregistered health service means while waiting for it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unregistered {
    /// This vLLM lacks the service.
    Fatal,
    /// vLLM removes its health services just before its listener closes.
    Retry,
    /// The service is optional.
    Absent,
}

impl Unregistered {
    fn on_connect(is_reconnect: bool) -> Self {
        if is_reconnect {
            Self::Retry
        } else {
            Self::Fatal
        }
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}

/// Publish `state` only when it changes.
pub(crate) fn report(availability: &watch::Sender<EngineAvailability>, state: EngineAvailability) {
    availability.send_if_modified(|current| std::mem::replace(current, state) != state);
}

/// Follow vLLM's health on `channel` until the engine is lost, then report
/// `Lost`. Uses `liveness` and `readiness`, or `vllm.Inference` on servers
/// without them. A NOT_SERVING readiness is `NotReady`; any stream end or error
/// is a loss.
pub(crate) async fn monitor_health(
    channel: Channel,
    availability: &watch::Sender<EngineAvailability>,
    open_timeout: Duration,
) {
    let reason = follow_health(HealthClient::new(channel), availability, open_timeout).await;
    tracing::warn!(reason = %reason.escape_debug(), "vLLM engine lost");
    report(availability, EngineAvailability::Lost);
}

async fn follow_health(
    mut health: HealthClient<Channel>,
    availability: &watch::Sender<EngineAvailability>,
    open_timeout: Duration,
) -> String {
    let mut liveness = match watch_health(&mut health, LIVENESS_SERVICE, open_timeout).await {
        Ok(stream) => stream,
        Err(status) if status.code() == tonic::Code::NotFound => {
            tracing::info!(
                "vLLM reports no {LIVENESS_SERVICE} health; watching {INFERENCE_SERVICE}"
            );
            let mut inference =
                match watch_health(&mut health, INFERENCE_SERVICE, open_timeout).await {
                    Ok(stream) => stream,
                    Err(status) => return format!("{INFERENCE_SERVICE} health watch: {status}"),
                };
            loop {
                match next_serving(&mut inference, INFERENCE_SERVICE).await {
                    Ok(true) => report(availability, EngineAvailability::Ready),
                    Ok(false) => report(availability, EngineAvailability::NotReady),
                    Err(reason) => return reason,
                }
            }
        }
        Err(status) => return format!("{LIVENESS_SERVICE} health watch: {status}"),
    };
    tracing::info!("watching vLLM {LIVENESS_SERVICE} and {READINESS_SERVICE} health");
    let mut readiness = match watch_health(&mut health, READINESS_SERVICE, open_timeout).await {
        Ok(stream) => stream,
        Err(status) => return format!("{READINESS_SERVICE} health watch: {status}"),
    };
    // Report once both streams have sent their current status.
    let (mut is_live, mut is_ready) = (false, None);
    loop {
        tokio::select! {
            serving = next_serving(&mut liveness, LIVENESS_SERVICE) => match serving {
                Ok(true) => is_live = true,
                Ok(false) => return format!("{LIVENESS_SERVICE} reported NOT_SERVING"),
                Err(reason) => return reason,
            },
            serving = next_serving(&mut readiness, READINESS_SERVICE) => match serving {
                Ok(serving) => is_ready = Some(serving),
                Err(reason) => return reason,
            },
        }
        match is_ready {
            Some(true) if is_live => report(availability, EngineAvailability::Ready),
            Some(false) if is_live => report(availability, EngineAvailability::NotReady),
            _ => {}
        }
    }
}

async fn watch_health(
    health: &mut HealthClient<Channel>,
    service: &str,
    open_timeout: Duration,
) -> Result<tonic::Streaming<HealthCheckResponse>, tonic::Status> {
    let request = HealthCheckRequest {
        service: service.to_string(),
    };
    tokio::time::timeout(open_timeout, health.watch(request))
        .await
        .map_err(|_| tonic::Status::deadline_exceeded(format!("no answer in {open_timeout:?}")))?
        .map(tonic::Response::into_inner)
}

/// Whether the next status is SERVING, or why the stream stopped.
async fn next_serving(
    stream: &mut tonic::Streaming<HealthCheckResponse>,
    service: &str,
) -> Result<bool, String> {
    match stream.message().await {
        Ok(Some(response)) => {
            Ok(ServingStatus::try_from(response.status) == Ok(ServingStatus::Serving))
        }
        Ok(None) => Err(format!("{service} health stream ended")),
        Err(status) => Err(format!("{service} health stream failed: {status}")),
    }
}

// Bound how long a stalled RPC can hold the lifecycle lock.
pub(crate) const LORA_RPC_DEADLINE: Duration = Duration::from_secs(60);

const ABORT_RPC_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub(crate) struct LoraRpcError {
    pub(crate) rpc: &'static str,
    pub(crate) code: tonic::Code,
    pub(crate) message: String,
}

impl LoraRpcError {
    // Other errors may follow a committed mutation; reconcile them with ListLoras.
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
