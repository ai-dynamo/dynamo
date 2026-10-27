// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for an LPX engine's OpenEngine gRPC services (`openengine.v1`).

use std::time::Duration;

use dynamo_backend_common::DynamoError;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};
use tokio::time::{Instant, sleep_until, timeout, timeout_at};
use tonic::transport::Channel;

pub(crate) use dynamo_sidecar_common::{
    engine_shutdown, invalid_argument, protocol_error, status_to_dynamo,
};

use crate::pb;
use crate::pb::control_client::ControlClient;
use crate::pb::inference_client::InferenceClient;

/// Bound on a `Control` RPC or cleanup dispatch, so an unresponsive server
/// cannot hang startup, cancellation, or rejection.
pub(crate) const RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a still-starting engine is reported while the bootstrap waits.
const WAIT_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// The peer name in connection and protocol errors until the server has
/// reported its engine's name, and in place of an empty one.
pub(crate) const UNIDENTIFIED_PEER: &str = "LPX OpenEngine";

pub(crate) struct OpenEngineClient {
    pool: GrpcChannelPool,
    /// The engine's name in errors and logs.
    peer: String,
}

impl OpenEngineClient {
    pub(crate) async fn connect(
        peer: &str,
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        // `false`: retries are reported through tracing, which the sidecar
        // runner installs before discovery starts.
        let pool = timeout_at(
            deadline,
            GrpcChannelPool::connect(peer, endpoint, transport, false),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "the {peer} gRPC connection pool to {endpoint} exceeded the gRPC startup deadline"
            ))
        })??;
        Ok(Self {
            pool,
            peer: peer.to_owned(),
        })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    fn inference(&self) -> InferenceClient<Channel> {
        InferenceClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    fn control(&self) -> ControlClient<Channel> {
        ControlClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    /// Opens a `Generate` stream. The status comes back raw because its code
    /// decides how the failure reaches the frontend (see `engine.rs`).
    ///
    /// `trace_headers` become W3C Trace Context metadata, which the hybrid
    /// engine reads into its request tracing and the LPU-only engine ignores.
    /// No routing metadata is sent: the sidecar rejects nonzero data-parallel
    /// ranks and priorities before dispatch.
    pub(crate) async fn generate(
        &self,
        request: pb::GenerateRequest,
        trace_headers: Vec<(&'static str, String)>,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, tonic::Status> {
        let mut request = tonic::Request::new(request);
        for (key, value) in trace_headers {
            match value.parse() {
                Ok(value) => {
                    request.metadata_mut().insert(key, value);
                }
                Err(_) => tracing::debug!(key, "dropping trace metadata that is not ASCII"),
            }
        }
        self.inference()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
    }

    pub(crate) async fn server_info(
        &self,
        deadline: Instant,
    ) -> Result<pb::ServerInfo, DynamoError> {
        let attempt_deadline = deadline.min(Instant::now() + RPC_TIMEOUT);
        timeout_at(
            attempt_deadline,
            self.control().get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            let peer = &self.peer;
            connection_timeout(if attempt_deadline < deadline {
                format!("the {peer} server did not answer GetServerInfo within {RPC_TIMEOUT:?}")
            } else {
                format!(
                    "the {peer} server did not answer GetServerInfo within \
                     --grpc-startup-deadline-secs"
                )
            })
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetServerInfo", status))
    }

    /// An LPX engine serves one model and accepts an empty model name. The
    /// sidecar does not know the name until this call returns it, so it asks
    /// for none.
    async fn get_model_info(&self) -> Result<pb::ModelInfo, tonic::Status> {
        self.control()
            .get_model_info(pb::GetModelInfoRequest {
                model: String::new(),
            })
            .await
            .map(tonic::Response::into_inner)
    }

    /// Polls `GetModelInfo` until the engine answers or `deadline` passes.
    ///
    /// This wait keeps the worker from registering before the engine admits
    /// requests. It relies on the engine answering UNAVAILABLE until then.
    ///
    /// Transport failures and UNAVAILABLE are retried, and so is a call that
    /// gets no answer within `attempt_timeout`. An answer saying the request
    /// itself is wrong (no Control service, or no such model) cannot change by
    /// waiting, so it ends the wait at once.
    pub(crate) async fn wait_for_model_info(
        &self,
        deadline: Instant,
        retry_interval: Duration,
        attempt_timeout: Duration,
    ) -> Result<pb::ModelInfo, DynamoError> {
        let peer = self.peer.as_str();
        let mut last = "it never answered".to_string();
        let mut last_logged: Option<(Instant, String)> = None;
        loop {
            let attempt_deadline = deadline.min(Instant::now() + attempt_timeout);
            match timeout_at(attempt_deadline, self.get_model_info()).await {
                Ok(Ok(info)) => return Ok(info),
                Ok(Err(status)) if is_request_itself_wrong(&status) => {
                    return Err(status_to_dynamo("GetModelInfo", status));
                }
                Ok(Err(status)) => {
                    last = format!("{:?}: {}", status.code(), status.message().escape_debug());
                }
                // A connected engine that stalls on one call is asked again.
                Err(_) if attempt_deadline < deadline => {
                    last = format!("no answer within {attempt_timeout:?}");
                }
                Err(_) => break,
            }
            let now = Instant::now();
            let next_attempt = now + retry_interval;
            if next_attempt >= deadline {
                break;
            }
            let should_log = last_logged.as_ref().is_none_or(|(at, reason)| {
                *reason != last || now.duration_since(*at) >= WAIT_LOG_INTERVAL
            });
            if should_log {
                tracing::info!(peer, reason = %last, "waiting for the server to answer GetModelInfo");
                last_logged = Some((now, last.clone()));
            }
            sleep_until(next_attempt).await;
        }
        Err(connection_timeout(format!(
            "the {peer} server never answered GetModelInfo within \
             --grpc-startup-deadline-secs ({last})"
        )))
    }

    /// The hybrid engine's abort detaches delivery: a request it already
    /// picked up keeps running in its scheduler until it ends on its own. The
    /// LPU-only engine's cancels the request.
    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let response = timeout(
            RPC_TIMEOUT,
            self.control().abort(pb::AbortRequest {
                target: Some(pb::abort_request::Target::RequestId(request_id)),
            }),
        )
        .await
        .map_err(|_| connection_timeout(format!("Abort did not respond within {RPC_TIMEOUT:?}")))?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("Abort", status))?;
        // A request already gone is not an error; an unspecified status is drift.
        match pb::AbortStatus::try_from(response.status) {
            Ok(pb::AbortStatus::Aborted | pb::AbortStatus::AlreadyFinished) => Ok(()),
            _ => Err(protocol_error(
                &self.peer,
                format!(
                    "the server returned an unexpected abort status {}: {}",
                    response.status, response.message
                ),
            )),
        }
    }
}

fn is_request_itself_wrong(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::Unimplemented
    )
}
