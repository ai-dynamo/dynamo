// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto as pb;
use crate::proto::open_engine_client::OpenEngineClient;

pub type Client = OpenEngineClient<Channel>;

#[derive(Clone, Debug)]
pub struct Discovery {
    pub engine: pb::EngineInfo,
    pub model: pb::ModelInfo,
    pub selected_model: String,
}

pub struct Pool {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl Pool {
    pub async fn connect(
        uri: &str,
        config: &TransportConfig,
        size: usize,
    ) -> Result<Self, DynamoError> {
        let mut channels = Vec::with_capacity(size.max(1));
        for _ in 0..size.max(1) {
            channels.push(connect_channel(uri, config).await?);
        }
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn stream_client(&self) -> Client {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        OpenEngineClient::new(self.channels[index].clone())
    }

    pub fn control_client(&self) -> Client {
        OpenEngineClient::new(self.channels[0].clone())
    }

    pub fn channel(&self) -> Channel {
        self.channels[0].clone()
    }
}

pub async fn connect(uri: &str, config: &TransportConfig) -> Result<Client, DynamoError> {
    Ok(OpenEngineClient::new(connect_channel(uri, config).await?))
}

async fn connect_channel(uri: &str, config: &TransportConfig) -> Result<Channel, DynamoError> {
    let deadline = Instant::now() + config.deadline;
    loop {
        let result = async {
            Endpoint::from_shared(uri.to_string())
                .map_err(|error| error.to_string())?
                .connect_timeout(config.connect_timeout)
                .connect()
                .await
                .map_err(|error| error.to_string())
        }
        .await;
        match result {
            Ok(channel) => return Ok(channel),
            Err(error) if Instant::now() < deadline => {
                tracing::debug!(%error, %uri, "OpenEngine connection attempt failed");
                tokio::time::sleep(config.poll_interval).await;
            }
            Err(error) => {
                return Err(cannot_connect(format!(
                    "could not reach OpenEngine at {uri} within {:?}: {error}",
                    config.deadline
                )));
            }
        }
    }
}

pub async fn discover(
    client: &mut Client,
    requested_model: Option<&str>,
    expected_engine: Option<&str>,
    deadline: Duration,
) -> Result<Discovery, DynamoError> {
    let deadline_at = Instant::now() + deadline;
    let mut engine = timeout_discovery_rpc(
        "GetEngineInfo",
        deadline_at,
        client.get_engine_info(pb::GetEngineInfoRequest {}),
    )
    .await?
    .map_err(|status| status_to_dynamo("GetEngineInfo", status))?
    .into_inner();

    validate_schema(&engine)?;
    if let Some(expected) = expected_engine
        && engine.engine_name != expected
    {
        return Err(invalid_arg(format!(
            "OpenEngine engine mismatch: expected `{expected}`, discovered `{}`",
            engine.engine_name
        )));
    }

    let selected_model = select_model(&engine.supported_models, requested_model)?;
    // Model enumeration is part of GetEngineInfo in revision 2; there is no
    // separate ListModels RPC. Selection therefore shares GetEngineInfo's
    // bounded deadline above.
    let model = timeout_discovery_rpc(
        "GetModelInfo",
        deadline_at,
        client.get_model_info(pb::GetModelInfoRequest {
            model: selected_model.clone(),
        }),
    )
    .await?
    .map_err(|status| status_to_dynamo("GetModelInfo", status))?
    .into_inner();
    if model.model_id != selected_model {
        return Err(invalid_arg(format!(
            "OpenEngine GetModelInfo returned model_id `{}` for selected model `{selected_model}`",
            model.model_id
        )));
    }
    let connector = timeout_discovery_rpc(
        "GetKvConnectorInfo",
        deadline_at,
        client.get_kv_connector_info(pb::GetKvConnectorInfoRequest {}),
    )
    .await?
    .map_err(|status| status_to_dynamo("GetKvConnectorInfo", status))?
    .into_inner();
    engine.kv_connector = Some(connector);
    Ok(Discovery {
        engine,
        model,
        selected_model,
    })
}

async fn timeout_discovery_rpc<F, T>(
    rpc: &str,
    deadline_at: Instant,
    future: F,
) -> Result<T, DynamoError>
where
    F: std::future::Future<Output = T>,
{
    let remaining = deadline_at.saturating_duration_since(Instant::now());
    tokio::time::timeout(remaining, future).await.map_err(|_| {
        cannot_connect(format!(
            "OpenEngine bootstrap {rpc} exceeded the configured discovery deadline"
        ))
    })
}

fn select_model(models: &[String], requested: Option<&str>) -> Result<String, DynamoError> {
    if let Some(requested) = requested {
        if models.iter().any(|model| model == requested) {
            return Ok(requested.to_string());
        }
        return Err(invalid_arg(format!(
            "requested model `{requested}` is not advertised by OpenEngine (available: {})",
            models.join(", ")
        )));
    }
    match models {
        [only] => Ok(only.clone()),
        [] => Err(invalid_arg(
            "OpenEngine advertised no models; pass --model only after fixing server discovery",
        )),
        _ => Err(invalid_arg(format!(
            "OpenEngine advertises multiple models ({}); select one with --model",
            models.join(", ")
        ))),
    }
}

fn validate_schema(engine: &pb::EngineInfo) -> Result<(), DynamoError> {
    let client_revision = openengine_proto::SCHEMA_REVISION;
    if engine.schema_revision == 0 {
        return Err(invalid_arg("OpenEngine reported invalid schema revision 0"));
    }
    if engine.minimum_client_revision > client_revision {
        return Err(invalid_arg(format!(
            "OpenEngine requires client schema revision {}, but this sidecar supports revision {client_revision}",
            engine.minimum_client_revision
        )));
    }
    if engine.schema_revision > client_revision {
        return Err(invalid_arg(format!(
            "OpenEngine schema revision {} is newer than this sidecar's tested revision {client_revision}",
            engine.schema_revision,
        )));
    }
    const COMPILED_OPENENGINE_COMMIT: &str = env!("OPENENGINE_PROTO_COMMIT");
    if engine.schema_release != COMPILED_OPENENGINE_COMMIT {
        return Err(invalid_arg(format!(
            "OpenEngine schema_release `{}` does not match the sidecar's pinned contract commit `{COMPILED_OPENENGINE_COMMIT}`",
            engine.schema_release
        )));
    }
    Ok(())
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_arg(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists
        | tonic::Code::Unimplemented => BackendError::InvalidArgument,
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

pub fn engine_error_to_dynamo(error: &pb::EngineError) -> DynamoError {
    let code = pb::ErrorCode::try_from(error.code).unwrap_or(pb::ErrorCode::Unspecified);
    let kind = match code {
        pb::ErrorCode::InvalidArgument
        | pb::ErrorCode::UnsupportedFeature
        | pb::ErrorCode::RoleMismatch
        | pb::ErrorCode::ModelNotFound
        | pb::ErrorCode::KvSessionNotFound
        | pb::ErrorCode::RequestNotFound
        | pb::ErrorCode::DuplicateRequest => BackendError::InvalidArgument,
        pb::ErrorCode::Cancelled => BackendError::Cancelled,
        pb::ErrorCode::Draining => BackendError::EngineShutdown,
        pb::ErrorCode::KvTransferFailed => BackendError::Disconnected,
        pb::ErrorCode::Overloaded | pb::ErrorCode::Internal | pb::ErrorCode::Unspecified => {
            BackendError::Unknown
        }
    };
    backend(kind, format!("OpenEngine [{code:?}]: {}", error.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_model_for_multi_model_server() {
        let models = vec!["a".to_string(), "b".to_string()];
        assert!(select_model(&models, None).is_err());
        assert_eq!(select_model(&models, Some("b")).unwrap(), "b");
    }

    #[test]
    fn rejects_nonimmutable_schema_release() {
        let mut engine = pb::EngineInfo {
            schema_revision: openengine_proto::SCHEMA_REVISION,
            minimum_client_revision: 1,
            schema_release: "unreleased".to_string(),
            ..Default::default()
        };
        assert!(validate_schema(&engine).is_err());
        engine.schema_release = String::new();
        assert!(validate_schema(&engine).is_err());
        engine.schema_release = "main".to_string();
        assert!(validate_schema(&engine).is_err());
        engine.schema_release = "cea19cb".to_string();
        assert!(validate_schema(&engine).is_err());
        engine.schema_release = env!("OPENENGINE_PROTO_COMMIT").to_string();
        assert!(validate_schema(&engine).is_ok());
        engine.schema_release = "0123456789abcdef0123456789abcdef01234567".to_string();
        assert!(validate_schema(&engine).is_err());
        engine.schema_release = "v0.2.0".to_string();
        assert!(validate_schema(&engine).is_err());
    }
}
