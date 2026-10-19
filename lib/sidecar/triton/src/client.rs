// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-neutral Triton client over the enabled KServe v2 transports.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_backend_common::DynamoError;
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_sidecar_common::{
    GrpcEndpoint, GrpcTransportConfig, HttpEndpoint, connection_timeout, protocol_error,
};
use tokio::time::{Instant, sleep_until, timeout_at};

use crate::grpc::GrpcClient;
use crate::http::HttpClient;

#[derive(Clone)]
enum Transport {
    Grpc(GrpcClient),
    Http(HttpClient),
}

impl Transport {
    async fn model_ready(&self, model: &str) -> Result<bool, DynamoError> {
        match self {
            Self::Grpc(client) => client.model_ready(model).await,
            Self::Http(client) => client.model_ready(model).await,
        }
    }

    async fn ready_models(&self) -> Result<Vec<String>, DynamoError> {
        match self {
            Self::Grpc(client) => client.ready_models().await,
            Self::Http(client) => client.ready_models().await,
        }
    }

    async fn model_config(&self, model: &str) -> Result<inference::ModelConfig, DynamoError> {
        match self {
            Self::Grpc(client) => client.model_config(model).await,
            Self::Http(client) => client.model_config(model).await,
        }
    }

    async fn infer(
        &self,
        request: inference::ModelInferRequest,
    ) -> Result<inference::ModelInferResponse, DynamoError> {
        match self {
            Self::Grpc(client) => client.infer(request).await,
            Self::Http(client) => client.infer(request).await,
        }
    }
}

/// Triton client spanning every enabled transport.
///
/// Control-plane calls (discovery and model configuration) use the first
/// transport, preferring gRPC. Inference requests rotate across all enabled
/// transports, so each one must be ready before a model is served.
#[derive(Clone)]
pub(crate) struct TritonClient {
    transports: Arc<[Transport]>,
    next: Arc<AtomicUsize>,
}

impl TritonClient {
    pub(crate) async fn connect(
        grpc_endpoint: Option<&GrpcEndpoint>,
        http_endpoint: Option<&HttpEndpoint>,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let grpc = async {
            match grpc_endpoint {
                Some(endpoint) => GrpcClient::connect(endpoint, transport).await.map(Some),
                None => Ok(None),
            }
        };
        let http = async {
            match http_endpoint {
                Some(endpoint) => HttpClient::connect(endpoint, transport).await.map(Some),
                None => Ok(None),
            }
        };
        let (grpc, http) = tokio::try_join!(grpc, http)?;
        let transports: Vec<_> = grpc
            .map(Transport::Grpc)
            .into_iter()
            .chain(http.map(Transport::Http))
            .collect();
        if transports.is_empty() {
            return Err(dynamo_sidecar_common::invalid_argument(
                "at least one of `--grpc-endpoint` and `--http-endpoint` is required",
            ));
        }
        Ok(Self {
            transports: transports.into(),
            next: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub(crate) fn protocols(&self) -> Vec<&'static str> {
        self.transports
            .iter()
            .map(|transport| match transport {
                Transport::Grpc(_) => "grpc",
                Transport::Http(_) => "http",
            })
            .collect()
    }

    fn control(&self) -> &Transport {
        &self.transports[0]
    }

    /// Whether `model` is ready on every enabled transport.
    pub(crate) async fn model_ready(&self, model: &str) -> Result<bool, DynamoError> {
        for transport in self.transports.iter() {
            if !transport.model_ready(model).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Names of the models Triton reports as READY, sorted and de-duplicated.
    pub(crate) async fn ready_models(&self) -> Result<Vec<String>, DynamoError> {
        self.control().ready_models().await
    }

    pub(crate) async fn model_config(
        &self,
        model: &str,
    ) -> Result<inference::ModelConfig, DynamoError> {
        self.control().model_config(model).await
    }

    pub(crate) async fn infer(
        &self,
        request: inference::ModelInferRequest,
    ) -> Result<inference::ModelInferResponse, DynamoError> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.transports.len();
        self.transports[index].infer(request).await
    }
}

/// Poll `probe` until it reports ready or the startup deadline passes.
///
/// `probe` returns `Ok(true)` when Triton is ready, `Ok(false)` when Triton
/// answered but is not yet ready, and `Err` with a description of any failure.
pub(crate) async fn wait_until_ready<F, Fut>(
    protocol: &str,
    transport: GrpcTransportConfig,
    mut probe: F,
) -> Result<(), DynamoError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<bool, String>>,
{
    let deadline = dynamo_sidecar_common::startup_deadline(transport.startup_deadline)?;
    let mut last_error = None;
    loop {
        match timeout_at(deadline, probe()).await {
            Ok(Ok(true)) => return Ok(()),
            Ok(Ok(false)) => last_error = Some("Triton reported ready=false".to_string()),
            Ok(Err(error)) => last_error = Some(error),
            Err(_) => {
                let detail = last_error
                    .as_deref()
                    .map(|error| format!("; last error: {error}"))
                    .unwrap_or_default();
                return Err(connection_timeout(format!(
                    "Triton {protocol} endpoint did not become ready within {:?}{detail}",
                    transport.startup_deadline
                )));
            }
        }
        let retry_at = Instant::now()
            .checked_add(transport.retry_interval)
            .ok_or_else(|| protocol_error("Triton", "retry interval overflowed"))?;
        sleep_until(retry_at.min(deadline)).await;
    }
}
