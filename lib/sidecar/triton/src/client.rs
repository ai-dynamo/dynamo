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
/// Control-plane calls (discovery and model configuration) use the first transport, preferring gRPC.
/// Inference requests rotate across all enabled transports, so each one must be ready before a model is served.
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
                "at least one of `--grpc` and `--http` is required",
            ));
        }
        Ok(Self {
            transports: transports.into(),
            next: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Client with no transports, for tests that never reach Triton.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self {
            transports: Vec::new().into(),
            next: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Client over one HTTP transport, for tests against a fake Triton.
    #[cfg(test)]
    pub(crate) fn from_http(client: HttpClient) -> Self {
        Self {
            transports: vec![Transport::Http(client)].into(),
            next: Arc::new(AtomicUsize::new(0)),
        }
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
/// `probe` returns `Ok(true)` when Triton is ready, `Ok(false)` when Triton answered but is not yet ready, and `Err` with a description of any failure.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use dynamo_backend_common::{BackendError, ErrorType};
    use dynamo_llm::grpc::service::kserve::inference;
    use dynamo_sidecar_common::GrpcTransportConfig;
    use serde_json::json;

    use super::{Transport, TritonClient, wait_until_ready};
    use crate::http::tests::{Reply, http_client, infer_reply, transport};

    /// A client that alternates between two HTTP transports, as with `--grpc` and `--http`.
    fn two_transports(
        first: crate::http::HttpClient,
        second: crate::http::HttpClient,
    ) -> TritonClient {
        TritonClient {
            transports: vec![Transport::Http(first), Transport::Http(second)].into(),
            next: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn short_deadline() -> GrpcTransportConfig {
        GrpcTransportConfig {
            retry_interval: Duration::from_millis(5),
            startup_deadline: Duration::from_millis(50),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn connect_requires_a_transport() {
        let error = match TritonClient::connect(None, None, transport()).await {
            Ok(_) => panic!("connected without a transport"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("at least one of `--grpc` and `--http`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn waits_until_the_probe_reports_ready() {
        let calls = AtomicUsize::new(0);
        wait_until_ready("HTTP", transport(), || {
            let call = calls.fetch_add(1, Ordering::Relaxed);
            async move {
                match call {
                    0 => Err("connection refused".to_string()),
                    1 => Ok(false),
                    _ => Ok(true),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn readiness_timeout_reports_the_last_failure() {
        // Each probe yields once, as a network round trip does: `timeout_at` checks the deadline only after its future returns pending.
        let error = wait_until_ready("HTTP", short_deadline(), || async {
            tokio::task::yield_now().await;
            Err::<bool, _>("connection refused".to_string())
        })
        .await
        .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::ConnectionTimeout)
        );
        let message = error.to_string();
        assert!(
            message.contains("Triton HTTP endpoint did not become ready"),
            "{message}"
        );
        assert!(
            message.contains("last error: connection refused"),
            "{message}"
        );

        let error = wait_until_ready("gRPC", short_deadline(), || async {
            tokio::task::yield_now().await;
            Ok(false)
        })
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("Triton gRPC endpoint"), "{error}");
        assert!(error.contains("ready=false"), "{error}");
    }

    #[tokio::test]
    async fn model_must_be_ready_on_every_transport() {
        let (first, first_server) = http_client(vec![
            Reply::json("200 OK", json!({})),
            Reply::json("400 Bad Request", json!({"error": "loading"})),
        ])
        .await;
        let (second, second_server) = http_client(vec![Reply::json(
            "400 Bad Request",
            json!({"error": "loading"}),
        )])
        .await;
        let client = two_transports(first, second);
        assert_eq!(client.protocols(), ["http", "http"]);

        // Ready on the first transport but not the second.
        assert!(!client.model_ready("model").await.unwrap());
        // Not ready on the first transport: the second is not asked, since its fake server has no reply left.
        assert!(!client.model_ready("model").await.unwrap());

        assert_eq!(first_server.await.unwrap().len(), 3);
        assert_eq!(second_server.await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn inference_alternates_between_transports() {
        let reply =
            |value: i32| infer_reply("OUTPUT0", "INT32", &[1], value.to_le_bytes().to_vec());
        let (first, first_server) = http_client(vec![reply(1), reply(3)]).await;
        let (second, second_server) = http_client(vec![reply(2)]).await;
        let client = two_transports(first, second);

        let request = inference::ModelInferRequest {
            model_name: "model".to_string(),
            ..Default::default()
        };
        let mut values = Vec::new();
        for _ in 0..3 {
            let response = client.infer(request.clone()).await.unwrap();
            values.push(i32::from_le_bytes(
                response.raw_output_contents[0]
                    .as_slice()
                    .try_into()
                    .unwrap(),
            ));
        }
        assert_eq!(values, [1, 2, 3]);
        assert_eq!(first_server.await.unwrap().len(), 3);
        assert_eq!(second_server.await.unwrap().len(), 2);
    }
}
