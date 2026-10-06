// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Result};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS;
use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType, match_error_chain},
    pipeline::{
        ManyOut, PipelineError, ResponseStream, SingleIn,
        network::{Ingress, PushWorkHandler, ResponsePlaneMode},
    },
    protocols::maybe_error::MaybeError,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TestResponse {
    #[serde(default)]
    error: Option<DynamoError>,
}

impl MaybeError for TestResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self {
            error: Some(DynamoError::from(
                Box::new(err) as Box<dyn std::error::Error + 'static>
            )),
        }
    }

    fn err(&self) -> Option<DynamoError> {
        self.error.clone()
    }
}

struct NeverConnectsHandler {
    request_received: Arc<Notify>,
}

#[async_trait]
impl PushWorkHandler for NeverConnectsHandler {
    async fn handle_payload(
        &self,
        _payload: Bytes,
        _request_id: Option<String>,
    ) -> Result<(), PipelineError> {
        self.request_received.notify_one();
        Ok(())
    }

    fn add_metrics(
        &self,
        _endpoint: &dynamo_runtime::component::Endpoint,
        _metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        Ok(())
    }
}

/// `Ingress` connects back to the frontend before calling `generate`, so
/// stalling here leaves the response stream connected but without a prologue.
struct StalledEngine {
    request_received: Arc<Notify>,
    release_request: Arc<Notify>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<TestResponse>, Error> for StalledEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<TestResponse>, Error> {
        self.request_received.notify_one();
        self.release_request.notified().await;
        let (_request, context) = input.into_parts();
        Ok(ResponseStream::new(
            Box::pin(futures::stream::empty()),
            context.context(),
        ))
    }
}

async fn assert_establish_timeout(
    distributed: &DistributedRuntime,
    endpoint_name: &str,
    handler: Arc<dyn PushWorkHandler>,
    request_received: Arc<Notify>,
    before_shutdown: impl FnOnce(),
) {
    let endpoint = distributed
        .namespace("response_stream_establish_timeout".to_string())
        .unwrap()
        .component("backend".to_string())
        .unwrap()
        .endpoint(endpoint_name.to_string());
    let started = endpoint
        .clone()
        .endpoint_builder()
        .handler(handler)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
        .unwrap();

    let client = endpoint.client().await.unwrap();
    let instance_id = client.wait_for_instances().await.unwrap()[0].id();
    let router =
        PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
            .await
            .unwrap();

    let request = tokio::spawn(async move { router.generate(SingleIn::new(42)).await });
    tokio::time::timeout(Duration::from_secs(5), request_received.notified())
        .await
        .expect("worker did not receive the request");

    let error = tokio::time::timeout(Duration::from_secs(10), request)
        .await
        .expect("request remained blocked past the response timeout")
        .expect("request task panicked")
        .expect_err("request unexpectedly succeeded without a response stream");
    assert!(
        match_error_chain(error.as_ref(), &[ErrorType::ResponseTimeout], &[]),
        "{endpoint_name}: expected a response timeout, got: {error:#}"
    );
    assert!(
        !client.instance_ids_avail().contains(&instance_id),
        "{endpoint_name}: worker that never established a response stream should be quarantined"
    );

    before_shutdown();
    tokio::time::timeout(Duration::from_secs(5), started.shutdown())
        .await
        .expect("worker endpoint did not shut down")
        .unwrap();
}

// Both cases share one runtime: the TCP request-plane server is process-wide
// and does not outlive the tokio runtime that started it.
#[tokio::test]
async fn tcp_request_fails_when_response_stream_is_never_established() {
    temp_env::async_with_vars([(DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS, Some("1"))], async {
        let runtime = Runtime::from_current().unwrap();
        let config = DistributedConfig {
            response_plane: Some(ResponsePlaneMode::Tcp),
            ..DistributedConfig::process_local()
        };
        let distributed = DistributedRuntime::new(runtime.clone(), config)
            .await
            .unwrap();

        let request_received = Arc::new(Notify::new());
        assert_establish_timeout(
            &distributed,
            "never_connects_back",
            Arc::new(NeverConnectsHandler {
                request_received: request_received.clone(),
            }),
            request_received,
            || {},
        )
        .await;

        let request_received = Arc::new(Notify::new());
        let release_request = Arc::new(Notify::new());
        let stalled: Arc<dyn PushWorkHandler> = Ingress::for_engine(Arc::new(StalledEngine {
            request_received: request_received.clone(),
            release_request: release_request.clone(),
        }))
        .unwrap();
        assert_establish_timeout(
            &distributed,
            "stalls_before_prologue",
            stalled,
            request_received,
            || release_request.notify_one(),
        )
        .await;

        runtime.shutdown();
    })
    .await;
}
