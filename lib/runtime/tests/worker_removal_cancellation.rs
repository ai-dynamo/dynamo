// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for discovery-driven cancellation while establishing a
//! response stream and graceful draining after the stream is established.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType, match_error_chain},
    pipeline::{ManyOut, ResponseStream, SingleIn, network::Ingress},
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

struct StalledEngine {
    request_received: Arc<Notify>,
    release_request: Arc<Notify>,
}

struct HangingResponseEngine {
    release_stream: Arc<Notify>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<TestResponse>, Error> for HangingResponseEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<TestResponse>, Error> {
        let (_request, context) = input.into_parts();
        let release_stream = self.release_stream.clone();
        let stream = futures::stream::once(async move {
            release_stream.notified().await;
            TestResponse { error: None }
        });
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
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

async fn assert_removal_cancels_request_waiting_for_response_stream(
    distributed: &DistributedRuntime,
) {
    let endpoint = distributed
        .namespace("worker_removal_cancellation".to_string())
        .unwrap()
        .component("backend".to_string())
        .unwrap()
        .endpoint("stalled_generate".to_string());

    let request_received = Arc::new(Notify::new());
    let release_request = Arc::new(Notify::new());
    let ingress = Ingress::for_engine(Arc::new(StalledEngine {
        request_received: request_received.clone(),
        release_request: release_request.clone(),
    }))
    .unwrap();
    let started = endpoint
        .clone()
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
        .unwrap();

    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    let mut instance_updates = client.instance_source.as_ref().clone();
    let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();

    let request = tokio::spawn(async move { router.generate(SingleIn::new(42)).await });
    tokio::time::timeout(Duration::from_secs(5), request_received.notified())
        .await
        .expect("worker did not receive the request");

    endpoint.unregister_endpoint_instance().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !instance_updates.borrow_and_update().is_empty() {
            instance_updates.changed().await.unwrap();
        }
    })
    .await
    .expect("client did not observe the worker leaving discovery");

    let error = tokio::time::timeout(Duration::from_secs(5), request)
        .await
        .expect("request remained blocked after its worker left discovery")
        .expect("request task panicked")
        .expect_err("request unexpectedly succeeded after its worker was removed");
    assert!(
        match_error_chain(error.as_ref(), &[ErrorType::Disconnected], &[]),
        "expected a disconnected error, got: {error:#}"
    );

    release_request.notify_one();
    tokio::time::timeout(Duration::from_secs(5), started.shutdown())
        .await
        .expect("worker endpoint did not shut down after releasing the request")
        .unwrap();
}

async fn assert_removal_drains_established_response_stream(distributed: &DistributedRuntime) {
    let endpoint = distributed
        .namespace("active_worker_removal_cancellation".to_string())
        .unwrap()
        .component("backend".to_string())
        .unwrap()
        .endpoint("active_generate".to_string());

    let release_stream = Arc::new(Notify::new());
    let ingress = Ingress::for_engine(Arc::new(HangingResponseEngine {
        release_stream: release_stream.clone(),
    }))
    .unwrap();
    let started = endpoint
        .clone()
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
        .unwrap();

    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    let mut instance_updates = client.instance_source.as_ref().clone();
    let router = PushRouter::<u64, TestResponse>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let mut response =
        tokio::time::timeout(Duration::from_secs(5), router.generate(SingleIn::new(42)))
            .await
            .expect("worker did not establish its response stream")
            .unwrap();

    endpoint.unregister_endpoint_instance().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !instance_updates.borrow_and_update().is_empty() {
            instance_updates.changed().await.unwrap();
        }
    })
    .await
    .expect("client did not observe the worker leaving discovery");

    release_stream.notify_one();
    let response = tokio::time::timeout(Duration::from_secs(5), response.next())
        .await
        .expect("established response stream did not drain after worker removal")
        .expect("established response stream ended before yielding its response");
    assert!(
        response.err().is_none(),
        "established response stream unexpectedly failed: {:?}",
        response.err()
    );

    tokio::time::timeout(Duration::from_secs(5), started.shutdown())
        .await
        .expect("worker endpoint did not shut down after draining the stream")
        .unwrap();
}

#[tokio::test]
async fn worker_removal_cancels_request_waiting_for_response_stream() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();

    assert_removal_cancels_request_waiting_for_response_stream(&distributed).await;

    runtime.shutdown();
}

#[tokio::test]
async fn worker_removal_drains_established_response_stream() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();

    assert_removal_drains_established_response_stream(&distributed).await;

    runtime.shutdown();
}
