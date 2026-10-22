// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Error, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, watch};

use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS;
use dynamo_runtime::config::environment_names::response_plane::DYN_RESPONSE_STREAM_ESTABLISH_TIMEOUT_SECS;
use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::{Endpoint, Instance},
    discovery::EndpointInstanceId,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType, match_error_chain},
    pipeline::{
        AddressedPushRouter, AddressedRequest, ManyIn, ManyOut, PipelineError, ResponseStream,
        SingleIn, StreamingDispatch,
        network::{Ingress, PushWorkHandler, ResponsePlaneMode},
    },
    protocols::maybe_error::MaybeError,
};

const ESTABLISH_TIMEOUT: Duration = Duration::from_secs(1);

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

/// Sends its prologue (by returning a stream) but produces no response until
/// released.
struct SilentEngine {
    release: Arc<Notify>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<TestResponse>, Error> for SilentEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<TestResponse>, Error> {
        let (_request, context) = input.into_parts();
        let release = self.release.clone();
        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(async move {
                release.notified().await;
                TestResponse { error: None }
            })),
            context.context(),
        ))
    }
}

/// Answers every request with a single successful response.
struct CountingEngine {
    requests: Arc<AtomicUsize>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u64>, ManyOut<TestResponse>, Error> for CountingEngine {
    async fn generate(&self, input: SingleIn<u64>) -> Result<ManyOut<TestResponse>, Error> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let (_request, context) = input.into_parts();
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([TestResponse { error: None }])),
            context.context(),
        ))
    }
}

/// TCP proxy between the frontend and the worker's request plane. Frontend
/// bytes pass through and are counted. Worker bytes, which carry the request
/// ACKs, are held until `release` is set.
struct HoldingProxy {
    addr: SocketAddr,
    forwarded: Arc<AtomicUsize>,
    release: watch::Sender<bool>,
}

async fn start_holding_proxy(upstream: SocketAddr) -> HoldingProxy {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let forwarded = Arc::new(AtomicUsize::new(0));
    let (release, release_rx) = watch::channel(false);
    let counter = forwarded.clone();
    tokio::spawn(async move {
        while let Ok((inbound, _)) = listener.accept().await {
            let outbound = TcpStream::connect(upstream).await.unwrap();
            let (mut inbound_read, mut inbound_write) = inbound.into_split();
            let (mut outbound_read, mut outbound_write) = outbound.into_split();
            let counter = counter.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 64 * 1024];
                loop {
                    let n = match inbound_read.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    counter.fetch_add(n, Ordering::SeqCst);
                    if outbound_write.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            });
            let mut release_rx = release_rx.clone();
            tokio::spawn(async move {
                let _ = release_rx.wait_for(|released| *released).await;
                let _ = tokio::io::copy(&mut outbound_read, &mut inbound_write).await;
            });
        }
    });
    HoldingProxy {
        addr,
        forwarded,
        release,
    }
}

/// Request-plane dispatch that sends every request through a proxy.
struct ProxiedDispatch {
    inner: Arc<AddressedPushRouter>,
    proxy: SocketAddr,
}

#[async_trait]
impl StreamingDispatch<u64, TestResponse> for ProxiedDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<u64>>,
    ) -> Result<ManyOut<TestResponse>, Error> {
        let (addressed, context) = request.transfer(());
        let establish_timeout = addressed.establish_timeout();
        let (payload, address, instance) = addressed.into_parts();
        let (_, endpoint_path) = address.split_once('/').unwrap();
        let proxied = AddressedRequest::with_instance(
            payload,
            format!("{}/{endpoint_path}", self.proxy),
            instance.unwrap(),
        )
        .with_establish_timeout(establish_timeout);
        let (_, request) = context.transfer(proxied);
        StreamingDispatch::<u64, TestResponse>::generate(self.inner.as_ref(), request).await
    }

    async fn generate_bidirectional(
        &self,
        instance: Instance,
        address: String,
        input: ManyIn<u64>,
    ) -> Result<ManyOut<TestResponse>, Error> {
        StreamingDispatch::<u64, TestResponse>::generate_bidirectional(
            self.inner.as_ref(),
            instance,
            address,
            input,
        )
        .await
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        StreamingDispatch::<u64, TestResponse>::on_instance_removed(self.inner.as_ref(), id).await;
    }

    async fn on_instance_added(&self, id: &EndpointInstanceId) {
        StreamingDispatch::<u64, TestResponse>::on_instance_added(self.inner.as_ref(), id).await;
    }
}

fn test_endpoint(distributed: &DistributedRuntime, endpoint_name: &str) -> Endpoint {
    distributed
        .namespace("response_stream_establish_timeout".to_string())
        .unwrap()
        .component("backend".to_string())
        .unwrap()
        .endpoint(endpoint_name.to_string())
}

async fn poll_until(mut pred: impl FnMut() -> bool) -> bool {
    for _ in 0..250 {
        if pred() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    pred()
}

async fn assert_establish_timeout(
    distributed: &DistributedRuntime,
    endpoint_name: &str,
    handler: Arc<dyn PushWorkHandler>,
    request_received: Arc<Notify>,
    before_shutdown: impl FnOnce(),
) {
    let endpoint = test_endpoint(distributed, endpoint_name);
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

/// The establish timeout stops at the prologue. A worker that sends its
/// prologue but no response is bounded by the inactivity timeout instead.
async fn assert_first_response_uses_inactivity_timeout(distributed: &DistributedRuntime) {
    let endpoint = test_endpoint(distributed, "prologue_without_first_response");
    let release = Arc::new(Notify::new());
    let handler: Arc<dyn PushWorkHandler> = Ingress::for_engine(Arc::new(SilentEngine {
        release: release.clone(),
    }))
    .unwrap();
    let started = endpoint
        .clone()
        .endpoint_builder()
        .handler(handler)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
        .unwrap();
    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    let router =
        PushRouter::<u64, TestResponse>::from_client(client.clone(), RouterMode::RoundRobin)
            .await
            .unwrap();

    let mut stream =
        tokio::time::timeout(Duration::from_secs(10), router.generate(SingleIn::new(7)))
            .await
            .expect("dispatch did not complete after the prologue")
            .expect("a worker that sent its prologue must not hit the establish timeout");
    let item = tokio::time::timeout(Duration::from_secs(15), stream.next())
        .await
        .expect("inactivity timeout did not fire")
        .expect("stream ended without the inactivity timeout error");
    let error = item.err().expect("expected the inactivity timeout error");
    assert!(
        match_error_chain(&error, &[ErrorType::ResponseTimeout], &[]),
        "expected a response timeout, got: {error}"
    );
    assert!(
        error.to_string().contains("inactivity"),
        "the first response must be bounded by the inactivity timeout, got: {error}"
    );

    drop(stream);
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), started.shutdown())
        .await
        .expect("worker endpoint did not shut down")
        .unwrap();
}

/// Requests waiting on local admission or on the worker's ACK have not
/// reached the post-ACK phase. They must not time out or quarantine the worker,
/// however long the wait.
async fn assert_local_admission_wait_is_not_bounded(distributed: &DistributedRuntime) {
    let endpoint = test_endpoint(distributed, "saturated_local_admission");
    let requests = Arc::new(AtomicUsize::new(0));
    let handler: Arc<dyn PushWorkHandler> = Ingress::for_engine(Arc::new(CountingEngine {
        requests: requests.clone(),
    }))
    .unwrap();
    let started = endpoint
        .clone()
        .endpoint_builder()
        .handler(handler)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
        .unwrap();
    let client = endpoint.client().await.unwrap();
    let instance = client.wait_for_instances().await.unwrap()[0].clone();
    let instance_id = instance.id();
    let worker_address = instance.transport.address();
    let worker_address = worker_address
        .strip_prefix("tcp://")
        .unwrap_or(worker_address);
    let (worker_socket, _) = worker_address.split_once('/').unwrap();
    let proxy = start_holding_proxy(worker_socket.parse().unwrap()).await;

    let dispatch = Arc::new(ProxiedDispatch {
        inner: AddressedPushRouter::from_runtime_provider(&endpoint)
            .await
            .unwrap(),
        proxy: proxy.addr,
    });
    let router = Arc::new(
        PushRouter::<u64, TestResponse>::from_client_with_dispatch(
            client.clone(),
            RouterMode::RoundRobin,
            dispatch,
        )
        .await
        .unwrap(),
    );
    assert!(
        poll_until(|| client.instance_ids_avail().contains(&instance_id)).await,
        "precondition: worker should be available"
    );

    // The proxy holds the first request's ACK, so it keeps the client's only
    // admission permit and the second request waits on local admission.
    let first = tokio::spawn({
        let router = router.clone();
        async move { router.generate(SingleIn::new(1)).await }
    });
    assert!(
        poll_until(|| requests.load(Ordering::SeqCst) == 1).await,
        "worker did not receive the first request"
    );
    let forwarded_before = proxy.forwarded.load(Ordering::SeqCst);
    let second = tokio::spawn({
        let router = router.clone();
        async move { router.generate(SingleIn::new(2)).await }
    });

    tokio::time::sleep(ESTABLISH_TIMEOUT * 3).await;
    assert!(!first.is_finished(), "unACKed request must keep waiting");
    assert!(
        !second.is_finished(),
        "request waiting on local admission must keep waiting"
    );
    assert_eq!(
        proxy.forwarded.load(Ordering::SeqCst),
        forwarded_before,
        "request waiting on local admission must not send bytes"
    );
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    assert!(
        client.instance_ids_avail().contains(&instance_id),
        "local admission wait must not quarantine the worker"
    );

    proxy.release.send(true).unwrap();
    for request in [first, second] {
        let mut stream = tokio::time::timeout(Duration::from_secs(10), request)
            .await
            .expect("request did not complete after the ACKs were released")
            .expect("request task panicked")
            .expect("request failed after the ACKs were released");
        let item = stream.next().await.expect("missing response");
        assert!(item.err().is_none(), "unexpected error: {:?}", item.err());
    }
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert!(client.instance_ids_avail().contains(&instance_id));

    tokio::time::timeout(Duration::from_secs(5), started.shutdown())
        .await
        .expect("worker endpoint did not shut down")
        .unwrap();
}

// All cases share one runtime: the TCP request-plane server is process-wide
// and does not outlive the tokio runtime that started it. One request-plane
// connection with one admission permit lets a held ACK saturate local admission.
#[tokio::test]
async fn response_stream_establish_timeout_starts_at_ack() {
    let establish_secs = ESTABLISH_TIMEOUT.as_secs().to_string();
    temp_env::async_with_vars(
        [
            (
                DYN_RESPONSE_STREAM_ESTABLISH_TIMEOUT_SECS,
                Some(establish_secs.as_str()),
            ),
            (DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS, Some("4")),
            ("DYN_TCP_POOL_SIZE", Some("1")),
            ("DYN_TCP_CHANNEL_BUFFER", Some("1")),
            ("DYN_TCP_REQUEST_TIMEOUT", Some("30")),
        ],
        async {
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

            assert_first_response_uses_inactivity_timeout(&distributed).await;
            assert_local_admission_wait_is_not_bounded(&distributed).await;

            runtime.shutdown();
        },
    )
    .await;
}
