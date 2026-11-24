// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The sidecar engine driven against the in-process fake.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use dynamo_backend_common::FinishReason;
use dynamo_sidecar_common::startup_deadline;
use tracing::Instrument;
use tracing_subscriber::{Layer, layer::SubscriberExt};

use super::fake::Behavior;
use super::*;
use crate::client::OpenEngineClient;
use crate::pb;

async fn open(
    engine: &LpxSidecarEngine,
    request: PreprocessedRequest,
    context: Arc<dyn dynamo_backend_common::AsyncEngineContext>,
) -> futures::stream::BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
    engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .expect("generate opens a stream")
}

async fn next_within(
    stream: &mut futures::stream::BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
) -> Result<LLMEngineOutput, DynamoError> {
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("an item within the deadline")
        .expect("the stream has an item")
}

/// A client for `server`, as the engine opens one.
async fn client(server: &FakeServer) -> OpenEngineClient {
    let endpoint = GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").unwrap();
    let deadline = startup_deadline(Duration::from_secs(5)).unwrap();
    OpenEngineClient::connect("lpx-test", &endpoint, transport(), deadline)
        .await
        .expect("connect")
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/// A connected engine that stalls or reports UNAVAILABLE once is asked again,
/// instead of failing or holding the bootstrap until the startup deadline.
#[tokio::test]
async fn a_stalled_or_unavailable_model_info_call_is_retried() {
    for (stalls, status) in [(1, None), (0, Some(tonic::Code::Unavailable))] {
        let fake = FakeEngine::new(AGG);
        *fake.0.model_info_stalls.lock().unwrap() = stalls;
        *fake.0.model_info_status.lock().unwrap() = status;
        let server = FakeServer::start(fake).await;
        let deadline = startup_deadline(Duration::from_secs(5)).unwrap();
        let info = client(&server)
            .await
            .wait_for_model_info(
                deadline,
                Duration::from_millis(10),
                Duration::from_millis(200),
            )
            .await
            .expect("the second call answers before the deadline");
        assert_eq!(info.served_model_name, SERVED);
        assert_eq!(server.fake.0.model_info_calls.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn a_stalled_server_info_call_stops_at_the_startup_deadline() {
    let fake = FakeEngine::new(AGG);
    *fake.0.server_info_stalls.lock().unwrap() = 1;
    let server = FakeServer::start(fake).await;
    let client = client(&server).await;
    let deadline = startup_deadline(Duration::from_millis(100)).unwrap();
    let error = client
        .server_info(deadline)
        .await
        .expect_err("GetServerInfo must stop at the startup deadline");
    assert_backend(&error, BackendError::ConnectionTimeout);
    assert!(
        error.to_string().contains("--grpc-startup-deadline-secs"),
        "{error}"
    );
}

/// An answer that cannot change by waiting ends the wait at once.
#[tokio::test]
async fn unimplemented_and_not_found_end_the_wait() {
    for code in [tonic::Code::Unimplemented, tonic::Code::NotFound] {
        let fake = FakeEngine::new(AGG);
        *fake.0.model_info_status.lock().unwrap() = Some(code);
        let server = FakeServer::start(fake).await;
        let Err(error) = discover(argv(&server.endpoint, &[])).await else {
            panic!("the bootstrap must fail");
        };
        assert!(error.to_string().contains("GetModelInfo"), "{error}");
        assert_eq!(server.fake.0.model_info_calls.load(Ordering::SeqCst), 1);
    }
}

/// An agent template written for the TensorRT-LLM sidecar passes
/// `--served-model-name`; the bootstrap checks it, a blank one included,
/// against the name the engine reports.
#[tokio::test]
async fn the_bootstrap_rejects_a_served_model_name_the_engine_does_not_report() {
    let server = FakeServer::start(FakeEngine::new(AGG)).await;
    for name in ["llama32-3b", " "] {
        let error = discover(argv(&server.endpoint, &["--served-model-name", name]))
            .await
            .err()
            .expect("the names differ");
        assert_backend(&error, BackendError::InvalidArgument);
        let message = error.to_string();
        for needle in [
            format!("served_model_name {SERVED:?}"),
            format!("--served-model-name is {:?}", name.trim()),
        ] {
            assert!(message.contains(&needle), "{message} should name {needle}");
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming. Status failures belong to the testkit's wire baseline; the
// conversion tests map each in-band error.
// ---------------------------------------------------------------------------

/// A failure the engine reports in-band after streaming tokens ends the
/// stream there, as a migratable error when the engine marks it retryable.
#[tokio::test]
async fn an_in_band_error_ends_the_stream() {
    let fake = FakeEngine::new(AGG).with_behavior(Behavior::ErrorAfterFirstToken);
    let server = FakeServer::start(fake).await;
    let engine = started(&server, AGG).await;
    let outputs = collect(&engine, request()).await;
    let [Ok(token), Err(error)] = outputs.as_slice() else {
        panic!("one token, then the error, then nothing: {outputs:?}");
    };
    assert!(token.finish_reason.is_none());
    assert_backend(error, BackendError::EngineShutdown);
}

/// A local rejection still dispatches the handoff, then drops the stream
/// without waiting for tokens or for the frontend to consume the error.
#[tokio::test]
async fn a_rejected_decode_leg_still_dispatches_its_handoff() {
    let fake = FakeEngine::new(DECODE).with_behavior(Behavior::HangBeforeFirstToken);
    let server = FakeServer::start(fake).await;
    let engine = started(&server, DECODE).await;
    let context = dynamo_backend_common::testing::mock_context();
    let mut request = decode_request();
    request.output_options.logprobs = Some(1);
    request.routing_mut().lora_name = Some("adapter".to_string());
    let expected_error =
        crate::convert::build_generate_request(&request, context.id(), &discovered(DECODE))
            .expect_err("logprobs are unsupported");
    let token_ids = request.token_ids.clone();
    let session = crate::handoff::decode_session(request.prefill_result.as_ref().unwrap())
        .expect("valid handoff");
    let mut stream = open(&engine, request, context.clone()).await;
    server.fake.wait_for_drop(context.id()).await;
    let error = next_within(&mut stream).await.expect_err("local rejection");
    assert_backend(&error, BackendError::InvalidArgument);
    assert_eq!(error.to_string(), expected_error.to_string());
    assert!(stream.next().await.is_none());

    assert_eq!(
        server.fake.requests(),
        [pb::GenerateRequest {
            request_id: context.id().to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: token_ids.as_ref().clone(),
            })),
            kv: Some(pb::KvOptions {
                session: Some(session),
                ..Default::default()
            }),
            sampling: Some(pb::SamplingParams {
                num_sequences: Some(1),
                ..Default::default()
            }),
            stopping: Some(pb::StoppingOptions {
                max_tokens: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        }]
    );
}

#[tokio::test]
async fn a_rejected_decode_leg_returns_its_error_when_cleanup_dispatch_times_out() {
    let fake = FakeEngine::new(DECODE).with_behavior(Behavior::NeverOpens);
    let server = FakeServer::start(fake).await;
    let engine = started(&server, DECODE).await;
    let context = dynamo_backend_common::testing::mock_context();
    let mut request = decode_request();
    request.output_options.logprobs = Some(1);
    let expected_error =
        crate::convert::build_generate_request(&request, context.id(), &discovered(DECODE))
            .expect_err("logprobs are unsupported");

    tokio::time::pause();
    let dispatch = open(&engine, request, context);
    tokio::pin!(dispatch);
    let waiting_since = std::time::Instant::now();
    while server.fake.requests().is_empty() {
        assert!(
            waiting_since.elapsed() < Duration::from_secs(5),
            "Generate was sent"
        );
        assert!(futures::poll!(&mut dispatch).is_pending());
        // Keep the runtime runnable so networking cannot advance paused time.
        tokio::task::yield_now().await;
    }

    tokio::time::advance(crate::client::RPC_TIMEOUT).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(1), dispatch)
        .await
        .expect("cleanup dispatch stops at its deadline");
    let error = next_within(&mut stream).await.expect_err("local rejection");
    assert_backend(&error, BackendError::InvalidArgument);
    assert_eq!(error.to_string(), expected_error.to_string());
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_cancelled_decode_leg_finishes_when_cleanup_dispatch_times_out() {
    let fake = FakeEngine::new(DECODE).with_behavior(Behavior::NeverOpens);
    let server = FakeServer::start(fake).await;
    let engine = started(&server, DECODE).await;
    let context = dynamo_backend_common::testing::mock_context();
    context.stop_generating();

    tokio::time::pause();
    let dispatch = open(&engine, decode_request(), context);
    tokio::pin!(dispatch);
    let waiting_since = std::time::Instant::now();
    while server.fake.requests().is_empty() {
        assert!(
            waiting_since.elapsed() < Duration::from_secs(5),
            "Generate was sent"
        );
        assert!(futures::poll!(&mut dispatch).is_pending());
        // Keep the runtime runnable so networking cannot advance paused time.
        tokio::task::yield_now().await;
    }

    tokio::time::advance(crate::client::RPC_TIMEOUT).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(1), dispatch)
        .await
        .expect("cleanup dispatch stops at its deadline");
    let terminal = next_within(&mut stream).await.expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    assert!(stream.next().await.is_none());
}

// ---------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------

#[tokio::test]
async fn abort_is_sent_with_the_request_id() {
    let server = FakeServer::start(FakeEngine::new(AGG)).await;
    let engine = started(&server, AGG).await;
    let context = dynamo_backend_common::testing::mock_context();
    let request_id = context.id().to_string();
    engine.abort(context).await;
    assert_eq!(
        server.fake.0.aborts.lock().unwrap().as_slice(),
        [request_id]
    );
}

// ---------------------------------------------------------------------------
// Decode-leg cancellation. Aggregated cancellation and cleanup belong to the
// testkit's wire baseline, which runs them for the LPX sidecar.
// ---------------------------------------------------------------------------

/// The frontend dispatches a decode leg whose prefill finished even after the
/// client left, and only that leg's `Generate` releases the prefill worker's
/// blocks. Known cancellation before dispatch uses a one-token cleanup;
/// cancellation after dispatch cannot reduce the budget already sent. Once the
/// engine has the cleanup, the sidecar drops the stream without waiting for a token.
#[tokio::test]
async fn a_decode_leg_cancelled_before_dispatch_is_still_sent() {
    let fake = FakeEngine::new(DECODE).with_behavior(Behavior::HangBeforeFirstToken);
    let server = FakeServer::start(fake).await;
    let engine = started(&server, DECODE).await;
    let context = dynamo_backend_common::testing::mock_context();
    context.stop_generating();
    let mut stream = open(&engine, decode_request(), context.clone()).await;
    let terminal = next_within(&mut stream).await.expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    let sent = server.fake.requests();
    assert_eq!(sent.len(), 1, "Generate was sent");
    let stopping = sent[0].stopping.as_ref().expect("stopping");
    assert_eq!(stopping.max_tokens, Some(1));
    assert_eq!(stopping.min_tokens, None);
    server.fake.wait_for_drop(context.id()).await;
}

/// A cancellation that arrives while `Generate` is in flight must not abandon
/// a decode leg the engine may already hold. Shutdown still wins.
#[tokio::test]
async fn a_cancelled_decode_dispatch_is_not_abandoned_mid_flight() {
    let fake = FakeEngine::new(DECODE).with_behavior(Behavior::NeverOpens);
    let server = FakeServer::start(fake).await;
    let engine = Arc::new(started(&server, DECODE).await);
    let context = dynamo_backend_common::testing::mock_context();
    let dispatch = tokio::spawn({
        let engine = Arc::clone(&engine);
        let context = context.clone();
        async move {
            engine
                .generate(decode_request(), GenerateContext::new(context, None))
                .await
        }
    });
    while server.fake.requests().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    context.stop_generating();

    engine.cleanup().await.expect("cleanup");
    let mut stream = tokio::time::timeout(Duration::from_secs(5), dispatch)
        .await
        .expect("shutdown releases the dispatch")
        .expect("dispatch task")
        .expect("generate");
    let terminal = next_within(&mut stream).await.expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

// ---------------------------------------------------------------------------
// Trace metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trace_context_rides_in_generate_metadata() {
    let server = FakeServer::start(FakeEngine::new(AGG)).await;
    let engine = started(&server, AGG).await;
    collect(&engine, request()).await;
    let recorded = server.fake.0.metadata.lock().unwrap().pop().unwrap();
    assert!(recorded.is_empty());
    // Only the request span has the trace fields this layer requires.
    let subscriber = tracing_subscriber::registry().with(
        dynamo_runtime::logging::DistributedTraceIdLayer.with_filter(
            tracing_subscriber::filter::filter_fn(|metadata| metadata.name() == "handle_payload"),
        ),
    );
    let _default = tracing::subscriber::set_default(subscriber);
    let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    for (tracestate, expected) in [
        ("vendor=value", Some("vendor=value")),
        ("caf\u{e9}=1", None),
    ] {
        let span = tracing::info_span!(
            "handle_payload",
            trace_id = "0af7651916cd43dd8448eb211c80319c",
            span_id = "b7ad6b7169203331",
            tracestate,
        );
        collect(&engine, request()).instrument(span).await;
        let recorded = server.fake.0.metadata.lock().unwrap().pop().unwrap();
        assert_eq!(
            recorded.get("traceparent").map(String::as_str),
            Some(traceparent)
        );
        assert_eq!(recorded.get("tracestate").map(String::as_str), expected);
    }
}
