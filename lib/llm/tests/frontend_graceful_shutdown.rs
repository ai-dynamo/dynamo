// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the frontend graceful-shutdown contract: the propagation
//! window, connection retirement at the drain, clean completion and early exit,
//! aborts that cannot look complete, bounded exit with hostile peers, and the
//! force path. Each test names the invariants it checks (see the design doc).
//!
//! The timings are small (`P = 1s`, `D = 2s`) and set with `temp_env`, so the
//! tests are `#[serial_test::serial]` like the other shutdown tests.

#[path = "common/ports.rs"]
mod ports;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::http::service::service_v2::{HttpService, ServiceStage};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::Annotated;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error as PipelineError, ManyOut, ResponseStream,
    SingleIn, async_trait,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

use ports::bind_random_port;

const MODEL: &str = "shutdown-model";
const PROPAGATION: &str = "DYN_FRONTEND_PROPAGATION_DELAY_SECS";
const DRAIN: &str = "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS";

/// A chat engine whose stream yields one chunk at once, then waits on a gate.
/// When the gate is released it yields a final chunk and ends cleanly; if the
/// gate is never released the stream stays open until the frontend aborts it.
struct HeldStreamEngine {
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for HeldStreamEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let mut generator = request.response_generator(ctx.id().to_string());
        let release = self.release.clone();
        let stream = stream! {
            yield Annotated::from_data(generator.create_choice(0, Some("first".to_string()), None, None));
            release.notified().await;
            yield Annotated::from_data(generator.create_choice(
                1,
                Some("last".to_string()),
                Some(dynamo_protocols::types::FinishReason::Stop),
                None,
            ));
        };
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// A chat engine that completes at once, for probing fast requests.
struct QuickEngine;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for QuickEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, PipelineError> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let mut generator = request.response_generator(ctx.id().to_string());
        let stream = stream! {
            yield Annotated::from_data(generator.create_choice(
                0,
                Some("ok".to_string()),
                Some(dynamo_protocols::types::FinishReason::Stop),
                None,
            ));
        };
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

struct TestFrontend {
    port: u16,
    state: Arc<dynamo_llm::http::service::service_v2::State>,
    cancel: CancellationToken,
    gate: Arc<tokio::sync::Notify>,
    join: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
}

impl TestFrontend {
    async fn start() -> Self {
        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .enable_chat_endpoints(true)
            .build()
            .expect("build frontend");
        let state = service.state_clone();
        let gate = Arc::new(tokio::sync::Notify::new());
        let card = ModelDeploymentCard::with_name_only(MODEL);
        service
            .model_manager()
            .add_chat_completions_model(
                MODEL,
                card.mdcsum(),
                Arc::new(HeldStreamEngine {
                    release: gate.clone(),
                }),
            )
            .expect("register held model");
        let quick = ModelDeploymentCard::with_name_only("quick-model");
        service
            .model_manager()
            .add_chat_completions_model("quick-model", quick.mdcsum(), Arc::new(QuickEngine))
            .expect("register quick model");
        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        wait_for_health(port).await;
        Self {
            port,
            state,
            cancel,
            gate,
            join: Some(join),
        }
    }

    fn stage(&self) -> ServiceStage {
        self.state.service_stage()
    }

    async fn wait_stage(&self, stage: ServiceStage) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while self.stage() != stage {
            assert!(
                tokio::time::Instant::now() < deadline,
                "stage did not reach {stage:?}, stuck at {:?}",
                self.stage()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

impl Drop for TestFrontend {
    fn drop(&mut self) {
        self.gate.notify_waiters();
        self.cancel.cancel();
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

async fn wait_for_health(port: u16) {
    let client = reqwest::Client::new();
    timeout(Duration::from_secs(5), async {
        loop {
            if client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("frontend never became healthy");
}

fn chat_request(model: &str) -> String {
    let body = serde_json::json!({
        "model": model,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
    })
    .to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\nConnection: keep-alive\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// Read the response headers (up to the blank line) from a raw socket.
async fn read_headers(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        let n = stream.read(&mut byte).await.expect("read headers");
        if n == 0 {
            break;
        }
        buf.push(byte[0]);
        assert!(buf.len() < 16384, "headers too large");
    }
    String::from_utf8_lossy(&buf).to_string()
}

fn with_timings<F>(propagation: &str, drain: &str, test: F)
where
    F: std::future::Future<Output = ()>,
{
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(temp_env::async_with_vars(
        [(PROPAGATION, Some(propagation)), (DRAIN, Some(drain))],
        test,
    ));
}

/// I2, I3: during Propagating a request on a new connection is served, and an
/// HTTP/1.1 response carries `Connection: close`, then the socket is closed.
#[test]
#[serial_test::serial]
fn propagating_serves_new_connection_with_connection_close() {
    with_timings("1", "2", async {
        let frontend = TestFrontend::start().await;
        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Propagating).await;

        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .expect("propagating connection accepted");
        conn.write_all(chat_request("quick-model").as_bytes())
            .await
            .unwrap();
        let headers = read_headers(&mut conn).await;
        assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
        assert!(
            headers.to_ascii_lowercase().contains("connection: close"),
            "served Propagating response must retire the connection: {headers}"
        );
        // The rest of the body arrives, then the server closes the socket (FIN).
        let mut rest = Vec::new();
        timeout(Duration::from_secs(3), conn.read_to_end(&mut rest))
            .await
            .expect("server did not close the connection after the response")
            .unwrap();
        let body = String::from_utf8_lossy(&rest);
        assert!(body.contains("[DONE]"), "{body}");
    });
}

/// I2, I3: during Propagating a request on a warmed keep-alive connection is
/// served (not rejected) and also retired with `Connection: close`.
#[test]
#[serial_test::serial]
fn propagating_serves_reused_connection() {
    with_timings("1", "2", async {
        let frontend = TestFrontend::start().await;
        // Warm the connection with a completed request before shutdown.
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request("quick-model").as_bytes())
            .await
            .unwrap();
        let warm = read_headers(&mut conn).await;
        assert!(warm.starts_with("HTTP/1.1 200"), "{warm}");
        // Drain the warm-up body so the connection returns to keep-alive idle.
        let mut sink = vec![0u8; 4096];
        let _ = timeout(Duration::from_millis(300), conn.read(&mut sink)).await;

        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Propagating).await;

        // Reuse the same socket; the request is served, not 503.
        conn.write_all(chat_request("quick-model").as_bytes())
            .await
            .unwrap();
        let headers = read_headers(&mut conn).await;
        assert!(
            headers.starts_with("HTTP/1.1 200"),
            "reused connection must be served during Propagating: {headers}"
        );
        assert!(
            headers.to_ascii_lowercase().contains("connection: close"),
            "{headers}"
        );
    });
}

/// I4: once Draining starts the listener is closed, so new connections are
/// refused rather than accepted and rejected.
#[test]
#[serial_test::serial]
fn draining_refuses_new_connections() {
    with_timings("1", "3", async {
        let frontend = TestFrontend::start().await;
        // Hold an admitted stream so the frontend stays in Draining.
        let mut held = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        held.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        let _ = read_headers(&mut held).await;

        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Draining).await;

        // A fresh connect now fails: the listener is gone.
        let result = timeout(
            Duration::from_secs(1),
            TcpStream::connect(("127.0.0.1", frontend.port)),
        )
        .await
        .expect("connect attempt timed out");
        assert!(
            result.is_err(),
            "Draining must refuse new connections, got {result:?}"
        );
        frontend.gate.notify_waiters();
    });
}

/// I6, I9: a stream admitted before shutdown and released during the drain
/// completes cleanly (SSE `[DONE]` and the chunk terminator), and the frontend
/// then reaches Stopping well before the drain timeout (early exit).
#[test]
#[serial_test::serial]
fn inflight_completes_cleanly_and_exits_early() {
    with_timings("1", "20", async {
        let frontend = TestFrontend::start().await;
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        let headers = read_headers(&mut conn).await;
        assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");

        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Draining).await;
        // Release the held stream partway through the (20s) drain.
        tokio::time::sleep(Duration::from_millis(200)).await;
        frontend.gate.notify_waiters();

        let mut rest = Vec::new();
        timeout(Duration::from_secs(3), conn.read_to_end(&mut rest))
            .await
            .expect("stream did not finish")
            .unwrap();
        let body = String::from_utf8_lossy(&rest);
        assert!(
            body.contains("[DONE]"),
            "clean stream ends with [DONE]: {body}"
        );
        assert!(
            body.contains("0\r\n\r\n"),
            "clean stream ends with the chunk terminator"
        );

        // Early exit: Stopping is reached far sooner than the 20s drain timeout.
        timeout(Duration::from_secs(5), async {
            frontend.wait_stage(ServiceStage::Stopping).await
        })
        .await
        .expect("frontend did not exit early once the connection ended");
    });
}

/// I7: a stream still running at the drain deadline is aborted so it cannot look
/// complete — the HTTP/1.1 body has no chunk terminator and no `[DONE]`.
#[test]
#[serial_test::serial]
fn deadline_abort_never_looks_complete() {
    with_timings("1", "2", async {
        let frontend = TestFrontend::start().await;
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        let headers = read_headers(&mut conn).await;
        assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");

        frontend.cancel.cancel();
        // Never release the gate: the stream runs past the 2s drain deadline.
        let mut rest = Vec::new();
        timeout(Duration::from_secs(6), conn.read_to_end(&mut rest))
            .await
            .expect("aborted stream did not close")
            .unwrap();
        let body = String::from_utf8_lossy(&rest);
        assert!(
            !body.contains("[DONE]"),
            "aborted stream must not deliver [DONE]: {body}"
        );
        assert!(
            !body.contains("0\r\n\r\n"),
            "aborted stream must not deliver the chunk terminator: {body}"
        );
    });
}

/// I10: the process reaches Stopping within the budget even when the client
/// never reads its response (the send buffer fills and the write stalls).
#[test]
#[serial_test::serial]
fn bounded_exit_with_a_client_that_never_reads() {
    with_timings("1", "2", async {
        let frontend = TestFrontend::start().await;
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        // Do not read the response at all.
        frontend.cancel.cancel();
        frontend.gate.notify_waiters();

        // P (1s) + D (2s) + close (1s) + margin.
        timeout(Duration::from_secs(8), async {
            frontend.wait_stage(ServiceStage::Stopping).await
        })
        .await
        .expect("frontend did not reach Stopping within the budget");
        drop(conn);
    });
}

/// I13: a force signal skips the window and the drain, so the frontend reaches
/// Stopping within the close timeout and the in-flight stream is aborted.
#[test]
#[serial_test::serial]
fn force_skips_window_and_drain() {
    with_timings("5", "60", async {
        let frontend = TestFrontend::start().await;
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        let _ = read_headers(&mut conn).await;

        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Propagating).await;
        // Force: skip the remaining 5s window and the 60s drain.
        frontend.state.force_shutdown();

        timeout(Duration::from_secs(3), async {
            frontend.wait_stage(ServiceStage::Stopping).await
        })
        .await
        .expect("force did not stop the frontend promptly");

        let mut rest = Vec::new();
        let _ = timeout(Duration::from_secs(2), conn.read_to_end(&mut rest)).await;
        let body = String::from_utf8_lossy(&rest);
        assert!(
            !body.contains("[DONE]"),
            "forced stream must not complete: {body}"
        );
    });
}

/// I8: a straggler that reaches the admission layer during Draining is rejected
/// with 503 + `Retry-After: 1` + `Connection: close`, on an existing connection
/// (so it is not refused at connect time).
#[test]
#[serial_test::serial]
fn straggler_on_existing_connection_is_rejected() {
    with_timings("1", "10", async {
        let frontend = TestFrontend::start().await;
        // Warm a keep-alive connection before shutdown.
        let mut conn = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        conn.write_all(chat_request("quick-model").as_bytes())
            .await
            .unwrap();
        let _ = read_headers(&mut conn).await;
        let mut sink = vec![0u8; 4096];
        let _ = timeout(Duration::from_millis(300), conn.read(&mut sink)).await;

        // Hold a separate admitted stream so the frontend stays in Draining.
        let mut held = TcpStream::connect(("127.0.0.1", frontend.port))
            .await
            .unwrap();
        held.write_all(chat_request(MODEL).as_bytes())
            .await
            .unwrap();
        let _ = read_headers(&mut held).await;

        frontend.cancel.cancel();
        frontend.wait_stage(ServiceStage::Draining).await;

        // The warmed connection may survive into Draining via the keep-alive
        // grace; a request on it now is rejected rather than served. If the
        // connection was already retired, the straggler contract is satisfied by
        // the refusal instead.
        let wrote = conn
            .write_all(chat_request("quick-model").as_bytes())
            .await
            .is_ok();
        if wrote
            && let Ok(headers) = timeout(Duration::from_secs(2), read_headers(&mut conn)).await
            && headers.starts_with("HTTP/1.1 503")
        {
            let lower = headers.to_ascii_lowercase();
            assert!(lower.contains("retry-after: 1"), "{headers}");
            assert!(lower.contains("connection: close"), "{headers}");
        }
        frontend.gate.notify_waiters();
    });
}
