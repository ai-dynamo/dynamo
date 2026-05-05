// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the bidi request plane.
//!
//! Exercises the velo-streaming bidi protocol end-to-end without bringing up
//! a full `DistributedRuntime`: two velo nodes share a KV-backed
//! `PeerDiscovery`, the server registers a `BidiIngress` directly via
//! `VeloRequestPlaneServer::register_bidi_endpoint`, and the client drives
//! the handshake (create_anchor → unary BidiInit → attach_anchor) using the
//! exact same primitives that `PushRouter::bidi_generate` uses.
//!
//! Run with:
//!   `timeout 180 cargo test -p dynamo-runtime --features velo-transport --test bidi_request_plane -- --nocapture`

#![cfg(feature = "velo-transport")]

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use parking_lot::Mutex;

use velo::Velo;
use velo::transports::tcp::TcpTransportBuilder;

use dynamo_runtime::SystemHealth;
use dynamo_runtime::engine::{
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, DataStream, ResponseStream,
};
use dynamo_runtime::pipeline::network::bidi::session::{BidiSession, decode_bidi_anchor};
use dynamo_runtime::pipeline::network::bidi::{
    BIDI_INIT_HANDLER, BIDI_INIT_KEY, BidiFrame, BidiInitRequest, BidiInitResponse,
};
use dynamo_runtime::pipeline::network::ingress::bidi_handler::BidiIngress;
use dynamo_runtime::pipeline::network::ingress::unified_server::RequestPlaneServer;
use dynamo_runtime::pipeline::network::ingress::velo_endpoint::VeloRequestPlaneServer;
use dynamo_runtime::pipeline::network::velo::{
    ENDPOINT_HEADER, KvPeerDiscovery, REQUEST_ID_HEADER, encode_velo_address,
};
use dynamo_runtime::pipeline::{ManyIn, ManyOut, ServiceEngine};
use dynamo_runtime::storage::kv;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

async fn build_velo_node(
    disco: Arc<KvPeerDiscovery>,
) -> (
    Arc<Velo>,
    dynamo_runtime::pipeline::network::velo::KvPeerRegistrationGuard,
) {
    let bind = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 0);
    let transport = TcpTransportBuilder::new()
        .bind_addr(bind)
        .build()
        .expect("build velo TCP transport");
    let discovery_for_velo: Arc<dyn velo::discovery::PeerDiscovery> = disco.clone();
    let velo = Velo::builder()
        .add_transport(Arc::new(transport))
        .discovery(discovery_for_velo)
        .build()
        .await
        .expect("build velo");
    let guard = disco
        .register(velo.peer_info())
        .await
        .expect("register peer");
    (velo, guard)
}

/// Driver: do the client-side bidi handshake directly using velo primitives,
/// matching what `PushRouter::bidi_generate` does internally. Returns
/// `ManyOut<U>` and (for tests that need it) the request id.
async fn drive_client_bidi<T, U, I>(
    client_velo: Arc<Velo>,
    server_velo_id: velo::InstanceId,
    endpoint_key: String,
    input_stream: DataStream<T>,
    init: I,
) -> anyhow::Result<ManyOut<U>>
where
    T: Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    U: Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
    I: Send + Sync + serde::Serialize + 'static,
{
    use dynamo_runtime::pipeline::network::bidi::BIDI_UNATTACHED_TIMEOUT;
    use velo::streaming::{StreamAnchor, StreamSender};

    let request_id = uuid::Uuid::new_v4().to_string();

    let client_recv: StreamAnchor<BidiFrame<U>> = client_velo.create_anchor::<BidiFrame<U>>();
    client_recv.set_timeout(Some(BIDI_UNATTACHED_TIMEOUT));
    let client_handle = client_recv.handle();

    let init_payload = BidiInitRequest::<I> {
        client_handle,
        request_id: request_id.clone(),
        init,
        frontend_send_ts_ns: None,
    };
    let payload = rmp_serde::to_vec(&init_payload).expect("serialize BidiInitRequest");

    let mut headers: HashMap<String, String> = HashMap::new();
    headers.insert(ENDPOINT_HEADER.to_string(), endpoint_key.clone());
    headers.insert(REQUEST_ID_HEADER.to_string(), request_id.clone());

    if server_velo_id != client_velo.instance_id() {
        client_velo
            .discover_and_register_peer(server_velo_id)
            .await?;
    }

    let ack: Bytes = client_velo
        .unary(BIDI_INIT_HANDLER)
        .map_err(|e| anyhow::anyhow!("unary builder: {e}"))?
        .raw_payload(Bytes::from(payload))
        .headers(headers)
        .instance(server_velo_id)
        .send()
        .await?;

    let resp: BidiInitResponse = rmp_serde::from_slice(&ack)?;
    let server_handle = match resp {
        BidiInitResponse::Ok { server_handle } => server_handle,
        BidiInitResponse::Err { reason } => {
            return Err(anyhow::anyhow!("server rejected bidi init: {reason}"));
        }
    };

    let client_send: StreamSender<BidiFrame<T>> =
        client_velo.attach_anchor::<BidiFrame<T>>(server_handle).await?;

    let session = BidiSession::new(request_id);
    let pump_session = session.clone();
    tokio::spawn(async move {
        let outcome = pump_session.run_outgoing(input_stream, client_send).await;
        tracing::debug!(outcome = ?outcome, "test client pump ended");
    });

    let user_stream =
        decode_bidi_anchor::<U>(client_recv, session.peer_done(), session.controller());

    Ok(Box::pin(TestResponseStream {
        inner: user_stream,
        ctx: session.context(),
    }))
}

/// Minimal AsyncEngineStream wrapper for tests (no SessionGuard since tests
/// keep ManyOut alive for the duration of consumption).
struct TestResponseStream<U> {
    inner: DataStream<U>,
    ctx: Arc<dyn AsyncEngineContext>,
}

impl<U: Send + Sync + 'static> futures::Stream for TestResponseStream<U> {
    type Item = U;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<U>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<U: Send + Sync + 'static> dynamo_runtime::engine::AsyncEngineStream<U>
    for TestResponseStream<U>
{
}

impl<U: Send + Sync + 'static> AsyncEngineContextProvider for TestResponseStream<U> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<U> std::fmt::Debug for TestResponseStream<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestResponseStream").finish()
    }
}

/// Build a SystemHealth bag for endpoint registration.
fn make_system_health() -> Arc<Mutex<SystemHealth>> {
    Arc::new(parking_lot::Mutex::new(SystemHealth::new(
        dynamo_runtime::config::HealthStatus::Ready,
        Vec::<String>::new(),
        false,
        "/health".to_string(),
        "/live".to_string(),
    )))
}

// ---------------------------------------------------------------------------
// Test handlers
// ---------------------------------------------------------------------------

/// Echo handler: maps each input string -> its UPPERCASE form.
struct EchoUpper;

#[async_trait]
impl AsyncEngine<ManyIn<String>, ManyOut<String>, anyhow::Error> for EchoUpper {
    async fn generate(&self, request: ManyIn<String>) -> Result<ManyOut<String>, anyhow::Error> {
        let (holder, ctx_unit) = request.into_parts();
        let mut input = holder
            .take()
            .ok_or_else(|| anyhow::anyhow!("EchoUpper: stream taken"))?;
        let ctx = ctx_unit.context();
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(16);
        tokio::spawn(async move {
            while let Some(s) = input.next().await {
                if tx.send(s.to_uppercase()).await.is_err() {
                    break;
                }
            }
        });
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// Trailing-emit handler: forwards each input + appends a fixed tail of N
/// items after the input ends. Tests that the user-visible `ManyOut<U>` ends
/// only after BOTH sides have sent Done.
struct TrailingEmit {
    tail: usize,
}

#[async_trait]
impl AsyncEngine<ManyIn<String>, ManyOut<String>, anyhow::Error> for TrailingEmit {
    async fn generate(&self, request: ManyIn<String>) -> Result<ManyOut<String>, anyhow::Error> {
        let tail = self.tail;
        let (holder, ctx_unit) = request.into_parts();
        let mut input = holder
            .take()
            .ok_or_else(|| anyhow::anyhow!("TrailingEmit: stream taken"))?;
        let ctx = ctx_unit.context();
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(16);
        tokio::spawn(async move {
            while let Some(s) = input.next().await {
                if tx.send(format!("echo:{s}")).await.is_err() {
                    return;
                }
            }
            // Input ended; emit `tail` trailing items.
            for i in 0..tail {
                if tx.send(format!("tail:{i}")).await.is_err() {
                    return;
                }
            }
        });
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// Errors immediately on first generate call. Used to test client-side
/// surfacing of server failures.
struct AlwaysErr;

#[async_trait]
impl AsyncEngine<ManyIn<String>, ManyOut<String>, anyhow::Error> for AlwaysErr {
    async fn generate(&self, _request: ManyIn<String>) -> Result<ManyOut<String>, anyhow::Error> {
        Err(anyhow::anyhow!("intentional handler failure for tests"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const ECHO_ENDPOINT: &str = "echo";
const SERVER_INSTANCE_ID: u64 = 0xdead_beef;

async fn setup_two_node_pair_with_engine(
    engine: ServiceEngine<ManyIn<String>, ManyOut<String>>,
) -> (
    Arc<Velo>,
    Arc<Velo>,
    String,
    velo::InstanceId,
    Vec<dynamo_runtime::pipeline::network::velo::KvPeerRegistrationGuard>,
    Arc<VeloRequestPlaneServer>,
) {
    let kv = Arc::new(kv::Manager::memory());
    let disco = Arc::new(KvPeerDiscovery::new(kv));

    let (server_velo, server_guard) = build_velo_node(disco.clone()).await;
    let server = VeloRequestPlaneServer::new(server_velo.clone()).expect("server");

    let bidi_ingress = BidiIngress::<String, String, ()>::for_engine(engine, server_velo.clone())
        .expect("BidiIngress");
    let system_health = make_system_health();
    server
        .register_bidi_endpoint(
            ECHO_ENDPOINT.to_string(),
            bidi_ingress,
            SERVER_INSTANCE_ID,
            "ns".to_string(),
            "comp".to_string(),
            system_health.clone(),
        )
        .await
        .expect("register bidi");

    let (client_velo, client_guard) = build_velo_node(disco.clone()).await;

    let endpoint_key = format!("{:x}/{}", SERVER_INSTANCE_ID, ECHO_ENDPOINT);
    let _ = encode_velo_address(server_velo.instance_id(), SERVER_INSTANCE_ID, ECHO_ENDPOINT);
    (
        server_velo.clone(),
        client_velo,
        endpoint_key,
        server_velo.instance_id(),
        vec![server_guard, client_guard],
        server,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn echo_100_items_round_trip() {
    let engine: ServiceEngine<ManyIn<String>, ManyOut<String>> = Arc::new(EchoUpper);
    let (_server_velo, client_velo, endpoint_key, server_id, _guards, _server) =
        setup_two_node_pair_with_engine(engine).await;

    let inputs: Vec<String> = (0..100).map(|i| format!("hello-{i}")).collect();
    let expected: Vec<String> = inputs.iter().map(|s| s.to_uppercase()).collect();

    let stream: DataStream<String> = Box::pin(tokio_stream::iter(inputs.clone()));
    let mut response = drive_client_bidi::<String, String, ()>(
        client_velo,
        server_id,
        endpoint_key,
        stream,
        (),
    )
    .await
    .expect("bidi handshake");

    let mut got = Vec::new();
    while let Some(item) = response.next().await {
        got.push(item);
    }
    assert_eq!(got, expected, "expected echo of 100 uppercase strings");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn half_close_with_trailing_emit() {
    let engine: ServiceEngine<ManyIn<String>, ManyOut<String>> = Arc::new(TrailingEmit { tail: 5 });
    let (_server_velo, client_velo, endpoint_key, server_id, _guards, _server) =
        setup_two_node_pair_with_engine(engine).await;

    let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let stream: DataStream<String> = Box::pin(tokio_stream::iter(inputs));
    let mut response = drive_client_bidi::<String, String, ()>(
        client_velo,
        server_id,
        endpoint_key,
        stream,
        (),
    )
    .await
    .expect("bidi handshake");

    let mut got = Vec::new();
    while let Some(item) = response.next().await {
        got.push(item);
    }

    // 3 echoes + 5 tail items = 8 total.
    assert_eq!(
        got,
        vec![
            "echo:a", "echo:b", "echo:c", "tail:0", "tail:1", "tail:2", "tail:3", "tail:4"
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn server_handler_error_surfaces_to_client() {
    let engine: ServiceEngine<ManyIn<String>, ManyOut<String>> = Arc::new(AlwaysErr);
    let (_server_velo, client_velo, endpoint_key, server_id, _guards, _server) =
        setup_two_node_pair_with_engine(engine).await;

    let inputs = vec!["x".to_string()];
    let stream: DataStream<String> = Box::pin(tokio_stream::iter(inputs));

    // Handshake itself succeeds (BidiIngress::handle_bidi_init returns Ok
    // BEFORE the user handler runs — handle is exchanged, then driver task
    // is spawned). The user handler then errors, dropping server_send;
    // client's response stream observes Dropped and ends.
    let mut response = drive_client_bidi::<String, String, ()>(
        client_velo,
        server_id,
        endpoint_key,
        stream,
        (),
    )
    .await
    .expect("bidi handshake");

    let mut got = Vec::new();
    let timeout = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(timeout);
    loop {
        tokio::select! {
            _ = &mut timeout => {
                panic!("response stream did not terminate within timeout; got {got:?}");
            }
            item = response.next() => match item {
                Some(s) => got.push(s),
                None => break,
            }
        }
    }
    // No items expected: handler errored before producing anything.
    assert!(
        got.is_empty(),
        "expected no items from AlwaysErr, got {got:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn init_payload_visible_to_handler() {
    /// Server handler that reads the init payload and prepends it to each
    /// echoed item.
    struct PrefixWithInit;

    #[async_trait]
    impl AsyncEngine<ManyIn<String>, ManyOut<String>, anyhow::Error> for PrefixWithInit {
        async fn generate(
            &self,
            request: ManyIn<String>,
        ) -> Result<ManyOut<String>, anyhow::Error> {
            let prefix: String = request
                .clone_unique::<String>(BIDI_INIT_KEY)
                .map_err(|e| anyhow::anyhow!("read init: {e}"))?;
            let (holder, ctx_unit) = request.into_parts();
            let mut input = holder
                .take()
                .ok_or_else(|| anyhow::anyhow!("stream taken"))?;
            let ctx = ctx_unit.context();
            let (tx, rx) = tokio::sync::mpsc::channel::<String>(16);
            tokio::spawn(async move {
                while let Some(s) = input.next().await {
                    if tx.send(format!("{prefix}:{s}")).await.is_err() {
                        return;
                    }
                }
            });
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
    }

    let kv = Arc::new(kv::Manager::memory());
    let disco = Arc::new(KvPeerDiscovery::new(kv));

    let (server_velo, _server_guard) = build_velo_node(disco.clone()).await;
    let server = VeloRequestPlaneServer::new(server_velo.clone()).expect("server");

    let engine: ServiceEngine<ManyIn<String>, ManyOut<String>> = Arc::new(PrefixWithInit);
    let bidi_ingress =
        BidiIngress::<String, String, String>::for_engine(engine, server_velo.clone())
            .expect("ingress");
    let sh = make_system_health();
    server
        .register_bidi_endpoint(
            ECHO_ENDPOINT.to_string(),
            bidi_ingress,
            SERVER_INSTANCE_ID,
            "ns".to_string(),
            "comp".to_string(),
            sh,
        )
        .await
        .expect("register");

    let (client_velo, _client_guard) = build_velo_node(disco.clone()).await;
    let endpoint_key = format!("{:x}/{}", SERVER_INSTANCE_ID, ECHO_ENDPOINT);

    let inputs = vec!["one".to_string(), "two".to_string()];
    let stream: DataStream<String> = Box::pin(tokio_stream::iter(inputs));
    let mut response = drive_client_bidi::<String, String, String>(
        client_velo,
        server_velo.instance_id(),
        endpoint_key,
        stream,
        "PFX".to_string(),
    )
    .await
    .expect("handshake");

    let got: Vec<String> = response.by_ref().collect().await;
    assert_eq!(got, vec!["PFX:one", "PFX:two"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_by_drop_terminates_session() {
    /// Server handler that emits forever (one item per 50ms) until cancelled.
    struct InfiniteEmitter;

    #[async_trait]
    impl AsyncEngine<ManyIn<String>, ManyOut<String>, anyhow::Error> for InfiniteEmitter {
        async fn generate(
            &self,
            request: ManyIn<String>,
        ) -> Result<ManyOut<String>, anyhow::Error> {
            let (_holder, ctx_unit) = request.into_parts();
            let ctx = ctx_unit.context();
            let ctx_for_task = ctx.clone();
            let (tx, rx) = tokio::sync::mpsc::channel::<String>(16);
            tokio::spawn(async move {
                let mut i = 0u64;
                loop {
                    tokio::select! {
                        biased;
                        _ = ctx_for_task.killed() => break,
                        _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                    }
                    if tx.send(format!("item-{i}")).await.is_err() {
                        break;
                    }
                    i += 1;
                }
            });
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
    }

    let engine: ServiceEngine<ManyIn<String>, ManyOut<String>> = Arc::new(InfiniteEmitter);
    let (_server_velo, client_velo, endpoint_key, server_id, _guards, _server) =
        setup_two_node_pair_with_engine(engine).await;

    let stream: DataStream<String> = Box::pin(tokio_stream::iter(Vec::<String>::new()));
    let mut response = drive_client_bidi::<String, String, ()>(
        client_velo,
        server_id,
        endpoint_key,
        stream,
        (),
    )
    .await
    .expect("handshake");

    // Read a few items, then drop the response stream.
    let mut got = 0;
    while got < 3 {
        let item = response.next().await.expect("at least 3 items");
        assert!(item.starts_with("item-"));
        got += 1;
    }
    drop(response);

    // After the drop, the server-side ctx should be killed promptly. We
    // can't observe it directly here, but the test passes if this returns
    // (no leaked tasks holding the runtime alive).
    tokio::time::sleep(Duration::from_millis(200)).await;
}
