// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP-configured carrier-feed clients.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::{StreamExt, stream::BoxStream};
use serde::Deserialize;
use tokio_util::bytes::Bytes;
use tokio_util::sync::CancellationToken;

use crate::carrier_feed::{
    CARRIER_FEED_TOPIC, CarrierFeedError, CarrierFeedReplica, FeedApply, decode_frame,
    decode_snapshot,
};
use crate::carrier_routing::CarrierFeedConnector;
use crate::services::common::zmq::create_sub_socket_topics;

const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const WARNING_INTERVAL: Duration = Duration::from_secs(1);
const RECEIVE_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct HubFeedConfig {
    #[serde(default = "default_event_plane")]
    pub event_plane: String,
    #[serde(default)]
    pub feed_endpoint: String,
    #[serde(default)]
    pub nats_subject_prefix: String,
}

fn default_event_plane() -> String {
    "zmq".to_string()
}

#[async_trait]
pub trait CarrierFeedTransport: Send + Sync + 'static {
    fn event_plane(&self) -> &'static str;

    /// The subscription is live when this returns.
    async fn subscribe(&self, config: &HubFeedConfig) -> Result<BoxStream<'static, Result<Bytes>>>;
}

#[derive(Debug, Default)]
pub struct ZmqFeedTransport;

#[async_trait]
impl CarrierFeedTransport for ZmqFeedTransport {
    fn event_plane(&self) -> &'static str {
        "zmq"
    }

    async fn subscribe(&self, config: &HubFeedConfig) -> Result<BoxStream<'static, Result<Bytes>>> {
        if config.feed_endpoint.is_empty() {
            anyhow::bail!("hub advertised an empty carrier feed endpoint");
        }
        let mut socket = create_sub_socket_topics(&[CARRIER_FEED_TOPIC])
            .context("create carrier feed subscriber")?;
        socket.connect(&config.feed_endpoint).with_context(|| {
            format!(
                "connect carrier feed subscriber to {}",
                config.feed_endpoint
            )
        })?;
        Ok(
            futures_util::stream::unfold(socket, |mut socket| async move {
                loop {
                    match socket.recv_multipart().await {
                        Ok(frames) => match frames.as_slice() {
                            [topic, payload] if topic == CARRIER_FEED_TOPIC => {
                                return Some((Ok(Bytes::copy_from_slice(payload)), socket));
                            }
                            _ => tracing::warn!(
                                "carrier feed received an unexpected multipart message"
                            ),
                        },
                        Err(error) => {
                            return Some((
                                Err(anyhow!(error).context("receive carrier feed frame")),
                                socket,
                            ));
                        }
                    }
                }
            })
            .boxed(),
        )
    }
}

pub struct HttpCarrierFeedConnector<T> {
    transport: Arc<T>,
}

impl<T> HttpCarrierFeedConnector<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
        }
    }
}

impl<T: Default> Default for HttpCarrierFeedConnector<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

pub type ZmqCarrierFeedConnector = HttpCarrierFeedConnector<ZmqFeedTransport>;

impl<T: CarrierFeedTransport> CarrierFeedConnector for HttpCarrierFeedConnector<T> {
    fn connect(
        &self,
        hub_url: &str,
        replica: std::sync::Arc<CarrierFeedReplica>,
        cancel: CancellationToken,
    ) {
        let hub_url = hub_url.to_string();
        let transport = Arc::clone(&self.transport);
        tokio::spawn(async move {
            run_feed_client(hub_url, replica, cancel, RECEIVE_IDLE_TIMEOUT, transport).await;
        });
    }
}

async fn run_feed_client<T: CarrierFeedTransport>(
    hub_url: String,
    replica: std::sync::Arc<CarrierFeedReplica>,
    cancel: CancellationToken,
    idle_timeout: Duration,
    transport: Arc<T>,
) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            tracing::error!(%error, hub_url, "failed to configure carrier feed HTTP client");
            return;
        }
    };
    let mut backoff = INITIAL_BACKOFF;
    let mut last_warning = None;
    while !cancel.is_cancelled() {
        let result = tokio::select! {
            _ = cancel.cancelled() => break,
            result = run_session(
                &client,
                &hub_url,
                &replica,
                &cancel,
                backoff,
                idle_timeout,
                transport.as_ref(),
                &mut last_warning,
            ) => result,
        };
        match result {
            Ok(()) => break,
            Err(_error) if cancel.is_cancelled() => break,
            Err(error) => {
                if should_warn(&mut last_warning) {
                    tracing::warn!(%error, hub_url, "carrier feed connection failed");
                }
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(backoff) => {}
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}

async fn run_session<T: CarrierFeedTransport>(
    client: &reqwest::Client,
    hub_url: &str,
    replica: &CarrierFeedReplica,
    cancel: &CancellationToken,
    backoff: Duration,
    idle_timeout: Duration,
    transport: &T,
    last_warning: &mut Option<Instant>,
) -> Result<()> {
    let config = fetch_feed_config(client, hub_url).await?;
    if config.event_plane != transport.event_plane() {
        if should_warn(last_warning) {
            tracing::error!(
                hub_event_plane = %config.event_plane,
                local_event_plane = transport.event_plane(),
                "carrier feed event-plane mismatch; refusing subscription"
            );
        }
        return Err(anyhow!(
            "carrier feed event-plane mismatch: hub={} local={}",
            config.event_plane,
            transport.event_plane()
        ));
    }
    let mut frames = transport.subscribe(&config).await?;
    install_snapshot(client, hub_url, replica).await?;
    let mut last_snapshot = Instant::now();

    loop {
        let receive = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = tokio::time::timeout(idle_timeout, frames.next()) => result,
        };
        let payload = match receive {
            Ok(Some(result)) => result.context("receive carrier feed frame")?,
            Ok(None) => return Err(anyhow!("carrier feed subscription ended")),
            Err(_) => {
                let current_config = fetch_feed_config(client, hub_url).await?;
                if current_config != config {
                    return Err(anyhow!("carrier feed config changed while idle"));
                }
                continue;
            }
        };
        let frame = match decode_frame(&payload) {
            Ok(frame) => frame,
            Err(error) => {
                if should_warn(last_warning) {
                    tracing::warn!(%error, hub_url, "failed to decode carrier feed frame");
                }
                continue;
            }
        };
        match replica.apply(frame) {
            FeedApply::Applied | FeedApply::Stale => {}
            FeedApply::UnsupportedVersion => {
                if should_warn(last_warning) {
                    tracing::warn!(hub_url, "unsupported carrier feed frame version");
                }
            }
            FeedApply::NeedsSnapshot => {
                let wait = backoff.saturating_sub(last_snapshot.elapsed());
                if !wait.is_zero() {
                    tokio::select! {
                        _ = cancel.cancelled() => return Ok(()),
                        _ = tokio::time::sleep(wait) => {}
                    }
                }
                install_snapshot(client, hub_url, replica).await?;
                last_snapshot = Instant::now();
            }
        }
    }
}

async fn fetch_feed_config(client: &reqwest::Client, hub_url: &str) -> Result<HubFeedConfig> {
    let url = hub_url_path(hub_url, "/v1/features/indexer/config");
    client
        .get(url)
        .send()
        .await
        .context("request carrier feed config")?
        .error_for_status()
        .context("carrier feed config returned an error")?
        .json::<HubFeedConfig>()
        .await
        .context("decode carrier feed config")
}

pub fn nats_feed_subject(prefix: &str) -> String {
    let topic = std::str::from_utf8(CARRIER_FEED_TOPIC).expect("carrier feed topic is valid UTF-8");
    format!("{prefix}.{topic}")
}

async fn install_snapshot(
    client: &reqwest::Client,
    hub_url: &str,
    replica: &CarrierFeedReplica,
) -> Result<()> {
    let url = hub_url_path(hub_url, "/v1/features/indexer/feed/snapshot");
    let bytes = client
        .get(url)
        .send()
        .await
        .context("request carrier feed snapshot")?
        .error_for_status()
        .context("carrier feed snapshot returned an error")?
        .bytes()
        .await
        .context("read carrier feed snapshot")?;
    let snapshot = decode_snapshot(&bytes).context("decode carrier feed snapshot")?;
    match replica
        .install_snapshot(snapshot)
        .map_err(|error: CarrierFeedError| anyhow!(error))
        .context("install carrier feed snapshot")?
    {
        FeedApply::Applied | FeedApply::Stale => Ok(()),
        FeedApply::NeedsSnapshot => Err(anyhow!(
            "carrier feed snapshot could not reconcile the pending frame tail"
        )),
        FeedApply::UnsupportedVersion => {
            Err(anyhow!("carrier feed snapshot used an unsupported version"))
        }
    }
}

fn hub_url_path(hub_url: &str, path: &str) -> String {
    format!("{}{}", hub_url.trim_end_matches('/'), path)
}

fn should_warn(last_warning: &mut Option<Instant>) -> bool {
    if last_warning.is_none_or(|last| last.elapsed() >= WARNING_INTERVAL) {
        *last_warning = Some(Instant::now());
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::{Json, Router};
    use dynamo_tokens::PositionalLineageHash;

    use super::*;
    use crate::carrier_feed::{
        CARRIER_FEED_VERSION, CarrierFeedFrame, CarrierFeedOp, CarrierFeedSnapshot, HolderSnapshot,
        ManifestSnapshot, encode_frame, encode_snapshot,
    };
    use crate::services::common::zmq::create_bound_pub_socket;
    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

    #[derive(Clone)]
    struct TestHttpState {
        feed_endpoint: Arc<Mutex<String>>,
        event_plane: Option<String>,
        snapshots: Arc<Mutex<VecDeque<Vec<u8>>>>,
    }

    async fn config(State(state): State<TestHttpState>) -> impl IntoResponse {
        let mut config = serde_json::json!({
            "feed_endpoint": state.feed_endpoint.lock().unwrap().clone(),
        });
        if let Some(event_plane) = state.event_plane {
            config["event_plane"] = serde_json::json!(event_plane);
        }
        Json(config)
    }

    async fn snapshot_handler(State(state): State<TestHttpState>) -> impl IntoResponse {
        let mut snapshots = state.snapshots.lock().unwrap();
        let body = snapshots
            .pop_front()
            .or_else(|| snapshots.back().cloned())
            .unwrap_or_default();
        axum::body::Body::from(body)
    }

    struct ChannelFeedTransport {
        event_plane: &'static str,
        subscribe_count: Arc<AtomicUsize>,
        receiver: Mutex<Option<UnboundedReceiver<Result<Bytes>>>>,
    }

    #[async_trait]
    impl CarrierFeedTransport for ChannelFeedTransport {
        fn event_plane(&self) -> &'static str {
            self.event_plane
        }

        async fn subscribe(
            &self,
            _config: &HubFeedConfig,
        ) -> Result<BoxStream<'static, Result<Bytes>>> {
            self.subscribe_count.fetch_add(1, Ordering::Relaxed);
            let receiver = self
                .receiver
                .lock()
                .unwrap()
                .take()
                .expect("transport can only be subscribed once");
            Ok(
                futures_util::stream::unfold(receiver, |mut receiver| async move {
                    receiver.recv().await.map(|item| (item, receiver))
                })
                .boxed(),
            )
        }
    }

    fn channel_transport(
        event_plane: &'static str,
    ) -> (
        ChannelFeedTransport,
        UnboundedSender<Result<Bytes>>,
        Arc<AtomicUsize>,
    ) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let subscribe_count = Arc::new(AtomicUsize::new(0));
        (
            ChannelFeedTransport {
                event_plane,
                subscribe_count: Arc::clone(&subscribe_count),
                receiver: Mutex::new(Some(receiver)),
            },
            sender,
            subscribe_count,
        )
    }

    async fn start_http_hub(
        event_plane: Option<String>,
        snapshots: Vec<Vec<u8>>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/features/indexer/config", get(config))
            .route("/v1/features/indexer/feed/snapshot", get(snapshot_handler))
            .with_state(TestHttpState {
                feed_endpoint: Arc::new(Mutex::new(String::new())),
                event_plane,
                snapshots: Arc::new(Mutex::new(VecDeque::from(snapshots))),
            });
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), server)
    }

    fn plhs(blocks: u32) -> Vec<PositionalLineageHash> {
        dynamo_kv_hashing::Request::builder()
            .tokens((0..blocks * 4).collect::<Vec<_>>())
            .build()
            .unwrap()
            .positional_lineage_hashes(4)
            .unwrap()
    }

    fn snapshot_bytes(seq: u64, hashes: Vec<PositionalLineageHash>) -> Vec<u8> {
        encode_snapshot(&CarrierFeedSnapshot {
            version: CARRIER_FEED_VERSION,
            epoch: 1,
            seq,
            manifests: vec![ManifestSnapshot {
                manifest: [1; 32],
                kind: crate::carrier_feed::FeedKind::Carrier,
                max_positions: 8,
                holders: vec![HolderSnapshot { holder: 5, hashes }],
            }],
        })
        .unwrap()
    }

    fn unused_tcp_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
        drop(listener);
        endpoint
    }

    async fn publish_until(
        publisher: &mut crate::services::common::zmq::ZmqSocket,
        frame: &CarrierFeedFrame,
        replica: &CarrierFeedReplica,
        cursor: (u64, u64),
    ) {
        let payload = encode_frame(frame).unwrap();
        for _ in 0..100 {
            publisher
                .send_multipart(vec![CARRIER_FEED_TOPIC.to_vec(), payload.clone()])
                .await
                .unwrap();
            if replica.cursor() == Some(cursor) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("carrier replica did not reach cursor {cursor:?}");
    }

    #[tokio::test]
    async fn snapshot_live_frame_and_gap_refetch_converge() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
        drop(listener);
        let mut publisher = create_bound_pub_socket(&endpoint).unwrap();

        let snapshots = Arc::new(Mutex::new(VecDeque::from([
            snapshot_bytes(0, plhs(1)),
            snapshot_bytes(3, plhs(3)),
        ])));
        let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_addr = http_listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/features/indexer/config", get(config))
            .route("/v1/features/indexer/feed/snapshot", get(snapshot_handler))
            .with_state(TestHttpState {
                feed_endpoint: Arc::new(Mutex::new(endpoint)),
                event_plane: Some("zmq".to_string()),
                snapshots,
            });
        let server = tokio::spawn(async move {
            axum::serve(http_listener, app).await.unwrap();
        });

        let replica = Arc::new(CarrierFeedReplica::new(16));
        let cancel = CancellationToken::new();
        let client = reqwest::Client::new();
        let session_client = client.clone();
        let session_replica = Arc::clone(&replica);
        let session_cancel = cancel.clone();
        let transport = ZmqFeedTransport;
        let hub_url = format!("http://{http_addr}");
        let session_hub_url = hub_url.clone();
        let session = tokio::spawn(async move {
            let mut last_warning = None;
            run_session(
                &session_client,
                &session_hub_url,
                &session_replica,
                &session_cancel,
                INITIAL_BACKOFF,
                RECEIVE_IDLE_TIMEOUT,
                &transport,
                &mut last_warning,
            )
            .await
        });
        let hashes = plhs(3);
        publish_until(
            &mut publisher,
            &CarrierFeedFrame {
                version: CARRIER_FEED_VERSION,
                epoch: 1,
                seq: 1,
                manifest: [1; 32],
                kind: crate::carrier_feed::FeedKind::Carrier,
                max_positions: 8,
                holder: 5,
                op: CarrierFeedOp::ReplaceHolder(hashes[..2].to_vec()),
            },
            &replica,
            (1, 1),
        )
        .await;
        publish_until(
            &mut publisher,
            &CarrierFeedFrame {
                version: CARRIER_FEED_VERSION,
                epoch: 1,
                seq: 3,
                manifest: [1; 32],
                kind: crate::carrier_feed::FeedKind::Carrier,
                max_positions: 8,
                holder: 5,
                op: CarrierFeedOp::ReplaceHolder(hashes),
            },
            &replica,
            (1, 3),
        )
        .await;

        assert_eq!(
            replica.deepest(&[1; 32], &plhs(3)).unwrap().hash.position(),
            2
        );
        cancel.cancel();
        session.await.unwrap().unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn matching_transport_installs_snapshot_then_applies_live_frames() {
        let (hub_url, server) =
            start_http_hub(Some("nats".to_string()), vec![snapshot_bytes(0, plhs(1))]).await;
        let (transport, sender, subscribe_count) = channel_transport("nats");
        let replica = Arc::new(CarrierFeedReplica::new(16));
        let cancel = CancellationToken::new();
        let client = reqwest::Client::new();
        let session_client = client.clone();
        let session_hub_url = hub_url.clone();
        let session_replica = Arc::clone(&replica);
        let session_cancel = cancel.clone();
        let session = tokio::spawn(async move {
            let mut last_warning = None;
            run_session(
                &session_client,
                &session_hub_url,
                &session_replica,
                &session_cancel,
                INITIAL_BACKOFF,
                RECEIVE_IDLE_TIMEOUT,
                &transport,
                &mut last_warning,
            )
            .await
        });

        for _ in 0..100 {
            if replica.cursor() == Some((1, 0)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(replica.cursor(), Some((1, 0)));
        assert_eq!(subscribe_count.load(Ordering::Relaxed), 1);

        sender
            .send(Ok(Bytes::from(
                encode_frame(&CarrierFeedFrame {
                    version: CARRIER_FEED_VERSION,
                    epoch: 1,
                    seq: 1,
                    manifest: [1; 32],
                    kind: crate::carrier_feed::FeedKind::Carrier,
                    max_positions: 8,
                    holder: 5,
                    op: CarrierFeedOp::ReplaceHolder(plhs(2)),
                })
                .unwrap(),
            )))
            .unwrap();
        for _ in 0..100 {
            if replica.cursor() == Some((1, 1)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(replica.cursor(), Some((1, 1)));
        cancel.cancel();
        session.await.unwrap().unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn mismatched_transport_skips_subscription_and_keeps_replica_empty() {
        let (hub_url, server) =
            start_http_hub(Some("nats".to_string()), vec![snapshot_bytes(0, plhs(1))]).await;
        let (transport, _sender, subscribe_count) = channel_transport("zmq");
        let replica = CarrierFeedReplica::new(16);
        let client = reqwest::Client::new();
        let mut last_warning = None;

        assert!(
            run_session(
                &client,
                &hub_url,
                &replica,
                &CancellationToken::new(),
                INITIAL_BACKOFF,
                RECEIVE_IDLE_TIMEOUT,
                &transport,
                &mut last_warning,
            )
            .await
            .is_err()
        );
        assert_eq!(subscribe_count.load(Ordering::Relaxed), 0);
        assert_eq!(replica.cursor(), None);
        server.abort();
    }

    #[tokio::test]
    async fn legacy_config_without_event_plane_defaults_to_zmq() {
        let (hub_url, server) = start_http_hub(None, vec![snapshot_bytes(0, plhs(1))]).await;
        let (transport, _sender, subscribe_count) = channel_transport("zmq");
        let replica = Arc::new(CarrierFeedReplica::new(16));
        let cancel = CancellationToken::new();
        let session_cancel = cancel.clone();
        let session_replica = Arc::clone(&replica);
        let session = tokio::spawn(async move {
            let client = reqwest::Client::new();
            let mut last_warning = None;
            run_session(
                &client,
                &hub_url,
                &session_replica,
                &session_cancel,
                INITIAL_BACKOFF,
                RECEIVE_IDLE_TIMEOUT,
                &transport,
                &mut last_warning,
            )
            .await
        });

        for _ in 0..100 {
            if replica.cursor() == Some((1, 0)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(replica.cursor(), Some((1, 0)));
        assert_eq!(subscribe_count.load(Ordering::Relaxed), 1);
        cancel.cancel();
        session.await.unwrap().unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn changed_endpoint_reconnects_to_new_publisher() {
        let old_endpoint = unused_tcp_endpoint();
        let _old_publisher = create_bound_pub_socket(&old_endpoint).unwrap();
        let new_endpoint = unused_tcp_endpoint();
        let mut new_publisher = create_bound_pub_socket(&new_endpoint).unwrap();

        let feed_endpoint = Arc::new(Mutex::new(old_endpoint));
        let snapshot = snapshot_bytes(0, plhs(1));
        let snapshots = Arc::new(Mutex::new(VecDeque::from([snapshot.clone(), snapshot])));
        let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_addr = http_listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/features/indexer/config", get(config))
            .route("/v1/features/indexer/feed/snapshot", get(snapshot_handler))
            .with_state(TestHttpState {
                feed_endpoint: Arc::clone(&feed_endpoint),
                event_plane: Some("zmq".to_string()),
                snapshots,
            });
        let server = tokio::spawn(async move {
            axum::serve(http_listener, app).await.unwrap();
        });

        let replica = Arc::new(CarrierFeedReplica::new(16));
        let cancel = CancellationToken::new();
        let session_cancel = cancel.clone();
        let session_replica = Arc::clone(&replica);
        let hub_url = format!("http://{http_addr}");
        let session = tokio::spawn(async move {
            run_feed_client(
                hub_url,
                session_replica,
                session_cancel,
                Duration::from_millis(50),
                Arc::new(ZmqFeedTransport),
            )
            .await;
        });

        for _ in 0..100 {
            if replica.cursor() == Some((1, 0)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(replica.cursor(), Some((1, 0)));
        *feed_endpoint.lock().unwrap() = new_endpoint;

        let hashes = plhs(2);
        publish_until(
            &mut new_publisher,
            &CarrierFeedFrame {
                version: CARRIER_FEED_VERSION,
                epoch: 1,
                seq: 1,
                manifest: [1; 32],
                kind: crate::carrier_feed::FeedKind::Carrier,
                max_positions: 8,
                holder: 5,
                op: CarrierFeedOp::ReplaceHolder(hashes),
            },
            &replica,
            (1, 1),
        )
        .await;
        assert_eq!(replica.cursor(), Some((1, 1)));

        cancel.cancel();
        session.await.unwrap();
        server.abort();
    }
}
