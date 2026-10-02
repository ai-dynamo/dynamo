// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone ZMQ client for the KVBM carrier feed.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
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

#[derive(Debug, Default)]
pub struct ZmqCarrierFeedConnector;

impl CarrierFeedConnector for ZmqCarrierFeedConnector {
    fn connect(
        &self,
        hub_url: &str,
        replica: std::sync::Arc<CarrierFeedReplica>,
        cancel: CancellationToken,
    ) {
        let hub_url = hub_url.to_string();
        tokio::spawn(async move {
            run_feed_client(hub_url, replica, cancel).await;
        });
    }
}

#[derive(Debug, Deserialize)]
struct FeedConfig {
    #[serde(default)]
    feed_endpoint: String,
}

async fn run_feed_client(
    hub_url: String,
    replica: std::sync::Arc<CarrierFeedReplica>,
    cancel: CancellationToken,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|error| {
            tracing::warn!(%error, "failed to configure carrier feed HTTP client");
            reqwest::Client::new()
        });
    let mut backoff = INITIAL_BACKOFF;
    let mut last_warning = None;
    while !cancel.is_cancelled() {
        let result = tokio::select! {
            _ = cancel.cancelled() => break,
            result = run_session(&client, &hub_url, &replica, &cancel, backoff) => result,
        };
        match result {
            Ok(()) => break,
            Err(_error) if cancel.is_cancelled() => break,
            Err(error) => {
                warn_rate_limited(
                    &mut last_warning,
                    &format!("carrier feed connection failed for {hub_url}: {error}"),
                );
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(backoff) => {}
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}

async fn run_session(
    client: &reqwest::Client,
    hub_url: &str,
    replica: &CarrierFeedReplica,
    cancel: &CancellationToken,
    backoff: Duration,
) -> Result<()> {
    let feed_endpoint = fetch_feed_endpoint(client, hub_url).await?;
    let mut socket = create_sub_socket_topics(&[CARRIER_FEED_TOPIC])
        .context("create carrier feed subscriber")?;
    socket
        .connect(&feed_endpoint)
        .with_context(|| format!("connect carrier feed subscriber to {feed_endpoint}"))?;

    install_snapshot(client, hub_url, replica).await?;
    let mut last_snapshot = Instant::now();
    let mut last_warning = None;

    loop {
        let frames = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = socket.recv_multipart() => result.context("receive carrier feed frame")?,
        };
        let payload = match frames.as_slice() {
            [topic, payload] if topic == CARRIER_FEED_TOPIC => payload,
            _ => {
                warn_rate_limited(
                    &mut last_warning,
                    "carrier feed received an unexpected multipart message",
                );
                continue;
            }
        };
        let frame = match decode_frame(payload) {
            Ok(frame) => frame,
            Err(error) => {
                warn_rate_limited(
                    &mut last_warning,
                    &format!("failed to decode carrier feed frame: {error}"),
                );
                continue;
            }
        };
        match replica.apply(frame) {
            FeedApply::Applied | FeedApply::Stale => {}
            FeedApply::UnsupportedVersion => {
                warn_rate_limited(&mut last_warning, "unsupported carrier feed frame version");
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

async fn fetch_feed_endpoint(client: &reqwest::Client, hub_url: &str) -> Result<String> {
    let url = hub_url_path(hub_url, "/v1/features/indexer/config");
    let config = client
        .get(url)
        .send()
        .await
        .context("request carrier feed config")?
        .error_for_status()
        .context("carrier feed config returned an error")?
        .json::<FeedConfig>()
        .await
        .context("decode carrier feed config")?;
    if config.feed_endpoint.is_empty() {
        return Err(anyhow!(
            "carrier feed config did not advertise feed_endpoint"
        ));
    }
    Ok(config.feed_endpoint)
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

fn warn_rate_limited(last_warning: &mut Option<Instant>, message: &str) {
    if last_warning.is_none_or(|last| last.elapsed() >= WARNING_INTERVAL) {
        tracing::warn!("{message}");
        *last_warning = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
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

    #[derive(Clone)]
    struct TestHttpState {
        feed_endpoint: String,
        snapshots: Arc<Mutex<VecDeque<Vec<u8>>>>,
    }

    async fn config(State(state): State<TestHttpState>) -> impl IntoResponse {
        Json(serde_json::json!({ "feed_endpoint": state.feed_endpoint }))
    }

    async fn snapshot_handler(State(state): State<TestHttpState>) -> impl IntoResponse {
        let mut snapshots = state.snapshots.lock().unwrap();
        let body = snapshots
            .pop_front()
            .or_else(|| snapshots.back().cloned())
            .unwrap_or_default();
        axum::body::Body::from(body)
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
                feed_endpoint: endpoint,
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
        let hub_url = format!("http://{http_addr}");
        let session_hub_url = hub_url.clone();
        let session = tokio::spawn(async move {
            run_session(
                &session_client,
                &session_hub_url,
                &session_replica,
                &session_cancel,
                INITIAL_BACKOFF,
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
}
