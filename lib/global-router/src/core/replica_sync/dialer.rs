// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Subscribes to each peer's `ReplicaSync` stream and applies what it
//! sends, reconnecting with a growing delay, as the Global Frontend follows
//! each pool's Relay.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::services::selection::SelectionService;
use tokio::task::{Id, JoinSet};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};
use url::Url;

use super::proto::SubscribeRequest;
use super::proto::replica_sync_client::ReplicaSyncClient;
use super::{CONTRACT_MARKER, ReplicaHub};
use crate::core::ROUTING_GROUP;

const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);
const FIRST_RETRY: Duration = Duration::from_millis(500);
const LAST_RETRY: Duration = Duration::from_secs(30);
/// A stream that lasted this long resets the retry delay.
const HEALTHY_STREAM: Duration = Duration::from_secs(30);

/// Follows every peer in `hub`, those present now and those added later,
/// until `cancel` fires. Removing a peer ends its subscription; a peer
/// removed and added again is followed afresh once its old subscription
/// has ended.
pub async fn run_peers(
    hub: Arc<ReplicaHub>,
    service: Arc<SelectionService>,
    cancel: CancellationToken,
) {
    let mut tasks = JoinSet::new();
    // The peer each running task follows. A task ends when its peer is
    // removed (or its token fires), so a peer still here is not started
    // again until then.
    let mut following = HashMap::<Id, Url>::new();
    loop {
        for (peer, follow) in hub.peer_tokens() {
            if following.values().any(|followed| *followed == peer) {
                continue;
            }
            let task = tasks.spawn(follow_peer(
                peer.clone(),
                hub.clone(),
                service.clone(),
                follow,
            ));
            following.insert(task.id(), peer);
        }
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = hub.peers_changed() => {}
            Some(ended) = tasks.join_next_with_id() => {
                let id = match ended {
                    Ok((id, ())) => id,
                    Err(error) => error.id(),
                };
                following.remove(&id);
            }
        }
    }
    tasks.abort_all();
}

/// Streams one peer, reconnecting with a growing delay, until `cancel`
/// fires.
async fn follow_peer(
    peer: Url,
    hub: Arc<ReplicaHub>,
    service: Arc<SelectionService>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        // Stop as soon as the peer is removed, even while connecting.
        let Err(error) = tokio::select! {
            _ = cancel.cancelled() => return,
            result = stream_peer(&peer, &hub, &service) => result,
        };
        tracing::warn!(%peer, error = format!("{error:#}"), "replica stream ended");
        if started.elapsed() >= HEALTHY_STREAM {
            retry = FIRST_RETRY;
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(retry) => {}
        }
        retry = (retry * 2).min(LAST_RETRY);
    }
}

/// Applies each frame the peer sends until the stream fails or ends.
async fn stream_peer(
    peer: &Url,
    hub: &Arc<ReplicaHub>,
    service: &SelectionService,
) -> Result<Infallible> {
    let mut client = ReplicaSyncClient::new(connect(peer).await?);
    let request = SubscribeRequest {
        subscriber_id: hub.process_id(),
        routing_group: ROUTING_GROUP.to_owned(),
        contract_marker: CONTRACT_MARKER,
    };
    let mut frames =
        tokio::time::timeout(SUBSCRIBE_TIMEOUT, client.subscribe_replica_events(request))
            .await
            .context("subscribe timed out")?
            .context("subscribe")?
            .into_inner();
    let _connected = Connected::new(hub);
    loop {
        match frames.message().await.context("read a frame")? {
            Some(frame) => hub.apply_from(peer, frame, service),
            None => bail!("the peer closed the stream"),
        }
    }
}

/// Counts the stream in the hub's connected peers while it is open.
struct Connected<'a> {
    hub: &'a ReplicaHub,
}

impl<'a> Connected<'a> {
    fn new(hub: &'a ReplicaHub) -> Self {
        hub.peer_connected();
        Self { hub }
    }
}

impl Drop for Connected<'_> {
    fn drop(&mut self) {
        self.hub.peer_disconnected();
    }
}

async fn connect(peer: &Url) -> Result<Channel> {
    Endpoint::from_shared(peer.to_string())?
        .connect_timeout(Duration::from_secs(3))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_timeout(Duration::from_secs(5))
        .connect()
        .await
        .context("connect")
}
