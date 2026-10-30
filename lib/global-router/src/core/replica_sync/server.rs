// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serves the `ReplicaSync` stream over a [`ReplicaHub`], as routes for the
//! replica's API router.

use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{self, Stream, StreamExt};
use tokio::sync::broadcast::error::RecvError;
use tonic::{Request, Response, Status};

use super::proto::replica_frame::Frame;
use super::proto::replica_sync_server::{ReplicaSync, ReplicaSyncServer};
use super::proto::{ReplicaFrame, Snapshot, SubscribeRequest};
use super::{CONTRACT_MARKER, ReplicaHub};

/// The `ReplicaSync` service of one replica.
pub struct ReplicaSyncService {
    hub: Arc<ReplicaHub>,
}

type FrameStream = Pin<Box<dyn Stream<Item = Result<ReplicaFrame, Status>> + Send>>;

#[tonic::async_trait]
impl ReplicaSync for ReplicaSyncService {
    type SubscribeReplicaEventsStream = FrameStream;

    async fn subscribe_replica_events(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeReplicaEventsStream>, Status> {
        let request = request.into_inner();
        if request.contract_marker != CONTRACT_MARKER {
            return Err(Status::failed_precondition(format!(
                "contract marker {} does not match this publisher's {CONTRACT_MARKER}",
                request.contract_marker
            )));
        }
        tracing::info!(
            subscriber = request.subscriber_id,
            routing_group = request.routing_group,
            "a replica subscribed"
        );
        // Subscribed before the snapshot is taken, so no event falls between.
        let frames = self.hub.subscribe();
        // C2: snapshot export pending a core accessor. The core exports
        // neither its live bindings nor its open bookings, so a joining
        // peer starts from this replica's events only; bookings open at the
        // join are not seen until they free.
        let snapshot = ReplicaFrame {
            frame: Some(Frame::Snapshot(Snapshot::default())),
        };
        let hub = self.hub.clone();
        let live = stream::unfold((hub, frames), |(hub, mut frames)| async move {
            loop {
                let received = tokio::select! {
                    _ = hub.cancelled() => return None,
                    received = frames.recv() => received,
                };
                match received {
                    Ok(frame) => return Some((Ok(frame), (hub, frames))),
                    Err(RecvError::Lagged(missed)) => {
                        hub.subscriber_lagged(missed);
                        tracing::warn!(missed, "a replica subscriber fell behind and lost frames");
                    }
                    Err(RecvError::Closed) => return None,
                }
            }
        });
        Ok(Response::new(Box::pin(
            stream::once(async { Ok(snapshot) }).chain(live),
        )))
    }
}

/// The stream's routes, to merge into the replica's API router. The path is
/// the service's, `/dynamo.globalrouter.replica.v1.ReplicaSync/...`, so it
/// sits beside the OpenAI routes.
///
/// Built as [`tonic::service::Routes`] builds its router, but without its
/// fallback: [`axum::Router::merge`] refuses two routers that both have
/// one, and the API router has its own.
pub fn routes(hub: Arc<ReplicaHub>) -> axum::Router {
    let service = ReplicaSyncServer::new(ReplicaSyncService { hub });
    axum::Router::new().route_service(
        &format!(
            "/{}/{{*rest}}",
            <ReplicaSyncServer<ReplicaSyncService> as tonic::server::NamedService>::NAME
        ),
        service,
    )
}
