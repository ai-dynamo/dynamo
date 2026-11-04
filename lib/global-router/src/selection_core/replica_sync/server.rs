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
        // Subscribed before the snapshot is taken, so no binding falls
        // between.
        let frames = self.hub.subscribe();
        // The snapshot is empty until the core exports its live bindings: a
        // joining peer learns a session's pin when it next changes or is
        // refreshed.
        let snapshot = ReplicaFrame {
            frame: Some(Frame::Snapshot(Snapshot::default())),
        };
        // This replica's requests in flight follow at once, so the joining
        // peer counts them before the next interval's frame. Taken after
        // subscribing, so no later count is missed either.
        let opening = std::iter::once(snapshot)
            .chain(self.hub.inflight_frame())
            .map(Ok)
            .collect::<Vec<_>>();
        let group = request.routing_group;
        let hub = self.hub.clone();
        let live = stream::unfold(
            (hub, frames, group),
            |(hub, mut frames, group)| async move {
                loop {
                    let received = tokio::select! {
                        _ = hub.cancelled() => return None,
                        received = frames.recv() => received,
                    };
                    match received {
                        Ok(frame) => {
                            // A frame with nothing of the subscriber's
                            // group is not worth a write.
                            if let Some(frame) = in_group(frame, &group) {
                                return Some((Ok(frame), (hub, frames, group)));
                            }
                        }
                        Err(RecvError::Lagged(missed)) => {
                            hub.subscriber_lagged(missed);
                            tracing::warn!(
                                missed,
                                "a replica subscriber fell behind and lost frames"
                            );
                        }
                        Err(RecvError::Closed) => return None,
                    }
                }
            },
        );
        Ok(Response::new(Box::pin(stream::iter(opening).chain(live))))
    }
}

/// `frame` with only the bindings of routing group `group`, or all of them
/// when `group` is empty, the wildcard `SubscribeRequest` documents. `None`
/// for a binding frame left with nothing; a snapshot is always kept, since
/// the subscriber waits for it first. An in-flight count is per pool, not
/// per partition, and always passes.
fn in_group(frame: ReplicaFrame, group: &str) -> Option<ReplicaFrame> {
    if group.is_empty() {
        return Some(frame);
    }
    match frame.frame {
        Some(Frame::Snapshot(mut snapshot)) => {
            snapshot
                .bindings
                .retain(|binding| binding.routing_group == group);
            Some(ReplicaFrame {
                frame: Some(Frame::Snapshot(snapshot)),
            })
        }
        Some(Frame::Bindings(mut batch)) => {
            batch
                .bindings
                .retain(|binding| binding.routing_group == group);
            (!batch.bindings.is_empty()).then_some(ReplicaFrame {
                frame: Some(Frame::Bindings(batch)),
            })
        }
        Some(Frame::InflightCount(inflight)) => Some(ReplicaFrame {
            frame: Some(Frame::InflightCount(inflight)),
        }),
        None => None,
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

#[cfg(test)]
mod tests {
    use super::super::proto::{AffinityBinding, BindingBatch};
    use super::*;

    fn binding(group: &str) -> AffinityBinding {
        AffinityBinding {
            routing_group: group.to_owned(),
            ..AffinityBinding::default()
        }
    }

    fn bindings(groups: &[&str]) -> ReplicaFrame {
        ReplicaFrame {
            frame: Some(Frame::Bindings(BindingBatch {
                bindings: groups.iter().map(|group| binding(group)).collect(),
            })),
        }
    }

    #[test]
    fn a_subscriber_sees_only_its_routing_group() {
        let Some(ReplicaFrame {
            frame: Some(Frame::Bindings(batch)),
        }) = in_group(bindings(&["global", "other", "global"]), "global")
        else {
            panic!("a bindings frame")
        };
        assert_eq!(batch.bindings.len(), 2);
        assert!(
            batch
                .bindings
                .iter()
                .all(|binding| binding.routing_group == "global")
        );
    }

    #[test]
    fn a_binding_frame_with_nothing_of_the_group_is_skipped() {
        assert!(in_group(bindings(&["other"]), "global").is_none());
    }

    #[test]
    fn an_empty_group_is_the_wildcard() {
        let frame = bindings(&["a", "b"]);
        assert_eq!(in_group(frame.clone(), ""), Some(frame));
    }

    #[test]
    fn a_snapshot_is_kept_even_when_filtered_empty() {
        let snapshot = ReplicaFrame {
            frame: Some(Frame::Snapshot(Snapshot {
                bindings: vec![binding("other")],
            })),
        };
        let Some(ReplicaFrame {
            frame: Some(Frame::Snapshot(filtered)),
        }) = in_group(snapshot, "global")
        else {
            panic!("a snapshot frame")
        };
        assert!(filtered.bindings.is_empty());
    }
}
