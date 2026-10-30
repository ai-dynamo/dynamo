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
        Ok(Response::new(Box::pin(
            stream::once(async { Ok(snapshot) }).chain(live),
        )))
    }
}

/// `frame` with only the events and bindings of routing group `group`, or
/// all of them when `group` is empty, the wildcard `SubscribeRequest`
/// documents. `None` for an event frame left with nothing; a snapshot is
/// always kept, since the subscriber waits for it first.
fn in_group(frame: ReplicaFrame, group: &str) -> Option<ReplicaFrame> {
    if group.is_empty() {
        return Some(frame);
    }
    match frame.frame {
        Some(Frame::Snapshot(mut snapshot)) => {
            snapshot
                .bindings
                .retain(|binding| binding.routing_group == group);
            snapshot
                .open_bookings
                .retain(|event| event.routing_group == group);
            Some(ReplicaFrame {
                frame: Some(Frame::Snapshot(snapshot)),
            })
        }
        Some(Frame::Events(mut batch)) => {
            batch.events.retain(|event| event.routing_group == group);
            batch
                .bindings
                .retain(|binding| binding.routing_group == group);
            (!batch.events.is_empty() || !batch.bindings.is_empty()).then_some(ReplicaFrame {
                frame: Some(Frame::Events(batch)),
            })
        }
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
    use super::super::proto::{ActiveSequenceEvent, AffinityBinding, ReplicaEventBatch};
    use super::*;

    fn event(group: &str) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            routing_group: group.to_owned(),
            ..ActiveSequenceEvent::default()
        }
    }

    fn binding(group: &str) -> AffinityBinding {
        AffinityBinding {
            routing_group: group.to_owned(),
            ..AffinityBinding::default()
        }
    }

    fn events(groups: &[&str], binding_groups: &[&str]) -> ReplicaFrame {
        ReplicaFrame {
            frame: Some(Frame::Events(ReplicaEventBatch {
                events: groups.iter().map(|group| event(group)).collect(),
                bindings: binding_groups.iter().map(|group| binding(group)).collect(),
            })),
        }
    }

    #[test]
    fn a_subscriber_sees_only_its_routing_group() {
        let Some(ReplicaFrame {
            frame: Some(Frame::Events(batch)),
        }) = in_group(
            events(&["global", "other", "global"], &["other", "global"]),
            "global",
        )
        else {
            panic!("an events frame")
        };
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.bindings.len(), 1);
        assert!(
            batch
                .events
                .iter()
                .all(|event| event.routing_group == "global")
        );
    }

    #[test]
    fn an_event_frame_with_nothing_of_the_group_is_skipped() {
        assert!(in_group(events(&["other"], &["other"]), "global").is_none());
    }

    #[test]
    fn an_empty_group_is_the_wildcard() {
        let frame = events(&["a", "b"], &["c"]);
        assert_eq!(in_group(frame.clone(), ""), Some(frame));
    }

    #[test]
    fn a_snapshot_is_kept_even_when_filtered_empty() {
        let snapshot = ReplicaFrame {
            frame: Some(Frame::Snapshot(Snapshot {
                bindings: vec![binding("other")],
                open_bookings: vec![event("other")],
            })),
        };
        let Some(ReplicaFrame {
            frame: Some(Frame::Snapshot(filtered)),
        }) = in_group(snapshot, "global")
        else {
            panic!("a snapshot frame")
        };
        assert!(filtered.bindings.is_empty() && filtered.open_bookings.is_empty());
    }
}
