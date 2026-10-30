// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay-initiated pool presence updates for the Global Router.

use std::sync::Arc;
use std::time::Duration;

use crate::{RelayGatedPoolCatalog, RelayPoolState, RelaySourceKey, RelayStreamGuard};
use dynamo_global_common::grpc::pool_view_frame::Event;
use dynamo_global_common::grpc::pool_view_receiver_server::{
    PoolViewReceiver, PoolViewReceiverServer,
};
use dynamo_global_common::grpc::{Hello, PoolState, PoolViewFrame, PublishResult};
use dynamo_global_common::{PoolId, PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use dynamo_global_router::RelaySource;
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvPoolCatalogUpdate, KvPoolLoadUpdate, ServingLoadUpdate, ServingReadinessUpdate,
};
use prost::Message;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Routing coordinates and replica identity claimed in the first frame.
struct RelayHello {
    pub pool_key: PoolKey,
    pub logical_service: String,
    pub replica_id: String,
    pub incarnation: u64,
}

impl TryFrom<Hello> for RelayHello {
    type Error = Status;

    fn try_from(hello: Hello) -> Result<Self, Self::Error> {
        let key = hello
            .pool_key
            .ok_or_else(|| Status::invalid_argument("missing pool key"))?;
        let pool_key = PoolKey::new(key.cluster_id, key.namespace, key.dgd_name)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if hello.logical_service.is_empty()
            || hello.logical_service.trim() != hello.logical_service
            || hello.replica_id.is_empty()
            || hello.replica_id.trim() != hello.replica_id
            || hello.logical_service.chars().any(char::is_control)
            || hello.replica_id.chars().any(char::is_control)
            || hello.incarnation == 0
        {
            return Err(Status::invalid_argument("invalid Relay identity"));
        }
        Ok(Self {
            pool_key,
            logical_service: hello.logical_service,
            replica_id: hello.replica_id,
            incarnation: hello.incarnation,
        })
    }
}

struct ReceiverService {
    catalog: Arc<RelayGatedPoolCatalog>,
    shutdown: CancellationToken,
}

#[tonic::async_trait]
impl PoolViewReceiver for ReceiverService {
    async fn publish(
        &self,
        request: Request<Streaming<PoolViewFrame>>,
    ) -> Result<Response<PublishResult>, Status> {
        let mut frames = request.into_inner();
        let first = tokio::select! {
            _ = self.shutdown.cancelled() => return Err(Status::unavailable("receiver shutting down")),
            frame = tokio::time::timeout(Duration::from_secs(5), frames.message()) =>
                frame.map_err(|_| Status::deadline_exceeded("Hello timed out"))??,
        };
        let hello = match first {
            Some(PoolViewFrame {
                event: Some(Event::Hello(hello)),
            }) => RelayHello::try_from(hello)?,
            _ => return Err(Status::invalid_argument("first frame must be Hello")),
        };
        let pool = V1PoolIdDeriver.derive(&hello.pool_key);
        let source = RelaySourceKey {
            pool: pool.clone(),
            service: hello.logical_service.clone(),
            replica: hello.replica_id.clone(),
        };
        let mut stream = self.catalog.open(source);
        let mut revision = None;
        loop {
            let frame = tokio::select! {
                _ = self.shutdown.cancelled() => return Err(Status::unavailable("receiver shutting down")),
                frame = frames.message() => frame?,
            };
            let Some(frame) = frame else { break };
            apply_frame(&mut stream, &pool, &hello, &mut revision, frame)?;
        }
        let accepted_revision =
            revision.ok_or_else(|| Status::failed_precondition("snapshot required"))?;
        Ok(Response::new(PublishResult { accepted_revision }))
    }
}

fn decode_state(state: PoolState, hello: &RelayHello) -> anyhow::Result<RelayPoolState> {
    let catalog = Arc::new(KvPoolCatalogUpdate::decode(state.catalog)?);
    let readiness = ServingReadinessUpdate::decode(state.readiness)?;
    let identity = catalog
        .relay
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing Relay identity"))?;
    anyhow::ensure!(
        hello.replica_id == identity.drt_instance_id.to_string(),
        "state does not belong to the stream's Relay"
    );
    RelayPoolState::from_updates(
        Arc::clone(&catalog),
        &readiness,
        (!state.serving_load.is_empty())
            .then(|| ServingLoadUpdate::decode(state.serving_load).map(Arc::new))
            .transpose()?,
        (!state.kv_load.is_empty())
            .then(|| KvPoolLoadUpdate::decode(state.kv_load).map(Arc::new))
            .transpose()?,
        RelaySource {
            runtime_id: identity.drt_instance_id,
            incarnation: hello.incarnation,
        },
    )
}

// Keep tonic's status at this RPC boundary, as in the Relay's existing gRPC service.
#[allow(clippy::result_large_err)]
fn apply_frame(
    stream: &mut RelayStreamGuard,
    pool: &PoolId,
    hello: &RelayHello,
    revision: &mut Option<u64>,
    frame: PoolViewFrame,
) -> Result<(), Status> {
    match frame.event {
        Some(Event::Snapshot(snapshot)) => {
            if snapshot.pool_present && snapshot.state.is_none() {
                return Err(Status::invalid_argument(
                    "present pool requires catalog and readiness state",
                ));
            }
            let state = snapshot
                .state
                .map(|state| decode_state(state, hello).map(Arc::new))
                .transpose()
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            if !snapshot.pool_present && state.is_some() {
                return Err(Status::invalid_argument("withdrawn pool has state"));
            }
            let pools = state.into_iter().map(|state| (pool.clone(), state));
            stream
                .replace_state_snapshot(snapshot.revision, pools)
                .map_err(|error| Status::failed_precondition(error.to_string()))?;
            *revision = Some(snapshot.revision);
        }
        Some(Event::Upsert(update)) => {
            let state = update.state.ok_or_else(|| {
                Status::invalid_argument("upsert requires catalog and readiness state")
            })?;
            let state = decode_state(state, hello)
                .map(Arc::new)
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            stream
                .upsert_state(update.revision, pool.clone(), state)
                .map_err(|error| Status::failed_precondition(error.to_string()))?;
            *revision = Some(update.revision);
        }
        Some(Event::Remove(update)) => {
            stream
                .remove(update.revision, pool)
                .map_err(|error| Status::failed_precondition(error.to_string()))?;
            *revision = Some(update.revision);
        }
        _ => return Err(Status::invalid_argument("unexpected Relay frame")),
    }
    Ok(())
}

/// Serve trusted-network Relay streams alongside the HTTP frontend.
pub async fn serve_relay_views(
    listener: TcpListener,
    catalog: Arc<RelayGatedPoolCatalog>,
    shutdown: CancellationToken,
) -> Result<(), tonic::transport::Error> {
    let service = PoolViewReceiverServer::new(ReceiverService {
        catalog,
        shutdown: shutdown.clone(),
    })
    .max_decoding_message_size(MAX_FRAME_BYTES)
    .max_encoding_message_size(MAX_FRAME_BYTES);
    Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(10)))
        .http2_keepalive_timeout(Some(Duration::from_secs(5)))
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown.cancelled())
        .await
}
