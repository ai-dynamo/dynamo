// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serves published DWM snapshots on a local Unix socket. No Agent RPC or watch.
#![allow(clippy::result_large_err)]

use super::proto::topology_service_server::{TopologyService, TopologyServiceServer};
use super::proto::{GetSnapshotRequest, GetSnapshotResponse, Identity};
use super::{check_expected, payload::decode_payload, require_expected, socket};
use anyhow::Context as _;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoveryQuery};
use std::{path::Path, sync::Arc};
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, transport::Server};

#[derive(Clone)]
pub(crate) struct Consumer {
    discovery: Arc<dyn Discovery>,
    namespace: String,
    cancel: CancellationToken,
}

impl Consumer {
    pub(crate) fn new(
        discovery: Arc<dyn Discovery>,
        namespace: String,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            discovery,
            namespace,
            cancel,
        }
    }
}

pub(crate) fn select_snapshot(
    expected: &Identity,
    records: Vec<DiscoveryInstance>,
) -> Result<GetSnapshotResponse, Status> {
    let mut records = records.into_iter();
    let record = records
        .next()
        .ok_or_else(|| Status::unavailable("no published LPX topology for this allocation"))?;
    if records.next().is_some() {
        return Err(Status::failed_precondition(
            "ambiguous LPX topology authorities for this allocation",
        ));
    }
    let DiscoveryInstance::LpxTopology {
        deployment,
        allocation,
        build,
        payload,
        ..
    } = record
    else {
        return Err(Status::failed_precondition(
            "discovery returned a non-topology record",
        ));
    };
    check_expected(
        &Identity {
            deployment,
            allocation,
            build,
        },
        Some(expected),
    )
    .map_err(Status::failed_precondition)?;
    let snapshot = decode_payload(&payload)?;
    check_expected(
        snapshot
            .identity
            .as_ref()
            .expect("decode_payload checked identity"),
        Some(expected),
    )
    .map_err(Status::failed_precondition)?;
    Ok(GetSnapshotResponse {
        snapshot: Some(snapshot),
    })
}

#[tonic::async_trait]
impl TopologyService for Consumer {
    async fn get_snapshot(
        &self,
        request: Request<GetSnapshotRequest>,
    ) -> Result<Response<GetSnapshotResponse>, Status> {
        let expected = require_expected(request.get_ref()).map_err(Status::invalid_argument)?;
        let query = DiscoveryQuery::LpxTopologies {
            namespace: self.namespace.clone(),
            deployment: expected.deployment.clone(),
            allocation: expected.allocation.clone(),
        };
        let records = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => return Err(Status::unavailable("sidecar is shutting down")),
            records = self.discovery.list(query) => records.map_err(|error| Status::unavailable(format!("topology discovery unavailable: {error:#}")))?,
        };
        select_snapshot(expected, records).map(Response::new)
    }
}

pub(crate) async fn serve(
    consumer: Consumer,
    path: &Path,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (listener, file) = socket::bind(path).await?;
    tracing::info!(socket = %path.display(), "Serving the local LPX topology snapshot API");
    let result = Server::builder()
        .add_service(TopologyServiceServer::new(consumer))
        .serve_with_incoming_shutdown(
            UnixListenerStream::new(listener),
            shutdown.cancelled_owned(),
        )
        .await;
    drop(file);
    result.context("the LPX topology API failed")
}
