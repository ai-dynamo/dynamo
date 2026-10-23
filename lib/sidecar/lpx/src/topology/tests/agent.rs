// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::pb;
use crate::topology::proto::{
    GetSnapshotRequest, TopologyPayload, topology_service_server::TopologyService,
};
use crate::topology::{agent::AgentClient, consumer::Consumer, publisher::publish};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_runtime::{
    discovery::{Discovery, DiscoveryQuery, KVStoreDiscovery},
    storage::kv,
};
use prost::Message;
use std::{sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

struct MetadataOnly(String);
#[tonic::async_trait]
impl pb::control_server::Control for MetadataOnly {
    async fn get_server_info(
        &self,
        _: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new(pb::ServerInfo {
            extra: Some(prost_types::Struct {
                fields: [(
                    "lpx_topology".into(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue(self.0.clone())),
                    },
                )]
                .into(),
            }),
            ..Default::default()
        }))
    }
    async fn get_model_info(
        &self,
        _: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        panic!("topology must not call get_model_info");
    }
    async fn abort(
        &self,
        _: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        panic!("topology must not call abort");
    }
    async fn get_load(
        &self,
        _: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        panic!("topology must not call get_load");
    }
    async fn health(
        &self,
        _: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        panic!("topology must not call health");
    }
    async fn load_lora(
        &self,
        _: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        panic!("topology must not call load_lora");
    }
    async fn unload_lora(
        &self,
        _: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        panic!("topology must not call unload_lora");
    }
    async fn list_loras(
        &self,
        _: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        panic!("topology must not call list_loras");
    }
    async fn get_kv_event_sources(
        &self,
        _: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        panic!("topology must not call get_kv_event_sources");
    }
    type SubscribeKvEventsStream =
        futures::stream::Empty<Result<pb::SubscribeKvEventsResponse, Status>>;
    async fn subscribe_kv_events(
        &self,
        _: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        panic!("topology must not subscribe to KV events");
    }
}

#[tokio::test]
async fn metadata_only_agent_publishes_once_and_startup_survives_its_loss() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("agent.sock");
    let listener = tokio::net::UnixListener::bind(&path).unwrap();
    let payload = STANDARD.encode(
        TopologyPayload {
            schema_version: 1,
            snapshot: Some(snapshot()),
        }
        .encode_to_vec(),
    );
    let stop_agent = CancellationToken::new();
    let stopped = stop_agent.clone();
    let agent = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(pb::control_server::ControlServer::new(MetadataOnly(
                payload,
            )))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::UnixListenerStream::new(listener),
                stopped.cancelled_owned(),
            )
            .await
            .unwrap();
    });
    let discovery = Arc::new(KVStoreDiscovery::new(
        kv::Manager::memory(),
        CancellationToken::new(),
    ));
    let stop_publisher = CancellationToken::new();
    let publisher = tokio::spawn({
        let discovery = discovery.clone();
        let stop = stop_publisher.clone();
        async move {
            publish(discovery, "test", AgentClient::new(&path).unwrap(), stop)
                .await
                .unwrap();
        }
    });
    let expected = identity();
    let query = DiscoveryQuery::LpxTopologies {
        namespace: "test".into(),
        deployment: expected.deployment.clone(),
        allocation: expected.allocation.clone(),
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while discovery.list(query.clone()).await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Agent metadata should publish before any inference calls");
    stop_agent.cancel();
    agent.await.unwrap();
    stop_publisher.cancel();
    publisher.await.unwrap();
    let consumer = Consumer::new(discovery.clone(), "test".into(), CancellationToken::new());
    assert_eq!(
        consumer
            .get_snapshot(Request::new(GetSnapshotRequest {
                expected: Some(expected)
            }))
            .await
            .unwrap()
            .into_inner()
            .snapshot,
        Some(snapshot())
    );
    assert!(
        discovery
            .list(DiscoveryQuery::AllEndpoints)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        discovery
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap()
            .is_empty()
    );
}
