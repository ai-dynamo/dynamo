// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_common::{PoolKey, grpc};
use dynamo_llm::pool_relay::wan::grpc::protocol as relay;
use prost::Message;
mod common;
use dynamo_global_frontend::{
    Metrics, PoolSnapshot, RelayGatedPoolCatalog, TrustedNetworkAdmission, app, serve,
    serve_relay_views,
};
use dynamo_global_router::{InFlightTracker, Pool, RoundRobinSelector};
use grpc::pool_view_frame::Event;
use grpc::pool_view_receiver_client::PoolViewReceiverClient;
use grpc::{Hello, PoolViewFrame, Snapshot};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

type Publisher =
    tokio::task::JoinHandle<Result<tonic::Response<grpc::PublishResult>, tonic::Status>>;

fn replica_id(replica: &str) -> u64 {
    match replica {
        "relay-a" => 1,
        "relay-b" => 2,
        "relay-c" => 3,
        _ => panic!("unknown test replica"),
    }
}

fn replica_state(replica: &str) -> grpc::PoolState {
    let mut state = common::state(true, 1, 1, 1);
    let identity = relay::RelayIdentity {
        drt_instance_id: replica_id(replica),
        relay_incarnation: 1,
    };
    let mut catalog = relay::KvPoolCatalogUpdate::decode(state.catalog).unwrap();
    catalog.relay = Some(identity);
    state.catalog = catalog.encode_to_vec().into();
    let mut ready = relay::ServingReadinessUpdate::decode(state.readiness).unwrap();
    ready.relay = Some(identity);
    state.readiness = ready.encode_to_vec().into();
    let mut serving = relay::ServingLoadUpdate::decode(state.serving_load).unwrap();
    serving.relay = Some(identity);
    state.serving_load = serving.encode_to_vec().into();
    let mut kv = relay::KvPoolLoadUpdate::decode(state.kv_load).unwrap();
    kv.relay = Some(identity);
    state.kv_load = kv.encode_to_vec().into();
    state
}

fn hello(replica: &str) -> PoolViewFrame {
    PoolViewFrame {
        event: Some(Event::Hello(Hello {
            pool_key: Some(grpc::PoolKey {
                cluster_id: "east".into(),
                namespace: "dynamo".into(),
                dgd_name: "blue".into(),
            }),
            logical_service: "blue-relay".into(),
            replica_id: replica_id(replica).to_string(),
            incarnation: 1,
        })),
    }
}

fn snapshot(revision: u64, replica: &str) -> PoolViewFrame {
    PoolViewFrame {
        event: Some(Event::Snapshot(Snapshot {
            revision,
            pool_present: true,
            state: Some(replica_state(replica)),
        })),
    }
}

async fn publish(
    address: std::net::SocketAddr,
    replica: &str,
) -> (mpsc::Sender<PoolViewFrame>, Publisher) {
    let mut client = PoolViewReceiverClient::connect(format!("http://{address}"))
        .await
        .unwrap();
    let (sender, frames) = mpsc::channel(8);
    let task = tokio::spawn(async move { client.publish(ReceiverStream::new(frames)).await });
    sender.send(hello(replica)).await.unwrap();
    sender.send(snapshot(0, replica)).await.unwrap();
    (sender, task)
}

async fn wait_count(pools: &mut watch::Receiver<Arc<PoolSnapshot>>, count: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while pools.borrow().pools().len() != count {
            pools.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn two_routers_forward_only_while_a_relay_is_live_and_reconnect_recovers() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let pool_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let pool_url = format!("http://{}", pool_listener.local_addr().unwrap());
        let backend = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async { "ok" }),
        );
        let pool_server =
            tokio::spawn(async move { axum::serve(pool_listener, backend).await.unwrap() });
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        // This discovery list remains unchanged throughout every disconnect.
        let (discovery, pools) = watch::channel(Arc::new(PoolSnapshot::new(vec![Pool::new(
            &key,
            "qwen",
            "east",
            vec![pool_url.parse().unwrap()],
        )])));
        let mut routers = Vec::new();
        for _ in 0..2 {
            let cancel = CancellationToken::new();
            let catalog = Arc::new(RelayGatedPoolCatalog::new(pools.clone(), cancel.clone()));
            let current = catalog.subscribe();
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let shutdown = cancel.clone();
            let receiver = tokio::spawn(serve_relay_views(
                listener,
                catalog,
                Arc::new(TrustedNetworkAdmission),
                shutdown,
            ));
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!(
                "http://{}/v1/chat/completions",
                listener.local_addr().unwrap()
            );
            let in_flight = Arc::new(InFlightTracker::new());
            let metrics = Metrics::new(None, in_flight.clone());
            let frontend = app(
                current.clone(),
                Arc::new(RoundRobinSelector::new()),
                in_flight,
                reqwest::Client::new(),
                metrics,
            );
            let http = tokio::spawn(serve(listener, frontend, cancel.clone()));
            routers.push((cancel, current, address, url, receiver, http));
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        for (_, current, address, url, _, _) in &mut routers {
            let post = || {
                http.post(url.as_str())
                    .json(&serde_json::json!({"model": "qwen"}))
                    .send()
            };
            assert!(!post().await.unwrap().status().is_success());
            let (first, first_task) = publish(*address, "relay-a").await;
            wait_count(current, 1).await;
            assert!(post().await.unwrap().status().is_success());
            let (second, second_task) = publish(*address, "relay-b").await;
            drop(first);
            first_task.await.unwrap().unwrap();
            wait_count(current, 1).await;
            assert!(post().await.unwrap().status().is_success());
            drop(second);
            second_task.await.unwrap().unwrap();
            wait_count(current, 0).await;
            assert!(!post().await.unwrap().status().is_success());
            assert_eq!(discovery.borrow().pools().len(), 1);

            let (old, old_task) = publish(*address, "relay-a").await;
            wait_count(current, 1).await;
            let (replacement, replacement_task) = publish(*address, "relay-a").await;
            // Closing the replacement waits until its snapshot was received and its guard dropped.
            // First use a distinct replica to keep the pool live while testing the old stream.
            let (survivor, survivor_task) = publish(*address, "relay-c").await;
            drop(replacement);
            replacement_task.await.unwrap().unwrap();
            // This old source can no longer restore its view after replacement ended.
            old.send(snapshot(1, "relay-a")).await.unwrap();
            assert_eq!(
                old_task.await.unwrap().unwrap_err().code(),
                tonic::Code::FailedPrecondition
            );
            drop(old);
            wait_count(current, 1).await;
            assert!(post().await.unwrap().status().is_success());
            drop(survivor);
            survivor_task.await.unwrap().unwrap();
            wait_count(current, 0).await;
        }
        for (cancel, _, address, _, receiver, server) in routers {
            // Shutdown also terminates a connection that has not sent a Hello.
            let mut client = PoolViewReceiverClient::connect(format!("http://{address}"))
                .await
                .unwrap();
            let (_sender, frames) = mpsc::channel::<PoolViewFrame>(1);
            let pending =
                tokio::spawn(async move { client.publish(ReceiverStream::new(frames)).await });
            cancel.cancel();
            tokio::time::timeout(Duration::from_secs(5), receiver)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            server.await.unwrap().unwrap();
            let _ = pending.await.unwrap();
        }
        pool_server.abort();
    })
    .await
    .expect("receiver transport test timed out");
}

#[tokio::test]
async fn present_snapshots_and_upserts_require_state() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let (_discovery, candidates) =
            watch::channel(Arc::new(PoolSnapshot::new(vec![Pool::new(
                &key,
                "qwen",
                "east",
                vec!["http://pool:8000".parse().unwrap()],
            )])));
        let cancel = CancellationToken::new();
        let catalog = Arc::new(RelayGatedPoolCatalog::new(candidates, cancel.clone()));
        let current = catalog.subscribe();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(serve_relay_views(
            listener,
            catalog,
            Arc::new(TrustedNetworkAdmission),
            cancel.clone(),
        )));
        for upsert in [false, true] {
            let mut client = PoolViewReceiverClient::connect(format!("http://{address}"))
                .await
                .unwrap();
            let mut frames = vec![hello("relay-a")];
            if upsert {
                frames.push(snapshot(0, "relay-a"));
                frames.push(PoolViewFrame {
                    event: Some(Event::Upsert(grpc::Upsert {
                        revision: 1,
                        state: None,
                    })),
                });
            } else {
                frames.push(PoolViewFrame {
                    event: Some(Event::Snapshot(Snapshot {
                        revision: 0,
                        pool_present: true,
                        state: None,
                    })),
                });
            }
            assert_eq!(
                client
                    .publish(tokio_stream::iter(frames))
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::InvalidArgument
            );
            assert!(current.borrow().pools().is_empty());
        }
        cancel.cancel();
        drop(server);
    })
    .await
    .unwrap();
}
