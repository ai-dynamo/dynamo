// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_global_common::{PoolIdDeriver, PoolKey, PoolsResponse, V1PoolIdDeriver, grpc};
use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_frontend::{
    RelayGatedPoolCatalog, TrustedNetworkAdmission, discover_pools, serve_relay_views,
};
use dynamo_global_router::{InFlightTracker, PoolSignals};
use dynamo_llm::pool_relay::wan::grpc::protocol as relay;
use grpc::pool_view_frame::Event;
use grpc::pool_view_receiver_client::PoolViewReceiverClient;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

mod common;
use common::state;

async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

async fn publisher(
    address: std::net::SocketAddr,
    snapshot: grpc::PoolState,
) -> (
    mpsc::Sender<grpc::PoolViewFrame>,
    tokio::task::JoinHandle<Result<tonic::Response<grpc::PublishResult>, tonic::Status>>,
) {
    let mut client = PoolViewReceiverClient::connect(format!("http://{address}"))
        .await
        .unwrap();
    let (sender, frames) = mpsc::channel(8);
    let task = tokio::spawn(async move { client.publish(ReceiverStream::new(frames)).await });
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Hello(grpc::Hello {
                pool_key: Some(grpc::PoolKey {
                    cluster_id: "east".into(),
                    namespace: "dynamo".into(),
                    dgd_name: "blue".into(),
                }),
                logical_service: "blue-relay".into(),
                replica_id: "1".into(),
                incarnation: 1,
            })),
        })
        .await
        .unwrap();
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Snapshot(grpc::Snapshot {
                revision: 0,
                pool_present: true,
                state: Some(snapshot),
            })),
        })
        .await
        .unwrap();
    (sender, task)
}

#[tokio::test]
async fn readiness_generation_and_relay_loss_gate_live_gds_candidates() {
    let listed: PoolsResponse = serde_json::from_value(serde_json::json!({
        "complete": true, "pools": [{
            "key": {"cluster_id": "east", "namespace": "dynamo", "dgd_name": "blue"},
            "location": {"region": "east"}, "model": "qwen", "frontends": ["http://pool:8000"]
        }]
    }))
    .unwrap();
    let failed = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let gds = Router::new().route(
        "/pools",
        get({
            let failed = failed.clone();
            let calls = calls.clone();
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                let listed = listed.clone();
                let failed = failed.load(Ordering::SeqCst);
                async move {
                    if failed {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    } else {
                        Json(listed).into_response()
                    }
                }
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap())
        .parse()
        .unwrap();
    let _gds = AbortOnDropHandle::new(tokio::spawn(async move {
        axum::serve(listener, gds).await.unwrap()
    }));
    let cancel = CancellationToken::new();
    let discovered = discover_pools(
        GlobalDiscoveryClient::new(url, Duration::from_secs(1)).unwrap(),
        Duration::from_millis(20),
        cancel.clone(),
    );
    until(|| discovered.borrow().pools().len() == 1).await;
    let pool = V1PoolIdDeriver.derive(&PoolKey::new("east", "dynamo", "blue").unwrap());
    let signals = Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new())));
    let catalog = Arc::new(RelayGatedPoolCatalog::with_signals(
        discovered.clone(),
        cancel.clone(),
        Some(signals.clone()),
    ));
    let current = catalog.subscribe();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = tokio::spawn(serve_relay_views(
        listener,
        catalog,
        Arc::new(TrustedNetworkAdmission),
        cancel.clone(),
    ));
    let (sender, task) = publisher(address, state(false, 1, 1, 1)).await;
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 1,
                state: Some(state(true, 1, 1, 2)),
            })),
        })
        .await
        .unwrap();
    until(|| current.borrow().pools().len() == 1).await;
    let load = signals.occupancy(&pool).unwrap();
    assert_eq!(load.slots, 0.125);
    assert_eq!(load.kv, Some(0.2));
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 2,
                state: Some(state(false, 1, 1, 2)),
            })),
        })
        .await
        .unwrap();
    until(|| current.borrow().pools().is_empty()).await;
    assert!(signals.occupancy(&pool).is_none());
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 3,
                state: Some(state(true, 1, 1, 2)),
            })),
        })
        .await
        .unwrap();
    until(|| current.borrow().pools().len() == 1).await;

    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 4,
                state: Some(state(true, 2, 1, 2)),
            })),
        })
        .await
        .unwrap();
    until(|| signals.occupancy(&pool).is_none()).await;
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 5,
                state: Some(state(true, 2, 2, 3)),
            })),
        })
        .await
        .unwrap();
    until(|| signals.occupancy(&pool).is_some()).await;

    let (replacement, replacement_task) = publisher(address, state(true, 2, 2, 4)).await;
    replacement
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 1,
                state: Some(state(false, 2, 2, 4)),
            })),
        })
        .await
        .unwrap();
    until(|| current.borrow().pools().is_empty()).await;
    replacement
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 2,
                state: Some(state(true, 2, 2, 4)),
            })),
        })
        .await
        .unwrap();
    until(|| current.borrow().pools().len() == 1).await;
    drop(sender);
    task.await.unwrap().unwrap();
    assert!(
        signals.occupancy(&pool).is_some(),
        "an old guard must not clear its replacement's load"
    );
    let sender = replacement;
    let task = replacement_task;

    failed.store(true, Ordering::SeqCst);
    let before = calls.load(Ordering::SeqCst);
    until(|| calls.load(Ordering::SeqCst) > before).await;
    drop(sender);
    task.await.unwrap().unwrap();
    until(|| current.borrow().pools().is_empty()).await;
    assert_eq!(discovered.borrow().pools().len(), 1);
    assert!(signals.occupancy(&pool).is_none());

    let (sender, task) = publisher(address, state(true, 2, 2, 4)).await;
    until(|| current.borrow().pools().len() == 1).await;
    let mut mismatched = state(true, 2, 2, 5);
    let mut load = relay::KvPoolLoadUpdate::decode(mismatched.kv_load).unwrap();
    load.relay.as_mut().unwrap().relay_incarnation = 2;
    mismatched.kv_load = load.encode_to_vec().into();
    sender
        .send(grpc::PoolViewFrame {
            event: Some(Event::Upsert(grpc::Upsert {
                revision: 1,
                state: Some(mismatched),
            })),
        })
        .await
        .unwrap();
    assert_eq!(
        task.await.unwrap().unwrap_err().code(),
        tonic::Code::InvalidArgument
    );
    until(|| current.borrow().pools().is_empty()).await;
    assert!(signals.occupancy(&pool).is_none());
    drop(sender);
    cancel.cancel();
    receiver.await.unwrap().unwrap();
}

type TestStream<T> =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, tonic::Status>> + Send + 'static>>;

#[derive(Clone)]
struct SubscriptionRelay {
    state: tokio::sync::watch::Receiver<Option<grpc::PoolState>>,
    peers: Arc<parking_lot::Mutex<Vec<std::net::SocketAddr>>>,
}

impl SubscriptionRelay {
    fn stream<T: Send + 'static>(
        &self,
        decode: fn(grpc::PoolState) -> Result<T, prost::DecodeError>,
    ) -> TestStream<T> {
        Box::pin(futures::stream::unfold(
            (self.state.clone(), true),
            move |(mut state, first)| async move {
                if !first && state.changed().await.is_err() {
                    return None;
                }
                let current = state.borrow_and_update().clone()?;
                Some((
                    decode(current).map_err(|error| tonic::Status::internal(error.to_string())),
                    (state, false),
                ))
            },
        ))
    }

    fn peer<T>(&self, request: &tonic::Request<T>) {
        self.peers.lock().push(request.remote_addr().unwrap());
    }
}

#[tonic::async_trait]
impl relay::KvEventRelay for SubscriptionRelay {
    type WatchKvPoolCatalogStream = TestStream<relay::KvPoolCatalogUpdate>;
    type SubscribeKvPoolStream = TestStream<relay::FilterUpdate>;
    type SubscribeServingReadinessStream = TestStream<relay::ServingReadinessUpdate>;
    type SubscribeKvPoolLoadStream = TestStream<relay::KvPoolLoadUpdate>;
    type SubscribeServingLoadStream = TestStream<relay::ServingLoadUpdate>;

    async fn get_relay_info(
        &self,
        _: tonic::Request<relay::RelayInfoRequest>,
    ) -> Result<tonic::Response<relay::RelayInfo>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used"))
    }
    async fn subscribe_kv_pool(
        &self,
        _: tonic::Request<relay::SubscribeKvPoolRequest>,
    ) -> Result<tonic::Response<Self::SubscribeKvPoolStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used"))
    }
    async fn watch_kv_pool_catalog(
        &self,
        request: tonic::Request<relay::WatchKvPoolCatalogRequest>,
    ) -> Result<tonic::Response<Self::WatchKvPoolCatalogStream>, tonic::Status> {
        self.peer(&request);
        Ok(tonic::Response::new(self.stream(|state| {
            relay::KvPoolCatalogUpdate::decode(state.catalog)
        })))
    }
    async fn subscribe_serving_readiness(
        &self,
        request: tonic::Request<relay::SubscribeServingReadinessRequest>,
    ) -> Result<tonic::Response<Self::SubscribeServingReadinessStream>, tonic::Status> {
        self.peer(&request);
        Ok(tonic::Response::new(self.stream(|state| {
            relay::ServingReadinessUpdate::decode(state.readiness)
        })))
    }
    async fn subscribe_serving_load(
        &self,
        request: tonic::Request<relay::SubscribeServingLoadRequest>,
    ) -> Result<tonic::Response<Self::SubscribeServingLoadStream>, tonic::Status> {
        self.peer(&request);
        Ok(tonic::Response::new(self.stream(|state| {
            relay::ServingLoadUpdate::decode(state.serving_load)
        })))
    }
    async fn subscribe_kv_pool_load(
        &self,
        request: tonic::Request<relay::SubscribeKvPoolLoadRequest>,
    ) -> Result<tonic::Response<Self::SubscribeKvPoolLoadStream>, tonic::Status> {
        self.peer(&request);
        Ok(tonic::Response::new(self.stream(|state| {
            relay::KvPoolLoadUpdate::decode(state.kv_load)
        })))
    }
}

enum StateInput {
    Push {
        address: std::net::SocketAddr,
        sender: Option<mpsc::Sender<grpc::PoolViewFrame>>,
        task: Option<
            tokio::task::JoinHandle<Result<tonic::Response<grpc::PublishResult>, tonic::Status>>,
        >,
        revision: u64,
    },
    Pull(tokio::sync::watch::Sender<Option<grpc::PoolState>>),
}

impl StateInput {
    async fn update(&mut self, state: grpc::PoolState) {
        match self {
            Self::Push {
                address,
                sender,
                task,
                revision,
            } => {
                if let Some(sender) = sender {
                    *revision += 1;
                    sender
                        .send(grpc::PoolViewFrame {
                            event: Some(Event::Upsert(grpc::Upsert {
                                revision: *revision,
                                state: Some(state),
                            })),
                        })
                        .await
                        .unwrap();
                } else {
                    let (next_sender, next_task) = publisher(*address, state).await;
                    *sender = Some(next_sender);
                    *task = Some(next_task);
                    *revision = 0;
                }
            }
            Self::Pull(sender) => {
                sender.send_replace(Some(state));
            }
        }
    }
    async fn disconnect(&mut self) {
        match self {
            Self::Push { sender, task, .. } => {
                sender.take();
                task.take().unwrap().await.unwrap().unwrap();
            }
            Self::Pull(sender) => {
                sender.send_replace(None);
            }
        }
    }
}

async fn shared_state_contract(router_initiated: bool) {
    let cancel = CancellationToken::new();
    let key = PoolKey::new("east", "dynamo", "blue").unwrap();
    let pool = V1PoolIdDeriver.derive(&key);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut candidate = dynamo_global_router::Pool::new(
        &key,
        "qwen",
        "east",
        vec!["http://pool:8000".parse().unwrap()],
    );
    candidate.relay = Some(format!("http://{address}").parse().unwrap());
    let (_discovery, discovered) =
        tokio::sync::watch::channel(Arc::new(dynamo_global_frontend::PoolSnapshot::new(vec![
            candidate,
        ])));
    let signals = Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new())));
    let catalog = Arc::new(RelayGatedPoolCatalog::with_signals(
        discovered.clone(),
        cancel.clone(),
        Some(signals.clone()),
    ));
    let current = catalog.subscribe();
    let initial = state(false, 1, 1, 1);
    let peers = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let server;
    let mut follower = None;
    let mut input = if router_initiated {
        let (sender, receiver) = tokio::sync::watch::channel(Some(initial));
        let service = SubscriptionRelay {
            state: receiver,
            peers: peers.clone(),
        };
        let shutdown = cancel.clone();
        server = AbortOnDropHandle::new(tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(relay::KvEventRelayServer::new(service))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    shutdown.cancelled(),
                )
                .await
        }));
        follower = Some(AbortOnDropHandle::new(tokio::spawn(
            dynamo_global_frontend::follow_relays(
                discovered.clone(),
                catalog.clone(),
                cancel.clone(),
            ),
        )));
        StateInput::Pull(sender)
    } else {
        server = AbortOnDropHandle::new(tokio::spawn(serve_relay_views(
            listener,
            catalog.clone(),
            Arc::new(TrustedNetworkAdmission),
            cancel.clone(),
        )));
        let (sender, task) = publisher(address, initial).await;
        StateInput::Push {
            address,
            sender: Some(sender),
            task: Some(task),
            revision: 0,
        }
    };
    // Confirm that all four pull subscriptions opened before transitioning.
    if router_initiated {
        until(|| peers.lock().len() == 4).await;
    }
    assert!(current.borrow().pools().is_empty());
    input.update(state(true, 1, 1, 2)).await;
    until(|| {
        current.borrow().pools().len() == 1
            && signals
                .occupancy(&pool)
                .is_some_and(|load| load.kv == Some(0.2))
    })
    .await;
    assert_eq!(signals.occupancy(&pool).unwrap().slots, 0.125);
    if router_initiated {
        let peers = peers.lock();
        assert_eq!(peers.len(), 4);
        assert!(
            peers.iter().all(|peer| *peer == peers[0]),
            "all four subscriptions must reach the same Relay replica"
        );
    }
    input.update(state(false, 1, 1, 2)).await;
    until(|| current.borrow().pools().is_empty()).await;
    assert!(signals.occupancy(&pool).is_none());
    input.update(state(true, 2, 1, 2)).await;
    until(|| current.borrow().pools().len() == 1).await;
    assert!(
        signals.occupancy(&pool).is_none(),
        "old generation must remain unknown"
    );
    input.update(state(true, 2, 2, 3)).await;
    until(|| {
        signals
            .occupancy(&pool)
            .is_some_and(|load| load.kv == Some(0.2))
    })
    .await;
    input.disconnect().await;
    until(|| current.borrow().pools().is_empty()).await;
    assert_eq!(discovered.borrow().pools().len(), 1);
    assert!(signals.occupancy(&pool).is_none());
    input.update(state(true, 2, 2, 4)).await;
    until(|| current.borrow().pools().len() == 1 && signals.occupancy(&pool).is_some()).await;
    let mut mismatched = state(true, 2, 2, 5);
    let mut load = relay::KvPoolLoadUpdate::decode(mismatched.kv_load).unwrap();
    load.relay.as_mut().unwrap().relay_incarnation = 2;
    mismatched.kv_load = load.encode_to_vec().into();
    input.update(mismatched).await;
    until(|| current.borrow().pools().is_empty()).await;
    assert!(signals.occupancy(&pool).is_none());
    match &mut input {
        StateInput::Push { sender, task, .. } => {
            sender.take();
            assert_eq!(
                task.take().unwrap().await.unwrap().unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
        }
        StateInput::Pull(sender) => {
            sender.send_replace(None);
        }
    }
    cancel.cancel();
    drop(follower);
    drop(server);
}

#[tokio::test]
async fn router_initiated_state_obeys_shared_readiness_generation_and_loss_contract() {
    tokio::time::timeout(Duration::from_secs(15), shared_state_contract(true))
        .await
        .unwrap();
}

#[tokio::test]
async fn relay_initiated_state_obeys_shared_readiness_generation_and_loss_contract() {
    tokio::time::timeout(Duration::from_secs(15), shared_state_contract(false))
        .await
        .unwrap();
}
