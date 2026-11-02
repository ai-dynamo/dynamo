// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicas sharing in-flight counts and session pins over gRPC.

use super::*;

#[tokio::test]
async fn least_in_flight_adds_what_peers_report() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let green = pool("green", "qwen", &["http://green:8000"]);
    let in_flight = Arc::new(InFlightTracker::new());
    let counts = Arc::new(FrontendSignals::new(in_flight.clone()));
    let policy = least_in_flight_counting(&counts);
    let (core, _catalog) = selector(policy, &[blue.clone(), green.clone()]).await;

    // This process has one on green; a peer reports three on blue.
    let _ours = in_flight.start(&green.id);
    counts.record_peer(
        7,
        HashMap::from([(worker_id(&blue.id), 3)]),
        Duration::from_secs(60),
    );
    for _ in 0..5 {
        assert_eq!(
            picks(&core, "qwen", &[&blue, &green]).await,
            Some(green.id.clone())
        );
    }
}

#[tokio::test]
async fn least_in_flight_replicas_share_their_counts_over_grpc() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let green = pool("green", "qwen", &["http://green:8000"]);
    let pools = [blue.clone(), green.clone()];
    let (tracker_a, tracker_b) = (
        Arc::new(InFlightTracker::new()),
        Arc::new(InFlightTracker::new()),
    );
    // grpc_replicas builds replica a first, then b.
    let policies = std::sync::Mutex::new(vec![
        least_in_flight(&tracker_b),
        least_in_flight(&tracker_a),
    ]);
    let (_a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
        grpc_replicas(|| policies.lock().unwrap().pop().unwrap(), &pools).await;

    // Replica a's requests on blue steer b to green once a's next
    // report arrives; nothing is booked on either.
    let held = [tracker_a.start(&blue.id), tracker_a.start(&blue.id)];
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let mut all_green = true;
            for _ in 0..4 {
                all_green &= picks(&b, "qwen", &[&blue, &green]).await == Some(green.id.clone());
            }
            if all_green {
                return;
            }
            tokio::time::sleep(replica_sync::transport::IN_FLIGHT_INTERVAL).await;
        }
    })
    .await
    .expect("b saw a's requests on blue");
    assert_eq!(b.booked(&blue), 0);

    // Once they end, b sees blue free again and picks it too.
    drop(held);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let mut seen = std::collections::HashSet::new();
            for _ in 0..4 {
                seen.insert(picks(&b, "qwen", &[&blue, &green]).await);
            }
            if seen.len() == 2 {
                return;
            }
            tokio::time::sleep(replica_sync::transport::IN_FLIGHT_INTERVAL).await;
        }
    })
    .await
    .expect("b saw a's requests on blue end");
}

/// A replica's `ReplicaSync` routes served on a port of their own, as
/// the Global Frontend serves them on its API port. Dropping it stops
/// accepting; open streams end when the replica drops.
struct Served {
    port: u16,
    task: tokio::task::JoinHandle<()>,
}

impl Served {
    async fn new(selector: &CoreSelector, port: u16) -> Self {
        let routes = selector.replica_sync_routes().expect("gRPC routes");
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        // As the Global Frontend's `serve` does: the frames are small
        // writes, which Nagle's algorithm would hold for the delayed ACK.
        let listener = listener.tap_io(|tcp| tcp.set_nodelay(true).unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, routes).await.unwrap();
        });
        Self { port, task }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn connected(selector: &CoreSelector, peers: u64) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while selector.replica_sync_stats().unwrap().connected_peers != peers {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the replica did not reach {peers} open streams"));
}

/// Two cores on the same pools, each serving the gRPC stream and
/// subscribed to the other's, once both streams are open.
async fn grpc_replicas(
    policy: impl Fn() -> CorePolicy,
    pools: &[Pool],
) -> (CoreSelector, CoreSelector, Catalog, Catalog, Served, Served) {
    let grpc = || Some(GrpcReplicaSync::default());
    let (a, catalog_a) = synced_selector(policy(), pools, grpc()).await;
    let (b, catalog_b) = synced_selector(policy(), pools, grpc()).await;
    let served_a = Served::new(&a, 0).await;
    let served_b = Served::new(&b, 0).await;
    assert!(a.register_peer(&served_b.url()).unwrap());
    assert!(!a.register_peer(&served_b.url()).unwrap());
    assert!(b.register_peer(&served_a.url()).unwrap());
    connected(&a, 1).await;
    connected(&b, 1).await;
    (a, b, catalog_a, catalog_b, served_a, served_b)
}

async fn assert_replicas_agree_on_a_session_s_pool(
    a: &CoreSelector,
    b: &CoreSelector,
    pools: &[Pool],
    signals: &Arc<PoolSignals>,
) {
    // A pins the session to ohio-a while it is the emptiest local pool.
    set_occupancy(pools, signals, [0.1, 0.5, 0.5, 0.5]);
    assert_eq!(host_for(a, pools, Some("s")).await, "ohio-a");
    // Now ohio-b is emptier, but not by twice the room. Without the
    // replicated pin B would start the session on ohio-b.
    set_occupancy(pools, signals, [0.5, 0.3, 0.5, 0.5]);
    assert_eq!(host_for(b, pools, Some("fresh")).await, "ohio-b");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if host_for(b, pools, Some("s")).await == "ohio-a" {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("B learned A's pin");
    // The pin moving on B is seen by A: ohio-b fills, Tokyo empties.
    set_occupancy(pools, signals, [0.9, 0.9, 0.9, 0.1]);
    assert_eq!(host_for(b, pools, Some("s")).await, "tokyo-a");
    set_occupancy(pools, signals, [0.5, 0.5, 0.5, 0.5]);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if host_for(a, pools, Some("s")).await == "tokyo-a" {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("A learned B's move");
}

#[tokio::test]
async fn grpc_replicas_agree_on_a_session_s_pool() {
    let (pools, signals) = fleet();
    let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
        grpc_replicas(|| affinity(&signals), &pools).await;
    assert_replicas_agree_on_a_session_s_pool(&a, &b, &pools, &signals).await;
    assert_eq!(a.replica_sync_stats().unwrap().bindings_rejected, 0);
}

#[tokio::test]
async fn grpc_affinity_shares_pins_but_keeps_its_bookings_local() {
    let (pools, signals) = fleet();
    let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
        grpc_replicas(|| affinity(&signals), &pools).await;
    set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
    let candidates = pools.iter().collect::<Vec<_>>();

    // A new session books on A, and the booking is held.
    let request = RoutingRequest::new("qwen").with_session(Some("held"));
    let selection = a.select(&request, &candidates).await.unwrap();
    assert!(selection.lifecycle.is_some(), "a new session books");
    let pool = selection.target.pool;
    assert_eq!(a.booked(pool), 1);

    // B learns the pin, but not the booking: affinity's picker reads
    // occupancy, not bookings, so they are not sent.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if host_for(&b, &pools, Some("held")).await == host_for(&a, &pools, Some("held")).await
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("B learned A's pin");
    assert_eq!(b.booked(pool), 0);
    drop(selection);
}

async fn subscribe(
    served: &Served,
    marker: u32,
) -> Result<tonic::Streaming<replica_sync::proto::ReplicaFrame>, tonic::Status> {
    let channel = tonic::transport::Endpoint::from_shared(served.url())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = replica_sync::proto::replica_sync_client::ReplicaSyncClient::new(channel);
    client
        .subscribe_replica_events(replica_sync::proto::SubscribeRequest {
            subscriber_id: 1,
            routing_group: ROUTING_GROUP.to_owned(),
            contract_marker: marker,
        })
        .await
        .map(tonic::Response::into_inner)
}

#[tokio::test]
async fn grpc_a_new_subscriber_gets_the_in_flight_count_right_after_the_snapshot() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let in_flight = Arc::new(InFlightTracker::new());
    let grpc = Some(GrpcReplicaSync::default());
    let (a, _catalog) = synced_selector(
        least_in_flight(&in_flight),
        std::slice::from_ref(&blue),
        grpc,
    )
    .await;
    let served = Served::new(&a, 0).await;
    let _held = [in_flight.start(&blue.id), in_flight.start(&blue.id)];

    let mut frames = subscribe(&served, replica_sync::CONTRACT_MARKER)
        .await
        .unwrap();
    let first = frames.message().await.unwrap().unwrap();
    assert!(matches!(
        first.frame,
        Some(replica_sync::proto::replica_frame::Frame::Snapshot(_))
    ));
    let second = frames.message().await.unwrap().unwrap();
    let Some(replica_sync::proto::replica_frame::Frame::InflightCount(counts)) = second.frame
    else {
        panic!("the in-flight count follows the snapshot: {second:?}");
    };
    assert_eq!(
        counts.pools,
        vec![replica_sync::proto::PoolInflightCount {
            worker_id: worker_id(&blue.id),
            count: 2,
        }]
    );
}

#[tokio::test]
async fn grpc_rejects_a_contract_mismatch() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let grpc = Some(GrpcReplicaSync::default());
    let (a, _catalog) = synced_selector(CorePolicy::RoundRobin, &[blue], grpc).await;
    let served = Served::new(&a, 0).await;

    let status = subscribe(&served, 0).await.expect_err("rejected");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert!(
        subscribe(&served, replica_sync::CONTRACT_MARKER)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn grpc_sends_a_snapshot_first_then_bindings() {
    use replica_sync::proto::replica_frame::Frame;

    let (pools, signals) = fleet();
    let grpc = Some(GrpcReplicaSync::default());
    let (a, _catalog) = synced_selector(affinity(&signals), &pools, grpc).await;
    let served = Served::new(&a, 0).await;

    let mut frames = subscribe(&served, replica_sync::CONTRACT_MARKER)
        .await
        .unwrap();
    let first = frames.message().await.unwrap().unwrap();
    assert!(matches!(first.frame, Some(Frame::Snapshot(_))), "{first:?}");

    set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
    let pool = host_for(&a, &pools, Some("s")).await;
    let second = tokio::time::timeout(Duration::from_secs(5), frames.message())
        .await
        .expect("a bindings frame")
        .unwrap()
        .unwrap();
    let Some(Frame::Bindings(batch)) = second.frame else {
        panic!("expected bindings, got {second:?}");
    };
    assert_eq!(batch.bindings.len(), 1);
    assert_eq!(batch.bindings[0].session_id, "s");
    assert_eq!(batch.bindings[0].routing_group, ROUTING_GROUP);
    let bound = pools
        .iter()
        .find(|candidate| candidate.name.ends_with(&pool));
    assert_eq!(batch.bindings[0].worker_id, worker_id(&bound.unwrap().id));
}

#[tokio::test]
async fn grpc_peer_reconnects_after_the_server_restarts() {
    let (pools, signals) = fleet();
    let (a, b, _catalog_a, catalog_b, _served_a, served_b) =
        grpc_replicas(|| affinity(&signals), &pools).await;
    let port_b = served_b.port;

    // B goes away: its listener closes and its open streams end.
    drop(served_b);
    drop(b);
    drop(catalog_b);
    connected(&a, 0).await;

    // A new B on the same port: A's dialer finds it within its retry.
    let grpc = Some(GrpcReplicaSync::default());
    let (b, _catalog_b) = synced_selector(affinity(&signals), &pools, grpc).await;
    let served_b = Served::new(&b, port_b).await;
    assert!(b.register_peer(&_served_a.url()).unwrap());
    connected(&a, 1).await;
    connected(&b, 1).await;
    assert_replicas_agree_on_a_session_s_pool(&a, &b, &pools, &signals).await;
    drop(served_b);
}

#[tokio::test]
async fn grpc_peers_are_followed_only_once_the_first_snapshot_is_in() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let pools = [blue.clone()];
    let grpc = || Some(GrpcReplicaSync::default());
    let (b, _catalog_b) = synced_selector(CorePolicy::RoundRobin, &pools, grpc()).await;
    let served_b = Served::new(&b, 0).await;

    // A knows B before it follows any catalog: nothing is dialed yet.
    let a = CoreSelector::new(
        CorePolicy::RoundRobin,
        Some(GrpcReplicaSync {
            peers: vec![replica_sync::peer_url(&served_b.url()).unwrap()],
        }),
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(a.replica_sync_stats().unwrap().connected_peers, 0);

    // Following the catalog takes the first snapshot, then dials.
    let (_catalog_a, subscription) = watch::channel(Arc::new(pools.to_vec()));
    let (first_snapshot, reconcile) = a.reconcile_catalog(subscription, CancellationToken::new());
    tokio::spawn(reconcile);
    first_snapshot.await.unwrap();
    connected(&a, 1).await;
    drop(served_b);
}

#[tokio::test]
async fn grpc_a_peer_removed_and_added_back_at_once_is_followed_again() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let green = pool("green", "qwen", &["http://green:8000"]);
    let pools = [blue.clone(), green.clone()];
    let (tracker_a, tracker_b) = (
        Arc::new(InFlightTracker::new()),
        Arc::new(InFlightTracker::new()),
    );
    let (counts_a, counts_b) = (
        Arc::new(FrontendSignals::new(tracker_a)),
        Arc::new(FrontendSignals::new(tracker_b.clone())),
    );
    // grpc_replicas builds replica a first, then b.
    let policies = std::sync::Mutex::new(vec![
        least_in_flight_counting(&counts_b),
        least_in_flight_counting(&counts_a),
    ]);
    let (a, _b, _catalog_a, _catalog_b, _served_a, served_b) =
        grpc_replicas(|| policies.lock().unwrap().pop().unwrap(), &pools).await;

    // The old subscription is still winding down when the peer is
    // back; the dialer must start a new one once it has ended.
    let hub = &a.grpc.as_ref().unwrap().hub;
    let peer_b = replica_sync::peer_url(&served_b.url()).unwrap();
    assert!(hub.remove_peer(&peer_b));
    assert!(hub.add_peer(peer_b));
    tokio::time::sleep(Duration::from_millis(200)).await;

    // B's requests in flight reach A again over the new subscription.
    let _held = [tracker_b.start(&blue.id), tracker_b.start(&blue.id)];
    tokio::time::timeout(Duration::from_secs(5), async {
        while counts_a.peers().get(&worker_id(&blue.id)) != Some(&2) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("A counts B's requests again");
}

#[tokio::test]
async fn grpc_stop_replication_ends_the_open_streams() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let green = pool("green", "qwen", &["http://green:8000"]);
    let pools = [blue.clone(), green.clone()];
    let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
        grpc_replicas(|| CorePolicy::RoundRobin, &pools).await;

    b.stop_replication();
    connected(&a, 0).await;
    connected(&b, 0).await;
}

#[tokio::test]
async fn grpc_a_frame_from_an_earlier_following_of_a_peer_is_not_applied() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let counts = Arc::new(FrontendSignals::new(Arc::new(InFlightTracker::new())));
    let policy = least_in_flight_counting(&counts);
    let grpc = Some(GrpcReplicaSync::default());
    let (a, _catalog) = synced_selector(policy, std::slice::from_ref(&blue), grpc).await;
    let hub = a.grpc.as_ref().unwrap().hub.clone();
    let peer: Url = "http://peer:8000".parse().unwrap();
    assert!(hub.add_peer(peer.clone()));
    let old = hub.follows()[0].1.generation;
    assert!(hub.remove_peer(&peer));
    assert!(hub.add_peer(peer.clone()));
    let current = hub.follows()[0].1.generation;
    assert_ne!(old, current);

    let frame = |count: u64| replica_sync::proto::ReplicaFrame {
        frame: Some(replica_sync::proto::replica_frame::Frame::InflightCount(
            replica_sync::proto::InflightCount {
                replica_id: 99,
                pools: vec![replica_sync::proto::PoolInflightCount {
                    worker_id: worker_id(&blue.id),
                    count,
                }],
            },
        )),
    };
    hub.apply_from(&peer, old, frame(5), &a.service);
    assert!(counts.peers().is_empty());
    hub.apply_from(&peer, current, frame(1), &a.service);
    assert_eq!(counts.peers().get(&worker_id(&blue.id)), Some(&1));
}

#[tokio::test]
async fn grpc_refuses_tls_peers_and_other_schemes() {
    let blue = pool("blue", "qwen", &["http://blue:8000"]);
    let grpc = Some(GrpcReplicaSync::default());
    let (a, _catalog) = synced_selector(CorePolicy::RoundRobin, &[blue], grpc).await;
    for peer in ["https://peer:8000", "tcp://peer:5600", "peer:8000"] {
        assert!(a.register_peer(peer).is_err(), "{peer}");
    }
    assert!(a.register_peer("http://peer:8000").unwrap());
    assert_eq!(a.replica_sync_stats().unwrap().connected_peers, 0);
}
