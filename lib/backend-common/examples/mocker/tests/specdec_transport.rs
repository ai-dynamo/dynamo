// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::TcpListener;
use std::time::Duration;

use dynamo_backend_common::{EndpointId, WorkerWithDpRank};
use dynamo_mocker_backend::specdec::PROTOCOL;
use dynamo_mocker_backend::specdec::protocol::{
    Cleanup, DraftIdentity, Envelope, ErrorCode, FailureState, Hello, Message, Proposal, Start,
    StartAck,
};
use dynamo_mocker_backend::specdec::queue::{SchedulerConfig, TokenMode};
use dynamo_mocker_backend::specdec::transport::{
    DraftClient, DraftClientConfig, DraftServer, DraftServerConfig, TransportErrorKind,
};
use futures::{SinkExt, StreamExt};
use serial_test::serial;
use tmq::{AsZmqSocket, Context, Multipart, dealer, router};
use tokio::task::JoinSet;
use uuid::Uuid;

fn unused_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
    let address = listener.local_addr().expect("read loopback port");
    drop(listener);
    format!("tcp://{address}")
}

fn identity(address: String, cleanup_ms: u32) -> DraftIdentity {
    DraftIdentity {
        endpoint: EndpointId::from("specdec/draft/generate"),
        worker: WorkerWithDpRank::new(17, 0),
        draft_incarnation_id: 23,
        protocol: PROTOCOL.to_string(),
        address,
        orphan_cleanup_timeout_ms: cleanup_ms,
    }
}

fn server_config(address: String) -> DraftServerConfig {
    DraftServerConfig {
        bind_address: address,
        transport_hwm: 64,
        outbound_capacity: 64,
        prefill_duration: Duration::from_millis(1),
        token_interval: Duration::from_millis(1),
        token_mode: TokenMode::Echo,
        scheduler: SchedulerConfig {
            queue_capacity: 32,
            concurrency: 8,
            output_capacity: 8,
        },
    }
}

fn client_config() -> DraftClientConfig {
    DraftClientConfig {
        handshake_timeout: Duration::from_secs(5),
        start_timeout: Duration::from_secs(5),
        inactivity_timeout: Duration::from_secs(5),
        cleanup_timeout: Duration::from_secs(5),
        ..DraftClientConfig::default()
    }
}

fn start(prompt_token_ids: Vec<u32>, max_output_tokens: u32) -> Start {
    Start {
        prompt_token_ids,
        max_output_tokens,
    }
}

#[tokio::test]
#[serial]
async fn real_router_dealer_exchange_verifies_digest_and_cleanup() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let server = DraftServer::bind(server_config(address), identity.clone())
        .await
        .unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();

    let mut session = client
        .start(Uuid::from_u128(101), start(vec![41, 42], 3))
        .await
        .unwrap();
    let proposal = session.collect().await.unwrap();
    assert_eq!(proposal.token_ids, vec![41, 42, 41]);
    assert_eq!(proposal.proposal_digest.len(), 64);
    session.cleanup().await.unwrap();
    assert_eq!(server.active_sessions(), 0);

    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn cleanup_drains_proposals_already_buffered_after_start() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let server = DraftServer::bind(server_config(address), identity.clone())
        .await
        .unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();

    let mut session = client
        .start(Uuid::from_u128(102), start(vec![41, 42], 4))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    session.cleanup().await.unwrap();
    assert_eq!(server.active_sessions(), 0);

    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn concurrent_sessions_do_not_cross_talk() {
    let address = unused_address();
    let identity = identity(address.clone(), 10_000);
    let server = DraftServer::bind(server_config(address), identity.clone())
        .await
        .unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();

    let mut starts = JoinSet::new();
    for index in 0..32_u32 {
        let request_id = Uuid::from_u128(1_000 + u128::from(index));
        let client = client.clone();
        starts.spawn(async move {
            let session = client
                .start(request_id, start(vec![index, index + 100], 3))
                .await
                .unwrap();
            (index, session)
        });
    }
    let mut sessions = Vec::new();
    while let Some(result) = starts.join_next().await {
        sessions.push(result.unwrap());
    }

    let mut tasks = JoinSet::new();
    for (index, mut session) in sessions {
        tasks.spawn(async move {
            let expected = vec![index, index + 100, index];
            let proposal = session.collect().await.unwrap();
            assert_eq!(proposal.token_ids, expected);
            session
        });
    }
    let mut sessions = Vec::new();
    while let Some(result) = tasks.join_next().await {
        sessions.push(result.unwrap());
    }
    for mut session in sessions {
        session.cleanup().await.unwrap();
    }
    assert_eq!(server.active_sessions(), 0);
    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn handshake_rejects_any_full_identity_mismatch() {
    let address = unused_address();
    let server_identity = identity(address.clone(), 500);
    let server = DraftServer::bind(server_config(address), server_identity.clone())
        .await
        .unwrap();
    let mut wrong_identity = server_identity;
    wrong_identity.worker = WorkerWithDpRank::new(18, 0);

    let error = match DraftClient::connect(wrong_identity, client_config()).await {
        Ok(_) => panic!("mismatched identity unexpectedly connected"),
        Err(error) => error,
    };
    assert_eq!(error.kind, TransportErrorKind::Identity);
    assert_eq!(error.state, FailureState::ProtocolInvalid);
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn heartbeat_lease_reaps_session_after_client_disconnect() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let mut config = server_config(address);
    config.prefill_duration = Duration::from_secs(10);
    let server = DraftServer::bind(config, identity.clone()).await.unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();
    let session = client
        .start(Uuid::from_u128(202), start(vec![1], 1))
        .await
        .unwrap();
    assert_eq!(server.active_sessions(), 1);

    client.shutdown().await.unwrap();
    drop(session);
    tokio::time::timeout(Duration::from_secs(2), async {
        while server.active_sessions() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn wrong_peer_cleanup_cannot_remove_an_owned_session() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let mut config = server_config(address.clone());
    config.prefill_duration = Duration::from_secs(1);
    let server = DraftServer::bind(config, identity.clone()).await.unwrap();
    let (mut sink_a, mut stream_a) = dealer(&Context::new())
        .set_linger(0)
        .set_identity(b"peer-a")
        .connect(&address)
        .unwrap()
        .split();
    let (mut sink_b, mut stream_b) = dealer(&Context::new())
        .set_linger(0)
        .set_identity(b"peer-b")
        .connect(&address)
        .unwrap()
        .split();

    for (sink, stream, hello_id) in [
        (&mut sink_a, &mut stream_a, Uuid::from_u128(301)),
        (&mut sink_b, &mut stream_b, Uuid::from_u128(302)),
    ] {
        let hello = Envelope::new(
            hello_id,
            0,
            Message::Hello(Hello {
                expected: identity.clone(),
            }),
        );
        sink.send(Multipart::from(vec![hello.encode().unwrap()]))
            .await
            .unwrap();
        let mut response = stream.next().await.unwrap().unwrap();
        let ack = Envelope::decode(response.pop_front().unwrap().as_ref()).unwrap();
        assert!(matches!(ack.message, Message::HelloAck(_)));
    }

    let request_id = Uuid::from_u128(303);
    let start = Envelope::new(request_id, 0, Message::Start(start(vec![7], 1)));
    sink_a
        .send(Multipart::from(vec![start.encode().unwrap()]))
        .await
        .unwrap();
    let mut response = stream_a.next().await.unwrap().unwrap();
    let ack = Envelope::decode(response.pop_front().unwrap().as_ref()).unwrap();
    assert!(matches!(ack.message, Message::StartAck(_)));
    assert_eq!(server.active_sessions(), 1);

    let wrong_cleanup = Envelope::new(request_id, 1, Message::Cleanup(Cleanup::default()));
    sink_b
        .send(Multipart::from(vec![wrong_cleanup.encode().unwrap()]))
        .await
        .unwrap();
    let mut response = stream_b.next().await.unwrap().unwrap();
    let rejection = Envelope::decode(response.pop_front().unwrap().as_ref()).unwrap();
    assert!(matches!(
        rejection.message,
        Message::Error(ref error) if error.code == ErrorCode::UnknownRequest
    ));
    assert_eq!(server.active_sessions(), 1);

    sink_a
        .send(Multipart::from(vec![
            Envelope::new(request_id, 1, Message::Cleanup(Cleanup::default()))
                .encode()
                .unwrap(),
        ]))
        .await
        .unwrap();
    loop {
        let mut response = stream_a.next().await.unwrap().unwrap();
        let envelope = Envelope::decode(response.pop_front().unwrap().as_ref()).unwrap();
        if matches!(envelope.message, Message::CleanupAck(_)) {
            break;
        }
    }
    assert_eq!(server.active_sessions(), 0);
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn cumulative_proposals_cannot_exceed_the_start_limit() {
    let socket = router(&Context::new())
        .set_linger(0)
        .bind("tcp://127.0.0.1:*")
        .unwrap();
    socket.set_router_mandatory(true).unwrap();
    let address = socket.get_socket().get_last_endpoint().unwrap().unwrap();
    let identity = identity(address, 500);
    let (mut sink, mut stream) = socket.split();
    let fake_identity = identity.clone();
    let (release_fake_server, released) = tokio::sync::oneshot::channel();
    let fake_server = tokio::spawn(async move {
        let mut hello_frames = stream.next().await.unwrap().unwrap();
        let peer = hello_frames.pop_front().unwrap().to_vec();
        let hello = Envelope::decode(hello_frames.pop_front().unwrap().as_ref()).unwrap();
        sink.send(Multipart::from(vec![
            peer.clone(),
            Envelope::new(
                hello.request_id,
                0,
                Message::HelloAck(dynamo_mocker_backend::specdec::protocol::HelloAck {
                    identity: fake_identity,
                }),
            )
            .encode()
            .unwrap(),
        ]))
        .await
        .unwrap();

        let (peer, start) = loop {
            let mut frames = stream.next().await.unwrap().unwrap();
            let peer = frames.pop_front().unwrap().to_vec();
            let envelope = Envelope::decode(frames.pop_front().unwrap().as_ref()).unwrap();
            if matches!(&envelope.message, Message::Start(_)) {
                break (peer, envelope);
            }
        };
        sink.send(Multipart::from(vec![
            peer.clone(),
            Envelope::new(start.request_id, 0, Message::StartAck(StartAck::default()))
                .encode()
                .unwrap(),
        ]))
        .await
        .unwrap();
        for (sequence, tokens) in [(1, vec![1, 2]), (2, vec![3])] {
            sink.send(Multipart::from(vec![
                peer.clone(),
                Envelope::new(
                    start.request_id,
                    sequence,
                    Message::Proposal(Proposal { token_ids: tokens }),
                )
                .encode()
                .unwrap(),
            ]))
            .await
            .unwrap();
        }
        let _ = released.await;
    });

    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();
    let mut session = client
        .start(Uuid::from_u128(401), start(vec![1], 2))
        .await
        .unwrap();
    let error = session.collect().await.unwrap_err();
    assert_eq!(error.kind, TransportErrorKind::Protocol);
    assert_eq!(error.state, FailureState::ProtocolInvalid);
    release_fake_server.send(()).unwrap();
    client.shutdown().await.unwrap();
    fake_server.await.unwrap();
}

#[tokio::test]
#[serial]
async fn cancelled_start_quarantines_the_connection_before_start_ack() {
    let socket = router(&Context::new())
        .set_linger(0)
        .bind("tcp://127.0.0.1:*")
        .unwrap();
    socket.set_router_mandatory(true).unwrap();
    let address = socket.get_socket().get_last_endpoint().unwrap().unwrap();
    let identity = identity(address, 500);
    let (mut sink, mut stream) = socket.split();
    let fake_identity = identity.clone();
    let (start_seen, start_received) = tokio::sync::oneshot::channel();
    let (release_fake_server, released) = tokio::sync::oneshot::channel();
    let fake_server = tokio::spawn(async move {
        let mut hello_frames = stream.next().await.unwrap().unwrap();
        let peer = hello_frames.pop_front().unwrap().to_vec();
        let hello = Envelope::decode(hello_frames.pop_front().unwrap().as_ref()).unwrap();
        sink.send(Multipart::from(vec![
            peer,
            Envelope::new(
                hello.request_id,
                0,
                Message::HelloAck(dynamo_mocker_backend::specdec::protocol::HelloAck {
                    identity: fake_identity,
                }),
            )
            .encode()
            .unwrap(),
        ]))
        .await
        .unwrap();

        loop {
            let mut frames = stream.next().await.unwrap().unwrap();
            let _peer = frames.pop_front().unwrap();
            let envelope = Envelope::decode(frames.pop_front().unwrap().as_ref()).unwrap();
            if matches!(envelope.message, Message::Start(_)) {
                start_seen.send(()).unwrap();
                break;
            }
        }
        let _ = released.await;
    });

    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();
    let starting_client = client.clone();
    let starting = tokio::spawn(async move {
        starting_client
            .start(Uuid::from_u128(402), start(vec![1], 1))
            .await
    });
    start_received.await.unwrap();
    starting.abort();
    let join_error = match starting.await {
        Ok(_) => panic!("cancelled START task unexpectedly completed"),
        Err(error) => error,
    };
    assert!(join_error.is_cancelled());
    assert!(client.is_closed());
    client.shutdown().await.unwrap();

    let reuse_error = match client.start(Uuid::from_u128(403), start(vec![2], 1)).await {
        Ok(_) => panic!("quarantined connection unexpectedly accepted another START"),
        Err(error) => error,
    };
    assert_eq!(reuse_error.kind, TransportErrorKind::Closed);
    release_fake_server.send(()).unwrap();
    fake_server.await.unwrap();
}

#[tokio::test]
#[serial]
async fn invalid_local_start_does_not_poison_the_shared_connection() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let server = DraftServer::bind(server_config(address), identity.clone())
        .await
        .unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();

    let error = match client
        .start(Uuid::from_u128(501), start(Vec::new(), 1))
        .await
    {
        Ok(_) => panic!("invalid START unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.kind, TransportErrorKind::Protocol);
    assert_eq!(error.state, FailureState::NotStarted);

    let mut session = client
        .start(Uuid::from_u128(502), start(vec![9], 1))
        .await
        .unwrap();
    assert_eq!(session.collect().await.unwrap().token_ids, vec![9]);
    session.cleanup().await.unwrap();
    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn completed_sessions_expire_while_the_client_stays_connected() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let mut config = server_config(address);
    config.scheduler.queue_capacity = 1;
    config.scheduler.concurrency = 1;
    let server = DraftServer::bind(config, identity.clone()).await.unwrap();
    let client = DraftClient::connect(identity, client_config())
        .await
        .unwrap();

    let mut first = client
        .start(Uuid::from_u128(601), start(vec![1], 1))
        .await
        .unwrap();
    first.collect().await.unwrap();
    let mut second = client
        .start(Uuid::from_u128(602), start(vec![2], 1))
        .await
        .unwrap();
    second.collect().await.unwrap();
    assert_eq!(server.active_sessions(), 2);

    let error = match client.start(Uuid::from_u128(603), start(vec![3], 1)).await {
        Ok(_) => panic!("session bound unexpectedly admitted another request"),
        Err(error) => error,
    };
    assert_eq!(error.kind, TransportErrorKind::Queue);
    assert_eq!(error.state, FailureState::NotStarted);
    tokio::time::timeout(Duration::from_secs(2), async {
        while server.active_sessions() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();

    drop(first);
    drop(second);
    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
#[serial]
async fn client_session_admission_is_hard_bounded() {
    let address = unused_address();
    let identity = identity(address.clone(), 500);
    let server = DraftServer::bind(server_config(address), identity.clone())
        .await
        .unwrap();
    let client = DraftClient::connect(
        identity,
        DraftClientConfig {
            max_sessions: 1,
            ..client_config()
        },
    )
    .await
    .unwrap();

    let mut admitted = client
        .start(Uuid::from_u128(701), start(vec![1], 1))
        .await
        .unwrap();
    let error = match client.start(Uuid::from_u128(702), start(vec![2], 1)).await {
        Ok(_) => panic!("client session limit unexpectedly admitted another request"),
        Err(error) => error,
    };
    assert_eq!(error.kind, TransportErrorKind::Backpressure);
    assert_eq!(error.state, FailureState::NotStarted);
    admitted.collect().await.unwrap();
    admitted.cleanup().await.unwrap();
    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}
