// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic shutdown test scenarios parameterized on ShutdownTestClient

use super::*;
use std::time::Duration;
use tokio::time::{sleep, timeout};

pub async fn drain_rejects_messages<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    let mut stream =
        C::connect_and_send_frame(&handle, MessageType::Message, b"req-header", b"req-payload")
            .await;
    let (msg_type, header, payload) = C::read_one_frame(&mut stream).await;
    assert_eq!(msg_type, MessageType::ShuttingDown);
    assert_eq!(&header[..], b"req-header");
    assert_eq!(payload.len(), 0);

    handle.streams.shutdown_state.teardown_token().cancel();
}

pub async fn drain_accepts_responses<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    C::connect_and_send_frame(
        &handle,
        MessageType::Response,
        b"resp-header",
        b"resp-payload",
    )
    .await;

    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"resp-header");
    assert_eq!(&payload[..], b"resp-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

pub async fn drain_accepts_events<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    C::connect_and_send_frame(&handle, MessageType::Event, b"evt-header", b"evt-payload").await;

    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.event_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"evt-header");
    assert_eq!(&payload[..], b"evt-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

pub async fn new_connection_during_drain<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    C::connect_and_send_frame(&handle, MessageType::Response, b"new-resp", b"new-payload").await;

    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"new-resp");
    assert_eq!(&payload[..], b"new-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

pub async fn graceful_shutdown_lifecycle<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();

    // Verify normal operation
    C::connect_and_send_frame(&handle, MessageType::Message, b"normal-msg", b"normal-pay").await;
    let (header, _payload) = timeout(
        Duration::from_secs(2),
        handle.streams.message_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"normal-msg");

    // Acquire in-flight guard
    let guard = handle.streams.shutdown_state.acquire();
    assert_eq!(handle.streams.shutdown_state.in_flight_count(), 1);

    // Phase 1: begin drain
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    // Verify rejection
    let mut stream =
        C::connect_and_send_frame(&handle, MessageType::Message, b"reject-me", b"").await;
    let (msg_type, _, _) = C::read_one_frame(&mut stream).await;
    assert_eq!(msg_type, MessageType::ShuttingDown);

    // Verify responses still flow
    C::connect_and_send_frame(&handle, MessageType::Response, b"still-ok", b"data").await;
    let (header, _) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"still-ok");

    // Phase 2+3: wait for drain then teardown
    let shutdown_state = handle.streams.shutdown_state.clone();
    let shutdown_handle = tokio::spawn(async move {
        shutdown_state.wait_for_drain().await;
        shutdown_state.teardown_token().cancel();
    });

    sleep(Duration::from_millis(100)).await;
    assert!(!shutdown_handle.is_finished());

    drop(guard);

    timeout(Duration::from_secs(2), shutdown_handle)
        .await
        .expect("shutdown should complete")
        .unwrap();

    assert!(
        handle
            .streams
            .shutdown_state
            .teardown_token()
            .is_cancelled()
    );
}

pub async fn shutdown_timeout_forces_teardown<C: ShutdownTestClient>() {
    let handle = C::new_handle().await.unwrap();
    let _guard = handle.streams.shutdown_state.acquire();

    let shutdown_state = handle.streams.shutdown_state.clone();
    let shutdown_handle = tokio::spawn(async move {
        shutdown_state.begin_drain();
        let _ =
            tokio::time::timeout(Duration::from_millis(100), shutdown_state.wait_for_drain()).await;
        shutdown_state.teardown_token().cancel();
    });

    timeout(Duration::from_secs(2), shutdown_handle)
        .await
        .expect("shutdown should complete via timeout")
        .unwrap();

    assert!(
        handle
            .streams
            .shutdown_state
            .teardown_token()
            .is_cancelled()
    );
    assert_eq!(handle.streams.shutdown_state.in_flight_count(), 1);
}

pub async fn outbound_sends_during_drain<C: ShutdownTestClient>()
where
    C::Transport: 'static,
{
    let handle_a = C::new_handle().await.unwrap();
    let handle_b = C::new_handle().await.unwrap();

    handle_a.register_peer(&handle_b).unwrap();
    handle_b.register_peer(&handle_a).unwrap();

    handle_a.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    handle_a.send(
        handle_b.instance_id,
        b"response-hdr".to_vec(),
        b"response-pay".to_vec(),
        MessageType::Response,
    );

    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle_b.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    assert_eq!(&header[..], b"response-hdr");
    assert_eq!(&payload[..], b"response-pay");

    handle_a.streams.shutdown_state.teardown_token().cancel();
    handle_b.streams.shutdown_state.teardown_token().cancel();
}

pub async fn connection_writer_exits_on_teardown<C: ShutdownTestClient>()
where
    C::Transport: 'static,
{
    let handle_a = C::new_handle().await.unwrap();
    let handle_b = C::new_handle().await.unwrap();

    handle_a.register_peer(&handle_b).unwrap();

    handle_a.send(
        handle_b.instance_id,
        b"setup".to_vec(),
        b"data".to_vec(),
        MessageType::Message,
    );

    timeout(
        Duration::from_secs(2),
        handle_b.streams.message_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    handle_a.transport.shutdown();
    sleep(Duration::from_millis(200)).await;

    handle_a.error_handler.clear();
    handle_a.send(
        handle_b.instance_id,
        b"should-fail".to_vec(),
        b"data".to_vec(),
        MessageType::Message,
    );

    sleep(Duration::from_millis(100)).await;

    assert!(
        handle_a.error_handler.error_count() >= 1,
        "error handler should be invoked for post-shutdown send"
    );

    let not_delivered = timeout(
        Duration::from_millis(100),
        handle_b.streams.message_stream.recv_async(),
    )
    .await;
    assert!(
        not_delivered.is_err(),
        "post-shutdown message must not arrive at handle_b"
    );

    handle_b.streams.shutdown_state.teardown_token().cancel();
}
