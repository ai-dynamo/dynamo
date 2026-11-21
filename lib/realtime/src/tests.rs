// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_protocols::types::realtime::{DynamoRealtimeClientEvent, RealtimeServerEvent};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::{
    net::TcpListener,
    sync::{Notify, mpsc},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_tungstenite::{WebSocketStream, tungstenite::Message};

use crate::{Client, Limits, Protocol, audio::AsrAudio};

type Socket = WebSocketStream<tokio::net::TcpStream>;
const DEADLINE: Duration = Duration::from_secs(5);
const MODEL: &str = "test-audio-model";

fn asr_session() -> Value {
    json!({"type": "session.update", "session": {
        "type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": MODEL}, "turn_detection": null
        }}
    }})
}

fn omni_session() -> Value {
    json!({"type": "session.update", "session": {
        "type": "realtime", "model": MODEL, "output_modalities": ["audio"],
    }})
}

fn typed_client_event(event: Value) -> Value {
    // Match the frontend's deserialize/serialize boundary: omitted optional
    // session fields must stay absent, rather than acquire test-only defaults.
    serde_json::to_value(serde_json::from_value::<DynamoRealtimeClientEvent>(event).unwrap())
        .unwrap()
}

fn audio() -> Value {
    json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(vec![0u8; 9600])})
}

async fn send(socket: &mut Socket, value: Value) {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

async fn receive(socket: &mut Socket) -> Value {
    let message = tokio::time::timeout(DEADLINE, socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    match message {
        Message::Text(text) => serde_json::from_str(&text).unwrap(),
        other => panic!("expected JSON, got {other:?}"),
    }
}

async fn next(session: &mut crate::Session) -> Value {
    let event = tokio::time::timeout(DEADLINE, session.next())
        .await
        .unwrap()
        .unwrap();
    // Exercise the actual frontend response decoder, not only JSON fixtures.
    serde_json::from_value::<RealtimeServerEvent>(event.clone())
        .unwrap_or_else(|error| panic!("invalid public event {event}: {error}"));
    event
}

async fn setup(protocol: Protocol) -> (TcpListener, Client, mpsc::Sender<Value>, crate::Session) {
    setup_until(protocol, std::future::pending()).await
}

async fn setup_until(
    protocol: Protocol,
    cancellation: impl Future<Output = ()> + Send + 'static,
) -> (TcpListener, Client, mpsc::Sender<Value>, crate::Session) {
    setup_with_limits(protocol, Limits::for_protocol(protocol), cancellation).await
}

async fn setup_with_limits(
    protocol: Protocol,
    limits: Limits,
    cancellation: impl Future<Output = ()> + Send + 'static,
) -> (TcpListener, Client, mpsc::Sender<Value>, crate::Session) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::new(
        format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        MODEL.into(),
        limits,
        None,
    )
    .unwrap();
    let (sender, receiver) = mpsc::channel(16);
    let session = client.session_with_cancellation(
        protocol,
        ReceiverStream::new(receiver).boxed(),
        cancellation,
    );
    (listener, client, sender, session)
}

#[tokio::test]
async fn asr_reads_commit_and_clear_while_precommit_output_is_blocked() {
    for control in ["input_audio_buffer.commit", "input_audio_buffer.clear"] {
        let limits = Limits {
            response_timeout: Duration::from_millis(300),
            event_capacity: 1.try_into().unwrap(),
            close_timeout: Duration::from_millis(100),
            ..Limits::for_protocol(Protocol::VllmTranscription)
        };
        let (listener, client, input, mut session) =
            setup_with_limits(Protocol::VllmTranscription, limits, std::future::pending()).await;
        let sent = Arc::new(Notify::new());
        let server_sent = sent.clone();
        let server = tokio::spawn(async move {
            let mut socket = accept_asr(&listener).await;
            assert_eq!(
                receive(&mut socket).await["type"],
                "input_audio_buffer.append"
            );
            for index in 0..3 {
                send(
                    &mut socket,
                    json!({"type": "transcription.delta", "delta": index.to_string()}),
                )
                .await;
            }
            socket
                .send(Message::Ping(b"output-full".to_vec().into()))
                .await
                .unwrap();
            loop {
                match tokio::time::timeout(DEADLINE, socket.next()).await.unwrap() {
                    Some(Ok(Message::Pong(payload))) => {
                        assert_eq!(payload.as_ref(), b"output-full");
                        break;
                    }
                    Some(Ok(Message::Text(text))) => {
                        let event: Value = serde_json::from_str(&text).unwrap();
                        assert_eq!(event["type"], "input_audio_buffer.append");
                    }
                    other => panic!("expected the backpressure barrier pong, got {other:?}"),
                }
            }
            server_sent.notify_one();
            loop {
                match tokio::time::timeout(DEADLINE, socket.next()).await.unwrap() {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(text))) => {
                        let event: Value = serde_json::from_str(&text).unwrap();
                        assert!(matches!(
                            event["type"].as_str(),
                            Some("input_audio_buffer.append" | "input_audio_buffer.commit")
                        ));
                    }
                    other => panic!("unexpected upstream event: {other:?}"),
                }
            }
        });
        input.send(asr_session()).await.unwrap();
        assert_eq!(next(&mut session).await["type"], "session.updated");
        input.send(audio()).await.unwrap();
        tokio::time::timeout(DEADLINE, sent.notified())
            .await
            .unwrap();
        // The pong proves three deltas reached the one-slot public queue,
        // the coordinator's pending send, and the one-slot turn queue.
        input.send(json!({"type": control})).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .expect("pre-commit output backpressure blocked client control and upstream closure")
            .unwrap();
        if control == "input_audio_buffer.clear" {
            assert_eq!(next(&mut session).await["delta"], "0");
            assert_eq!(
                next(&mut session).await["type"],
                "input_audio_buffer.cleared"
            );
        } else {
            let mut index = 0;
            loop {
                let event = next(&mut session).await;
                if event["type"] == "conversation.item.input_audio_transcription.failed" {
                    break;
                }
                if event["type"] == "conversation.item.input_audio_transcription.delta" {
                    assert_eq!(event["delta"], index.to_string());
                    index += 1;
                } else {
                    assert_eq!(event["type"], "input_audio_buffer.committed");
                }
            }
            assert_eq!(index, 3);
        }
        drop(input);
        assert!(
            tokio::time::timeout(DEADLINE, session.next())
                .await
                .unwrap()
                .is_none()
        );
        client.shutdown().await;
    }
}

#[tokio::test]
async fn fatal_session_error_closes_omni_upstream_before_blocked_error_delivery() {
    let limits = Limits {
        event_capacity: 1.try_into().unwrap(),
        max_buffered_output_bytes: 1024.try_into().unwrap(),
        close_timeout: Duration::from_millis(100),
        ..Limits::for_protocol(Protocol::Omni)
    };
    let (listener, client, input, mut session) =
        setup_with_limits(Protocol::Omni, limits, std::future::pending()).await;
    let full = Arc::new(Notify::new());
    let server_full = full.clone();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
        send(&mut socket, json!({"type": "session.created"})).await;
        assert_eq!(receive(&mut socket).await["type"], "session.update");
        send(&mut socket, json!({"type": "session.updated"})).await;
        for kind in [
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ] {
            assert_eq!(receive(&mut socket).await["type"], kind);
        }
        for index in 0..3 {
            send(
                &mut socket,
                json!({
                    "type": "response.output_text.delta", "event_id": format!("delta-{index}"),
                    "response_id": "upstream", "item_id": "item", "output_index": 0,
                    "content_index": 0, "delta": index.to_string()
                }),
            )
            .await;
        }
        socket
            .send(Message::Ping(b"output-full".to_vec().into()))
            .await
            .unwrap();
        assert!(
            matches!(tokio::time::timeout(DEADLINE, socket.next()).await.unwrap(),
            Some(Ok(Message::Pong(payload))) if payload.as_ref() == b"output-full")
        );
        server_full.notify_one();
        loop {
            match tokio::time::timeout(DEADLINE, socket.next()).await.unwrap() {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(Message::Text(text))) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(event["type"], "response.cancel");
                }
                other => panic!("expected upstream cancellation and closure, got {other:?}"),
            }
        }
    });
    input
        .send(typed_client_event(
            json!({"type": "session.update", "session": {
                "type": "realtime", "model": MODEL, "output_modalities": ["text"]
            }}),
        ))
        .await
        .unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    input.send(audio()).await.unwrap();
    input
        .send(json!({"type": "input_audio_buffer.commit"}))
        .await
        .unwrap();
    tokio::time::timeout(DEADLINE, full.notified())
        .await
        .unwrap();
    input
        .send(typed_client_event(
            json!({"type": "session.update", "session": {
                "type": "realtime", "instructions": "x".repeat(2048)
            }}),
        ))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("fatal error delivery kept the upstream socket alive")
        .unwrap();
    assert_eq!(next(&mut session).await["delta"], "0");
    let error = next(&mut session).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["code"], "session_error");
    assert!(
        tokio::time::timeout(DEADLINE, session.next())
            .await
            .unwrap()
            .is_none()
    );
    client.shutdown().await;
}

#[tokio::test]
async fn committed_asr_deadline_closes_upstream_while_output_is_blocked() {
    let limits = Limits {
        response_timeout: Duration::from_millis(300),
        event_capacity: 1.try_into().unwrap(),
        close_timeout: Duration::from_millis(100),
        ..Limits::for_protocol(Protocol::VllmTranscription)
    };
    let (listener, client, input, mut session) =
        setup_with_limits(Protocol::VllmTranscription, limits, std::future::pending()).await;
    let committed = Arc::new(Notify::new());
    let server_committed = committed.clone();
    let paused = Arc::new(Notify::new());
    let server_paused = paused.clone();
    let server = tokio::spawn(async move {
        let mut socket = accept_asr(&listener).await;
        loop {
            let event = receive(&mut socket).await;
            if event["type"] != "input_audio_buffer.append" {
                assert_eq!(
                    event,
                    json!({"type": "input_audio_buffer.commit", "final": true})
                );
                break;
            }
        }
        server_committed.notify_one();
        tokio::time::timeout(DEADLINE, server_paused.notified())
            .await
            .unwrap();
        for _ in 0..16 {
            send(
                &mut socket,
                json!({"type": "transcription.delta", "delta": "word"}),
            )
            .await;
        }
        assert!(matches!(
            tokio::time::timeout(DEADLINE, socket.next()).await.unwrap(),
            Some(Ok(Message::Close(_))) | None
        ));
    });
    input.send(asr_session()).await.unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    input.send(audio()).await.unwrap();
    input
        .send(json!({"type": "input_audio_buffer.commit"}))
        .await
        .unwrap();
    tokio::time::timeout(DEADLINE, committed.notified())
        .await
        .unwrap();
    assert_eq!(
        next(&mut session).await["type"],
        "input_audio_buffer.committed"
    );
    paused.notify_one();
    // Keep the public stream alive and unpolled until the upstream is closed.
    tokio::time::timeout(DEADLINE, server)
        .await
        .expect("committed turn held its upstream socket while output was blocked")
        .unwrap();
    loop {
        let event = next(&mut session).await;
        if event["type"] == "conversation.item.input_audio_transcription.failed" {
            break;
        }
        assert_eq!(
            event["type"],
            "conversation.item.input_audio_transcription.delta"
        );
    }
    drop(input);
    assert!(
        tokio::time::timeout(DEADLINE, session.next())
            .await
            .unwrap()
            .is_none()
    );
    client.shutdown().await;
}

#[tokio::test]
async fn omni_local_audio_preserves_order_limit_and_clear_beyond_event_capacity() {
    const CHUNK_BYTES: usize = 4800;
    const CHUNKS: u8 = 65;
    let limits = Limits {
        max_audio_chunk_bytes: CHUNK_BYTES.try_into().unwrap(),
        max_turn_audio_bytes: (usize::from(CHUNKS) * CHUNK_BYTES).try_into().unwrap(),
        ..Limits::for_protocol(Protocol::Omni)
    };
    let (listener, client, input, mut session) =
        setup_with_limits(Protocol::Omni, limits, std::future::pending()).await;
    input.send(omni_session()).await.unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    let mut expected = Vec::new();
    for phase in 0..3 {
        for index in 0..CHUNKS {
            let pcm = vec![index; CHUNK_BYTES];
            if phase == 1 {
                expected.extend_from_slice(&pcm);
            }
            input
                .send(json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(pcm)}))
                .await
                .unwrap();
        }
        match phase {
            0 => {
                input
                    .send(json!({"type": "input_audio_buffer.clear"}))
                    .await
                    .unwrap();
                tokio::select! {
                    biased;
                    _ = listener.accept() => panic!("uncommitted Omni audio opened an upstream socket"),
                    event = next(&mut session) => assert_eq!(event["type"], "input_audio_buffer.cleared"),
                }
            }
            1 => {
                input
                    .send(json!({"type": "input_audio_buffer.commit"}))
                    .await
                    .unwrap();
                let (tcp, _) = tokio::time::timeout(DEADLINE, listener.accept())
                    .await
                    .unwrap()
                    .unwrap();
                let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
                send(&mut socket, json!({"type": "session.created"})).await;
                assert_eq!(receive(&mut socket).await["type"], "session.update");
                send(&mut socket, json!({"type": "session.updated"})).await;
                let mut actual = Vec::new();
                loop {
                    let event = receive(&mut socket).await;
                    if event["type"] == "input_audio_buffer.commit" {
                        break;
                    }
                    assert_eq!(event["type"], "input_audio_buffer.append");
                    let chunk = BASE64_STANDARD
                        .decode(event["audio"].as_str().unwrap())
                        .unwrap();
                    assert!(!chunk.is_empty() && chunk.len() <= CHUNK_BYTES);
                    assert_eq!(chunk.len() % 2, 0);
                    actual.extend(chunk);
                }
                assert_eq!(actual, expected);
                assert_eq!(receive(&mut socket).await["type"], "response.create");
                send(&mut socket, json!({"type": "response.done", "event_id": "done", "response": {
                    "id": "upstream-response", "object": "realtime.response", "output": [],
                    "output_modalities": ["audio"], "status": "completed", "max_output_tokens": "inf"
                }})).await;
                assert_eq!(next(&mut session).await["type"], "response.done");
                assert!(matches!(
                    tokio::time::timeout(DEADLINE, socket.next()).await.unwrap(),
                    Some(Ok(Message::Close(_))) | None
                ));
            }
            _ => {
                input.send(json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode([0, 0])})).await.unwrap();
                assert_eq!(next(&mut session).await["error"]["code"], "invalid_audio");
            }
        }
    }
    drop(input);
    tokio::select! {
        biased;
        _ = listener.accept() => panic!("cleared or oversized Omni audio opened an upstream socket"),
        event = tokio::time::timeout(DEADLINE, session.next()) => assert!(event.unwrap().is_none()),
    }
    client.shutdown().await;
}

#[tokio::test]
async fn cancellation_closes_upstream_without_polling_buffered_output() {
    let cancel = tokio_util::sync::CancellationToken::new();
    let (listener, client, input, session) = setup_until(
        Protocol::VllmTranscription,
        cancel.clone().cancelled_owned(),
    )
    .await;
    let opened = Arc::new(Notify::new());
    let server_opened = opened.clone();
    let server = tokio::spawn(async move {
        let mut socket = accept_asr(&listener).await;
        assert_eq!(
            receive(&mut socket).await["type"],
            "input_audio_buffer.append"
        );
        server_opened.notify_one();
        assert!(matches!(
            tokio::time::timeout(DEADLINE, socket.next()).await.unwrap(),
            Some(Ok(Message::Close(_))) | None
        ));
    });
    input.send(asr_session()).await.unwrap();
    input.send(audio()).await.unwrap();
    tokio::time::timeout(DEADLINE, opened.notified())
        .await
        .unwrap();
    cancel.cancel();
    tokio::time::timeout(DEADLINE, server)
        .await
        .unwrap()
        .unwrap();
    drop(session);
    client.shutdown().await;
}

async fn accept_asr(listener: &TcpListener) -> Socket {
    let (tcp, _) = listener.accept().await.unwrap();
    let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
    send(
        &mut socket,
        json!({"type": "session.created", "id": "upstream"}),
    )
    .await;
    assert_eq!(
        receive(&mut socket).await,
        json!({"type": "session.update", "model": MODEL})
    );
    assert_eq!(
        receive(&mut socket).await,
        json!({"type": "input_audio_buffer.commit", "final": false})
    );
    socket
}

#[tokio::test]
async fn asr_streams_before_commit_and_drains_queued_turns_on_input_eof() {
    let (listener, client, input, mut session) = setup(Protocol::VllmTranscription).await;
    let server = tokio::spawn(async move {
        for turn in 0..2 {
            let mut socket = accept_asr(&listener).await;
            let first = receive(&mut socket).await;
            assert_eq!(first["type"], "input_audio_buffer.append");
            let mut pcm = BASE64_STANDARD
                .decode(first["audio"].as_str().unwrap())
                .unwrap();
            send(
                &mut socket,
                json!({"type": "transcription.delta", "delta": format!("turn-{turn}")}),
            )
            .await;
            loop {
                let event = receive(&mut socket).await;
                if event["type"] == "input_audio_buffer.append" {
                    pcm.extend(
                        BASE64_STANDARD
                            .decode(event["audio"].as_str().unwrap())
                            .unwrap(),
                    );
                } else {
                    assert_eq!(
                        event,
                        json!({"type": "input_audio_buffer.commit", "final": true})
                    );
                    break;
                }
            }
            // 0.2 seconds of public 24 kHz PCM becomes exactly 16 kHz PCM.
            assert_eq!(pcm.len(), 6400);
            send(
                &mut socket,
                json!({"type": "transcription.done", "text": format!("turn-{turn}"),
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}),
            )
            .await;
        }
    });
    input.send(asr_session()).await.unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    input.send(audio()).await.unwrap();
    assert_eq!(next(&mut session).await["delta"], "turn-0");
    input
        .send(json!({"type": "input_audio_buffer.commit"}))
        .await
        .unwrap();
    input.send(audio()).await.unwrap();
    // Input EOF flushes the second accepted turn; it is not cancellation.
    drop(input);
    let mut completed = Vec::new();
    while let Some(event) = tokio::time::timeout(DEADLINE, session.next())
        .await
        .unwrap()
    {
        serde_json::from_value::<RealtimeServerEvent>(event.clone()).unwrap();
        if event["type"] == "conversation.item.input_audio_transcription.completed" {
            assert_eq!(event["usage"]["total_tokens"], 6);
            completed.push(event["transcript"].clone());
        }
    }
    assert_eq!(completed, vec![json!("turn-0"), json!("turn-1")]);
    server.await.unwrap();
    client.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn asr_admits_committed_audio_before_a_later_uncommitted_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::new(
        format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        MODEL.into(),
        Limits::for_protocol(Protocol::VllmTranscription),
        None,
    )
    .unwrap();
    let input = futures::stream::iter([
        asr_session(),
        audio(),
        json!({"type": "input_audio_buffer.commit"}),
        json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(vec![1; 9600])}),
    ]).chain(futures::stream::pending()).boxed();
    let mut session = client.session(Protocol::VllmTranscription, input);
    let mut socket = tokio::time::timeout(DEADLINE, accept_asr(&listener))
        .await
        .unwrap();
    let first = receive(&mut socket).await;
    assert_eq!(first["type"], "input_audio_buffer.append");
    assert!(
        BASE64_STANDARD
            .decode(first["audio"].as_str().unwrap())
            .unwrap()
            .iter()
            .all(|sample| *sample == 0),
        "the committed first turn must acquire the only upstream slot"
    );
    while receive(&mut socket).await["type"] == "input_audio_buffer.append" {}
    send(
        &mut socket,
        json!({"type": "transcription.done", "text": "first"}),
    )
    .await;
    assert_eq!(next(&mut session).await["type"], "session.updated");
    assert_eq!(
        next(&mut session).await["type"],
        "input_audio_buffer.committed"
    );
    assert_eq!(next(&mut session).await["transcript"], "first");
    drop(session);
    client.shutdown().await;
}

#[tokio::test]
async fn asr_clear_acknowledgements_follow_prior_transcription_and_precede_later_turns() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::new(
        format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        MODEL.into(),
        Limits::for_protocol(Protocol::VllmTranscription),
        None,
    )
    .unwrap();
    let handled = Arc::new(Notify::new());
    let input_handled = handled.clone();
    let input = futures::stream::iter([
        asr_session(),
        audio(),
        json!({"type": "input_audio_buffer.commit"}),
        audio(),
        json!({"type": "input_audio_buffer.clear"}),
        json!({"type": "input_audio_buffer.clear"}),
        audio(),
        json!({"type": "input_audio_buffer.commit"}),
    ])
    .chain(futures::stream::once(async move {
        input_handled.notify_one();
        std::future::pending().await
    }))
    .boxed();
    let mut session = client.session(Protocol::VllmTranscription, input);
    tokio::time::timeout(DEADLINE, handled.notified())
        .await
        .unwrap();
    for turn in ["first", "last"] {
        let mut socket = tokio::time::timeout(DEADLINE, accept_asr(&listener))
            .await
            .unwrap();
        while receive(&mut socket).await["type"] == "input_audio_buffer.append" {}
        send(
            &mut socket,
            json!({"type": "transcription.done", "text": turn}),
        )
        .await;
        if turn == "first" {
            assert_eq!(next(&mut session).await["type"], "session.updated");
        }
        assert_eq!(
            next(&mut session).await["type"],
            "input_audio_buffer.committed"
        );
        assert_eq!(next(&mut session).await["transcript"], turn);
        if turn == "first" {
            for _ in 0..2 {
                assert_eq!(
                    next(&mut session).await["type"],
                    "input_audio_buffer.cleared"
                );
            }
        }
    }
    drop(session);
    client.shutdown().await;
}

#[tokio::test]
async fn omni_cleared_audio_does_not_open_an_upstream_connection_on_eof() {
    let (listener, client, input, mut session) = setup(Protocol::Omni).await;
    for event in [
        omni_session(),
        audio(),
        json!({"type": "input_audio_buffer.clear"}),
    ] {
        input.send(event).await.unwrap();
    }
    drop(input);
    for kind in [
        Some("session.updated"),
        Some("input_audio_buffer.cleared"),
        None,
    ] {
        let event = tokio::time::timeout(DEADLINE, async {
            tokio::select! {
                biased;
                _ = listener.accept() => panic!("cleared Omni audio must stay local"),
                event = session.next() => event,
            }
        })
        .await
        .unwrap();
        assert_eq!(
            event.as_ref().and_then(|event| event["type"].as_str()),
            kind
        );
    }
    client.shutdown().await;
}

#[tokio::test]
async fn clear_and_silent_stream_drop_close_upstream_connections() {
    let (listener, client, input, mut session) = setup(Protocol::VllmTranscription).await;
    let (opened_tx, mut opened_rx) = mpsc::channel(2);
    let (closed_tx, mut closed_rx) = mpsc::channel(2);
    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let mut socket = accept_asr(&listener).await;
            assert_eq!(
                receive(&mut socket).await["type"],
                "input_audio_buffer.append"
            );
            opened_tx.send(()).await.unwrap();
            while let Some(message) = socket.next().await {
                match message {
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => {}
                }
            }
            closed_tx.send(()).await.unwrap();
        }
    });
    input.send(asr_session()).await.unwrap();
    next(&mut session).await;
    input.send(audio()).await.unwrap();
    tokio::time::timeout(DEADLINE, opened_rx.recv())
        .await
        .unwrap()
        .unwrap();
    input
        .send(json!({"type": "input_audio_buffer.clear"}))
        .await
        .unwrap();
    assert_eq!(
        next(&mut session).await["type"],
        "input_audio_buffer.cleared"
    );
    tokio::time::timeout(DEADLINE, closed_rx.recv())
        .await
        .unwrap()
        .unwrap();
    input.send(audio()).await.unwrap();
    tokio::time::timeout(DEADLINE, opened_rx.recv())
        .await
        .unwrap()
        .unwrap();
    drop(session);
    tokio::time::timeout(DEADLINE, closed_rx.recv())
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    client.shutdown().await;
}

#[tokio::test]
async fn omni_turns_generate_concurrently_but_output_does_not_interleave() {
    let (listener, client, input, mut session) = setup(Protocol::Omni).await;
    let second_finished = Arc::new(Notify::new());
    let release_first = Arc::new(Notify::new());
    let server_second = second_finished.clone();
    let server_release = release_first.clone();
    let server = tokio::spawn(async move {
        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..2 {
            let (tcp, _) = listener.accept().await.unwrap();
            let second_finished = server_second.clone();
            let release_first = server_release.clone();
            tasks.spawn(async move {
                let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
                send(
                    &mut socket,
                    json!({"type": "session.created", "session": {}}),
                )
                .await;
                send(
                    &mut socket,
                    json!({"type": "conversation.created", "conversation": {"id": "upstream"}}),
                )
                .await;
                let update = receive(&mut socket).await;
                send(
                    &mut socket,
                    json!({"type": "session.updated", "session": update["session"]}),
                )
                .await;
                let audio = receive(&mut socket).await;
                assert_eq!(audio["type"], "input_audio_buffer.append");
                let index = BASE64_STANDARD
                    .decode(audio["audio"].as_str().unwrap())
                    .unwrap()[0];
                let modality = if index == 0 { "audio" } else { "text" };
                assert_eq!(update["session"]["output_modalities"], json!([modality]));
                assert_eq!(
                    receive(&mut socket).await["type"],
                    "input_audio_buffer.commit"
                );
                assert_eq!(receive(&mut socket).await["type"], "response.create");
                if index == 0 {
                    release_first.notified().await;
                }
                let delta_kind = if index == 0 {
                    "response.output_audio.delta"
                } else {
                    "response.output_text.delta"
                };
                for event in [
                    json!({"type": "response.created", "event_id": format!("created-{index}"),
                        "response": {"id": "upstream-response", "object": "realtime.response",
                        "output": [], "output_modalities": [modality], "status": "in_progress",
                        "max_output_tokens": "inf"}}),
                    json!({"type": delta_kind, "event_id": format!("delta-{index}"),
                        "response_id": "upstream-response", "item_id": format!("item-{index}"),
                        "output_index": 0, "content_index": 0,
                        "delta": if index == 0 { "AQI=" } else { "hello" }}),
                    json!({"type": "response.done", "event_id": format!("done-{index}"),
                        "response": {"id": "upstream-response", "object": "realtime.response",
                        "output": [], "output_modalities": [modality], "status": "completed",
                        "max_output_tokens": "inf"}}),
                ] {
                    send(&mut socket, event).await;
                }
                if index == 1 {
                    second_finished.notify_one();
                }
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
    });
    input.send(omni_session()).await.unwrap();
    next(&mut session).await;
    for index in 0..2 {
        let audio = json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(vec![index; 9600])});
        input.send(audio).await.unwrap();
        if index == 0 {
            // The open turn keeps its first-append snapshot; only the next
            // turn uses the newly accepted configuration.
            let mut update = omni_session();
            update["session"]["output_modalities"] = json!(["text"]);
            input.send(update).await.unwrap();
            assert_eq!(next(&mut session).await["type"], "session.updated");
        }
        input
            .send(json!({"type": "input_audio_buffer.commit"}))
            .await
            .unwrap();
    }
    drop(input);
    tokio::time::timeout(DEADLINE, second_finished.notified())
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), session.next())
            .await
            .is_err(),
        "later turns must stay buffered while the first is pending"
    );
    release_first.notify_one();
    let mut response_ids = Vec::new();
    for index in 0..2 {
        let created = next(&mut session).await;
        let delta = next(&mut session).await;
        let done = next(&mut session).await;
        assert_eq!(created["type"], "response.created");
        assert_eq!(done["type"], "response.done");
        assert_eq!(created["response"]["id"], done["response"]["id"]);
        assert_eq!(created["response"]["id"], delta["response_id"]);
        assert_eq!(delta["delta"], if index == 0 { "AQI=" } else { "hello" });
        response_ids.push(created["response"]["id"].clone());
    }
    assert_ne!(response_ids[0], response_ids[1]);
    assert!(
        tokio::time::timeout(DEADLINE, session.next())
            .await
            .unwrap()
            .is_none()
    );
    server.await.unwrap();
    client.shutdown().await;
}

#[tokio::test]
async fn omni_next_turn_uses_last_accepted_session_update() {
    let (listener, client, input, mut session) = setup(Protocol::Omni).await;
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
        send(
            &mut socket,
            json!({"type": "session.created", "session": {}}),
        )
        .await;
        let update = receive(&mut socket).await;
        assert_eq!(update["type"], "session.update");
        assert_eq!(update["session"]["model"], MODEL);
        assert_eq!(update["session"]["output_modalities"], json!(["text"]));
        assert_eq!(update["session"]["instructions"], "Keep replies short.");
        assert_eq!(update["session"]["max_output_tokens"], 123);
        assert!(update["session"].get("audio").is_none());
        send(
            &mut socket,
            json!({"type": "session.updated", "session": update["session"]}),
        )
        .await;
        assert_eq!(
            receive(&mut socket).await["type"],
            "input_audio_buffer.append"
        );
        assert_eq!(
            receive(&mut socket).await["type"],
            "input_audio_buffer.commit"
        );
        assert_eq!(receive(&mut socket).await["type"], "response.create");
        send(
            &mut socket,
            json!({"type": "response.done", "event_id": "done", "response": {
                "id": "upstream-response", "object": "realtime.response",
                "output": [], "output_modalities": ["text"], "status": "completed",
                "max_output_tokens": 123
            }}),
        )
        .await;
    });

    input
        .send(typed_client_event(
            json!({"type": "session.update", "session": {
                "type": "realtime", "model": MODEL, "output_modalities": ["text"],
                "instructions": "Keep replies short."
            }}),
        ))
        .await
        .unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    let patch = json!({"type": "session.update", "session": {
        "type": "realtime", "max_output_tokens": 123
    }});
    input.send(typed_client_event(patch)).await.unwrap();
    assert_eq!(next(&mut session).await["type"], "session.updated");
    input
        .send(typed_client_event(
            json!({"type": "session.update", "session": {
                "type": "realtime", "model": "other-model", "instructions": "discard"
            }}),
        ))
        .await
        .unwrap();
    assert_eq!(next(&mut session).await["error"]["code"], "invalid_session");
    input.send(typed_client_event(audio())).await.unwrap();
    drop(input);
    let done = next(&mut session).await;
    assert_eq!(done["type"], "response.done");
    assert_eq!(done["response"]["status"], "completed");
    assert_eq!(done["response"]["output_modalities"], json!(["text"]));
    assert!(
        tokio::time::timeout(DEADLINE, session.next())
            .await
            .unwrap()
            .is_none()
    );
    tokio::time::timeout(DEADLINE, server)
        .await
        .unwrap()
        .unwrap();
    client.shutdown().await;
}

#[tokio::test]
async fn omni_session_update_accepts_a_direct_acknowledgement() {
    let (listener, client, input, session) = setup(Protocol::Omni).await;
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
        assert_eq!(receive(&mut socket).await, omni_session());
        send(&mut socket, json!({"type": "session.updated"})).await;
    });
    let mut socket = client.connect().await.unwrap();
    client
        .update_omni_session(&mut socket, omni_session()["session"].clone())
        .await
        .unwrap();
    server.await.unwrap();
    drop(input);
    drop(session);
    client.shutdown().await;
}

#[tokio::test]
async fn omni_session_update_rejects_errors_unexpected_events_and_duplicate_greetings() {
    for events in [
        vec![json!({"type": "error"})],
        vec![json!({"type": "response.created"})],
        vec![
            json!({"type": "conversation.created"}),
            json!({"type": "conversation.created"}),
        ],
        vec![
            json!({"type": "conversation.created"}),
            json!({"type": "error"}),
        ],
    ] {
        let (listener, client, input, session) = setup(Protocol::Omni).await;
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
            assert_eq!(receive(&mut socket).await, omni_session());
            for event in events {
                send(&mut socket, event).await;
            }
        });
        let mut socket = client.connect().await.unwrap();
        assert!(matches!(
            client
                .update_omni_session(&mut socket, omni_session()["session"].clone())
                .await,
            Err(crate::Error::Protocol(_))
        ));
        server.await.unwrap();
        drop(input);
        drop(session);
        client.shutdown().await;
    }
}

#[tokio::test]
async fn omni_session_update_uses_one_deadline_across_the_greeting_and_acknowledgement() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::new(
        format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        MODEL.into(),
        Limits {
            connect_timeout: Duration::from_millis(200),
            ..Limits::for_protocol(Protocol::Omni)
        },
        None,
    )
    .unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(tcp).await.unwrap();
        assert_eq!(receive(&mut socket).await, omni_session());
        // Both receives would succeed under separate 200 ms deadlines. Their
        // combined duration exceeds the one allowed session-update deadline.
        tokio::time::sleep(Duration::from_millis(120)).await;
        send(&mut socket, json!({"type": "conversation.created"})).await;
        tokio::time::sleep(Duration::from_millis(120)).await;
        send(&mut socket, json!({"type": "session.updated"})).await;
    });
    let mut socket = client.connect().await.unwrap();
    assert!(matches!(
        client
            .update_omni_session(&mut socket, omni_session()["session"].clone())
            .await,
        Err(crate::Error::Timeout)
    ));
    tokio::time::timeout(DEADLINE, server)
        .await
        .unwrap()
        .unwrap();
    client.shutdown().await;
}

#[tokio::test]
async fn malformed_audio_is_recoverable_and_upstream_failure_is_not_success() {
    let (listener, client, input, mut session) = setup(Protocol::VllmTranscription).await;
    input.send(asr_session()).await.unwrap();
    next(&mut session).await;
    input.send(json!({"type": "input_audio_buffer.append", "audio": "invalid!", "event_id": "bad_audio"})).await.unwrap();
    let error = next(&mut session).await;
    assert_eq!(error["error"]["event_id"], "bad_audio");
    assert_eq!(error["error"]["code"], "invalid_audio");
    let server = tokio::spawn(async move {
        let mut socket = accept_asr(&listener).await;
        receive(&mut socket).await;
        send(
            &mut socket,
            json!({"type": "error", "code": "engine_error", "error": "failed"}),
        )
        .await;
    });
    input.send(audio()).await.unwrap();
    drop(input);
    let mut failed = false;
    while let Some(event) = tokio::time::timeout(DEADLINE, session.next())
        .await
        .unwrap()
    {
        serde_json::from_value::<RealtimeServerEvent>(event.clone()).unwrap();
        assert_ne!(
            event["type"],
            "conversation.item.input_audio_transcription.completed"
        );
        if event["type"] == "conversation.item.input_audio_transcription.failed" {
            failed = true;
        }
    }
    assert!(failed);
    server.await.unwrap();
    client.shutdown().await;
}

#[test]
fn audio_resampling_is_chunk_invariant_and_filters_above_nyquist() {
    let waveform = |hz: f32| {
        (0..24000)
            .flat_map(|n| {
                let sample =
                    ((n as f32 * hz * std::f32::consts::TAU / 24000.0).sin() * 16000.0) as i16;
                sample.to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let resample = |audio: &[u8], size| {
        let mut converter = AsrAudio::new().unwrap();
        let mut output = Vec::new();
        for chunk in audio.chunks(size) {
            output.extend(converter.append(chunk).unwrap());
        }
        output.extend(converter.finish().unwrap());
        output
    };
    let low = waveform(1000.0);
    let single = resample(&low, low.len());
    assert_eq!(single.len(), 32000);
    assert_eq!(single, resample(&low, 142));
    let high = resample(&waveform(10000.0), 960);
    let rms = |audio: &[u8]| {
        let energy: f32 = audio
            .chunks_exact(2)
            .skip(100)
            .take(15800)
            .map(|x| (i16::from_le_bytes([x[0], x[1]]) as f32 / 32768.0).powi(2))
            .sum();
        (energy / 15800.0).sqrt()
    };
    assert!(rms(&single) > 0.3);
    assert!(rms(&high) < 0.01, "downsampling must attenuate aliases");
}

#[tokio::test]
async fn wss_connections_reuse_native_trust_without_requiring_it_for_http() {
    const CHILD_URL: &str = "DYNAMO_REALTIME_TEST_WSS_URL";
    if let Ok(url) = std::env::var(CHILD_URL) {
        let client = Client::new(
            url.parse().unwrap(),
            MODEL.into(),
            Limits {
                connect_timeout: DEADLINE,
                close_timeout: DEADLINE,
                ..Limits::for_protocol(Protocol::Omni)
            },
            None,
        )
        .unwrap();
        for turn in 0..2 {
            let mut socket = client.clone().connect().await.unwrap();
            if turn == 0 {
                std::fs::remove_file(std::env::var_os("SSL_CERT_FILE").unwrap()).unwrap();
            }
            assert_eq!(
                crate::client::receive(&mut socket).await.unwrap()["type"],
                "session.created"
            );
            client.close(&mut socket).await;
        }
        client.shutdown().await;
        // Speech uses HTTP only and must not require a native WSS trust store.
        Client::new(
            url.parse().unwrap(),
            MODEL.into(),
            Limits::for_protocol(Protocol::Omni),
            None,
        )
        .unwrap()
        .shutdown()
        .await;
        return;
    }

    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let cert_path = directory.path().join("ca.pem");
    std::fs::write(&cert_path, cert.pem()).unwrap();
    let empty_cert_dir = directory.path().join("empty");
    std::fs::create_dir(&empty_cert_dir).unwrap();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![cert.der().clone()],
        rustls::pki_types::PrivatePkcs8KeyDer::from(key_pair.serialize_der()).into(),
    )
    .unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "https://localhost:{}",
        listener.local_addr().unwrap().port()
    );
    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let (tcp, _) = listener.accept().await.unwrap();
            let tls = acceptor.accept(tcp).await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(tls).await.unwrap();
            socket
                .send(Message::Text(
                    json!({"type": "session.created"}).to_string().into(),
                ))
                .await
                .unwrap();
            assert!(matches!(socket.next().await, Some(Ok(Message::Close(_)))));
        }
    });

    // Isolate trust-store changes. Process startup gets its own deadline in
    // addition to the child's bounded connection/close operations.
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::wss_connections_reuse_native_trust_without_requiring_it_for_http",
                "--nocapture",
            ])
            .env(CHILD_URL, url)
            .env("SSL_CERT_FILE", cert_path)
            .env("SSL_CERT_DIR", empty_cert_dir)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    tokio::time::timeout(DEADLINE, server)
        .await
        .unwrap()
        .unwrap();
}
