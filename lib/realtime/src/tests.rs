// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_protocols::types::realtime::RealtimeServerEvent;
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
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::new(
        format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        MODEL.into(),
        Limits::for_protocol(protocol),
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
        if index == 1 {
            let mut update = omni_session();
            update["session"]["output_modalities"] = json!(["text"]);
            input.send(update).await.unwrap();
            assert_eq!(next(&mut session).await["type"], "session.updated");
        }
        let audio = json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(vec![index; 9600])});
        input.send(audio).await.unwrap();
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
