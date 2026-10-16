// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_protocols::types::realtime::RealtimeServerEvent;
use dynamo_realtime_ws_sidecar::{Bridge, Profile};
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{ManyIn, RequestStream, context::Context},
};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn bridge(profile: Profile, url: String) -> Bridge {
    Bridge {
        profile,
        upstream_url: url,
        model: "test-model".into(),
        input_rate: 24000,
        max_output_tokens: None,
    }
}

async fn listener() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/v1/realtime", listener.local_addr().unwrap());
    (listener, url)
}

fn input(events: Vec<Value>) -> ManyIn<Value> {
    Context::new(RequestStream::new(Box::pin(tokio_stream::iter(events))))
}

#[tokio::test]
async fn asr_translates_audio_and_commits_then_terminates() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        ws.send(Message::Text(
            json!({"type":"session.created","id":"native"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        let mut requests = vec![];
        while let Some(Ok(Message::Text(text))) = ws.next().await {
            let frame: Value = serde_json::from_str(&text).unwrap();
            let done = frame["type"] == "input_audio_buffer.commit" && frame["final"] == true;
            requests.push(frame);
            if done {
                break;
            }
        }
        assert_eq!(
            requests[0],
            json!({"type":"session.update","model":"test-model"})
        );
        assert_eq!(
            requests[1],
            json!({"type":"input_audio_buffer.commit","final":false})
        );
        assert_eq!(
            STANDARD
                .decode(requests[2]["audio"].as_str().unwrap())
                .unwrap()
                .len(),
            3200
        );
        ws.send(Message::Text(
            json!({"type":"transcription.delta","delta":"hello"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        ws.send(Message::Text(json!({"type":"transcription.done","text":"hello","usage":{"prompt_tokens":4,"completion_tokens":1}}).to_string().into())).await.unwrap();
        assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
    });
    let frames = vec![
        json!({"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"test-model"},"turn_detection":null}}}}),
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0;4800])}),
        json!({"type":"input_audio_buffer.commit"}),
    ];
    let result = bridge(Profile::Asr, url)
        .generate(input(frames))
        .await
        .unwrap();
    let events: Vec<_> = tokio::time::timeout(Duration::from_secs(3), result.collect())
        .await
        .unwrap();
    assert!(events.iter().all(|event| !event.is_error()), "{events:?}");
    for event in &events {
        serde_json::from_value::<RealtimeServerEvent>(event.data.clone().unwrap()).unwrap();
    }
    let kinds: Vec<_> = events
        .iter()
        .map(|event| event.data.as_ref().unwrap()["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "session.updated",
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed"
        ]
    );
    assert_eq!(events[3].data.as_ref().unwrap()["usage"]["input_tokens"], 4);
    server.await.unwrap();
}

#[tokio::test]
async fn tts_maps_binary_pcm_to_frontend_speech_chunks() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let mut requests = vec![];
        for _ in 0..3 {
            let Message::Text(text) = ws.next().await.unwrap().unwrap() else {
                panic!("expected JSON")
            };
            requests.push(serde_json::from_str::<Value>(&text).unwrap());
        }
        assert_eq!(requests[0]["stream_audio"], true);
        assert_eq!(requests[0]["seed"], 42);
        assert_eq!(requests[1], json!({"type":"input.text","text":"Hello"}));
        assert_eq!(requests[2], json!({"type":"input.done"}));
        ws.send(Message::Text(
            json!({"type":"audio.start","format":"pcm","sample_rate":24000})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        ws.send(Message::Binary(vec![1, 0, 2, 0].into()))
            .await
            .unwrap();
        ws.send(Message::Text(
            json!({"type":"audio.done"}).to_string().into(),
        ))
        .await
        .unwrap();
        ws.send(Message::Text(
            json!({"type":"session.done"}).to_string().into(),
        ))
        .await
        .unwrap();
        assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
    });
    let request = Context::new(
        json!({"input":"Hello","voice":"vivian","language":"English","response_format":"pcm", "extra_args":{"media_passthrough":{"seed":42,"stream":true}}}),
    );
    let result = bridge(Profile::Tts, url).generate(request).await.unwrap();
    let events: Vec<_> = tokio::time::timeout(Duration::from_secs(3), result.collect())
        .await
        .unwrap();
    assert_eq!(events.len(), 2, "{events:?}");
    for event in &events {
        assert!(!event.is_error());
        serde_json::from_value::<dynamo_llm::protocols::openai::audios::NvAudioSpeechResponse>(
            event.data.clone().unwrap(),
        )
        .unwrap();
    }
    assert_eq!(
        events[0].data.as_ref().unwrap()["data"][0]["b64_json"],
        "AQACAA=="
    );
    assert_eq!(events[1].data.as_ref().unwrap()["status"], "completed");
    server.await.unwrap();
}

#[tokio::test]
async fn cancellation_closes_idle_upstream_socket() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
    });
    let request: ManyIn<Value> =
        Context::new(RequestStream::new(Box::pin(futures::stream::pending())));
    let ctx = request.context();
    let result = bridge(Profile::Asr, url).generate(request).await.unwrap();
    ctx.stop_generating();
    let events: Vec<_> = tokio::time::timeout(Duration::from_secs(3), result.collect())
        .await
        .unwrap();
    assert!(events.is_empty());
    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn omni_resamples_and_creates_response_after_commit() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        ws.send(Message::Text(
            json!({"type":"conversation.created","event_id":"legacy-conversation","conversation":{"id":"native-conversation","object":"realtime.conversation"}})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        let mut requests = vec![];
        for _ in 0..4 {
            let Message::Text(text) = ws.next().await.unwrap().unwrap() else {
                panic!("expected JSON")
            };
            requests.push(serde_json::from_str::<Value>(&text).unwrap());
        }
        assert_eq!(
            requests[0]["session"]["audio"]["input"]["format"]["rate"],
            24000
        );
        assert_eq!(
            requests[0]["session"]["output_modalities"],
            json!(["audio"])
        );
        assert_eq!(
            STANDARD
                .decode(requests[1]["audio"].as_str().unwrap())
                .unwrap()
                .len(),
            4800
        );
        assert_eq!(requests[2]["type"], "input_audio_buffer.commit");
        assert_eq!(requests[3]["type"], "response.create");
        let events = [
            json!({"type":"response.output_audio.delta","event_id":"audio","response_id":"r","item_id":"i","output_index":0,"content_index":0,"delta":"AQACAA=="}),
            json!({"type":"response.done","event_id":"done","response":{"id":"r","object":"realtime.response","max_output_tokens":128,"output":[],"output_modalities":["audio"],"status":"completed"}}),
        ];
        for event in events {
            ws.send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
    });
    let mut config = bridge(Profile::Omni, url);
    config.input_rate = 16000;
    let frames = vec![
        json!({"type":"session.update","session":{"type":"realtime","model":"test-model","output_modalities":["audio","text"]}}),
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0;3200])}),
        json!({"type":"input_audio_buffer.commit"}),
    ];
    let result = config.generate(input(frames)).await.unwrap();
    let events: Vec<_> = tokio::time::timeout(Duration::from_secs(3), result.collect())
        .await
        .unwrap();
    assert_eq!(events.len(), 3, "{events:?}");
    for event in events {
        assert!(!event.is_error(), "{event:?}");
        serde_json::from_value::<RealtimeServerEvent>(event.data.unwrap()).unwrap();
    }
    server.await.unwrap();
}
