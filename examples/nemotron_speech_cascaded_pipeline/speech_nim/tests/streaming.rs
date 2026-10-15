// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_backend_common::{
    BackendError, ErrorType, GenerateContext, RawEngine, testing::mock_context,
};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_speech_nim_sidecar::{
    asr::AsrAdapter,
    client,
    config::Mode,
    proto::{
        AudioEncoding, asr::streaming_recognize_request::StreamingRequest,
        tts::SynthesizeSpeechResponse,
    },
    tts::TtsEngine,
};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::task::AbortOnDropHandle;
use tonic::Status;

use support::{Server, transcript, within};

fn session_update() -> Value {
    json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "nemotron-asr-streaming", "language": "en"},
        "turn_detection": null,
    }}}})
}

fn append() -> Value {
    json!({"type": "input_audio_buffer.append", "audio": STANDARD.encode([1, 0, 2, 0])})
}
fn commit() -> Value {
    json!({"type": "input_audio_buffer.commit"})
}

struct Session {
    input: mpsc::UnboundedSender<Value>,
    output: mpsc::Receiver<Value>,
    context: Arc<dyn AsyncEngineContext>,
    task: AbortOnDropHandle<()>,
}

impl Session {
    async fn start(adapter: &AsrAdapter) -> Self {
        let (input, rx) = mpsc::unbounded_channel();
        let (tx, output) = mpsc::channel(1024);
        let context = mock_context();
        let mut responses =
            adapter.session(UnboundedReceiverStream::new(rx).boxed(), context.clone());
        let task = AbortOnDropHandle::new(tokio::spawn(async move {
            while let Some(event) = responses.next().await {
                if tx.send(event).await.is_err() {
                    break;
                }
            }
        }));
        let mut result = Self {
            input,
            output,
            context,
            task,
        };
        result.send(session_update());
        assert_eq!(result.next().await["type"], "session.updated");
        result
    }

    fn send(&self, event: Value) {
        self.input.send(event).unwrap();
    }
    async fn next(&mut self) -> Value {
        within(self.output.recv())
            .await
            .expect("session ended early")
    }
}

#[tokio::test]
async fn asr_streams_before_commit_and_half_close_drains_results() {
    let mut server = Server::start().await;
    let mut config = server.asr_config();
    config.commit_padding_ms = 10;
    config.nim_model = "internal-model".into();
    let adapter = AsrAdapter::new(config, server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let mut call = within(server.asr.recv()).await.unwrap();
    let first = within(call.request.get_mut().message())
        .await
        .unwrap()
        .unwrap();
    let Some(StreamingRequest::StreamingConfig(config)) = first.streaming_request else {
        panic!("config must precede audio")
    };
    assert!(config.interim_results);
    let config = config.config.unwrap();
    assert_eq!(config.sample_rate_hertz, 24000);
    assert_eq!(config.encoding, AudioEncoding::LinearPcm as i32);
    assert_eq!(config.model, "internal-model");
    assert!(config.enable_automatic_punctuation);
    assert_eq!(
        within(call.request.get_mut().message())
            .await
            .unwrap()
            .unwrap()
            .streaming_request,
        Some(StreamingRequest::AudioContent(vec![1, 0, 2, 0]))
    );
    call.responses
        .send(Ok(transcript("hello", false)))
        .await
        .unwrap();
    assert_eq!(session.next().await["delta"], "hello");
    // Revisions do not retract emitted deltas. The final carries canonical text.
    call.responses
        .send(Ok(transcript("hullo", true)))
        .await
        .unwrap();
    call.responses
        .send(Ok(transcript("world", true)))
        .await
        .unwrap();
    assert_eq!(session.next().await["delta"], " world");
    session.send(commit());
    let committed = session.next().await;
    assert_eq!(committed["type"], "input_audio_buffer.committed");
    assert_eq!(
        within(call.request.get_mut().message())
            .await
            .unwrap()
            .unwrap()
            .streaming_request,
        Some(StreamingRequest::AudioContent(vec![0; 480]))
    );
    assert!(
        within(call.request.get_mut().message())
            .await
            .unwrap()
            .is_none()
    );
    drop(session.input);
    drop(call.responses);
    let completed = within(session.output.recv()).await.unwrap();
    assert_eq!(completed["transcript"], "hullo world");
    assert_eq!(completed["item_id"], committed["item_id"]);
    assert_eq!(completed["usage"]["total_tokens"], 0);
    assert!(within(session.output.recv()).await.is_none());
}

#[tokio::test]
async fn asr_eof_preserves_the_latest_interim_transcript() {
    for prefix in ["hello", ""] {
        let mut server = Server::start().await;
        let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
        let mut session = Session::start(&adapter).await;
        session.send(append());
        let call = within(server.asr.recv()).await.unwrap();
        session.send(commit());
        assert_eq!(session.next().await["type"], "input_audio_buffer.committed");
        if !prefix.is_empty() {
            call.responses
                .send(Ok(transcript(prefix, true)))
                .await
                .unwrap();
            assert_eq!(session.next().await["delta"], prefix);
        }
        call.responses
            .send(Ok(transcript("word", false)))
            .await
            .unwrap();
        assert_eq!(
            session.next().await["delta"],
            if prefix.is_empty() { "word" } else { " word" }
        );
        call.responses
            .send(Ok(transcript("world", false)))
            .await
            .unwrap();
        drop(call.responses);
        assert_eq!(
            session.next().await["transcript"],
            if prefix.is_empty() {
                "world"
            } else {
                "hello world"
            }
        );
    }
}

#[tokio::test]
async fn asr_input_eof_commits_open_turn_without_cancelling_rpc() {
    let mut server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let mut call = within(server.asr.recv()).await.unwrap();
    drop(session.input);
    while within(call.request.get_mut().message())
        .await
        .unwrap()
        .is_some()
    {}
    call.responses
        .send(Ok(transcript("half closed", true)))
        .await
        .unwrap();
    drop(call.responses);
    let events: Vec<_> = within(async {
        let mut result = Vec::new();
        while let Some(v) = session.output.recv().await {
            result.push(v);
        }
        result
    })
    .await;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["delta"], "half closed");
    assert_eq!(events[1]["transcript"], "half closed");
}

#[tokio::test]
async fn asr_clear_cancels_uncommitted_turn_and_session_recovers() {
    let mut server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    session.send(json!({"type": "input_audio_buffer.clear"}));
    assert_eq!(session.next().await["type"], "input_audio_buffer.cleared");
    within(call.responses.closed()).await;
    session.send(append());
    session.send(commit());
    let call = within(server.asr.recv()).await.unwrap();
    assert_eq!(session.next().await["type"], "input_audio_buffer.committed");
    call.responses
        .send(Ok(transcript("next turn", true)))
        .await
        .unwrap();
    drop(call.responses);
    assert_eq!(session.next().await["delta"], "next turn");
    assert_eq!(session.next().await["transcript"], "next turn");
}

#[tokio::test]
async fn asr_failure_before_commit_releases_input_and_capacity() {
    let mut server = Server::start().await;
    let mut config = server.asr_config();
    config.max_concurrent_turns = 1.try_into().unwrap();
    let adapter = AsrAdapter::new(config, server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    call.responses
        .send(Err(Status::unavailable("fake failure")))
        .await
        .unwrap();
    assert_eq!(
        session.next().await["error"]["message"],
        "Transcription failed"
    );
    within(call.responses.closed()).await;
    session.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    session.send(commit());
    assert_eq!(session.next().await["type"], "input_audio_buffer.committed");
    drop(call.responses);
    assert_eq!(session.next().await["transcript"], "");
}

#[tokio::test]
async fn asr_global_capacity_is_fail_fast_and_stop_releases_it() {
    let mut server = Server::start().await;
    let mut config = server.asr_config();
    config.max_concurrent_turns = 1.try_into().unwrap();
    let adapter = AsrAdapter::new(config, server.connection().await);
    let first = Session::start(&adapter).await;
    first.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    let mut second = Session::start(&adapter).await;
    second.send(append());
    assert_eq!(
        second.next().await["error"]["message"],
        "ASR capacity exhausted; retry the turn later"
    );
    first.context.stop_generating();
    within(call.responses.closed()).await;
    second.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    drop(second.task);
    within(call.responses.closed()).await;
}

#[tokio::test]
async fn asr_completion_deadline_starts_after_commit() {
    let mut server = Server::start().await;
    let mut config = server.asr_config();
    config.timeout_s = Duration::from_millis(40);
    let adapter = AsrAdapter::new(config, server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let call = within(server.asr.recv()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(!call.responses.is_closed());
    assert!(session.output.try_recv().is_err());
    session.send(commit());
    assert_eq!(session.next().await["type"], "input_audio_buffer.committed");
    assert_eq!(
        session.next().await["error"]["message"],
        "Transcription failed"
    );
    within(call.responses.closed()).await;
}

#[tokio::test]
async fn asr_committed_turns_remain_ordered_when_next_input_arrives() {
    let mut server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    let mut session = Session::start(&adapter).await;
    session.send(append());
    let first = within(server.asr.recv()).await.unwrap();
    session.send(commit());
    let first_id = session.next().await["item_id"].clone();
    session.send(append());
    session.send(commit());
    first
        .responses
        .send(Ok(transcript("first", true)))
        .await
        .unwrap();
    drop(first.responses);
    let second = within(server.asr.recv()).await.unwrap();
    second
        .responses
        .send(Ok(transcript("second", true)))
        .await
        .unwrap();
    drop(second.responses);
    assert_eq!(session.next().await["delta"], "first");
    assert_eq!(session.next().await["item_id"], first_id);
    let second_id = session.next().await["item_id"].clone();
    assert_ne!(first_id, second_id);
    assert_eq!(session.next().await["delta"], "second");
    let completed = session.next().await;
    assert_eq!(completed["transcript"], "second");
    assert_eq!(completed["item_id"], second_id);
}

#[tokio::test]
async fn asr_validates_session_and_audio_without_breaking_connection() {
    let server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    let mut session = Session::start(&adapter).await;
    for (audio, message) in [
        (json!(42), "audio must be a base64 string"),
        (json!("!"), "audio must be valid base64"),
        (json!(""), "audio chunk is empty"),
        (json!("AA=="), "PCM16 audio must be 2-byte aligned"),
    ] {
        session.send(
            json!({"type": "input_audio_buffer.append", "audio": audio, "event_id": "bad-audio"}),
        );
        let error = session.next().await;
        assert_eq!(error["error"]["message"], message);
        assert_eq!(error["error"]["event_id"], "bad-audio");
    }
    let mut invalid = session_update();
    invalid["session"]["audio"]["input"]["turn_detection"] = json!({"type": "server_vad"});
    session.send(invalid);
    assert_eq!(
        session.next().await["error"]["message"],
        "server turn detection is not supported; use local VAD"
    );
    session.send(commit());
    assert_eq!(
        session.next().await["error"]["message"],
        "input audio buffer is empty"
    );
}

#[tokio::test]
async fn tts_preserves_pcm_chunks_and_config() {
    let mut server = Server::start().await;
    let mut config = server.tts_config();
    config.sample_rate_hz = 16000;
    let engine = TtsEngine::new(config, server.connection().await);
    let card = engine.start(0).await.unwrap();
    assert_eq!(card.model, "nvidia/magpie-tts-multilingual");
    assert!(card.llm.is_none());
    let mut stream = engine
        .generate(
            json!({"input": "Hello", "voice": "custom", "language": "de-DE", "speed": 1,
        "response_format": "pcm", "data_source": "b64_json"}),
            GenerateContext::new(mock_context(), None),
        )
        .await
        .unwrap();
    let output = tokio::spawn(async move {
        let first = stream.next().await.unwrap().unwrap();
        (first, stream)
    });
    let mut call = within(server.tts.recv()).await.unwrap();
    let request = within(call.request.get_mut().message())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(request.text, "Hello");
    assert_eq!(request.voice_name, "custom");
    assert_eq!(request.language_code, "de-DE");
    assert_eq!(request.sample_rate_hz, 16000);
    assert_eq!(request.encoding, AudioEncoding::LinearPcm as i32);
    assert!(
        within(call.request.get_mut().message())
            .await
            .unwrap()
            .is_none()
    );
    call.responses
        .send(Ok(SynthesizeSpeechResponse {
            audio: vec![1, 0, 2, 0],
            ..Default::default()
        }))
        .await
        .unwrap();
    let (first, mut stream) = within(output).await.unwrap();
    assert_eq!(first["object"], "audio.speech");
    assert_eq!(first["data"][0]["b64_json"], "AQACAA==");
    call.responses
        .send(Ok(SynthesizeSpeechResponse {
            audio: vec![3, 0],
            ..Default::default()
        }))
        .await
        .unwrap();
    let second = within(stream.next()).await.unwrap().unwrap();
    assert_eq!(first["id"], second["id"]);
    assert_eq!(second["data"][0]["b64_json"], "AwA=");
    drop(call.responses);
    assert!(within(stream.next()).await.is_none());
}

#[tokio::test]
async fn tts_requires_explicit_pcm_format_before_rpc() {
    let mut server = Server::start().await;
    let engine = TtsEngine::new(server.tts_config(), server.connection().await);
    for request in [
        json!({"input": "Hello"}),
        json!({"input": "Hello", "response_format": null}),
        json!({"input": "Hello", "response_format": "wav"}),
    ] {
        let Err(error) = engine
            .generate(request, GenerateContext::new(mock_context(), None))
            .await
        else {
            panic!("non-PCM format was silently accepted")
        };
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(error.to_string().contains("response_format='pcm'"));
        assert!(server.tts.try_recv().is_err());
    }
}

#[tokio::test]
async fn tts_rejects_unsupported_options_before_rpc() {
    let server = Server::start().await;
    let engine = TtsEngine::new(server.tts_config(), server.connection().await);
    for (field, value) in [
        ("input", json!(" ")),
        ("model", json!("wrong")),
        ("speed", json!(2)),
        ("data_source", json!("url")),
        ("instructions", json!("whisper")),
        ("task_type", json!("Base")),
        ("ref_audio", json!("")),
        ("ref_text", json!("")),
        ("max_new_tokens", json!(0)),
        ("nvext", json!({"cfg_scale": 1.0})),
    ] {
        let mut request = json!({"input": "Hello", "response_format": "pcm"});
        request[field] = value;
        let result = engine
            .generate(request, GenerateContext::new(mock_context(), None))
            .await;
        let Err(error) = result else {
            panic!("{field} was silently accepted")
        };
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}

#[tokio::test]
async fn tts_maps_invalid_argument_and_cancels_open_rpc_on_stop() {
    let mut server = Server::start().await;
    let engine = TtsEngine::new(server.tts_config(), server.connection().await);
    let ctx = mock_context();
    let stream = engine
        .generate(
            json!({"input": "Hello", "response_format": "pcm"}),
            GenerateContext::new(ctx.clone(), None),
        )
        .await
        .unwrap();
    let output = tokio::spawn(stream.collect::<Vec<_>>());
    let call = within(server.tts.recv()).await.unwrap();
    ctx.stop_generating();
    assert!(within(output).await.unwrap().is_empty());
    within(call.responses.closed()).await;

    let stream = engine
        .generate(
            json!({"input": "Hello", "response_format": "pcm"}),
            GenerateContext::new(mock_context(), None),
        )
        .await
        .unwrap();
    let output = tokio::spawn(stream.collect::<Vec<_>>());
    let call = within(server.tts.recv()).await.unwrap();
    call.responses
        .send(Err(Status::invalid_argument("server-internal-details")))
        .await
        .unwrap();
    let output = within(output).await.unwrap();
    let error = output[0].as_ref().unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(!error.to_string().contains("server-internal-details"));
}

#[tokio::test]
async fn asr_requires_a_supported_initial_session_event() {
    let server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    for (event, code) in [
        (json!(null), "invalid_event"),
        (append(), "invalid_event"),
        (
            json!({"type": "session.update", "session": {"type": "conversation"}}),
            "unsupported_session",
        ),
    ] {
        let result = within(
            adapter
                .session(futures::stream::iter(vec![event]).boxed(), mock_context())
                .collect::<Vec<_>>(),
        )
        .await;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["error"]["code"], code);
    }
}

#[tokio::test]
async fn asr_cancels_even_if_transcript_output_is_backpressured() {
    let mut server = Server::start().await;
    let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
    let (input, rx) = mpsc::unbounded_channel();
    let ctx = mock_context();
    let mut output = adapter.session(UnboundedReceiverStream::new(rx).boxed(), ctx.clone());
    input.send(session_update()).unwrap();
    assert_eq!(
        within(output.next()).await.unwrap()["type"],
        "session.updated"
    );
    input.send(append()).unwrap();
    let call = within(server.asr.recv()).await.unwrap();
    // Stop polling output after session.updated. Fill the per-turn queue, then
    // verify cancellation independently of the response consumer advancing.
    let producer = call.responses.clone();
    let produce = AbortOnDropHandle::new(tokio::spawn(async move {
        for len in 1..=400 {
            if producer
                .send(Ok(transcript(&"a".repeat(len), false)))
                .await
                .is_err()
            {
                break;
            }
        }
    }));
    within(produce).await.unwrap();
    ctx.kill();
    within(call.responses.closed()).await;
    assert!(within(output.next()).await.is_none());
}

// More control acknowledgements than the bounded queue can hold must not
// prevent the input pump from reaching a commit or clear.
#[tokio::test]
async fn asr_control_backpressure_does_not_block_commit_or_clear() {
    for clear in [false, true] {
        let mut server = Server::start().await;
        let adapter = AsrAdapter::new(server.asr_config(), server.connection().await);
        let mut session = Session::start(&adapter).await;
        session.send(append());
        let mut call = within(server.asr.recv()).await.unwrap();
        for _ in 0..2 {
            within(call.request.get_mut().message())
                .await
                .unwrap()
                .unwrap();
        }
        for _ in 0..128 {
            session.send(session_update());
        }
        if clear {
            session.send(json!({"type": "input_audio_buffer.clear"}));
        } else {
            session.send(commit());
        }
        let expected = if clear {
            "input_audio_buffer.cleared"
        } else {
            "input_audio_buffer.committed"
        };
        let mut updates = 0;
        let mut handled = false;
        for _ in 0..129 {
            let event = session.next().await;
            if event["type"] == "session.updated" {
                updates += 1;
            } else {
                assert_eq!(event["type"], expected);
                assert!(!handled);
                handled = true;
            }
        }
        assert_eq!(updates, 128);
        assert!(handled);
        if clear {
            within(call.responses.closed()).await;
        } else {
            assert!(
                within(call.request.get_mut().message())
                    .await
                    .unwrap()
                    .is_none()
            );
            call.responses
                .send(Ok(transcript("still responsive", true)))
                .await
                .unwrap();
            drop(call.responses);
            assert_eq!(session.next().await["delta"], "still responsive");
            assert_eq!(session.next().await["transcript"], "still responsive");
        }
        session.context.kill();
    }
}

#[tokio::test]
async fn startup_waits_for_model_service_and_honors_deadline() {
    for tts in [false, true] {
        let server = Server::start().await;
        let mode = if tts {
            Mode::Tts(server.tts_config())
        } else {
            Mode::Asr(server.asr_config())
        };
        server.ready.store(false, Ordering::SeqCst);
        let connect = client::connect(&mode);
        tokio::pin!(connect);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut connect)
                .await
                .is_err()
        );
        server.ready.store(true, Ordering::SeqCst);
        let channels = within(connect).await.unwrap();
        assert_eq!(
            channels.len(),
            mode.worker().connection.grpc.grpc_connections.get()
        );
        server.ready.store(false, Ordering::SeqCst);
        let error = within(client::connect(&mode))
            .await
            .err()
            .expect("empty model service must time out");
        assert!(error.to_string().contains("startup deadline"), "{error:#}");
    }
}
