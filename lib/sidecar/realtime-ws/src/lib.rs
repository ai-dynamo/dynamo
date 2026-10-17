// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental one-utterance adapters. Dynamo still terminates the public API,
//! discovers this worker and carries requests over its normal runtime transport.
//! Only the worker-to-engine boundary changes to stock upstream WebSockets.

mod audio;

use std::time::Duration;

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{ManyIn, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config, tungstenite::Message,
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
const IO_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Profile {
    Asr,
    Omni,
    Tts,
}

#[derive(Clone, Debug)]
pub struct Bridge {
    pub profile: Profile,
    pub upstream_url: String,
    pub model: String,
    pub input_rate: u32,
    pub max_output_tokens: Option<u32>,
}

fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn event(mut value: Value) -> Value {
    value["event_id"] = Value::String(id("event"));
    value
}

async fn send(socket: &mut Socket, value: Value) -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(10),
        socket.send(Message::Text(value.to_string().into())),
    )
    .await
    .context("upstream write timed out")??;
    Ok(())
}

impl Bridge {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.upstream_url.starts_with("ws://"),
            "prototype supports ws:// upstream URLs only"
        );
        ensure!(
            matches!(self.input_rate, 16000 | 24000),
            "input rate must be 16000 or 24000"
        );
        Ok(())
    }

    async fn connect(&self) -> Result<Socket> {
        self.validate()?;
        // Realtime controls are small writes; Nagle can delay their delivery
        // while waiting for an ACK of the previous frame.
        let (socket, _) = tokio::time::timeout(
            Duration::from_secs(10),
            connect_async_with_config(&self.upstream_url, None, true),
        )
        .await
        .context("upstream WS handshake timed out")??;
        Ok(socket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn upstream_connection_disables_nagle() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            tokio_tungstenite::accept_async(stream).await.unwrap()
        });
        let bridge = Bridge {
            profile: Profile::Tts,
            upstream_url: format!("ws://{addr}/v1/audio/speech/stream"),
            model: "test-model".into(),
            input_rate: 24000,
            max_output_tokens: None,
        };
        let socket = bridge.connect().await.unwrap();
        let MaybeTlsStream::Plain(tcp) = socket.get_ref() else {
            panic!("expected plaintext test connection");
        };
        assert!(tcp.nodelay().unwrap());
        drop(server.await.unwrap());
    }
}

struct Session {
    bridge: Bridge,
    audio: audio::AudioConverter,
    started: bool,
    committed: bool,
    item_id: String,
}

impl Session {
    fn new(bridge: Bridge) -> Self {
        let output_rate = if bridge.profile == Profile::Asr {
            16000
        } else {
            24000
        };
        Self {
            audio: audio::AudioConverter::new(bridge.input_rate, output_rate),
            bridge,
            started: false,
            committed: false,
            item_id: id("item"),
        }
    }

    // Returns (upstream frames, downstream frames). Session/control semantics
    // are owned here; model-specific processing stays in the upstream server.
    fn input(&mut self, frame: Value) -> Result<(Vec<Value>, Vec<Value>)> {
        let mut upstream = vec![];
        let mut downstream = vec![];
        match frame["type"].as_str().unwrap_or_default() {
            "session.update" => {
                ensure!(
                    !self.started,
                    "prototype does not support session updates after audio starts"
                );
                let mut session = frame["session"].clone();
                ensure!(
                    session.is_object(),
                    "session.update requires a session object"
                );
                let rate = session
                    .pointer("/audio/input/format/rate")
                    .and_then(Value::as_u64)
                    .unwrap_or(self.bridge.input_rate as u64);
                ensure!(
                    rate == self.bridge.input_rate as u64,
                    "session input rate differs from --input-rate"
                );
                if self.bridge.profile == Profile::Asr {
                    upstream.push(json!({"type":"session.update", "model":self.bridge.model}));
                } else {
                    session["type"] = json!("realtime");
                    session["model"] = json!(self.bridge.model);
                    session["output_modalities"] = json!(["audio"]);
                    session["audio"] = json!({
                        "input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null},
                        "output":{"format":{"type":"audio/pcm","rate":24000}}
                    });
                    if let Some(max) = self.bridge.max_output_tokens {
                        session["max_output_tokens"] = json!(max);
                    }
                    upstream.push(json!({"type":"session.update","session":session}));
                }
                downstream.push(event(
                    json!({"type":"session.updated","session":frame["session"]}),
                ));
            }
            "input_audio_buffer.append" => {
                ensure!(
                    !self.committed,
                    "prototype supports one committed utterance per connection"
                );
                if !self.started {
                    self.started = true;
                    if self.bridge.profile == Profile::Asr {
                        upstream.push(json!({"type":"input_audio_buffer.commit","final":false}));
                    }
                }
                let raw = STANDARD.decode(
                    frame["audio"]
                        .as_str()
                        .context("audio must be base64 text")?,
                )?;
                ensure!(!raw.is_empty(), "empty audio chunk");
                let converted = self.audio.push(&raw, false)?;
                if !converted.is_empty() {
                    upstream.push(json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(converted)}));
                }
            }
            "input_audio_buffer.commit" => {
                ensure!(
                    self.started && !self.committed,
                    "commit requires an active uncommitted utterance"
                );
                self.committed = true;
                let tail = self.audio.push(&[], true)?;
                if !tail.is_empty() {
                    upstream.push(
                        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(tail)}),
                    );
                }
                if self.bridge.profile == Profile::Asr {
                    upstream.push(json!({"type":"input_audio_buffer.commit","final":true}));
                    downstream.push(event(json!({"type":"input_audio_buffer.committed","item_id":self.item_id,"previous_item_id":null})));
                } else {
                    upstream.push(json!({"type":"input_audio_buffer.commit"}));
                    upstream.push(json!({"type":"response.create","response":{"output_modalities":["audio"]}}));
                }
            }
            other => bail!("event {other:?} is outside the one-utterance prototype contract"),
        }
        Ok((upstream, downstream))
    }

    fn output(&self, frame: Value) -> Result<(Option<Value>, bool)> {
        let kind = frame["type"]
            .as_str()
            .context("upstream event missing type")?;
        if matches!(kind, "session.created" | "session.updated") {
            return Ok((None, false));
        }
        // Omni v0.31.0rc1 emits this obsolete handshake event; Dynamo's
        // current public schema does not expose it.
        if self.bridge.profile == Profile::Omni && kind == "conversation.created" {
            return Ok((None, false));
        }
        if kind == "error" {
            bail!("upstream error: {frame}");
        }
        if self.bridge.profile == Profile::Omni {
            let done = kind == "response.done";
            return Ok((Some(frame), done));
        }
        let mapped = match kind {
            "transcription.delta" => event(json!({
                "type":"conversation.item.input_audio_transcription.delta",
                "item_id":self.item_id,"content_index":0,"delta":frame["delta"],"logprobs":null
            })),
            "transcription.done" => {
                let input = frame
                    .pointer("/usage/prompt_tokens")
                    .or_else(|| frame.pointer("/usage/input_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let output = frame
                    .pointer("/usage/completion_tokens")
                    .or_else(|| frame.pointer("/usage/output_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                event(json!({
                    "type":"conversation.item.input_audio_transcription.completed",
                    "item_id":self.item_id,"content_index":0,"transcript":frame["text"],"logprobs":null,
                    "usage":{"type":"tokens","input_tokens":input,"output_tokens":output,"total_tokens":input+output,
                        "input_token_details":{"audio_tokens":input,"text_tokens":0}}
                }))
            }
            _ => bail!("unexpected ASR upstream event {kind}"),
        };
        Ok((Some(mapped), kind == "transcription.done"))
    }
}

#[async_trait]
impl AsyncEngine<ManyIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for Bridge {
    async fn generate(&self, input: ManyIn<Value>) -> Result<ManyOut<Annotated<Value>>> {
        let context = input.context();
        let (request, _) = input.into_parts();
        let mut incoming = request.take().context("input stream already consumed")?;
        let mut socket = tokio::select! {
            _ = context.stopped() => bail!("request cancelled during upstream connection"),
            result = self.connect() => result?,
        };
        let mut session = Session::new(self.clone());
        let ctx = context.clone();
        let stream = async_stream::stream! {
            let mut input_closed = false;
            loop {
                tokio::select! {
                    _ = ctx.stopped() => break,
                    input = incoming.next(), if !input_closed => {
                        let Some(input) = input else {
                            input_closed = true;
                            if !session.committed { yield Annotated::from_error("input stream closed before commit"); break; }
                            continue;
                        };
                        let (upstream, downstream) = match session.input(input) {
                            Ok(result) => result,
                            Err(error) => { yield Annotated::from_error(error.to_string()); break; },
                        };
                        let mut failed = false;
                        for frame in upstream {
                            let result = tokio::select! {
                                _ = ctx.stopped() => { failed = true; break; },
                                result = send(&mut socket, frame) => result,
                            };
                            if let Err(error) = result {
                                yield Annotated::from_error(error.to_string());
                                failed = true;
                                break;
                            }
                        }
                        if failed { break; }
                        for frame in downstream { yield Annotated::from_data(frame); }
                    }
                    received = tokio::time::timeout(IO_TIMEOUT, socket.next()) => {
                        let message = match received {
                            Ok(Some(Ok(message))) => message,
                            Ok(Some(Err(error))) => { yield Annotated::from_error(error.to_string()); break; },
                            Ok(None) => { yield Annotated::from_error("upstream closed before completion"); break; },
                            Err(_) => { yield Annotated::from_error("upstream response timed out"); break; },
                        };
                        match message {
                            Message::Text(text) => {
                                let result = serde_json::from_str(&text).map_err(anyhow::Error::from).and_then(|frame| session.output(frame));
                                match result {
                                    Ok((frame, done)) => {
                                        if let Some(frame) = frame { yield Annotated::from_data(frame); }
                                        if done { break; }
                                    }
                                    Err(error) => { yield Annotated::from_error(error.to_string()); break; }
                                }
                            }
                            Message::Ping(_) => {
                                let result = tokio::select! {
                                    _ = ctx.stopped() => break,
                                    result = tokio::time::timeout(Duration::from_secs(10), socket.flush()) => result,
                                };
                                if !matches!(result, Ok(Ok(()))) { yield Annotated::from_error("failed to flush upstream pong"); break; }
                            },
                            Message::Close(_) => { yield Annotated::from_error("upstream closed before completion"); break; },
                            Message::Binary(_) => { yield Annotated::from_error("unexpected binary realtime frame"); break; },
                            _ => {},
                        }
                    }
                }
            }
            // Dropping the response stream also drops this socket immediately;
            // normal completion/cancellation gets a bounded WS close handshake.
            let _ = tokio::time::timeout(Duration::from_secs(1), socket.close(None)).await;
        };
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

fn speech_response(model: &str, request_id: &str, bytes: Option<&[u8]>, completed: bool) -> Value {
    json!({"id":request_id,"object":"audio.speech","model":model,
        "status":if completed {"completed"} else {"in_progress"},
        "progress":if completed {100} else {0},"created":0,
        "data":bytes.map(|raw| vec![json!({"output_format":"pcm","b64_json":STANDARD.encode(raw)})]).unwrap_or_default()})
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for Bridge {
    async fn generate(&self, input: SingleIn<Value>) -> Result<ManyOut<Annotated<Value>>> {
        ensure!(
            self.profile == Profile::Tts,
            "single-input engine requires TTS profile"
        );
        let context = input.context();
        let (request, _) = input.into_parts();
        ensure!(
            request["response_format"].as_str().unwrap_or("wav") == "pcm",
            "prototype TTS requires response_format=pcm"
        );
        ensure!(
            request["speed"].as_f64().unwrap_or(1.0) == 1.0,
            "prototype TTS requires speed=1"
        );
        ensure!(
            request["data_source"].as_str().unwrap_or("b64_json") == "b64_json",
            "prototype TTS does not support URL output"
        );
        let text = request["input"]
            .as_str()
            .context("speech request requires input text")?;
        let mut config = json!({"type":"session.config","model":self.model,
            "response_format":"pcm","stream_audio":true,"split_granularity":"none"});
        for key in [
            "voice",
            "language",
            "task_type",
            "instructions",
            "ref_audio",
            "ref_text",
            "max_new_tokens",
        ] {
            if let Some(value) = request.get(key).filter(|v| !v.is_null()) {
                config[key] = value.clone();
            }
        }
        // The frontend deliberately nests noncanonical media fields here.
        if let Some(extra) = request
            .pointer("/extra_args/media_passthrough")
            .and_then(Value::as_object)
        {
            for (key, value) in extra {
                match key.as_str() {
                    "seed"
                    | "initial_codec_chunk_frames"
                    | "non_streaming_mode"
                    | "x_vector_only_mode"
                    | "speaker_embedding" => config[key] = value.clone(),
                    "stream" => {} // frontend delivery hint, streaming is forced above
                    _ => bail!("TTS option {key:?} is outside the prototype contract"),
                }
            }
        }
        let mut socket = tokio::select! {
            _ = context.stopped() => bail!("request cancelled during upstream connection"),
            result = self.connect() => result?,
        };
        for frame in [
            config,
            json!({"type":"input.text","text":text}),
            json!({"type":"input.done"}),
        ] {
            tokio::select! {
                _ = context.stopped() => bail!("request cancelled while sending to upstream"),
                result = send(&mut socket, frame) => result?,
            }
        }
        let ctx = context.clone();
        let model = self.model.clone();
        let request_id = context.id().to_string();
        let stream = async_stream::stream! {
            let mut received_audio = false;
            loop {
                let message = tokio::select! {
                    _ = ctx.stopped() => break,
                    result = tokio::time::timeout(IO_TIMEOUT, socket.next()) => result,
                };
                match message {
                    Ok(Some(Ok(Message::Binary(bytes)))) => {
                        received_audio = true;
                        yield Annotated::from_data(speech_response(&model, &request_id, Some(&bytes), false));
                    }
                    Ok(Some(Ok(Message::Text(text)))) => {
                        let frame: Value = match serde_json::from_str(&text) {
                            Ok(frame) => frame,
                            Err(error) => { yield Annotated::from_error(error.to_string()); break; }
                        };
                        match frame["type"].as_str().unwrap_or_default() {
                            "audio.start" => {
                                if frame["format"] != "pcm" || frame["sample_rate"] != 24000 || frame.get("channels").is_some_and(|v| v != 1) {
                                    yield Annotated::from_error(format!("unexpected audio format: {frame}")); break;
                                }
                            }
                            "audio.done" => {},
                            "session.done" => {
                                if !received_audio { yield Annotated::from_error("TTS completed without audio"); }
                                else { yield Annotated::from_data(speech_response(&model, &request_id, None, true)); }
                                break;
                            }
                            "error" => { yield Annotated::from_error(format!("upstream TTS error: {frame}")); break; },
                            other => { yield Annotated::from_error(format!("unexpected TTS event {other:?}")); break; }
                        }
                    }
                    Ok(Some(Ok(Message::Ping(_)))) => {
                        let result = tokio::select! {
                            _ = ctx.stopped() => break,
                            result = tokio::time::timeout(Duration::from_secs(10), socket.flush()) => result,
                        };
                        if !matches!(result, Ok(Ok(()))) { yield Annotated::from_error("failed to flush upstream pong"); break; }
                    },
                    Ok(Some(Ok(Message::Pong(_)))) => {},
                    Ok(Some(Err(error))) => { yield Annotated::from_error(error.to_string()); break; },
                    Err(_) => { yield Annotated::from_error("upstream TTS response timed out"); break; },
                    _ => { yield Annotated::from_error("upstream TTS closed before session.done"); break; },
                }
            }
            let _ = tokio::time::timeout(Duration::from_secs(1), socket.close(None)).await;
        };
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}
