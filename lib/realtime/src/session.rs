// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use base64::{Engine, prelude::BASE64_STANDARD};
use futures::{Stream, StreamExt, stream::BoxStream};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::{
    Client, Error, Limits, Protocol,
    audio::{AsrAudio, OPENAI_PCM_RATE, decode_pcm16},
    client::{Socket, receive, send},
    events,
};

/// Dropping the response stream closes every upstream turn, even when the
/// runtime did not explicitly mark the session cancelled.
pub struct Session {
    output: mpsc::Receiver<BufferedEvent>,
    cancel: CancellationToken,
}

impl Session {
    pub(crate) fn start(
        client: Client,
        protocol: Protocol,
        input: BoxStream<'static, Value>,
        cancellation: impl Future<Output = ()> + Send + 'static,
    ) -> Self {
        let cancel = client.inner.shutdown.child_token();
        let (output, receiver) = mpsc::channel(client.inner.limits.event_capacity.get());
        let emitter = Emitter {
            output,
            budget: Arc::new(Semaphore::new(
                client.inner.limits.max_buffered_output_bytes.get(),
            )),
            limit: client.inner.limits.max_buffered_output_bytes.get(),
        };
        let driver = Coordinator {
            permits: Arc::new(Semaphore::new(client.inner.limits.concurrent_turns.get())),
            client: client.clone(),
            protocol,
            input,
            output: emitter,
            cancel: cancel.clone(),
            turns: VecDeque::new(),
            tasks: JoinSet::new(),
            active: None,
            session: None,
        };
        client
            .inner
            .tasks
            .spawn(driver.run(cancellation).in_current_span());
        Self {
            output: receiver,
            cancel,
        }
    }
}

impl Stream for Session {
    type Item = Value;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Value>> {
        self.output
            .poll_recv(cx)
            .map(|event| event.map(|event| event.value))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

struct BufferedEvent {
    value: Value,
    _bytes: OwnedSemaphorePermit,
}

struct TurnOutput {
    events: mpsc::Receiver<BufferedEvent>,
    cancel: CancellationToken,
}

struct Turn {
    input: Arc<TurnInput>,
    cancel: CancellationToken,
}

#[derive(Default)]
struct InputState {
    audio: VecDeque<Vec<u8>>,
    received: usize,
    clear: bool,
    committed: bool,
    acknowledge_commit: bool,
    finished: bool,
}

/// Audio is bounded by both bytes and event count. Commit and clear have their
/// own slots so a full audio queue cannot prevent control messages.
#[derive(Default)]
struct TurnInput {
    state: Mutex<InputState>,
    changed: Notify,
}

enum Command {
    Audio(Vec<u8>),
    Clear,
    Commit { acknowledge: bool },
}

impl TurnInput {
    fn append(&self, pcm: Vec<u8>, limits: &Limits) -> Result<(), Error> {
        let mut state = self.state.lock();
        if pcm.len()
            > limits
                .max_turn_audio_bytes
                .get()
                .saturating_sub(state.received)
        {
            return Err(Error::Invalid(
                "input audio buffer exceeds the configured limit".into(),
            ));
        }
        let received = pcm.len();
        if state.audio.len() >= limits.event_capacity.get() {
            // A burst can fill the event slots before the upstream handshake
            // finishes. Preserve PCM order without increasing the queue bound.
            let tail = state.audio.back_mut().expect("nonzero full audio queue");
            if pcm.len()
                > limits
                    .max_audio_chunk_bytes
                    .get()
                    .saturating_sub(tail.len())
            {
                return Err(Error::Invalid(
                    "input audio buffer exceeds the configured limit".into(),
                ));
            }
            tail.extend_from_slice(&pcm);
        } else {
            state.audio.push_back(pcm);
        }
        state.received += received;
        drop(state);
        self.changed.notify_one();
        Ok(())
    }

    fn commit(&self, acknowledge: bool) {
        let mut state = self.state.lock();
        state.committed = true;
        state.acknowledge_commit = acknowledge;
        drop(state);
        self.changed.notify_one();
    }

    fn clear(&self) {
        let mut state = self.state.lock();
        state.audio.clear();
        state.received = 0;
        state.clear = true;
        drop(state);
        self.changed.notify_one();
    }

    async fn next(&self) -> Command {
        loop {
            let notified = self.changed.notified();
            {
                let mut state = self.state.lock();
                if state.clear {
                    state.clear = false;
                    return Command::Clear;
                }
                if let Some(pcm) = state.audio.pop_front() {
                    return Command::Audio(pcm);
                }
                if state.committed {
                    return Command::Commit {
                        acknowledge: state.acknowledge_commit,
                    };
                }
            }
            notified.await;
        }
    }
}

struct Coordinator {
    client: Client,
    protocol: Protocol,
    input: BoxStream<'static, Value>,
    output: Emitter,
    cancel: CancellationToken,
    permits: Arc<Semaphore>,
    turns: VecDeque<TurnOutput>,
    tasks: JoinSet<()>,
    active: Option<Turn>,
    session: Option<Value>,
}

impl Coordinator {
    async fn run(mut self, cancellation: impl Future<Output = ()>) {
        let cancel = self.cancel.clone();
        tokio::select! {
            _ = cancel.cancelled() => {},
            _ = cancellation => {},
            result = self.drive() => {
                if let Err(error) = result {
                    tracing::debug!(%error, "realtime session failed");
                    let event = events::event("error", json!({"error": {
                        "type": "server_error", "code": "session_error",
                        "message": "Realtime session failed",
                    }}));
                    tokio::select! {
                        _ = cancel.cancelled() => {},
                        _ = self.output.send(event) => {},
                    }
                }
            },
        }
        self.cancel.cancel();
        // Each turn owns its socket and gets a bounded opportunity to close it.
        while self.tasks.join_next().await.is_some() {}
    }

    async fn drive(&mut self) -> Result<(), Error> {
        let mut eof = false;
        loop {
            if eof && self.turns.is_empty() {
                return Ok(());
            }
            tokio::select! {
                event = self.input.next(), if !eof => {
                    match event {
                        Some(event) => self.handle(event).await?,
                        None => {
                            eof = true;
                            if let Some(turn) = self.active.take() {
                                turn.input.commit(false);
                            }
                        }
                    }
                }
                event = async { self.turns.front_mut().expect("guarded by nonempty").events.recv().await },
                    if !self.turns.is_empty() => {
                    if let Some(event) = event {
                        if !self.turns.front().expect("nonempty").cancel.is_cancelled() {
                            self.output.output.send(event).await.map_err(|_| Error::Closed)?;
                        }
                    } else {
                        self.turns.pop_front();
                    }
                }
                result = self.tasks.join_next(), if !self.tasks.is_empty() => {
                    if let Some(Err(error)) = result {
                        tracing::error!(%error, "realtime turn task failed");
                        return Err(Error::Protocol("realtime turn task failed".into()));
                    }
                }
            }
        }
    }

    async fn emit(&mut self, event: Value) -> Result<(), Error> {
        self.output.send(event).await
    }

    async fn handle(&mut self, event: Value) -> Result<(), Error> {
        if self
            .active
            .as_ref()
            .is_some_and(|turn| turn.input.state.lock().finished)
        {
            self.active = None;
        }
        match event["type"].as_str() {
            Some("session.update") => {
                if let Err(error) =
                    validate_session(&event["session"], self.protocol, &self.client.inner.model)
                {
                    return self
                        .emit(events::invalid(
                            "invalid_session",
                            &error.to_string(),
                            &event,
                        ))
                        .await;
                }
                let mut session = event["session"].clone();
                if self.protocol == Protocol::Omni && session["output_modalities"].is_null() {
                    session["output_modalities"] = json!(["audio"]);
                }
                self.session = Some(session.clone());
                self.emit(events::event(
                    "session.updated",
                    json!({"session": session}),
                ))
                .await?;
            }
            Some("input_audio_buffer.append") => {
                if self.session.is_none() {
                    return self
                        .emit(events::invalid(
                            "invalid_session",
                            "send a valid session.update first",
                            &event,
                        ))
                        .await;
                }
                let pcm = event["audio"]
                    .as_str()
                    .ok_or_else(|| Error::Invalid("audio must be a base64 string".into()))
                    .and_then(|audio| {
                        decode_pcm16(audio, self.client.inner.limits.max_audio_chunk_bytes.get())
                    });
                match pcm {
                    Ok(pcm) => {
                        if self.active.is_none() && !self.open_turn().await? {
                            return self
                                .emit(events::invalid(
                                    "session_busy",
                                    "too many pending turns",
                                    &event,
                                ))
                                .await;
                        }
                        let turn = self.active.as_ref().expect("turn opened");
                        if let Err(error) = turn.input.append(pcm, &self.client.inner.limits) {
                            turn.cancel.cancel();
                            self.active = None;
                            self.emit(events::invalid("invalid_audio", &error.to_string(), &event))
                                .await?;
                        }
                    }
                    Err(error) => {
                        self.emit(events::invalid("invalid_audio", &error.to_string(), &event))
                            .await?
                    }
                }
            }
            Some("input_audio_buffer.commit") => {
                if let Some(turn) = self.active.take() {
                    turn.input.commit(true);
                } else {
                    self.emit(events::invalid(
                        "invalid_audio",
                        "input audio buffer is empty",
                        &event,
                    ))
                    .await?;
                }
            }
            Some("input_audio_buffer.clear") => {
                if self.protocol == Protocol::VllmTranscription {
                    if let Some(turn) = self.active.take() {
                        turn.cancel.cancel();
                    }
                } else if let Some(turn) = &self.active {
                    turn.input.clear();
                }
                self.emit(events::event("input_audio_buffer.cleared", json!({})))
                    .await?;
            }
            // Preserve the existing Omni bridge's independent-utterance contract.
            Some(kind)
                if self.protocol == Protocol::Omni
                    && (kind.starts_with("conversation.item.") || kind == "response.create") => {}
            _ => {
                self.emit(events::invalid(
                    "unsupported_event",
                    "unsupported realtime event",
                    &event,
                ))
                .await?
            }
        }
        Ok(())
    }

    async fn open_turn(&mut self) -> Result<bool, Error> {
        let limits = &self.client.inner.limits;
        if self.turns.len() >= limits.concurrent_turns.get() + limits.queued_turns {
            return Ok(false);
        }
        let input = Arc::new(TurnInput::default());
        let cancel = self.cancel.child_token();
        let (output, events) = mpsc::channel(limits.event_capacity.get());
        let budget = Arc::new(Semaphore::new(limits.max_buffered_output_bytes.get()));
        let emitter = Emitter {
            output,
            budget,
            limit: limits.max_buffered_output_bytes.get(),
        };
        let driver = TurnDriver {
            client: self.client.clone(),
            protocol: self.protocol,
            input: input.clone(),
            cancel: cancel.clone(),
            session: self.session.clone().expect("session validated"),
            emitter,
            item: events::id("item"),
            response: events::id("resp"),
        };
        let permits = self.permits.clone();
        self.tasks
            .spawn(async move { driver.run(permits).await }.in_current_span());
        self.turns.push_back(TurnOutput {
            events,
            cancel: cancel.clone(),
        });
        self.active = Some(Turn { input, cancel });
        Ok(true)
    }
}

struct Emitter {
    output: mpsc::Sender<BufferedEvent>,
    budget: Arc<Semaphore>,
    limit: usize,
}

impl Emitter {
    async fn send(&self, value: Value) -> Result<(), Error> {
        let size = serde_json::to_vec(&value)?.len().max(1);
        if size > self.limit {
            return Err(Error::Protocol(
                "upstream event exceeds the output byte budget".into(),
            ));
        }
        let bytes = self
            .budget
            .clone()
            .acquire_many_owned(size as u32)
            .await
            .map_err(|_| Error::Closed)?;
        self.output
            .send(BufferedEvent {
                value,
                _bytes: bytes,
            })
            .await
            .map_err(|_| Error::Closed)
    }
}

struct TurnDriver {
    client: Client,
    protocol: Protocol,
    input: Arc<TurnInput>,
    cancel: CancellationToken,
    session: Value,
    emitter: Emitter,
    item: String,
    response: String,
}

impl TurnDriver {
    async fn run(self, permits: Arc<Semaphore>) {
        // Publish completion even if a connection attempt fails or a task is
        // dropped during runtime shutdown, so later input cannot enter a dead turn.
        struct Finished(Arc<TurnInput>);
        impl Drop for Finished {
            fn drop(&mut self) {
                self.0.state.lock().finished = true;
            }
        }
        let _finished = Finished(self.input.clone());
        let operation = async {
            let _slot = tokio::select! {
                _ = self.cancel.cancelled() => return Ok(()),
                permit = permits.acquire_owned() => permit.map_err(|_| Error::Closed)?,
            };
            let mut socket = tokio::select! {
                _ = self.cancel.cancelled() => return Ok(()),
                result = self.client.connect() => result?,
            };
            let result = tokio::select! {
                _ = self.cancel.cancelled() => Ok(()),
                result = self.drive(&mut socket) => result,
            };
            if self.cancel.is_cancelled() && self.protocol == Protocol::Omni {
                let _ = tokio::time::timeout(
                    self.client.inner.limits.close_timeout,
                    send(&mut socket, json!({"type": "response.cancel"})),
                )
                .await;
            }
            self.client.close(&mut socket).await;
            result
        };
        let result = operation.await;
        if let Err(error) = result {
            tracing::debug!(%error, "upstream realtime turn failed");
            let failure = match self.protocol {
                Protocol::VllmTranscription => events::transcription_failed(&self.item),
                Protocol::Omni => {
                    events::omni_failed(&self.response, &self.session["output_modalities"])
                }
            };
            tokio::select! {
                _ = self.cancel.cancelled() => {},
                _ = self.emitter.send(failure) => {},
            }
        }
    }

    async fn drive(&self, socket: &mut Socket) -> Result<(), Error> {
        let created =
            tokio::time::timeout(self.client.inner.limits.connect_timeout, receive(socket))
                .await
                .map_err(|_| Error::Timeout)??;
        if created["type"] != "session.created" {
            return Err(Error::Protocol("upstream did not create a session".into()));
        }
        match self.protocol {
            Protocol::VllmTranscription => {
                send(
                    socket,
                    json!({"type": "session.update", "model": self.client.inner.model}),
                )
                .await?;
                send(
                    socket,
                    json!({"type": "input_audio_buffer.commit", "final": false}),
                )
                .await?;
            }
            Protocol::Omni => {
                let mut session = self.session.clone();
                session["model"] = self.client.inner.model.clone().into();
                self.client.update_omni_session(socket, session).await?;
            }
        }
        let mut converter = (self.protocol == Protocol::VllmTranscription)
            .then(AsrAudio::new)
            .transpose()?;
        let mut committed = false;
        let deadline = tokio::time::sleep(self.client.inner.limits.response_timeout);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                command = self.input.next(), if !committed => {
                    match command {
                        Command::Audio(pcm) => {
                            let pcm = match &mut converter {
                                Some(converter) => converter.append(&pcm)?,
                                None => pcm,
                            };
                            append(socket, pcm).await?;
                        }
                        Command::Clear => send(socket, json!({"type": "input_audio_buffer.clear"})).await?,
                        Command::Commit { acknowledge } => {
                            if let Some(converter) = &mut converter {
                                append(socket, converter.finish()?).await?;
                                if acknowledge {
                                    self.emitter.send(events::event("input_audio_buffer.committed", json!({
                                        "item_id": self.item, "previous_item_id": null
                                    }))).await?;
                                }
                                send(socket, json!({"type": "input_audio_buffer.commit", "final": true})).await?;
                            } else {
                                send(socket, json!({"type": "input_audio_buffer.commit"})).await?;
                                send(socket, json!({"type": "response.create"})).await?;
                            }
                            committed = true;
                            deadline.as_mut().reset(tokio::time::Instant::now() + self.client.inner.limits.response_timeout);
                        }
                    }
                }
                event = receive(socket) => {
                    let mut event = event?;
                    let kind = event["type"].as_str().ok_or_else(|| Error::Protocol("event has no type".into()))?.to_owned();
                    if kind == "error" {
                        return Err(Error::Protocol("upstream rejected a realtime event".into()));
                    }
                    match self.protocol {
                        Protocol::VllmTranscription => {
                            let audio_bytes = self.input.state.lock().received;
                            if let Some(event) = events::asr_output(event, &self.item, audio_bytes)? {
                                self.emitter.send(event).await?;
                            }
                            if kind == "transcription.done" {
                                return Ok(());
                            }
                        }
                        Protocol::Omni => {
                            // The frontend owns the public session. Turn-local
                            // upstream session/control acknowledgements stay internal.
                            if kind.starts_with("response.") {
                                if kind == "response.created" || kind == "response.done" {
                                    event["response"]["id"] = self.response.clone().into();
                                } else if event.get("response_id").is_some() {
                                    event["response_id"] = self.response.clone().into();
                                }
                                self.emitter.send(event).await?;
                            }
                            if kind == "response.done" {
                                return Ok(());
                            }
                        }
                    }
                }
                _ = &mut deadline, if committed => return Err(Error::Timeout),
            }
        }
    }
}

async fn append(socket: &mut Socket, pcm: Vec<u8>) -> Result<(), Error> {
    if !pcm.is_empty() {
        send(
            socket,
            json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(pcm)}),
        )
        .await?;
    }
    Ok(())
}

fn validate_session(session: &Value, protocol: Protocol, model: &str) -> Result<(), Error> {
    let invalid = |message: &str| Error::Invalid(message.into());
    if !session.is_object() {
        return Err(invalid("session must be an object"));
    }
    if protocol == Protocol::VllmTranscription {
        if session["type"] != "transcription" {
            return Err(invalid("session.type must be transcription"));
        }
        let input = &session["audio"]["input"];
        let transcription = &input["transcription"];
        if transcription["model"] != model {
            return Err(invalid(
                "session transcription model must match the served model",
            ));
        }
        if input["format"]["type"] != "audio/pcm" || input["format"]["rate"] != OPENAI_PCM_RATE {
            return Err(invalid("input must be mono PCM16 at 24000 Hz"));
        }
        if !transcription["language"].is_null() && transcription["language"] != "en" {
            return Err(invalid("only English realtime transcription is supported"));
        }
        if !transcription["prompt"].is_null() && transcription["prompt"] != "" {
            return Err(invalid("transcription prompts are not supported"));
        }
        if !input["noise_reduction"].is_null() || !input["turn_detection"].is_null() {
            return Err(invalid(
                "noise reduction and server turn detection are not supported",
            ));
        }
    } else {
        if session["type"] != "realtime" || session["model"] != model {
            return Err(invalid("expected a realtime session for the served model"));
        }
        if !session["audio"]["input"]["turn_detection"].is_null() {
            return Err(invalid(
                "ordered Omni turns require explicit client commits",
            ));
        }
        let format = &session["audio"]["input"]["format"];
        if !format.is_null() && (format["type"] != "audio/pcm" || format["rate"] != OPENAI_PCM_RATE)
        {
            return Err(invalid("input must be mono PCM16 at 24000 Hz"));
        }
        if let Some(modalities) = session
            .get("output_modalities")
            .filter(|value| !value.is_null())
        {
            let valid = modalities.as_array().is_some_and(|values| {
                !values.is_empty()
                    && values.len() <= 2
                    && values
                        .iter()
                        .all(|value| value == "audio" || value == "text")
                    && (values.len() != 2 || values[0] != values[1])
            });
            if !valid {
                return Err(invalid("output_modalities must contain audio and/or text"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn audio_backpressure_does_not_block_clear_or_commit() {
        let mut limits = Limits::for_protocol(Protocol::Omni);
        limits.event_capacity = std::num::NonZeroUsize::new(1).unwrap();
        limits.max_audio_chunk_bytes = std::num::NonZeroUsize::new(4).unwrap();
        let input = TurnInput::default();
        input.append(vec![0, 0], &limits).unwrap();
        input.append(vec![0, 0], &limits).unwrap();
        assert!(input.append(vec![0, 0], &limits).is_err());
        input.clear();
        assert_eq!(input.state.lock().received, 0);
        input.append(vec![1, 1], &limits).unwrap();
        input.commit(true);
        assert!(matches!(input.next().await, Command::Clear));
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![1, 1]));
        assert!(matches!(
            input.next().await,
            Command::Commit { acknowledge: true }
        ));
    }

    #[tokio::test]
    async fn audio_burst_coalesces_tail_without_reordering_or_exceeding_event_capacity() {
        let limits = Limits::for_protocol(Protocol::VllmTranscription);
        let input = TurnInput::default();
        let mut expected = Vec::new();
        // The real 7.1-second burst uses 71 100 ms chunks, exceeding 64 slots
        // while remaining well below the default 60-second turn byte bound.
        for value in 0..71 {
            let pcm = vec![value; 4800];
            expected.extend_from_slice(&pcm);
            input.append(pcm, &limits).unwrap();
        }
        let queued = {
            let state = input.state.lock();
            assert_eq!(state.audio.len(), limits.event_capacity.get());
            assert_eq!(state.received, expected.len());
            assert!(
                state
                    .audio
                    .iter()
                    .all(|pcm| { pcm.len() <= limits.max_audio_chunk_bytes.get() })
            );
            state.audio.len()
        };
        input.commit(true);
        let mut actual = Vec::new();
        for _ in 0..queued {
            let Command::Audio(pcm) = input.next().await else {
                panic!("queued audio must precede commit");
            };
            actual.extend_from_slice(&pcm);
        }
        assert_eq!(actual, expected);
        assert!(matches!(
            input.next().await,
            Command::Commit { acknowledge: true }
        ));
    }

    #[tokio::test]
    async fn audio_tail_coalescing_preserves_chunk_and_total_turn_byte_limits() {
        let limits = Limits {
            event_capacity: std::num::NonZeroUsize::new(2).unwrap(),
            max_audio_chunk_bytes: std::num::NonZeroUsize::new(6).unwrap(),
            max_turn_audio_bytes: std::num::NonZeroUsize::new(10).unwrap(),
            ..Limits::for_protocol(Protocol::VllmTranscription)
        };
        let input = TurnInput::default();
        for pcm in [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]] {
            input.append(pcm, &limits).unwrap();
        }
        // Total bytes would fit, but the full queue's tail cannot exceed 6.
        assert!(input.append(vec![8, 9], &limits).is_err());
        assert_eq!(input.state.lock().received, 8);
        assert_eq!(input.state.lock().audio.len(), 2);
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![0, 1]));
        // Once a slot is consumed, the same chunk can occupy a separate slot.
        input.append(vec![8, 9], &limits).unwrap();
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![2, 3, 4, 5, 6, 7]));
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![8, 9]));
        // Consuming queued audio does not erase the cumulative turn byte cap.
        assert!(input.append(vec![10, 11], &limits).is_err());
        assert_eq!(input.state.lock().received, 10);
        assert!(input.state.lock().audio.is_empty());
        input.clear();
        input.append(vec![12, 13], &limits).unwrap();
        assert!(matches!(input.next().await, Command::Clear));
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![12, 13]));
        assert_eq!(input.state.lock().received, 2);
    }
}
