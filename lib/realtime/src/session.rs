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
            client: client.clone(),
            protocol,
            input,
            output: emitter,
            local_output: None,
            cancel: cancel.clone(),
            turns: VecDeque::new(),
            pending: VecDeque::new(),
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
    pending: Option<BufferedEvent>,
    clears: usize,
}

struct Turn {
    input: Arc<TurnInput>,
    cancel: CancellationToken,
}

enum AudioInput {
    Streaming(VecDeque<Vec<u8>>),
    Buffered(Vec<u8>),
}

impl Default for AudioInput {
    fn default() -> Self {
        Self::Streaming(VecDeque::new())
    }
}

impl AudioInput {
    fn clear(&mut self) {
        match self {
            Self::Streaming(audio) => audio.clear(),
            Self::Buffered(audio) => audio.clear(),
        }
    }
}

#[derive(Default)]
struct InputState {
    audio: AudioInput,
    received: usize,
    committed: bool,
    acknowledge_commit: bool,
    finished: bool,
}

/// ASR streams through a bounded queue; Omni buffers one byte-bounded utterance.
/// Commit has a separate slot so a full audio queue cannot block it.
#[derive(Default)]
struct TurnInput {
    state: Mutex<InputState>,
    changed: Notify,
    committed: Notify,
}

enum Command {
    Audio(Vec<u8>),
    Commit { acknowledge: bool },
}

impl TurnInput {
    fn new(protocol: Protocol) -> Self {
        Self {
            state: Mutex::new(InputState {
                audio: match protocol {
                    Protocol::VllmTranscription => AudioInput::default(),
                    Protocol::Omni => AudioInput::Buffered(Vec::new()),
                },
                ..InputState::default()
            }),
            ..Self::default()
        }
    }

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
        match &mut state.audio {
            AudioInput::Streaming(audio) => {
                if audio.len() >= limits.event_capacity.get() {
                    // A burst can fill the event slots before the upstream handshake
                    // finishes. Preserve PCM order without increasing the queue bound.
                    let tail = audio.back_mut().expect("nonzero full audio queue");
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
                    audio.push_back(pcm);
                }
            }
            AudioInput::Buffered(audio) => {
                if audio.is_empty() {
                    *audio = pcm;
                } else {
                    audio.extend_from_slice(&pcm);
                }
            }
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
        self.committed.notify_one();
    }

    async fn wait_for_commit(&self) {
        loop {
            let notified = self.committed.notified();
            if self.state.lock().committed {
                return;
            }
            notified.await;
        }
    }

    async fn next(&self) -> Command {
        loop {
            let notified = self.changed.notified();
            {
                let mut state = self.state.lock();
                let committed = state.committed;
                let pcm = match &mut state.audio {
                    AudioInput::Streaming(audio) => audio.pop_front(),
                    AudioInput::Buffered(audio) if committed && !audio.is_empty() => {
                        Some(std::mem::take(audio))
                    }
                    AudioInput::Buffered(_) => None,
                };
                if let Some(pcm) = pcm {
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
    local_output: Option<(Value, usize)>,
    cancel: CancellationToken,
    turns: VecDeque<TurnOutput>,
    pending: VecDeque<TurnDriver>,
    tasks: JoinSet<()>,
    active: Option<Turn>,
    session: Option<Value>,
}

impl Coordinator {
    async fn run(mut self, cancellation: impl Future<Output = ()>) {
        let cancel = self.cancel.clone();
        tokio::pin!(cancellation);
        let result = tokio::select! {
            _ = cancel.cancelled() => Ok(()),
            _ = &mut cancellation => Ok(()),
            result = self.drive() => result,
        };
        self.cancel.cancel();
        // Each turn owns its socket and gets a bounded opportunity to close it.
        while self.tasks.join_next().await.is_some() {}
        if let Err(error) = result {
            tracing::debug!(%error, "realtime session failed");
            let event = events::event(
                "error",
                json!({"error": {
                    "type": "server_error", "code": "session_error",
                    "message": "Realtime session failed",
                }}),
            );
            tokio::select! {
                _ = &mut cancellation => {},
                _ = self.client.inner.shutdown.cancelled() => {},
                _ = self.output.send(event) => {},
            }
        }
    }

    async fn drive(&mut self) -> Result<(), Error> {
        let mut eof = false;
        let output = self.output.output.clone();
        let budget = self.output.budget.clone();
        loop {
            self.start_ready_turns();
            if eof && self.turns.is_empty() && self.local_output.is_none() {
                return Ok(());
            }
            let pending_turn = self
                .turns
                .front()
                .is_some_and(|turn| turn.pending.is_some());
            let draining_clears = self.turns.front().is_some_and(|turn| {
                turn.pending.is_none()
                    && turn.clears > 0
                    && turn.events.is_closed()
                    && turn.events.is_empty()
            });
            let local_size = self.local_output.as_ref().map_or(0, |(_, size)| *size);
            tokio::select! {
                event = self.input.next(), if !eof && self.local_output.is_none() && !draining_clears => {
                    match event {
                        Some(event) => self.handle(event)?,
                        None => {
                            eof = true;
                            if let Some(turn) = self.active.take() {
                                turn.input.commit(false);
                            }
                        }
                    }
                }
                permit = output.reserve(), if pending_turn => {
                    let event = self.turns.front_mut().expect("pending turn").pending.take().expect("pending event");
                    permit.map_err(|_| Error::Closed)?.send(event);
                }
                permits = async {
                    let bytes = budget.clone().acquire_many_owned(local_size as u32)
                        .await.map_err(|_| Error::Closed)?;
                    let output = output.reserve().await.map_err(|_| Error::Closed)?;
                    Ok::<_, Error>((bytes, output))
                }, if self.local_output.is_some() && !pending_turn => {
                    let (bytes, output) = permits?;
                    let (value, _) = self.local_output.take().expect("pending local event");
                    output.send(BufferedEvent { value, _bytes: bytes });
                }
                event = async { self.turns.front_mut().expect("guarded by nonempty").events.recv().await },
                    if !self.turns.is_empty() && !pending_turn && self.local_output.is_none() => {
                    if let Some(event) = event {
                        self.turns.front_mut().expect("front turn").pending = Some(event);
                    } else {
                        let turn = self.turns.front_mut().expect("front turn");
                        if turn.clears > 0 {
                            turn.clears -= 1;
                            if turn.clears == 0 {
                                self.turns.pop_front();
                            }
                            self.emit(events::event("input_audio_buffer.cleared", json!({})))?;
                        } else {
                            self.turns.pop_front();
                        }
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

    fn emit(&mut self, event: Value) -> Result<(), Error> {
        let size = self.output.event_size(&event)?;
        self.local_output = Some((event, size));
        Ok(())
    }

    fn handle(&mut self, event: Value) -> Result<(), Error> {
        if self
            .active
            .as_ref()
            .is_some_and(|turn| turn.input.state.lock().finished)
        {
            self.active = None;
        }
        match event["type"].as_str() {
            Some("session.update") => {
                if let Err(error) = update_session(
                    &mut self.session,
                    event["session"].clone(),
                    self.protocol,
                    &self.client.inner.model,
                ) {
                    return self.emit(events::invalid(
                        "invalid_session",
                        &error.to_string(),
                        &event,
                    ));
                }
                self.emit(events::event(
                    "session.updated",
                    json!({"session": self.session}),
                ))?;
            }
            Some("input_audio_buffer.append") => {
                if self.session.is_none() {
                    return self.emit(events::invalid(
                        "invalid_session",
                        "send a valid session.update first",
                        &event,
                    ));
                }
                let pcm = event["audio"]
                    .as_str()
                    .ok_or_else(|| Error::Invalid("audio must be a base64 string".into()))
                    .and_then(|audio| {
                        decode_pcm16(audio, self.client.inner.limits.max_audio_chunk_bytes.get())
                    });
                match pcm {
                    Ok(pcm) => {
                        if self.active.is_none() && !self.open_turn() {
                            return self.emit(events::invalid(
                                "session_busy",
                                "too many pending turns",
                                &event,
                            ));
                        }
                        let turn = self.active.as_ref().expect("turn opened");
                        if let Err(error) = turn.input.append(pcm, &self.client.inner.limits) {
                            self.cancel_active();
                            self.emit(events::invalid(
                                "invalid_audio",
                                &error.to_string(),
                                &event,
                            ))?;
                        }
                    }
                    Err(error) => {
                        self.emit(events::invalid("invalid_audio", &error.to_string(), &event))?
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
                    ))?;
                }
            }
            Some("input_audio_buffer.clear") => {
                self.cancel_active();
                if self.protocol == Protocol::VllmTranscription
                    && let Some(turn) = self.turns.back_mut()
                {
                    // Recovery reconciles clears with earlier transcription
                    // output. Counts keep repeated clears allocation-bounded.
                    turn.clears = turn
                        .clears
                        .checked_add(1)
                        .ok_or_else(|| Error::Invalid("too many clear acknowledgements".into()))?;
                } else {
                    self.emit(events::event("input_audio_buffer.cleared", json!({})))?;
                }
            }
            // Preserve the existing Omni bridge's independent-utterance contract.
            Some(kind)
                if self.protocol == Protocol::Omni
                    && (kind.starts_with("conversation.item.") || kind == "response.create") => {}
            _ => self.emit(events::invalid(
                "unsupported_event",
                "unsupported realtime event",
                &event,
            ))?,
        }
        Ok(())
    }

    fn cancel_active(&mut self) {
        if let Some(turn) = self.active.take() {
            turn.cancel.cancel();
            turn.input.state.lock().audio.clear();
            // The only uncommitted turn is the tail. Discard its output before
            // acknowledging clear; a running task retains its slot until exit.
            self.turns.pop_back();
            if self
                .pending
                .back()
                .is_some_and(|driver| Arc::ptr_eq(&driver.input, &turn.input))
            {
                self.pending.pop_back();
            }
        }
    }

    fn start_ready_turns(&mut self) {
        while self.tasks.len() < self.client.inner.limits.concurrent_turns.get() {
            let Some(driver) = self.pending.front() else {
                break;
            };
            if self.protocol == Protocol::Omni && !driver.input.state.lock().committed {
                break;
            }
            if let Some(driver) = self.pending.pop_front() {
                self.tasks.spawn(driver.run().in_current_span());
            }
        }
    }

    fn open_turn(&mut self) -> bool {
        let limits = &self.client.inner.limits;
        if self.turns.len() >= limits.concurrent_turns.get() + limits.queued_turns {
            return false;
        }
        let input = Arc::new(TurnInput::new(self.protocol));
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
        self.pending.push_back(driver);
        self.turns.push_back(TurnOutput {
            events,
            pending: None,
            clears: 0,
        });
        self.active = Some(Turn { input, cancel });
        true
    }
}

struct Emitter {
    output: mpsc::Sender<BufferedEvent>,
    budget: Arc<Semaphore>,
    limit: usize,
}

impl Emitter {
    fn event_size(&self, value: &Value) -> Result<usize, Error> {
        let size = serde_json::to_vec(value)?.len().max(1);
        if size > self.limit {
            return Err(Error::Protocol(
                "upstream event exceeds the output byte budget".into(),
            ));
        }
        Ok(size)
    }

    async fn send(&self, value: Value) -> Result<(), Error> {
        let size = self.event_size(&value)?;
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
    async fn run(self) {
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
            let mut socket = tokio::select! {
                _ = self.cancel.cancelled() => return Ok(()),
                result = self.client.connect() => result?,
            };
            let deadline = async {
                self.input.wait_for_commit().await;
                tokio::time::sleep(self.client.inner.limits.response_timeout).await;
            };
            let result = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => Ok(()),
                _ = deadline => Err(Error::Timeout),
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
        loop {
            tokio::select! {
                command = self.input.next(), if !committed => {
                    match command {
                        Command::Audio(pcm) => {
                            if let Some(converter) = &mut converter {
                                append(socket, &converter.append(&pcm)?).await?;
                            } else {
                                // Keep PCM16 samples intact even for an odd append cap.
                                let chunk_bytes = self.client.inner.limits.max_audio_chunk_bytes.get() & !1;
                                for chunk in pcm.chunks(chunk_bytes.max(2)) {
                                    append(socket, chunk).await?;
                                }
                            }
                        }
                        Command::Commit { acknowledge } => {
                            if let Some(converter) = &mut converter {
                                append(socket, &converter.finish()?).await?;
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
            }
        }
    }
}

async fn append(socket: &mut Socket, pcm: &[u8]) -> Result<(), Error> {
    if !pcm.is_empty() {
        send(
            socket,
            json!({"type": "input_audio_buffer.append", "audio": BASE64_STANDARD.encode(pcm)}),
        )
        .await?;
    }
    Ok(())
}

fn merge_session_update(current: &mut Value, update: Value) {
    match (current, update) {
        (Value::Object(current), Value::Object(update)) => {
            for (key, value) in update {
                merge_session_update(current.entry(key).or_insert(Value::Null), value);
            }
        }
        (current, update) => *current = update,
    }
}

fn update_session(
    current: &mut Option<Value>,
    mut update: Value,
    protocol: Protocol,
    model: &str,
) -> Result<(), Error> {
    if protocol == Protocol::Omni {
        if let Some(session) = current.as_ref() {
            let mut candidate = session.clone();
            merge_session_update(&mut candidate, update);
            update = candidate;
        }
        if update.is_object() && update["output_modalities"].is_null() {
            update["output_modalities"] = json!(["audio"]);
        }
    }
    validate_session(&update, protocol, model)?;
    *current = Some(update);
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
        let format = &session["audio"]["output"]["format"];
        if !format.is_null() && (format["type"] != "audio/pcm" || format["rate"] != OPENAI_PCM_RATE)
        {
            return Err(invalid("output must be mono PCM16 at 24000 Hz"));
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

    #[test]
    fn omni_session_updates_preserve_only_accepted_configuration() {
        use dynamo_protocols::types::realtime::DynamoRealtimeClientEvent;

        let typed_session = |session: Value| {
            let event: DynamoRealtimeClientEvent =
                serde_json::from_value(json!({"type": "session.update", "session": session}))
                    .unwrap();
            serde_json::to_value(event).unwrap()["session"].take()
        };
        for repeated_model in [None, Some("model")] {
            let mut state = None;
            assert!(
                update_session(
                    &mut state,
                    typed_session(json!({
                        "type": "realtime", "max_output_tokens": 1,
                    })),
                    Protocol::Omni,
                    "model"
                )
                .is_err()
            );
            assert!(state.is_none());
            update_session(
                &mut state,
                typed_session(json!({
                    "type": "realtime", "model": "model", "output_modalities": ["text"],
                    "instructions": "Keep replies short."
                })),
                Protocol::Omni,
                "model",
            )
            .unwrap();
            let mut patch = json!({"type": "realtime", "max_output_tokens": 123});
            if let Some(model) = repeated_model {
                patch["model"] = model.into();
            }
            update_session(&mut state, typed_session(patch), Protocol::Omni, "model").unwrap();
            assert_eq!(
                state,
                Some(json!({
                    "type": "realtime", "model": "model", "output_modalities": ["text"],
                    "instructions": "Keep replies short.", "max_output_tokens": 123
                }))
            );
            let accepted = state.clone();
            for patch in [
                json!({"type": "realtime", "model": "other-model", "instructions": "discard"}),
                json!({"type": "realtime", "output_modalities": ["invalid"], "instructions": "discard"}),
                json!({"type": "realtime", "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcmu"}, "speed": 1.0, "voice": "alloy"}
                }, "instructions": "discard"}),
                json!({"type": "realtime", "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 16000}, "speed": 1.0, "voice": "alloy"}
                }, "instructions": "discard"}),
            ] {
                assert!(
                    update_session(&mut state, typed_session(patch), Protocol::Omni, "model")
                        .is_err()
                );
                assert_eq!(state, accepted);
            }
        }
    }

    #[test]
    fn omni_session_update_merges_nested_objects_and_replaces_other_values() {
        // The typed frontend schema requires complete audio input/output
        // objects and omits some explicit nulls. Pin the internal merge rule
        // separately without claiming that those nulls survive that boundary.
        let mut session = json!({
            "model": "test-model",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "transcriber", "language": "en"}
                },
                "output": {"voice": "alloy"}
            },
            "instructions": "Keep replies short.",
            "output_modalities": ["audio"],
            "max_output_tokens": "inf"
        });
        merge_session_update(
            &mut session,
            json!({
                "audio": {"input": {"transcription": null}},
                "instructions": null,
                "output_modalities": ["text"],
                "max_output_tokens": 123
            }),
        );
        assert_eq!(
            session,
            json!({
                "model": "test-model",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": null
                    },
                    "output": {"voice": "alloy"}
                },
                "instructions": null,
                "output_modalities": ["text"],
                "max_output_tokens": 123
            })
        );
    }

    #[tokio::test]
    async fn audio_backpressure_does_not_block_commit() {
        let mut limits = Limits::for_protocol(Protocol::VllmTranscription);
        limits.event_capacity = std::num::NonZeroUsize::new(1).unwrap();
        limits.max_audio_chunk_bytes = std::num::NonZeroUsize::new(4).unwrap();
        let input = TurnInput::default();
        input.append(vec![0, 0], &limits).unwrap();
        input.append(vec![0, 0], &limits).unwrap();
        assert!(input.append(vec![0, 0], &limits).is_err());
        input.commit(true);
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![0, 0, 0, 0]));
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
        // Exceed the 64 event slots while staying within the turn byte limit.
        for value in 0..71 {
            let pcm = vec![value; 4800];
            expected.extend_from_slice(&pcm);
            input.append(pcm, &limits).unwrap();
        }
        let queued = {
            let state = input.state.lock();
            let AudioInput::Streaming(audio) = &state.audio else {
                panic!("ASR input must stream");
            };
            assert_eq!(audio.len(), limits.event_capacity.get());
            assert_eq!(state.received, expected.len());
            assert!(
                audio
                    .iter()
                    .all(|pcm| { pcm.len() <= limits.max_audio_chunk_bytes.get() })
            );
            audio.len()
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
        assert!(
            matches!(&input.state.lock().audio, AudioInput::Streaming(audio) if audio.len() == 2)
        );
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![0, 1]));
        // Once a slot is consumed, the same chunk can occupy a separate slot.
        input.append(vec![8, 9], &limits).unwrap();
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![2, 3, 4, 5, 6, 7]));
        assert!(matches!(input.next().await, Command::Audio(pcm) if pcm == vec![8, 9]));
        // Consuming queued audio does not erase the cumulative turn byte cap.
        assert!(input.append(vec![10, 11], &limits).is_err());
        assert_eq!(input.state.lock().received, 10);
        assert!(
            matches!(&input.state.lock().audio, AudioInput::Streaming(audio) if audio.is_empty())
        );
    }
}
