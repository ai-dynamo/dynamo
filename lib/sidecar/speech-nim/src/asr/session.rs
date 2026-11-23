// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::engine::AsyncEngineContext;
use futures::{StreamExt, future::Either, stream::BoxStream};
use serde_json::Value;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinSet,
    time::Instant,
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};

use super::{AsrAdapter, turn, validation};
use crate::events;

enum Output {
    Turn {
        events: mpsc::Receiver<Value>,
        cancelled: CancellationToken,
    },
    Cleared,
}

struct OpenTurn {
    id: String,
    received: usize,
    // Unbounded in chunk count, bounded in bytes by validation::audio.
    audio: mpsc::UnboundedSender<Vec<u8>>,
    closed: oneshot::Sender<Instant>,
    commit_event: mpsc::OwnedPermit<Value>,
    cancelled: CancellationToken,
    finished: CancellationToken,
}

impl OpenTurn {
    fn close(self) {
        let _ = self.closed.send(Instant::now());
        // Dropping the sole audio sender half-closes the gRPC request stream.
    }
}

struct Session {
    adapter: AsrAdapter,
    control: mpsc::Sender<Value>,
    output: mpsc::Sender<Output>,
    active: Option<OpenTurn>,
    turns: JoinSet<()>,
    previous: Option<CancellationToken>,
    configured: bool,
}

impl Session {
    fn discard_finished_input(&mut self) {
        if self
            .active
            .as_ref()
            .is_some_and(|turn| turn.finished.is_cancelled())
        {
            self.active.take();
        }
    }

    async fn ensure_turn(&mut self) -> Option<&mut OpenTurn> {
        self.discard_finished_input();
        if self.active.is_none() {
            // One running and one queued turn per session. Completed task
            // handles are reaped here and by the input pump.
            while self.turns.len() >= 2 {
                self.turns.join_next().await;
            }
            let (audio_tx, audio_rx) = mpsc::unbounded_channel();
            let (close_tx, close_rx) = oneshot::channel();
            let (event_tx, event_rx) = mpsc::channel(256);
            let cancelled = CancellationToken::new();
            let finished = CancellationToken::new();
            let id = events::id("item");
            self.output
                .send(Output::Turn {
                    events: event_rx,
                    cancelled: cancelled.clone(),
                })
                .await
                .ok()?;
            let adapter = self.adapter.clone();
            let previous = self.previous.replace(finished.clone());
            let task_cancel = cancelled.clone();
            let done = finished.clone();
            let item_id = id.clone();
            // A full delta queue must never delay closing the input or arming its deadline.
            let commit_event = event_tx.clone().reserve_owned().await.ok()?;
            self.turns.spawn(async move {
                let finished = done.drop_guard();
                // Cancelled queued turns still preserve their predecessor's place.
                if let Some(previous) = previous {
                    previous.cancelled().await;
                }
                tokio::select! {
                    biased;
                    _ = task_cancel.cancelled() => {},
                    _ = async {
                        let terminal = turn::run(&adapter, &item_id, audio_rx, close_rx, &event_tx).await;
                        // Retire input and release RPC resources before publishing the result.
                        drop(finished);
                        let _ = event_tx.send(terminal).await;
                    } => {},
                }
            });
            self.active = Some(OpenTurn {
                id,
                received: 0,
                audio: audio_tx,
                closed: close_tx,
                commit_event,
                cancelled,
                finished,
            });
        }
        self.active.as_mut()
    }

    fn cancel_active(&mut self) {
        if let Some(turn) = self.active.take() {
            turn.cancelled.cancel();
        }
    }

    async fn emit(&self, event: Value) {
        let _ = self.control.send(event).await;
    }

    async fn handle(&mut self, event: Value) {
        self.discard_finished_input();
        let client_id = &event["event_id"];
        if !event.is_object() {
            self.emit(events::invalid(
                "invalid_event",
                "event must be an object",
                &Value::Null,
            ))
            .await;
            return;
        }
        match event["type"].as_str() {
            Some("session.update") => {
                let response = match validation::session(&event["session"], &self.adapter.config) {
                    Ok(()) => {
                        self.configured = true;
                        events::updated(event["session"].clone())
                    }
                    Err(message) => events::invalid("invalid_session", message, client_id),
                };
                self.emit(response).await;
            }
            Some("input_audio_buffer.append" | "input_audio_buffer.commit") if !self.configured => {
                self.emit(events::invalid(
                    "invalid_session",
                    "send a valid session.update before submitting audio",
                    client_id,
                ))
                .await;
            }
            Some("input_audio_buffer.append") => {
                let Some(turn) = self.ensure_turn().await else {
                    return;
                };
                match validation::audio(&event["audio"], &mut turn.received) {
                    Ok(audio) => {
                        if turn.audio.send(audio).is_err() {
                            self.emit(events::invalid(
                                "invalid_audio",
                                "transcription input closed before accepting audio",
                                client_id,
                            ))
                            .await;
                        }
                    }
                    Err(message) => {
                        self.cancel_active();
                        self.emit(events::invalid("invalid_audio", message, client_id))
                            .await;
                    }
                }
            }
            Some("input_audio_buffer.commit") => {
                if let Some(turn) = self.active.take() {
                    let samples = validation::SAMPLE_RATE
                        * self.adapter.config.commit_padding_ms as usize
                        / 1000;
                    if samples > 0 {
                        let _ = turn.audio.send(vec![0; samples * 2]);
                    }
                    turn.commit_event.send(events::committed(&turn.id));
                    let _ = turn.closed.send(Instant::now());
                } else {
                    self.emit(events::invalid(
                        "invalid_audio",
                        "input audio buffer is empty",
                        client_id,
                    ))
                    .await;
                }
            }
            Some("input_audio_buffer.clear") => {
                self.cancel_active();
                let _ = self.output.send(Output::Cleared).await;
            }
            _ => {
                self.emit(events::invalid(
                    "unsupported_event",
                    format!("unsupported event type: {}", event["type"]),
                    client_id,
                ))
                .await
            }
        }
    }

    async fn pump(mut self, mut input: BoxStream<'static, Value>) {
        let Some(first) = input.next().await else {
            return;
        };
        if !first.is_object() || first["type"] != "session.update" {
            self.emit(events::invalid(
                "invalid_event",
                "first event must be session.update",
                &first["event_id"],
            ))
            .await;
            return;
        }
        if first["session"]["type"] != "transcription" {
            self.emit(events::invalid(
                "unsupported_session",
                format!("unsupported session type: {}", first["session"]["type"]),
                &first["event_id"],
            ))
            .await;
            return;
        }
        self.handle(first).await;
        loop {
            tokio::select! {
                biased;
                result = self.turns.join_next(), if !self.turns.is_empty() => {
                    if let Some(Err(error)) = result {
                        tracing::error!(%error, "Realtime turn task failed");
                        return;
                    }
                    self.discard_finished_input();
                },
                event = input.next() => match event {
                    Some(event) => self.handle(event).await,
                    None => break,
                },
            }
        }
        // Input EOF is a half-close, not a disconnected response consumer.
        if let Some(turn) = self.active.take() {
            turn.close();
        }
        while let Some(result) = self.turns.join_next().await {
            if let Err(error) = result {
                tracing::error!(%error, "Realtime turn task failed");
            }
        }
    }
}

pub(super) fn serve(
    adapter: AsrAdapter,
    input: BoxStream<'static, Value>,
    context: Arc<dyn AsyncEngineContext>,
) -> BoxStream<'static, Value> {
    async_stream::stream! {
        let (control_tx, mut control_rx) = mpsc::channel(64);
        let (turn_tx, turn_rx) = mpsc::channel(2);
        let shutdown = adapter.shutdown.clone();
        let session = Session {
            adapter, control: control_tx, output: turn_tx, active: None, turns: JoinSet::new(),
            previous: None, configured: false,
        };
        // Dropping the response stream aborts the input pump, whose JoinSet
        // then aborts every turn. No detached tasks or RPCs outlive a session.
        let pump_context = context.clone();
        let _pump = AbortOnDropHandle::new(tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = pump_context.stopped() => {},
                _ = pump_context.killed() => {},
                _ = shutdown.cancelled() => {},
                _ = session.pump(input) => {},
            }
        }));
        // The initial session acknowledgement must precede every turn event.
        let first = tokio::select! {
            biased;
            _ = context.stopped() => return,
            _ = context.killed() => return,
            event = control_rx.recv() => event,
        };
        if let Some(event) = first { yield event; }
        // Keep turns and clear acknowledgments ordered, but drain control replies so they
        // cannot block the input pump from reaching commit or clear.
        let ordered = ReceiverStream::new(turn_rx).flat_map(|output| match output {
            Output::Turn { events, cancelled } => Either::Left(
                ReceiverStream::new(events).take_until(cancelled.cancelled_owned())
            ),
            Output::Cleared => Either::Right(futures::stream::once(async { events::cleared() })),
        });
        let mut output = futures::stream::select(ReceiverStream::new(control_rx), ordered).boxed();
        loop {
            let event = tokio::select! {
                biased;
                _ = context.stopped() => break,
                _ = context.killed() => break,
                event = output.next() => event,
            };
            match event { Some(event) => yield event, None => break }
        }
    }
    .boxed()
}
