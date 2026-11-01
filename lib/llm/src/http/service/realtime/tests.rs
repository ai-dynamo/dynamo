// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_runtime::engine::{AsyncEngine, ResponseStream, async_trait};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{ManyIn, ManyOut};
use dynamo_runtime::protocols::annotated::Annotated;
use serde_json::{Value, json};
use tokio::sync::Notify;

use super::*;

struct FailingEngine {
    error_type: ErrorType,
    calls: AtomicUsize,
    connect_delay: Option<Duration>,
    connecting: Notify,
    contexts: Mutex<Vec<dynamo_runtime::engine::Context>>,
}

impl FailingEngine {
    fn new(error_type: ErrorType, connect_delay: Option<Duration>) -> Self {
        Self {
            error_type,
            connect_delay,
            calls: AtomicUsize::new(0),
            connecting: Notify::new(),
            contexts: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl
    AsyncEngine<
        ManyIn<DynamoRealtimeClientEvent>,
        ManyOut<Annotated<RealtimeServerEvent>>,
        anyhow::Error,
    > for FailingEngine
{
    async fn generate(
        &self,
        input: ManyIn<DynamoRealtimeClientEvent>,
    ) -> anyhow::Result<ManyOut<Annotated<RealtimeServerEvent>>> {
        let context = input.context();
        self.contexts.lock().push(context.clone());
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call > 0
            && let Some(delay) = self.connect_delay
        {
            self.connecting.notify_one();
            tokio::time::sleep(delay).await;
            return Err(DynamoError::builder()
                .error_type(self.error_type)
                .message("worker unavailable")
                .build()
                .into());
        }
        let updated = Annotated::from_data(serde_json::from_value(json!({
            "type": "session.updated", "event_id": "setup", "session": configuration().session
        }))?);
        let error = Annotated {
            error: Some(
                DynamoError::builder()
                    .error_type(self.error_type)
                    .message("worker failure")
                    .build(),
            ),
            ..Annotated::from_error("worker failure")
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([updated, error])),
            context,
        ))
    }
}

fn configuration() -> RealtimeClientEventSessionUpdate {
    serde_json::from_value(json!({"session": {
        "type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "asr"}, "turn_detection": null
        }}
    }}))
    .unwrap()
}

fn session(engine: RealtimeBidirectionalEngine, limit: u32) -> RealtimeSession {
    RealtimeSession::new(engine, &configuration(), limit, CancellationToken::new())
}

#[tokio::test]
async fn typed_non_retryable_errors_and_attempt_exhaustion() {
    for (error_type, calls) in [(ErrorType::InvalidRequest, 1), (ErrorType::Disconnected, 3)] {
        let engine = Arc::new(FailingEngine::new(error_type, None));
        let mut session = session(engine.clone(), 2);
        let mut incoming = futures::stream::pending();
        let outgoing =
            futures::sink::unfold((), |(), _: Message| async { Ok::<_, axum::Error>(()) });
        tokio::pin!(outgoing);
        assert!(
            session
                .run(
                    &mut incoming,
                    &mut outgoing,
                    &Mutex::new(None),
                    configuration()
                )
                .await
                .is_err()
        );
        assert_eq!(engine.calls.load(Ordering::SeqCst), calls, "{error_type:?}");
        assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
    }
}

#[tokio::test(start_paused = true)]
async fn replacement_attempts_share_one_absolute_deadline() {
    let engine = Arc::new(FailingEngine::new(
        ErrorType::Disconnected,
        Some(Duration::from_secs(3)),
    ));
    let mut session = session(engine.clone(), 10);
    let mut incoming = futures::stream::pending();
    let outgoing = futures::sink::unfold((), |(), _: Message| async { Ok::<_, axum::Error>(()) });
    tokio::pin!(outgoing);
    let started = Instant::now();
    let error = session
        .run(
            &mut incoming,
            &mut outgoing,
            &Mutex::new(None),
            configuration(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "realtime recovery timed out");
    assert_eq!(started.elapsed(), Duration::from_secs(5));
    assert_eq!(engine.calls.load(Ordering::SeqCst), 3);
    assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
}

#[tokio::test]
async fn client_close_and_shutdown_cancel_a_connecting_replacement() {
    for shutdown in [false, true] {
        let engine = Arc::new(FailingEngine::new(
            ErrorType::Disconnected,
            Some(Duration::from_secs(60)),
        ));
        let mut session = session(engine.clone(), 1);
        let cancel = session.cancel.clone();
        let (tx, rx) = mpsc::channel(1);
        let mut incoming = ReceiverStream::new(rx).map(Ok);
        let outgoing =
            futures::sink::unfold((), |(), _: Message| async { Ok::<_, axum::Error>(()) });
        tokio::pin!(outgoing);
        let reason = Mutex::new(None);
        let task = session.run(&mut incoming, &mut outgoing, &reason, configuration());
        tokio::pin!(task);
        tokio::select! {
            _ = &mut task => panic!("session ended before reaching replacement"),
            _ = engine.connecting.notified() => {},
        }
        if shutdown {
            cancel.cancel();
        } else {
            tx.try_send(Message::Close(None)).unwrap();
        }
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
    }
}

#[tokio::test(start_paused = true)]
async fn blocked_socket_write_cannot_keep_a_disconnected_worker_running() {
    let engine = Arc::new(FailingEngine::new(ErrorType::InvalidRequest, None));
    let mut session = session(engine.clone(), 0);
    let (tx, rx) = mpsc::channel(2);
    let mut incoming = ReceiverStream::new(rx).map(Ok);
    let writing = Notify::new();
    let outgoing = futures::sink::unfold((), |(), _: Message| async {
        writing.notify_one();
        std::future::pending::<Result<(), axum::Error>>().await
    });
    tokio::pin!(outgoing);
    let reason = Mutex::new(None);
    let task = session.run(&mut incoming, &mut outgoing, &reason, configuration());
    tokio::pin!(task);
    tokio::select! {
        _ = &mut task => panic!("session ended before writing"),
        _ = writing.notified() => {},
    }
    tx.try_send(Message::Text(
        json!({"type": "input_audio_buffer.append", "audio": "AAAA"})
            .to_string()
            .into(),
    ))
    .unwrap();
    tx.try_send(Message::Close(None)).unwrap();
    tokio::time::timeout(Duration::from_secs(6), task)
        .await
        .expect("blocked output must not retain a disconnected session")
        .unwrap();
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
}

#[tokio::test]
async fn client_write_error_does_not_retry_workers() {
    let engine = Arc::new(FailingEngine::new(ErrorType::Disconnected, None));
    let mut session = session(engine.clone(), 1);
    let mut incoming = futures::stream::pending();
    let outgoing = futures::sink::unfold((), |(), _: Message| async {
        Err::<(), _>(axum::Error::new(
            DynamoError::builder()
                .error_type(ErrorType::Disconnected)
                .message("client disconnected")
                .build(),
        ))
    });
    tokio::pin!(outgoing);
    session
        .run(
            &mut incoming,
            &mut outgoing,
            &Mutex::new(None),
            configuration(),
        )
        .await
        .unwrap();
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
}

struct ReplayErrorEngine {
    frames: Arc<Vec<Value>>,
    error_before_ack: bool,
    stall_after_error: bool,
    calls: AtomicUsize,
    replacing: Notify,
    contexts: Mutex<Vec<dynamo_runtime::engine::Context>>,
}

#[async_trait]
impl
    AsyncEngine<
        ManyIn<DynamoRealtimeClientEvent>,
        ManyOut<Annotated<RealtimeServerEvent>>,
        anyhow::Error,
    > for ReplayErrorEngine
{
    async fn generate(
        &self,
        input: ManyIn<DynamoRealtimeClientEvent>,
    ) -> anyhow::Result<ManyOut<Annotated<RealtimeServerEvent>>> {
        let context = input.context();
        self.contexts.lock().push(context.clone());
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            call < 2,
            "disabled recovery must not dispatch another worker"
        );
        if call == 1 {
            self.replacing.notify_one();
        }
        let (requests, _) = input.into_parts();
        let mut requests = requests.take().unwrap();
        let frames = self.frames.clone();
        let error_before_ack = self.error_before_ack;
        let stall_after_error = self.stall_after_error;
        let output = async_stream::stream! {
            let setup = serde_json::to_value(requests.next().await.unwrap()).unwrap();
            assert_eq!(setup["type"], "session.update");
            let updated = json!({"type": "session.updated", "event_id": "setup",
                "session": configuration().session});
            if call == 0 || !error_before_ack {
                yield Annotated::from_data(serde_json::from_value(updated.clone()).unwrap());
            }
            if call == 1 {
                yield Annotated::from_data(serde_json::from_value(json!({
                    "type": "error", "event_id": "invalid", "error": {
                        "type": "invalid_request_error", "code": "invalid_audio",
                        "message": "invalid audio"
                    }
                })).unwrap());
                if stall_after_error {
                    std::future::pending::<()>().await;
                }
            }
            for expected in frames.iter() {
                let actual = serde_json::to_value(requests.next().await.unwrap()).unwrap();
                assert_eq!(&actual, expected, "accepted input must arrive once and in order");
            }
            if call == 0 {
                yield Annotated::from_data(serde_json::from_value(json!({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "event_id": "partial", "item_id": "original", "content_index": 0,
                    "delta": "partial"
                })).unwrap());
                yield Annotated {
                    error: Some(DynamoError::builder()
                        .error_type(ErrorType::Disconnected)
                        .message("worker lost")
                        .build()),
                    ..Annotated::from_error("worker lost")
                };
            } else {
                // The replayed commit precedes the queued open turn, even after an error.
                yield Annotated::from_data(serde_json::from_value(json!({
                    "type": "input_audio_buffer.committed", "event_id": "commit",
                    "item_id": "residual", "previous_item_id": null
                })).unwrap());
                if frames.last().unwrap()["type"] == "input_audio_buffer.clear" {
                    yield Annotated::from_data(serde_json::from_value(json!({
                        "type": "input_audio_buffer.cleared", "event_id": "clear"
                    })).unwrap());
                }
                let fresh = serde_json::to_value(requests.next().await.unwrap()).unwrap();
                assert_eq!(fresh["type"], "session.update");
                yield Annotated::from_data(serde_json::from_value(updated).unwrap());
            }
        };
        Ok(ResponseStream::new(Box::pin(output), context))
    }
}

#[tokio::test(start_paused = true)]
async fn application_error_during_replay_preserves_input_and_the_client_session() {
    for (error_before_ack, stall_after_error, clear_queued) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let mut frames: Vec<_> = (0..REQUEST_CHANNEL_CAPACITY + 8)
            .map(|i| json!({"type": "input_audio_buffer.append", "event_id": i.to_string(), "audio": "AAAA"}))
            .collect();
        frames.push(json!({"type": "input_audio_buffer.commit"}));
        frames.push(if clear_queued {
            json!({"type": "input_audio_buffer.clear"})
        } else {
            json!({"type": "input_audio_buffer.append", "audio": "BBBB"})
        });
        let engine = Arc::new(ReplayErrorEngine {
            frames: Arc::new(frames),
            error_before_ack,
            stall_after_error,
            calls: AtomicUsize::new(0),
            replacing: Notify::new(),
            contexts: Mutex::new(Vec::new()),
        });
        let mut session = session(engine.clone(), 2);
        let (tx, rx) = mpsc::channel(engine.frames.len() + 1);
        for frame in engine.frames.iter() {
            tx.try_send(Message::Text(frame.to_string().into()))
                .unwrap();
        }
        let mut incoming = ReceiverStream::new(rx).map(Ok);
        let output = Mutex::new(Vec::<Value>::new());
        let outgoing = futures::sink::unfold((), |(), message: Message| async {
            let Message::Text(text) = message else {
                panic!("expected an event")
            };
            output.lock().push(serde_json::from_str(&text).unwrap());
            Ok::<_, axum::Error>(())
        });
        tokio::pin!(outgoing);
        let reason = Mutex::new(None);
        let started = Instant::now();
        let run = session.run(&mut incoming, &mut outgoing, &reason, configuration());
        let send_live = async {
            engine.replacing.notified().await;
            tx.send(Message::Text(
                json!({"type": "session.update",
                "session": configuration().session})
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(6), async {
            tokio::join!(run, send_live)
        })
        .await
        .unwrap();
        if stall_after_error {
            assert_eq!(
                result.unwrap_err().to_string(),
                "realtime recovery timed out"
            );
            assert_eq!(started.elapsed(), Duration::from_secs(5));
        } else {
            result.unwrap();
            assert!(!session.recovery.is_replaying());
        }
        assert!(session.recovery.retry(UnexpectedEnd.into()).is_err());
        let output = output.lock();
        assert_eq!(output.iter().filter(|e| e["type"] == "error").count(), 1);
        if !stall_after_error {
            assert_eq!(
                output
                    .iter()
                    .filter(|e| e["type"] == "session.updated")
                    .count(),
                2
            );
            assert_eq!(
                output
                    .iter()
                    .find(|e| e["type"] == "input_audio_buffer.committed")
                    .unwrap()["item_id"],
                "residual"
            );
        }
        assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
        assert!(engine.contexts.lock().iter().all(|ctx| ctx.is_stopped()));
    }
}
