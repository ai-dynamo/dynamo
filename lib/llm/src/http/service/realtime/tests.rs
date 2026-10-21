// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_runtime::engine::{AsyncEngine, ResponseStream, async_trait};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{ManyIn, ManyOut};
use dynamo_runtime::protocols::annotated::Annotated;
use serde_json::json;
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
    RealtimeSession {
        engine,
        migration_limit: limit,
        attempts: 0,
        deadline: None,
        incoming: None,
        cancel: CancellationToken::new(),
        journal: Journal::new(&configuration()),
    }
}

#[tokio::test]
async fn typed_non_retryable_errors_and_attempt_exhaustion() {
    for (error_type, calls) in [
        (ErrorType::InvalidRequest, 1),
        (ErrorType::Cancelled, 1),
        (ErrorType::DeadlineExceeded, 1),
        (ErrorType::ResourceExhausted, 1),
        (ErrorType::Disconnected, 3),
    ] {
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
    assert_eq!(started.elapsed(), RECOVERY_TIMEOUT);
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
    session.journal = None;
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
