// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared native-sidecar contracts; enable only through dev-dependencies.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DynamoError, ErrorType, FinishReason, GenerateContext, LLMEngine,
    LLMEngineOutput, PreprocessedRequest, StopConditions,
};
use futures::{StreamExt, poll};

use crate::NativeStream;

pub struct Script<T> {
    pub messages: Vec<Result<T, tonic::Status>>,
    pub pending_open: bool,
    pub open_error: Option<tonic::Status>,
    pub pending_tail: bool,
}

impl<T> Script<T> {
    pub fn responses(messages: Vec<T>) -> Self {
        Self {
            messages: messages.into_iter().map(Ok).collect(),
            pending_open: false,
            open_error: None,
            pending_tail: false,
        }
    }
}

#[derive(Default)]
struct Observations {
    open_dropped: AtomicBool,
    read_pending: AtomicBool,
    stream_dropped: AtomicBool,
}

struct State<Q, R> {
    script: Option<Script<R>>,
    requests: Vec<Q>,
}

pub struct ScriptedClient<Q, R> {
    state: Arc<Mutex<State<Q, R>>>,
    observed: Arc<Observations>,
}

impl<Q, R> Clone for ScriptedClient<Q, R> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            observed: self.observed.clone(),
        }
    }
}

struct OpenGuard(Arc<Observations>);

impl Drop for OpenGuard {
    fn drop(&mut self) {
        self.0.open_dropped.store(true, Ordering::SeqCst);
    }
}

impl<Q, R> ScriptedClient<Q, R> {
    fn new(script: Script<R>) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                script: Some(script),
                requests: Vec::new(),
            })),
            observed: Arc::default(),
        }
    }

    pub async fn open(&self, request: Q) -> Result<NativeStream<R>, tonic::Status> {
        let script = {
            let mut state = self.state.lock().unwrap();
            state.requests.push(request);
            state.script.take().expect("one generation per fixture")
        };
        let _guard = OpenGuard(self.observed.clone());
        if script.pending_open {
            futures::future::pending::<()>().await;
        }
        if let Some(error) = script.open_error {
            return Err(error);
        }
        Ok(NativeStream::Scripted(ScriptedStream {
            messages: script.messages.into(),
            pending_tail: script.pending_tail,
            observed: self.observed.clone(),
        }))
    }
}

pub struct ScriptedStream<T> {
    messages: VecDeque<Result<T, tonic::Status>>,
    pending_tail: bool,
    observed: Arc<Observations>,
}

impl<T> ScriptedStream<T> {
    pub(crate) async fn message(&mut self) -> Result<Option<T>, tonic::Status> {
        if let Some(message) = self.messages.pop_front() {
            return message.map(Some);
        }
        if self.pending_tail {
            self.observed.read_pending.store(true, Ordering::SeqCst);
            futures::future::pending::<()>().await;
        }
        Ok(None)
    }
}

impl<T> Drop for ScriptedStream<T> {
    fn drop(&mut self) {
        self.observed.stream_dropped.store(true, Ordering::SeqCst);
    }
}

/// Native fixtures supply messages; the shared suite owns the behavioral assertions.
pub trait SidecarFixture {
    type Engine: LLMEngine;
    type Request: Send + 'static;
    type Response: Send + 'static;

    fn engine(client: Option<ScriptedClient<Self::Request, Self::Response>>) -> Self::Engine;
    fn first_token() -> Self::Response;
    /// Empty/intermediate messages followed by a length terminal for tokens [42, 43, 44].
    fn successful_responses() -> Vec<Self::Response>;
    fn request_id(request: &Self::Request) -> &str;
    fn eof_error() -> BackendError;
}

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test-model".to_string())
        .token_ids(vec![11, 22, 33])
        .sampling_options(Default::default())
        .output_options(Default::default())
        .stop_conditions(StopConditions {
            max_tokens: Some(3),
            ..Default::default()
        })
        .build()
        .unwrap()
}

type Outputs = Vec<Result<LLMEngineOutput, DynamoError>>;

async fn collect(engine: &impl LLMEngine, ctx: GenerateContext) -> Outputs {
    bounded(async {
        match engine.generate(request(), ctx).await {
            Ok(stream) => stream.collect().await,
            Err(error) => vec![Err(error)],
        }
    })
    .await
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(1), future)
        .await
        .expect("sidecar stalled")
}

fn assert_terminal(outputs: Outputs, tokens: &[u32], reason: FinishReason) {
    let outputs: Vec<_> = outputs
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("successful stream");
    assert_eq!(
        outputs
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens
    );
    assert_eq!(
        outputs.iter().filter(|o| o.finish_reason.is_some()).count(),
        1
    );
    let terminal = outputs.last().expect("terminal output");
    assert_eq!(terminal.finish_reason, Some(reason));
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (3, tokens.len() as u32, 3 + tokens.len() as u32)
    );
}

fn assert_failure(mut outputs: Outputs, tokens: &[u32], kind: BackendError) -> DynamoError {
    let error = outputs
        .pop()
        .expect("error output")
        .expect_err("truncation must fail");
    assert_eq!(error.error_type(), ErrorType::Backend(kind));
    let preceding: Vec<_> = outputs
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("only one error");
    assert!(preceding.iter().all(|o| o.finish_reason.is_none()));
    assert_eq!(
        preceding
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens
    );
    error
}

pub async fn stream_contract<F: SidecarFixture>() {
    let mut messages = F::successful_responses();
    messages.push(F::first_token());
    let client = ScriptedClient::new(Script::responses(messages));
    let engine = F::engine(Some(client.clone()));
    let ctx = mock_context();
    let outputs = collect(&engine, GenerateContext::new(ctx.clone(), None)).await;
    assert_terminal(outputs, &[42, 43, 44], FinishReason::Length);
    let state = client.state.lock().unwrap();
    assert_eq!(state.requests.len(), 1);
    assert_eq!(F::request_id(&state.requests[0]), ctx.id());
    assert!(client.observed.stream_dropped.load(Ordering::SeqCst));
}

pub async fn failure_contract<F: SidecarFixture>() {
    for failure in ["open", "eof", "read"] {
        let mut script = Script::responses(vec![F::first_token()]);
        let kind = match failure {
            "open" => {
                script.open_error = Some(tonic::Status::unavailable("scripted peer failure"));
                BackendError::CannotConnect
            }
            "read" => {
                script
                    .messages
                    .push(Err(tonic::Status::unavailable("scripted peer failure")));
                script.messages.push(Ok(F::first_token()));
                BackendError::CannotConnect
            }
            _ => F::eof_error(),
        };
        let client = ScriptedClient::new(script);
        let engine = F::engine(Some(client.clone()));
        let outputs = collect(&engine, GenerateContext::new(mock_context(), None)).await;
        let tokens = if failure == "open" {
            &[][..]
        } else {
            &[42][..]
        };
        let error = assert_failure(outputs, tokens, kind);
        if failure != "eof" {
            assert!(error.to_string().contains("scripted peer failure"));
            assert!(error.to_string().contains("Generate"));
        }
        if failure != "open" {
            assert!(client.observed.stream_dropped.load(Ordering::SeqCst));
        }
    }
}

pub async fn cancellation_contract<F: SidecarFixture>() {
    for stage in ["before_open", "pending_open", "pending_read"] {
        let mut script = Script::responses(vec![F::first_token()]);
        script.pending_open = stage == "pending_open";
        script.pending_tail = true;
        let client = ScriptedClient::new(script);
        let engine = F::engine(Some(client.clone()));
        let ctx = mock_context();
        if stage == "before_open" {
            ctx.stop_generating();
            assert_terminal(
                collect(&engine, GenerateContext::new(ctx, None)).await,
                &[],
                FinishReason::Cancelled,
            );
            assert!(client.state.lock().unwrap().requests.is_empty());
        } else if stage == "pending_open" {
            let collected = collect(&engine, GenerateContext::new(ctx.clone(), None));
            tokio::pin!(collected);
            assert!(poll!(&mut collected).is_pending());
            assert_eq!(client.state.lock().unwrap().requests.len(), 1);
            assert!(!client.observed.open_dropped.load(Ordering::SeqCst));
            ctx.stop_generating();
            assert_terminal(collected.await, &[], FinishReason::Cancelled);
            assert!(client.observed.open_dropped.load(Ordering::SeqCst));
        } else {
            let mut stream =
                bounded(engine.generate(request(), GenerateContext::new(ctx.clone(), None)))
                    .await
                    .unwrap();
            let first = bounded(stream.next()).await.unwrap();
            assert_eq!(first.as_ref().unwrap().token_ids, [42]);
            assert!(poll!(stream.next()).is_pending());
            assert!(client.observed.read_pending.load(Ordering::SeqCst));
            ctx.stop_generating();
            let mut outputs = vec![first];
            outputs.extend(bounded(stream.collect::<Outputs>()).await);
            assert_terminal(outputs, &[42], FinishReason::Cancelled);
            assert!(client.observed.stream_dropped.load(Ordering::SeqCst));
        }
    }
}

pub async fn cleanup_contract<F: SidecarFixture>() {
    let engine = F::engine(None);
    assert_failure(
        collect(&engine, GenerateContext::new(mock_context(), None)).await,
        &[],
        BackendError::EngineShutdown,
    );
    engine.cleanup().await.unwrap();
    engine.cleanup().await.unwrap();

    let mut script = Script::responses(vec![F::first_token()]);
    script.pending_tail = true;
    let client = ScriptedClient::new(script);
    let engine = F::engine(Some(client.clone()));
    let mut stream =
        bounded(engine.generate(request(), GenerateContext::new(mock_context(), None)))
            .await
            .unwrap();
    let first = bounded(stream.next()).await.unwrap();
    assert!(poll!(stream.next()).is_pending());
    assert!(client.observed.read_pending.load(Ordering::SeqCst));
    engine.cleanup().await.unwrap();
    engine.cleanup().await.unwrap();
    let mut outputs = vec![first];
    outputs.extend(bounded(stream.collect::<Outputs>()).await);
    assert_terminal(outputs, &[42], FinishReason::Cancelled);
    assert!(client.observed.stream_dropped.load(Ordering::SeqCst));
}

#[macro_export]
macro_rules! sidecar_contract_tests {
    ($fixture:ty) => {
        #[tokio::test(start_paused = true)]
        async fn stream_tokens_terminal_and_usage() {
            $crate::testing::stream_contract::<$fixture>().await;
        }
        #[tokio::test(start_paused = true)]
        async fn open_failure_early_eof_and_read_failure() {
            $crate::testing::failure_contract::<$fixture>().await;
        }
        #[tokio::test(start_paused = true)]
        async fn cancellation_before_open_during_open_and_during_read() {
            $crate::testing::cancellation_contract::<$fixture>().await;
        }
        #[tokio::test(start_paused = true)]
        async fn cleanup_before_start_and_during_read() {
            $crate::testing::cleanup_contract::<$fixture>().await;
        }
    };
}
