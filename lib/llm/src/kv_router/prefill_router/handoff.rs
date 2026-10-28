// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep consuming prefill while decode runs, and return failures to the caller.

use std::{future::Future, sync::Arc};

use dynamo_runtime::{
    error::{DynamoError, ErrorClass},
    pipeline::{AsyncEngineContext, AsyncEngineContextProvider, ManyOut, ResponseStream},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};
use futures::{StreamExt, stream};
use tokio::task::JoinHandle;
use tracing::Instrument;

use super::PrefillError;
use crate::protocols::{
    common::llm_backend::{FinishReason, LLMEngineOutput},
    response_merge::{PREFILL_OUTPUT_ANNOTATION, PrefillNeed},
};

/// Prefill's last `engine_data`.
pub(super) type PrefillOutput = Option<serde_json::Value>;

/// Dropping the observer detaches the task: client disconnects must not abort a
/// prefill whose KV transfer may still be needed by the decode worker.
pub(super) struct PrefillTask(JoinHandle<Result<PrefillOutput, PrefillError>>);

/// Prefill's output as a stream item tagged with [`PREFILL_OUTPUT_ANNOTATION`].
fn output_item(output: PrefillOutput) -> Option<Annotated<LLMEngineOutput>> {
    let engine_data = Some(output?);
    Some(Annotated {
        event: Some(PREFILL_OUTPUT_ANNOTATION.to_string()),
        ..Annotated::from_data(LLMEngineOutput {
            engine_data,
            ..Default::default()
        })
    })
}

impl PrefillTask {
    /// Observe completion without dropping staged KV on client disconnect.
    pub(super) fn spawn(
        future: impl Future<Output = Result<PrefillOutput, PrefillError>> + Send + 'static,
        prefill_context: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        Self::spawn_unlogged(async move {
            future
                .await
                .inspect_err(|error| Self::log_failure(error, prefill_context.as_ref()))
        })
    }

    /// Spawn a wrapper that delegates logging to its child, while logging its
    /// own failures explicitly so the same child error is not reported twice.
    pub(super) fn spawn_unlogged(
        future: impl Future<Output = Result<PrefillOutput, PrefillError>> + Send + 'static,
    ) -> Self {
        Self(tokio::spawn(future.instrument(tracing::Span::current())))
    }

    /// Keep intentional cancellation quiet without hiding unsolicited failures.
    pub(super) fn log_failure(error: &PrefillError, prefill_context: &dyn AsyncEngineContext) {
        if matches!(error, PrefillError::Cancelled) && prefill_context.is_stopped() {
            tracing::debug!(%error, "Prefill background task cancelled");
        } else {
            tracing::warn!(%error, "Prefill background task failed");
        }
    }

    /// Return the prefill result, including failures from the observer task.
    pub(super) async fn wait(self) -> Result<PrefillOutput, PrefillError> {
        self.0.await.map_err(|error| {
            tracing::warn!(%error, "Prefill background task did not complete");
            PrefillError::PrefillError(
                format!("Prefill background task did not complete: {error}"),
                Some(Box::new(error)),
            )
        })?
    }

    /// Reject backend failures and cancellation before accepting handoff completion.
    pub(super) fn check_output(output: &Annotated<LLMEngineOutput>) -> Result<(), PrefillError> {
        if let Some(error) = output.err() {
            return Err(PrefillError::PrefillError(
                format!("Prefill router returned error in output stream: {error}"),
                Some(Box::new(error)),
            ));
        }
        match output
            .data
            .as_ref()
            .and_then(|data| data.finish_reason.as_ref())
        {
            Some(FinishReason::Error(message)) => Err(PrefillError::PrefillError(
                format!("Prefill backend failed: {message}"),
                None,
            )),
            Some(FinishReason::Cancelled) => Err(PrefillError::Cancelled),
            _ => Ok(()),
        }
    }

    /// Observe prefill failures both during decode dispatch and while streaming.
    /// Successful prefill does not delay tokens; a successful decode terminal is
    /// held until prefill finishes so a late failure cannot follow success.
    /// `output` sends prefill's output into the stream as soon as prefill
    /// finishes (`BeforeLast`), or before any decode output (`BeforeFirst`).
    pub(super) async fn forward_decode(
        self,
        decode: impl Future<Output = anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>>>,
        request_context: Arc<dyn AsyncEngineContext>,
        output: PrefillNeed,
        prefill_context: Arc<dyn AsyncEngineContext>,
    ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let emit = output != PrefillNeed::None;
        let mut completion = Box::pin(self.wait());
        tokio::pin!(decode);
        let mut response = tokio::select! {
            biased;
            result = &mut completion => {
                let prefill = result.inspect_err(|_| request_context.stop_generating())?;
                let response = decode.await?;
                if !emit {
                    return Ok(response);
                }
                let context = response.context();
                let stream = stream::iter(output_item(prefill)).chain(response);
                return Ok(ResponseStream::new(Box::pin(stream), context));
            }
            result = &mut decode => match result {
                Ok(response) => response,
                Err(error) => {
                    cancel_prefill_on_decode_rejection(error.as_ref(), prefill_context.as_ref());
                    return Err(error);
                }
            },
        };
        // Move the completion future into the response stream. The task itself
        // keeps running if this stream is dropped during client cancellation.
        let context = response.context();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
            if output == PrefillNeed::BeforeFirst {
                // Admission refusals are the first decode stream item on this path.
                // Observe it while prefill is pending; buffer only one ordinary item
                // to preserve BeforeFirst ordering and backpressure.
                let mut first_decode_output = None;
                let mut decode_ended = false;
                let prefill_result = tokio::select! {
                    biased;
                    result = &mut completion => result,
                    item = response.next() => {
                        if let Some(error) = item.as_ref().and_then(|item| item.err())
                            && cancel_prefill_on_decode_rejection(&error, prefill_context.as_ref())
                        {
                            yield item.unwrap();
                            return;
                        }
                        decode_ended = item.is_none();
                        first_decode_output = item;
                        completion.as_mut().await
                    }
                };
                match prefill_result {
                    Ok(output) => {
                        if let Some(item) = output_item(output) {
                            yield item;
                        }
                    }
                    Err(error) => {
                        request_context.stop_generating();
                        stream_context.stop_generating();
                        yield Annotated::from_err(error);
                        return;
                    }
                }
                if let Some(output) = first_decode_output {
                    yield output;
                }
                // Prefill metadata/errors still need handling after decode EOF,
                // but an ended stream must not be polled again.
                if decode_ended {
                    return;
                }
                while let Some(output) = response.next().await {
                    yield output;
                }
                return;
            }
            loop {
                tokio::select! {
                    biased;
                    result = &mut completion => {
                        match result {
                            Ok(prefill) => if emit && let Some(item) = output_item(prefill) {
                                yield item;
                            },
                            Err(error) => {
                                request_context.stop_generating();
                                stream_context.stop_generating();
                                yield Annotated::from_err(error);
                                return;
                            }
                        }
                        while let Some(output) = response.next().await {
                            yield output;
                        }
                        return;
                    }
                    output = response.next() => {
                        if output.as_ref().is_some_and(|output| {
                            output.err().is_some() || matches!(
                                output.data.as_ref().and_then(|data| data.finish_reason.as_ref()),
                                Some(FinishReason::Error(_) | FinishReason::Cancelled)
                            )
                        }) {
                            if let Some(error) = output.as_ref().and_then(|output| output.err()) {
                                cancel_prefill_on_decode_rejection(&error, prefill_context.as_ref());
                            }
                            yield output.unwrap();
                            return;
                        }
                        let terminal = output.as_ref().is_none_or(|output| {
                            output.data.as_ref().is_some_and(|data| data.finish_reason.is_some())
                        });
                        if terminal {
                            match completion.as_mut().await {
                                Ok(prefill) => if emit && let Some(item) = output_item(prefill) {
                                    yield item;
                                },
                                Err(error) => {
                                    request_context.stop_generating();
                                    stream_context.stop_generating();
                                    yield Annotated::from_err(error);
                                    return;
                                }
                            }
                        }
                        match output {
                            Some(output) => yield output,
                            None => return,
                        }
                        if terminal {
                            return;
                        }
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

/// Cancel only an explicit decode admission refusal, which has no receiver
/// waiting for staged KV. Ambiguous transport errors retain the existing
/// independent-prefill transfer cleanup behavior.
fn cancel_prefill_on_decode_rejection(
    error: &(dyn std::error::Error + 'static),
    prefill_context: &dyn AsyncEngineContext,
) -> bool {
    let mut current = Some(error);
    while let Some(cause) = current {
        if let Some(error) = cause.downcast_ref::<DynamoError>()
            && error.class() == ErrorClass::CapacityExhausted
        {
            tracing::debug!(request_id = %prefill_context.id(), %error,
                "Cancelling remote prefill after decode admission rejection");
            prefill_context.stop_generating();
            return true;
        }
        current = cause.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{
        error::{BackendError, ErrorType},
        pipeline::context::Controller,
    };
    use futures::{FutureExt, stream};
    use tokio::sync::oneshot;

    fn pending_prefill() -> (
        oneshot::Sender<Result<PrefillOutput, PrefillError>>,
        PrefillTask,
    ) {
        let (tx, rx) = oneshot::channel();
        let task = PrefillTask::spawn(
            async move { rx.await.unwrap() },
            Arc::new(Controller::default()),
        );
        (tx, task)
    }

    fn failure() -> Result<PrefillOutput, PrefillError> {
        Err(PrefillError::PrefillError(
            "prefill connection lost".to_string(),
            None,
        ))
    }

    #[tokio::test]
    async fn failure_interrupts_pending_decode_dispatch() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let mut response = Box::pin(task.forward_decode(
            std::future::pending(),
            context.clone(),
            PrefillNeed::None,
            Arc::new(Controller::default()),
        ));
        assert!(response.as_mut().now_or_never().is_none());
        tx.send(failure()).unwrap();
        let error = tokio::time::timeout(std::time::Duration::from_secs(1), response)
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("prefill connection lost"));
        assert!(context.is_stopped());
    }

    #[tokio::test]
    async fn failure_interrupts_wait_for_decode_tokens() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> =
            ResponseStream::new(Box::pin(stream::pending()), context.clone());
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context.clone(),
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        tx.send(failure()).unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(1), response.next())
            .await
            .unwrap()
            .unwrap();
        assert!(
            output
                .err()
                .unwrap()
                .to_string()
                .contains("prefill connection lost")
        );
        assert!(context.is_stopped());
        assert!(response.next().await.is_none());
    }

    #[tokio::test]
    async fn tokens_flow_while_prefill_is_pending_but_success_waits() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let token = LLMEngineOutput {
            token_ids: vec![42],
            ..Default::default()
        };
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([
                Annotated::from_data(token),
                Annotated::from_data(LLMEngineOutput::length()),
            ])),
            context.clone(),
        );
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context.clone(),
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        assert_eq!(response.next().await.unwrap().data.unwrap().token_ids, [42]);
        assert!(response.next().now_or_never().is_none());
        tx.send(Ok(None)).unwrap();
        assert_eq!(
            response.next().await.unwrap().data.unwrap().finish_reason,
            Some(FinishReason::Length)
        );
        assert!(response.next().await.is_none());
        assert!(!context.is_stopped());
    }

    #[tokio::test]
    async fn late_prefill_failure_replaces_decode_success() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_data(
                LLMEngineOutput::length(),
            )])),
            context.clone(),
        );
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context,
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        assert!(response.next().now_or_never().is_none());
        tx.send(failure()).unwrap();
        assert!(
            response
                .next()
                .await
                .unwrap()
                .err()
                .unwrap()
                .to_string()
                .contains("prefill connection lost")
        );
        assert!(response.next().await.is_none());
    }

    #[tokio::test]
    async fn decode_failure_does_not_wait_for_prefill() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_error("decode failed")])),
            context.clone(),
        );
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context,
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        assert!(
            response
                .next()
                .now_or_never()
                .unwrap()
                .unwrap()
                .err()
                .unwrap()
                .to_string()
                .contains("decode failed")
        );
        tx.send(Ok(None)).unwrap();
    }

    #[tokio::test]
    async fn completed_prefill_leaves_decode_stream_unchanged() {
        let (tx, task) = pending_prefill();
        tx.send(Ok(None)).unwrap();
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_data(
                LLMEngineOutput::length(),
            )])),
            context.clone(),
        );
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context.clone(),
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        assert_eq!(
            response.next().await.unwrap().data.unwrap().finish_reason,
            Some(FinishReason::Length)
        );
        assert!(response.next().await.is_none());
        assert!(!context.is_stopped());
    }

    #[tokio::test]
    async fn disconnected_client_does_not_drop_prefill_transfer() {
        let (release_tx, release_rx) = oneshot::channel();
        let (finished_tx, finished_rx) = oneshot::channel();
        let task = PrefillTask::spawn(
            async move {
                release_rx.await.unwrap();
                finished_tx.send(()).unwrap();
                Ok(None)
            },
            Arc::new(Controller::default()),
        );
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> =
            ResponseStream::new(Box::pin(stream::pending()), context.clone());
        let response = task
            .forward_decode(
                async { Ok(decode) },
                context.clone(),
                PrefillNeed::None,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        context.stop_generating();
        drop(response);
        // Even the dropped response observer must leave prefill able to finish.
        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), finished_rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn emitted_prefill_output_precedes_ready_decode_tokens() {
        for prefill_first in [false, true] {
            let (tx, task) = pending_prefill();
            let mut tx = Some(tx);
            let output = serde_json::json!({"sglang_response": {}});
            if prefill_first {
                tx.take().unwrap().send(Ok(Some(output.clone()))).unwrap();
                tokio::task::yield_now().await;
            }
            let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
                Box::pin(stream::iter([Annotated::from_data(
                    LLMEngineOutput::length(),
                )])),
                Arc::new(Controller::default()),
            );
            let context = Arc::new(Controller::default());
            let mut response = task
                .forward_decode(
                    async { Ok(decode) },
                    context.clone(),
                    PrefillNeed::BeforeFirst,
                    Arc::new(Controller::default()),
                )
                .await
                .unwrap();
            if !prefill_first {
                assert!(response.next().now_or_never().is_none());
                tx.take().unwrap().send(Ok(Some(output.clone()))).unwrap();
            }
            let item = response.next().await.unwrap();
            assert_eq!(item.event.as_deref(), Some(PREFILL_OUTPUT_ANNOTATION));
            assert_eq!(item.data.unwrap().engine_data, Some(output));
            assert!(
                response
                    .next()
                    .await
                    .unwrap()
                    .data
                    .unwrap()
                    .finish_reason
                    .is_some()
            );
        }

        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> =
            ResponseStream::new(Box::pin(stream::pending()), context.clone());
        let response = task.forward_decode(
            async { Ok(decode) },
            context.clone(),
            PrefillNeed::BeforeFirst,
            Arc::new(Controller::default()),
        );
        let mut response = response.await.unwrap();
        tx.send(failure()).unwrap();
        assert!(response.next().await.unwrap().err().is_some());
        assert!(context.is_stopped());
    }

    #[tokio::test]
    async fn prefill_output_before_last_does_not_hold_decode_tokens() {
        let (tx, task) = pending_prefill();
        let token = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![7],
            ..Default::default()
        });
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([
                token,
                Annotated::from_data(LLMEngineOutput::length()),
            ])),
            Arc::new(Controller::default()),
        );
        let context = Arc::new(Controller::default());
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                context,
                PrefillNeed::BeforeLast,
                Arc::new(Controller::default()),
            )
            .await
            .unwrap();
        assert_eq!(response.next().await.unwrap().data.unwrap().token_ids, [7]);
        let terminal = response.next();
        tokio::pin!(terminal);
        assert!((&mut terminal).now_or_never().is_none());
        let output = serde_json::json!({"sglang_response": {}});
        tx.send(Ok(Some(output.clone()))).unwrap();
        let item = terminal.await.unwrap();
        assert_eq!(item.event.as_deref(), Some(PREFILL_OUTPUT_ANNOTATION));
        assert_eq!(item.data.unwrap().engine_data, Some(output));
        assert!(
            response
                .next()
                .await
                .unwrap()
                .data
                .unwrap()
                .finish_reason
                .is_some()
        );
    }

    /// Preserve prefill metadata and failures without polling an ended decode stream again.
    #[tokio::test]
    async fn before_first_does_not_repoll_ended_decode() {
        for result in [
            Ok(None),
            Ok(Some(serde_json::json!({"prefill": true}))),
            failure(),
        ] {
            let expected = result.as_ref().ok().cloned().flatten();
            let failed = result.is_err();
            let (tx, task) = pending_prefill();
            let context = Arc::new(Controller::default());
            let mut ended = false;
            let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
                Box::pin(stream::poll_fn(move |_| {
                    assert!(!ended, "decode stream polled after returning None");
                    ended = true;
                    std::task::Poll::Ready(None)
                })),
                context.clone(),
            );
            let mut response = task
                .forward_decode(
                    async { Ok(decode) },
                    context.clone(),
                    PrefillNeed::BeforeFirst,
                    Arc::new(Controller::default()),
                )
                .await
                .unwrap();
            assert!(response.next().now_or_never().is_none());
            tx.send(result).unwrap();
            let outputs = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                response.collect::<Vec<_>>(),
            )
            .await
            .unwrap();
            if failed {
                assert_eq!(outputs.len(), 1);
                assert!(outputs[0].err().is_some());
                assert!(context.is_stopped());
            } else if let Some(expected) = expected {
                assert_eq!(outputs.len(), 1);
                let output = outputs.into_iter().next().unwrap();
                assert_eq!(output.event.as_deref(), Some(PREFILL_OUTPUT_ANNOTATION));
                assert_eq!(output.data.unwrap().engine_data, Some(expected));
            } else {
                assert!(outputs.is_empty());
            }
        }
    }

    async fn assert_dispatch_rejection_cancels_prefill(output: PrefillNeed, error_type: ErrorType) {
        let case = format!("output={output:?}, error={error_type:?}");
        let prefill_context = Arc::new(Controller::default());
        let task_context = prefill_context.clone();
        let (cleaned_tx, cleaned_rx) = oneshot::channel();
        let task = PrefillTask::spawn(
            async move {
                task_context.stopped().await;
                let _ = cleaned_tx.send(());
                Ok(None)
            },
            prefill_context.clone(),
        );
        let error = dynamo_runtime::error::DynamoError::builder()
            .error_type(error_type)
            .message("decode admission rejected")
            .build();
        let error = anyhow::Error::new(error).context("decode admission rejected");
        let request_context = Arc::new(Controller::default());
        let result = task
            .forward_decode(
                async { Err(error) },
                request_context,
                output,
                prefill_context.clone(),
            )
            .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("decode admission rejected")
        );
        assert!(
            prefill_context.is_stopped(),
            "rejected decode left independent prefill running: {case}"
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), cleaned_rx)
            .await
            .unwrap()
            .unwrap();
    }

    async fn assert_stream_rejection_cancels_prefill(output: PrefillNeed, error_type: ErrorType) {
        let case = format!("output={output:?}, error={error_type:?}");
        let prefill_context = Arc::new(Controller::default());
        let task_context = prefill_context.clone();
        let (cleaned_tx, cleaned_rx) = oneshot::channel();
        let task = PrefillTask::spawn(
            async move {
                task_context.stopped().await;
                let _ = cleaned_tx.send(());
                Ok(None)
            },
            prefill_context.clone(),
        );
        let request_context = Arc::new(Controller::default());
        let error = dynamo_runtime::error::DynamoError::builder()
            .error_type(error_type)
            .message("decode stream admission rejected")
            .build();
        let decode: ManyOut<Annotated<LLMEngineOutput>> = ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_err(error)])),
            request_context.clone(),
        );
        let mut response = task
            .forward_decode(
                async { Ok(decode) },
                request_context,
                output,
                prefill_context.clone(),
            )
            .await
            .unwrap();
        let item = tokio::time::timeout(std::time::Duration::from_secs(1), response.next())
            .await
            .unwrap_or_else(|_| {
                panic!("decode admission error blocked behind prefill output: {case}")
            })
            .unwrap();
        assert!(
            item.err()
                .unwrap()
                .to_string()
                .contains("decode stream admission rejected")
        );
        assert!(
            prefill_context.is_stopped(),
            "stream rejection left independent prefill running: {case}"
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), cleaned_rx)
            .await
            .unwrap()
            .unwrap();
    }

    async fn assert_ambiguous_decode_error_preserves_prefill(output: PrefillNeed) {
        let (tx, task) = pending_prefill();
        let prefill_context = Arc::new(Controller::default());
        let request_context = Arc::new(Controller::default());
        let error = dynamo_runtime::error::DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::Disconnected)
            .message("decode connection lost after possible dispatch")
            .build();
        let result = task
            .forward_decode(
                async { Err(error.into()) },
                request_context,
                output,
                prefill_context.clone(),
            )
            .await;
        assert!(result.is_err());
        assert!(
            !prefill_context.is_stopped(),
            "ambiguous failure must preserve KV transfer: {output:?}"
        );
        tx.send(Ok(None)).unwrap();
    }

    #[tokio::test]
    async fn decode_admission_rejection_cancels_prefill() {
        for output in [
            PrefillNeed::None,
            PrefillNeed::BeforeLast,
            PrefillNeed::BeforeFirst,
        ] {
            for error_type in [
                ErrorType::WorkerOverloaded,
                ErrorType::ResourceExhausted,
                ErrorType::CapacityExhausted,
                ErrorType::Backend(BackendError::WorkerOverloaded),
            ] {
                assert_dispatch_rejection_cancels_prefill(output, error_type).await;
            }
        }
    }

    #[tokio::test]
    async fn decode_stream_admission_rejection_cancels_prefill() {
        for output in [
            PrefillNeed::None,
            PrefillNeed::BeforeLast,
            PrefillNeed::BeforeFirst,
        ] {
            for error_type in [
                ErrorType::WorkerOverloaded,
                ErrorType::ResourceExhausted,
                ErrorType::CapacityExhausted,
                ErrorType::Backend(BackendError::WorkerOverloaded),
            ] {
                assert_stream_rejection_cancels_prefill(output, error_type).await;
            }
        }
    }

    #[tokio::test]
    async fn ambiguous_decode_dispatch_error_preserves_prefill_transfer() {
        for output in [
            PrefillNeed::None,
            PrefillNeed::BeforeLast,
            PrefillNeed::BeforeFirst,
        ] {
            assert_ambiguous_decode_error_preserves_prefill(output).await;
        }
    }
}
