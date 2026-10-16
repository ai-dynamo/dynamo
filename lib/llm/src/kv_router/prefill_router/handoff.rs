// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep consuming prefill while decode runs, and return failures to the caller.

use std::{future::Future, sync::Arc};

use dynamo_runtime::{
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
    pub(super) fn spawn(
        future: impl Future<Output = Result<PrefillOutput, PrefillError>> + Send + 'static,
    ) -> Self {
        Self(tokio::spawn(
            async move {
                let result = future.await;
                if let Err(error) = &result {
                    // Keep evidence even if the client has already disconnected.
                    tracing::warn!(%error, "Prefill background task failed");
                }
                result
            }
            .instrument(tracing::Span::current()),
        ))
    }

    pub(super) async fn wait(self) -> Result<PrefillOutput, PrefillError> {
        self.0.await.map_err(|error| {
            PrefillError::PrefillError(
                format!("Prefill background task did not complete: {error}"),
                Some(Box::new(error)),
            )
        })?
    }

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
            Some(FinishReason::Cancelled) => Err(PrefillError::PrefillError(
                "Prefill backend cancelled before completing the handoff".to_string(),
                None,
            )),
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
            result = &mut decode => result?,
        };
        // Move the completion future into the response stream. The task itself
        // keeps running if this stream is dropped during client cancellation.
        let context = response.context();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
            if output == PrefillNeed::BeforeFirst {
                match completion.as_mut().await {
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::context::Controller;
    use futures::{FutureExt, stream};
    use tokio::sync::oneshot;

    fn pending_prefill() -> (
        oneshot::Sender<Result<PrefillOutput, PrefillError>>,
        PrefillTask,
    ) {
        let (tx, rx) = oneshot::channel();
        let task = PrefillTask::spawn(async move { rx.await.unwrap() });
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
            .forward_decode(async { Ok(decode) }, context.clone(), PrefillNeed::None)
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
            .forward_decode(async { Ok(decode) }, context.clone(), PrefillNeed::None)
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
            .forward_decode(async { Ok(decode) }, context, PrefillNeed::None)
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
            .forward_decode(async { Ok(decode) }, context, PrefillNeed::None)
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
            .forward_decode(async { Ok(decode) }, context.clone(), PrefillNeed::None)
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
        let task = PrefillTask::spawn(async move {
            release_rx.await.unwrap();
            finished_tx.send(()).unwrap();
            Ok(None)
        });
        let context = Arc::new(Controller::default());
        let decode: ManyOut<Annotated<LLMEngineOutput>> =
            ResponseStream::new(Box::pin(stream::pending()), context.clone());
        let response = task
            .forward_decode(async { Ok(decode) }, context.clone(), PrefillNeed::None)
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
            .forward_decode(async { Ok(decode) }, context, PrefillNeed::BeforeLast)
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
}
