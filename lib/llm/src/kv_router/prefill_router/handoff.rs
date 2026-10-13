// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep consuming prefill while decode runs, and return failures to the caller.

use std::{future::Future, sync::Arc};

use dynamo_runtime::{
    pipeline::{AsyncEngineContext, AsyncEngineContextProvider, ManyOut, ResponseStream},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};
use futures::StreamExt;
use tokio::task::JoinHandle;
use tracing::Instrument;

use super::PrefillError;
use crate::protocols::{
    common::llm_backend::{FinishReason, LLMEngineOutput},
    response_merge::PREFILL_OUTPUT_ANNOTATION,
};

/// Prefill's last `engine_data`, kept for frontends that join prompt-side
/// output into the decode response.
pub(super) type PrefillOutput = Option<serde_json::Value>;

/// Dropping the observer detaches the task: client disconnects must not abort a
/// prefill whose KV transfer may still be needed by the decode worker.
pub(super) struct PrefillTask(JoinHandle<Result<PrefillOutput, PrefillError>>);

/// Remember the newest `engine_data` a prefill output carries.
pub(super) fn keep_engine_data(output: &Annotated<LLMEngineOutput>, last: &mut PrefillOutput) {
    if let Some(engine_data) = output
        .data
        .as_ref()
        .and_then(|data| data.engine_data.as_ref())
    {
        *last = Some(engine_data.clone());
    }
}

impl PrefillTask {
    pub(super) fn ready(output: PrefillOutput) -> Self {
        Self::spawn(async move { Ok(output) })
    }

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
    pub(super) async fn forward_decode(
        self,
        decode: impl Future<Output = anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>>>,
        request_context: Arc<dyn AsyncEngineContext>,
    ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let mut completion = Box::pin(self.wait());
        tokio::pin!(decode);
        let mut response = tokio::select! {
            biased;
            result = &mut completion => {
                if let Err(error) = result {
                    request_context.stop_generating();
                    return Err(error.into());
                }
                return decode.await;
            }
            result = &mut decode => result?,
        };
        // Move the completion future into the response stream. The task itself
        // keeps running if this stream is dropped during client cancellation.
        let context = response.context();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    result = &mut completion => {
                        if let Err(error) = result {
                            request_context.stop_generating();
                            stream_context.stop_generating();
                            yield Annotated::from_err(error);
                            return;
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
                        if terminal
                            && let Err(error) = completion.as_mut().await
                        {
                            request_context.stop_generating();
                            stream_context.stop_generating();
                            yield Annotated::from_err(error);
                            return;
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

    /// Like [`Self::forward_decode`], but the stream first waits for prefill,
    /// then sends its output as a [`PREFILL_OUTPUT_ANNOTATION`] event, then the
    /// decode outputs. Prefill is done before any decode output, so a late
    /// prefill failure cannot follow decode tokens.
    pub(super) async fn forward_decode_with_output(
        self,
        decode: impl Future<Output = anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>>>,
        request_context: Arc<dyn AsyncEngineContext>,
    ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let mut completion = Box::pin(self.wait());
        tokio::pin!(decode);
        let mut finished = None;
        let mut response = tokio::select! {
            biased;
            result = &mut completion => {
                match result {
                    Ok(output) => finished = Some(output),
                    Err(error) => {
                        request_context.stop_generating();
                        return Err(error.into());
                    }
                }
                decode.await?
            }
            result = &mut decode => result?,
        };
        let context = response.context();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
            let output = match finished {
                Some(output) => Ok(output),
                None => completion.await,
            };
            match output {
                Err(error) => {
                    request_context.stop_generating();
                    stream_context.stop_generating();
                    yield Annotated::from_err(error);
                    return;
                }
                Ok(Some(output)) => {
                    match Annotated::from_annotation(PREFILL_OUTPUT_ANNOTATION, &output) {
                        Ok(annotation) => yield annotation,
                        Err(error) => tracing::warn!(%error, "cannot encode prefill output"),
                    }
                }
                Ok(None) => {}
            }
            while let Some(output) = response.next().await {
                yield output;
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
        let mut response = Box::pin(task.forward_decode(std::future::pending(), context.clone()));
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
            .forward_decode(async { Ok(decode) }, context.clone())
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
            .forward_decode(async { Ok(decode) }, context.clone())
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
            .forward_decode(async { Ok(decode) }, context)
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
            .forward_decode(async { Ok(decode) }, context)
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
            .forward_decode(async { Ok(decode) }, context.clone())
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
            .forward_decode(async { Ok(decode) }, context.clone())
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

    fn tokens_then_length() -> ManyOut<Annotated<LLMEngineOutput>> {
        let token = LLMEngineOutput {
            token_ids: vec![42],
            ..Default::default()
        };
        ResponseStream::new(
            Box::pin(stream::iter([
                Annotated::from_data(token),
                Annotated::from_data(LLMEngineOutput::length()),
            ])),
            Arc::new(Controller::default()),
        )
    }

    #[tokio::test]
    async fn prefill_output_precedes_ready_decode_tokens() {
        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode = tokens_then_length();
        let mut response = task
            .forward_decode_with_output(async { Ok(decode) }, context.clone())
            .await
            .unwrap();
        assert!(response.next().now_or_never().is_none());
        tx.send(Ok(Some(serde_json::json!({"prompt": [1, 2]}))))
            .unwrap();
        let annotation = response.next().await.unwrap();
        assert_eq!(annotation.event.as_deref(), Some(PREFILL_OUTPUT_ANNOTATION));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&annotation.comment.unwrap()[0]).unwrap(),
            serde_json::json!({"prompt": [1, 2]})
        );
        assert_eq!(response.next().await.unwrap().data.unwrap().token_ids, [42]);
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
        assert!(response.next().await.is_none());
    }

    #[tokio::test]
    async fn finished_prefill_output_is_kept_while_decode_dispatches() {
        let (tx, task) = pending_prefill();
        tx.send(Ok(Some(serde_json::json!({"done": true}))))
            .unwrap();
        tokio::task::yield_now().await;
        let (decode_tx, decode_rx) = oneshot::channel();
        let response = task.forward_decode_with_output(
            async move { Ok(decode_rx.await.unwrap()) },
            Arc::new(Controller::default()),
        );
        decode_tx.send(tokens_then_length()).ok();
        let events: Vec<_> = response.await.unwrap().collect().await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event.as_deref(), Some(PREFILL_OUTPUT_ANNOTATION));
    }

    #[tokio::test]
    async fn prefill_without_output_or_with_failure() {
        let task = PrefillTask::ready(None);
        let events: Vec<_> = task
            .forward_decode_with_output(
                async { Ok(tokens_then_length()) },
                Arc::new(Controller::default()),
            )
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| event.event.is_none()));

        let (tx, task) = pending_prefill();
        let context = Arc::new(Controller::default());
        let decode = tokens_then_length();
        let response = task
            .forward_decode_with_output(async { Ok(decode) }, context.clone())
            .await
            .unwrap();
        tx.send(failure()).unwrap();
        let events: Vec<_> = response.collect().await;
        assert_eq!(events.len(), 1);
        assert!(events[0].err().is_some());
        assert!(context.is_stopped());
    }
}
