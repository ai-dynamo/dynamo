// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;

use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, ManyOut},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter, handoff::PrefillTask};
use crate::protocols::common::{
    llm_backend::{FinishReason, LLMEngineOutput},
    timing::RequestTracker,
};

impl PrefillRouter {
    pub(super) async fn consume_prefill_stream(
        mut prefill_response: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
    ) -> Result<PrefillCompletion, PrefillError> {
        let Some(first_output) = prefill_response.next().await else {
            // The egress ends the response stream as soon as the request context is
            // stopped, before the worker's final item can arrive. Once a decode
            // rejection has stopped this prefill, an empty stream is that cancellation.
            if prefill_response.context().is_stopped() {
                return Err(PrefillError::Cancelled);
            }
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        if let Some(error) = first_output.err() {
            // Include the worker's text. `to_pyerr` keeps only `Display`, so a
            // `#[source]` is lost. See `PrefillError::PrefillError`.
            let detail = format!("Prefill router returned error in output: {error}");
            return Err(PrefillError::PrefillError(detail, Some(Box::new(error))));
        }

        let mut prompt_tokens_details = first_output
            .data
            .as_ref()
            .and_then(|output| output.completion_usage.as_ref())
            .and_then(|usage| usage.prompt_tokens_details.clone());

        // For SGLang, check if the first output is a bootstrap message.
        let is_bootstrap = first_output
            .data
            .as_ref()
            .and_then(|o| o.disaggregated_params.as_ref())
            .and_then(|p| p.as_object())
            .is_some_and(|obj| {
                obj.contains_key("bootstrap_host")
                    && obj.contains_key("bootstrap_port")
                    && obj.contains_key("bootstrap_room")
            });

        let completion = if !is_bootstrap {
            if let Some(ref tracker) = tracker {
                tracker.record_prefill_complete();
            }
            while let Some(next) = prefill_response.next().await {
                if let Some(error) = next.err() {
                    let detail = format!("Prefill router returned error in output stream: {error}");
                    return Err(PrefillError::PrefillError(detail, Some(Box::new(error))));
                }
                if let Some(output) = next.data.as_ref()
                    && prompt_tokens_details.is_none()
                {
                    prompt_tokens_details = output
                        .completion_usage
                        .as_ref()
                        .and_then(|usage| usage.prompt_tokens_details.clone());
                }
            }
            None
        } else {
            let prefill_context = prefill_response.context();
            Some(PrefillTask::spawn(
                async move {
                    let _task_guard = task_guard;
                    let mut prefill_output = None;
                    while let Some(output) = prefill_response.next().await {
                        PrefillTask::check_output(&output)?;
                        if let Some(engine_data) = output.data.and_then(|data| data.engine_data) {
                            prefill_output = Some(engine_data);
                        }
                    }
                    // Bootstrap details unblock decode, but do not mean prefill is
                    // complete. Record completion only after its stream finishes.
                    if let Some(tracker) = tracker {
                        tracker.record_prefill_complete();
                    }
                    Ok(prefill_output)
                },
                prefill_context,
            ))
        };

        // A CTX request that reaches EOS/stop during its one-token prefill step
        // is already complete and does not establish a KV-cache handoff. The
        // prefill protocol carries this classification on the first output's
        // data; a data-less first output is rejected below. A missing finish
        // reason is equivalent to TRT-LLM's "not_finished", while Length still
        // requires the normal GEN handoff.
        let is_terminal = first_output
            .data
            .as_ref()
            .and_then(|output| output.finish_reason.as_ref())
            .is_some_and(|reason| !matches!(reason, FinishReason::Length));
        if is_terminal {
            return Ok(PrefillCompletion::Terminal {
                output: Box::new(first_output),
            });
        }

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };
        let Some(disaggregated_params) = output.disaggregated_params.clone() else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };
        // TRT-LLM serializes ctx_request_id as null for a terminal context
        // response. Terminal responses returned above do not need a handoff;
        // any non-terminal response that explicitly carries a null ID cannot
        // be decoded safely. Refuse it here instead of dispatching GEN and
        // failing later inside the backend. Other backends do not expose this
        // TRT-LLM-specific field and are unaffected.
        let ctx_request_id = disaggregated_params.get("ctx_request_id");
        let is_trtllm_context_handoff = disaggregated_params
            .get("request_type")
            .and_then(serde_json::Value::as_str)
            == Some("context_only");
        if ctx_request_id.is_some_and(serde_json::Value::is_null)
            || (is_trtllm_context_handoff && ctx_request_id.is_none())
        {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no usable ctx_request_id for a non-terminal handoff"
                    .to_string(),
            ));
        }

        Ok(PrefillCompletion::Handoff {
            result: crate::protocols::common::preprocessor::PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            worker_link: output.worker_trace_link.clone(),
            completion,
        })
    }

    pub(super) fn spawn_prefill_task(
        &self,
        prefill_stream: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        phase_transition_permit: OwnedSemaphorePermit,
    ) -> PrefillTask {
        let task_guard = self.task_guard.clone();
        let prefill_context = prefill_stream.context();
        PrefillTask::spawn_unlogged(async move {
            drop(phase_transition_permit);
            let result =
                match Self::consume_prefill_stream(prefill_stream, tracker, task_guard).await {
                    Ok(PrefillCompletion::Handoff {
                        completion: Some(task),
                        ..
                    }) => {
                        // The inner observer already logs its own failures.
                        return task.wait().await;
                    }
                    Ok(PrefillCompletion::Terminal { output }) => {
                        PrefillTask::check_output(&output).map(|()| None)
                    }
                    Ok(_) => Ok(None),
                    Err(error) => Err(error),
                };
            result.inspect_err(|error| PrefillTask::log_failure(error, prefill_context.as_ref()))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::protocols::response_merge::PrefillNeed;
    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::pipeline::{AsyncEngineContext, ResponseStream, context::Controller};

    use super::*;

    fn prefill_stream(
        items: Vec<Annotated<LLMEngineOutput>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        ResponseStream::new(
            Box::pin(stream::iter(items)),
            Arc::new(Controller::default()),
        )
    }

    fn valid_prefill_output() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            disaggregated_params: Some(json!({})),
            ..Default::default()
        })
    }

    /// Intentional prefill aborts must be quiet without hiding real tail failures.
    #[test]
    fn bootstrap_cancellation_logging_distinguishes_expected_abort() {
        use std::sync::Mutex;
        use tracing_subscriber::{Layer, layer::SubscriberExt};

        struct Levels(Arc<Mutex<Vec<tracing::Level>>>);
        impl<S: tracing::Subscriber> Layer<S> for Levels {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event
                    .metadata()
                    .target()
                    .ends_with("prefill_router::handoff")
                {
                    self.0.lock().unwrap().push(*event.metadata().level());
                }
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for wrapped in [false, true] {
            for bootstrap in [true, false] {
                if !wrapped && !bootstrap {
                    continue;
                }
                for stopped in [true, false] {
                    // `None` ends the stream without a tail item, as the egress does
                    // once the context is stopped; it only reaches the observer when
                    // no bootstrap item arrived first.
                    let mut tails = vec![
                        Some(Annotated::from_data(LLMEngineOutput::cancelled())),
                        Some(Annotated::from_data(LLMEngineOutput::error(
                            "prefill failed".to_string(),
                        ))),
                        Some(Annotated::from_error("prefill transport failed")),
                    ];
                    if !bootstrap {
                        tails.push(None);
                    }
                    for tail in tails {
                        let ended = tail.is_none();
                        let cancelled = tail.as_ref().is_some_and(|tail| {
                            tail.data.as_ref().is_some_and(|data| {
                                data.finish_reason == Some(FinishReason::Cancelled)
                            })
                        });
                        let levels = Arc::new(Mutex::new(Vec::new()));
                        let subscriber =
                            tracing_subscriber::registry().with(Levels(levels.clone()));
                        let run = async {
                            let context = Arc::new(Controller::default());
                            let first = Annotated::from_data(LLMEngineOutput {
                                disaggregated_params: Some(json!({
                                    "bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7,
                                })),
                                ..Default::default()
                            });
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            let stream = stream::iter(bootstrap.then_some(first)).chain(
                                stream::once(async move { rx.await.unwrap() })
                                    .filter_map(|tail| async move { tail }),
                            );
                            let response = ResponseStream::new(Box::pin(stream), context.clone());
                            let task = if wrapped {
                                let router = super::super::tests::active_conditional_router();
                                let tracker = Arc::new(RequestTracker::new());
                                let permit = tracker
                                    .set_phase(
                                        crate::protocols::common::timing::RequestPhase::Prefill,
                                    )
                                    .await;
                                router.spawn_prefill_task(response, Some(tracker), permit)
                            } else {
                                let result =
                                    PrefillRouter::consume_prefill_stream(response, None, None)
                                        .await
                                        .unwrap();
                                let PrefillCompletion::Handoff {
                                    completion: Some(task),
                                    ..
                                } = result
                                else {
                                    panic!("expected bootstrap completion observer");
                                };
                                task
                            };
                            if stopped {
                                context.stop_generating();
                            }
                            tx.send(tail).unwrap();
                            assert!(
                                tokio::time::timeout(
                                    std::time::Duration::from_secs(1),
                                    task.wait()
                                )
                                .await
                                .unwrap()
                                .is_err()
                            );
                        };
                        tracing::subscriber::with_default(subscriber, || runtime.block_on(run));
                        let expected = if stopped && (cancelled || ended) {
                            tracing::Level::DEBUG
                        } else {
                            tracing::Level::WARN
                        };
                        assert_eq!(
                            *levels.lock().unwrap(),
                            vec![expected],
                            "wrapped={wrapped}, bootstrap={bootstrap}, stopped={stopped}, cancelled={cancelled}, ended={ended}"
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn bootstrap_tail_failures_reach_completion_observer() {
        for failure in [
            Annotated::from_error("connection closed during prefill"),
            Annotated::from_data(LLMEngineOutput::error("backend prefill failed".to_string())),
            Annotated::from_data(LLMEngineOutput::cancelled()),
        ] {
            let first = Annotated::from_data(LLMEngineOutput {
                disaggregated_params: Some(json!({
                    "bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7,
                })),
                ..Default::default()
            });
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let stream = stream::iter([first]).chain(stream::once(async move {
                release_rx.await.unwrap();
                failure
            }));
            let response = ResponseStream::new(Box::pin(stream), Arc::new(Controller::default()));
            let result = PrefillRouter::consume_prefill_stream(response, None, None)
                .await
                .unwrap();
            let PrefillCompletion::Handoff {
                completion: Some(completion),
                ..
            } = result
            else {
                panic!("expected an early handoff with a completion observer");
            };
            release_tx.send(()).unwrap();
            let context = Arc::new(Controller::default());
            let decode: ManyOut<Annotated<LLMEngineOutput>> =
                ResponseStream::new(Box::pin(stream::pending()), context.clone());
            match completion
                .forward_decode(
                    async { Ok(decode) },
                    context,
                    PrefillNeed::None,
                    Arc::new(Controller::default()),
                )
                .await
            {
                Err(error) => assert!(error.to_string().contains("Prefill")),
                Ok(mut response) => {
                    let output =
                        tokio::time::timeout(std::time::Duration::from_secs(1), response.next())
                            .await
                            .unwrap()
                            .unwrap();
                    assert!(output.err().is_some());
                    assert!(response.next().await.is_none());
                }
            }
        }
    }

    #[tokio::test]
    async fn bootstrap_drain_retains_teardown_guard_until_stream_finishes() {
        let first = Annotated::from_data(LLMEngineOutput {
            disaggregated_params: Some(json!({
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 1,
                "bootstrap_room": "test",
                "ctx_request_id": 42,
            })),
            ..Default::default()
        });
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let stream = stream::iter([first]).chain(stream::once(async move {
            let _ = release_rx.await;
            Annotated::from_data(LLMEngineOutput::default())
        }));
        let response = ResponseStream::new(Box::pin(stream), Arc::new(Controller::default()));
        let teardown = Arc::new(());
        let teardown_weak = Arc::downgrade(&teardown);
        let task_guard: dynamo_runtime::engine::EngineContextGuard = teardown.clone();
        drop(teardown);

        PrefillRouter::consume_prefill_stream(response, None, Some(task_guard))
            .await
            .unwrap();
        assert!(teardown_weak.upgrade().is_some());

        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while teardown_weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("bootstrap drain did not release its teardown guard");
    }

    #[tokio::test]
    async fn first_output_error_does_not_record_prefill_complete() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_error("prefill failed")]),
            Some(tracker.clone()),
            None,
        )
        .await;

        // The text must survive: `to_pyerr` only keeps `Display`.
        let Err(err) = result else {
            panic!("expected a first output error");
        };
        assert!(err.to_string().contains("prefill failed"), "{err}");
        assert!(tracker.record_prefill_complete());
    }

    #[tokio::test]
    async fn later_output_error_is_propagated_after_prefill_arrival() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![
                valid_prefill_output(),
                Annotated::from_error("prefill stream failed"),
            ]),
            Some(tracker.clone()),
            None,
        )
        .await;

        let Err(err) = result else {
            panic!("expected a later output error");
        };
        assert!(err.to_string().contains("prefill stream failed"), "{err}");
        assert!(!tracker.record_prefill_complete());
    }

    #[tokio::test]
    async fn terminal_finish_reasons_without_handoff_are_returned_to_caller() {
        for finish_reason in [
            FinishReason::EoS,
            FinishReason::Stop,
            FinishReason::Cancelled,
            FinishReason::Error("prefill failed".to_string()),
            FinishReason::ContentFilter,
        ] {
            let output = LLMEngineOutput {
                token_ids: vec![2],
                finish_reason: Some(finish_reason.clone()),
                ..Default::default()
            };
            let result = PrefillRouter::consume_prefill_stream(
                prefill_stream(vec![Annotated::from_data(output)]),
                None,
                None,
            )
            .await
            .unwrap();

            let PrefillCompletion::Terminal { output } = result else {
                panic!("expected terminal prefill completion for {finish_reason:?}");
            };
            let output = *output;
            assert_eq!(
                output.data.and_then(|data| data.finish_reason),
                Some(finish_reason)
            );
        }
    }

    #[tokio::test]
    async fn length_limited_prefill_still_requires_handoff() {
        let output = LLMEngineOutput {
            finish_reason: Some(FinishReason::Length),
            disaggregated_params: Some(json!({"ctx_request_id": 42})),
            ..Default::default()
        };
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_data(output)]),
            None,
            None,
        )
        .await
        .unwrap();

        let PrefillCompletion::Handoff { result, .. } = result else {
            panic!("expected prefill handoff");
        };
        assert_eq!(result.disaggregated_params, json!({"ctx_request_id": 42}));
    }

    #[tokio::test]
    async fn unfinished_prefill_still_requires_handoff() {
        let output = LLMEngineOutput {
            finish_reason: None,
            disaggregated_params: Some(json!({"ctx_request_id": 42})),
            ..Default::default()
        };
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_data(output)]),
            None,
            None,
        )
        .await
        .unwrap();

        let PrefillCompletion::Handoff { result, .. } = result else {
            panic!("expected prefill handoff");
        };
        assert_eq!(result.disaggregated_params, json!({"ctx_request_id": 42}));
    }

    #[tokio::test]
    async fn non_terminal_prefill_without_usable_ctx_request_id_fails_before_decode() {
        for disaggregated_params in [
            json!({"ctx_request_id": null}),
            json!({"request_type": "context_only"}),
        ] {
            let output = LLMEngineOutput {
                finish_reason: None,
                disaggregated_params: Some(disaggregated_params),
                ..Default::default()
            };
            let result = PrefillRouter::consume_prefill_stream(
                prefill_stream(vec![Annotated::from_data(output)]),
                None,
                None,
            )
            .await;
            let Err(error) = result else {
                panic!("expected a missing ctx_request_id to reject the handoff");
            };

            assert!(matches!(
                error,
                PrefillError::NoDisaggregatedParams(message)
                    if message.contains("no usable ctx_request_id")
            ));
        }
    }
}
