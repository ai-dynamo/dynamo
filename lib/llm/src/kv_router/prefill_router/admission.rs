// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_kv_router::selector::WorkerSelector;

use dynamo_runtime::{
    pipeline::{ManyOut, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::{
    kv_router::KvPushRouter,
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::{
        llm_backend::{FinishReason, LLMEngineOutput, PreprocessedRequest},
        timing::RequestTracker,
    },
    session_affinity::{AffinityTarget, SessionAffinityPushRouter},
};

pub(super) enum InnerPrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    KvRouter(Arc<KvPushRouter<Sel>>),
    SimpleRouter(Arc<SessionAffinityPushRouter>),
}

impl<Sel> InnerPrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(super) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>)>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M>,
    {
        match self {
            InnerPrefillRouter::KvRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
        }
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(super) async fn consume_prefill_stream(
        mut prefill_response: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Result<PrefillCompletion, PrefillError> {
        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        if let Some(error) = first_output.err() {
            return Err(PrefillError::PrefillError(
                "Prefill router returned error in output".to_string(),
                Some(Box::new(error)),
            ));
        }

        if let Some(ref tracker) = tracker {
            tracker.record_prefill_complete();
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

        if !is_bootstrap {
            while let Some(next) = prefill_response.next().await {
                if let Some(error) = next.err() {
                    return Err(PrefillError::PrefillError(
                        "Prefill router returned error in output stream".to_string(),
                        Some(Box::new(error)),
                    ));
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
        } else {
            tokio::spawn(async move { while prefill_response.next().await.is_some() {} });
        }

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
        })
    }

    pub(super) fn spawn_prefill_task(
        &self,
        prefill_stream: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        phase_transition_permit: OwnedSemaphorePermit,
    ) {
        let span = tracing::Span::current();
        tokio::spawn(
            async move {
                drop(phase_transition_permit);
                match Self::consume_prefill_stream(prefill_stream, tracker).await {
                    Ok(_) => tracing::debug!("Prefill background task completed"),
                    Err(error) => tracing::warn!("Prefill background task error: {error:?}"),
                }
            }
            .instrument(span),
        );
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::selector::DefaultWorkerSelector;
    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::pipeline::{ResponseStream, context::Controller};

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

    #[tokio::test]
    async fn first_output_error_does_not_record_prefill_complete() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_error("prefill failed")]),
            Some(tracker.clone()),
        )
        .await;

        assert!(result.is_err());
        assert!(tracker.record_prefill_complete());
    }

    #[tokio::test]
    async fn later_output_error_is_propagated_after_prefill_arrival() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
            prefill_stream(vec![
                valid_prefill_output(),
                Annotated::from_error("prefill stream failed"),
            ]),
            Some(tracker.clone()),
        )
        .await;

        assert!(result.is_err());
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
            let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
                prefill_stream(vec![Annotated::from_data(output)]),
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
        let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_data(output)]),
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
        let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_data(output)]),
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
            let result = PrefillRouter::<DefaultWorkerSelector>::consume_prefill_stream(
                prefill_stream(vec![Annotated::from_data(output)]),
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
