// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::{
    kv_router::KvPushRouter,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::RequestTracker,
    },
    session_affinity::{AffinityTarget, SessionAffinityPushRouter},
};

fn merge_generate_prefill_metadata(
    accumulated: &mut Option<crate::protocols::inference::generate::GeneratePrefillMetadata>,
    output: &LLMEngineOutput,
) -> Result<(), PrefillError> {
    let Some(metadata) = output.generate_metadata.as_ref() else {
        return Ok(());
    };
    let accumulated = accumulated.get_or_insert_with(Default::default);
    if let Some(prompt_logprobs) = metadata.prompt_logprobs.as_ref() {
        if accumulated
            .prompt_logprobs
            .as_ref()
            .is_some_and(|current| current != prompt_logprobs)
        {
            return Err(PrefillError::PrefillError(
                "Prefill workers returned inconsistent prompt logprobs".to_string(),
                None,
            ));
        }
        accumulated.prompt_logprobs = Some(prompt_logprobs.clone());
    }
    if let Some(routed_experts) = metadata.routed_experts.as_ref() {
        let choice = output.index.unwrap_or(0);
        if accumulated
            .routed_experts_by_choice
            .get(&choice)
            .is_some_and(|current| current != routed_experts)
        {
            return Err(PrefillError::PrefillError(
                format!("Prefill workers returned inconsistent routed experts for choice {choice}"),
                None,
            ));
        }
        accumulated
            .routed_experts_by_choice
            .insert(choice, routed_experts.clone());
    }
    Ok(())
}

#[derive(Clone)]
pub(super) enum InnerPrefillRouter {
    KvRouter(Arc<KvPushRouter>),
    SimpleRouter(Arc<SessionAffinityPushRouter>),
}

impl InnerPrefillRouter {
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

impl PrefillRouter {
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
        let mut generate_metadata = None;
        if let Some(output) = first_output.data.as_ref() {
            merge_generate_prefill_metadata(&mut generate_metadata, output)?;
        }

        while let Some(next) = prefill_response.next().await {
            if let Some(error) = next.err() {
                return Err(PrefillError::PrefillError(
                    "Prefill router returned error in output stream".to_string(),
                    Some(Box::new(error)),
                ));
            }
            if let Some(output) = next.data.as_ref() {
                if prompt_tokens_details.is_none() {
                    prompt_tokens_details = output
                        .completion_usage
                        .as_ref()
                        .and_then(|usage| usage.prompt_tokens_details.clone());
                }
                merge_generate_prefill_metadata(&mut generate_metadata, output)?;
            }
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

        Ok(PrefillCompletion {
            result: crate::protocols::common::preprocessor::PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
                generate_metadata,
            },
            worker_link: output.worker_trace_link.clone(),
        })
    }

    pub(super) fn spawn_prefill_task(
        &self,
        prefill_stream: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        phase_transition_permit: OwnedSemaphorePermit,
    ) -> tokio::task::JoinHandle<Result<PrefillCompletion, PrefillError>> {
        let span = tracing::Span::current();
        tokio::spawn(
            async move {
                drop(phase_transition_permit);
                let result = Self::consume_prefill_stream(prefill_stream, tracker).await;
                match &result {
                    Ok(_) => tracing::debug!("Prefill background task completed"),
                    Err(error) => tracing::warn!("Prefill background task error: {error:?}"),
                };
                result
            }
            .instrument(span),
        )
    }

    pub(super) fn attach_bootstrap_generate_metadata(
        mut decode_stream: ManyOut<Annotated<LLMEngineOutput>>,
        completion: tokio::task::JoinHandle<Result<PrefillCompletion, PrefillError>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        let context = decode_stream.context();
        let stream = async_stream::stream! {
            let mut completion = Some(completion);
            let mut completion_resolved = false;
            let mut metadata = None;
            while let Some(mut annotated) = decode_stream.next().await {
                let terminal = annotated
                    .data
                    .as_ref()
                    .is_some_and(|output| output.finish_reason.is_some());
                if terminal && !completion_resolved {
                    let result = completion
                        .take()
                        .expect("bootstrap completion awaited at most once")
                        .await;
                    match result {
                        Ok(Ok(prefill)) => {
                            completion_resolved = true;
                            metadata = prefill.result.generate_metadata;
                        }
                        Ok(Err(error)) => {
                            yield Annotated::from_error(format!(
                                "asynchronous prefill metadata failed: {error}"
                            ));
                            break;
                        }
                        Err(error) => {
                            yield Annotated::from_error(format!(
                                "asynchronous prefill metadata task failed: {error}"
                            ));
                            break;
                        }
                    }
                }
                if terminal
                    && let (Some(output), Some(metadata)) =
                        (annotated.data.as_mut(), metadata.as_ref())
                    && let Err(error) = attach_prefill_metadata(output, metadata)
                {
                    yield Annotated::from_error(error.to_string());
                    break;
                }
                yield annotated;
            }
        };
        ResponseStream::new(Box::pin(stream), context)
    }
}

fn attach_prefill_metadata(
    output: &mut LLMEngineOutput,
    prefill: &crate::protocols::inference::generate::GeneratePrefillMetadata,
) -> anyhow::Result<()> {
    let metadata = output
        .generate_metadata
        .get_or_insert_with(Default::default);
    if let Some(prompt_logprobs) = prefill.prompt_logprobs.as_ref() {
        if metadata
            .prompt_logprobs
            .as_ref()
            .is_some_and(|decode| decode != prompt_logprobs)
        {
            anyhow::bail!("prefill and decode returned inconsistent prompt logprobs");
        }
        metadata.prompt_logprobs = Some(prompt_logprobs.clone());
    }
    if let Some(routed_experts) = prefill
        .routed_experts_by_choice
        .get(&output.index.unwrap_or(0))
    {
        if metadata
            .prefill_routed_experts
            .as_ref()
            .is_some_and(|current| current != routed_experts)
        {
            anyhow::bail!("prefill returned inconsistent routed experts for one choice");
        }
        metadata.prefill_routed_experts = Some(routed_experts.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::pipeline::{ResponseStream, context::Controller};

    use super::*;
    use crate::protocols::inference::generate::{GenerateBackendMetadata, GenerateLogprob};

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
        let result = PrefillRouter::consume_prefill_stream(
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
        let result = PrefillRouter::consume_prefill_stream(
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
    async fn prefill_generate_metadata_is_typed_and_choice_aligned() {
        let prompt_logprobs = vec![Some(HashMap::from([(
            11,
            GenerateLogprob {
                logprob: -0.25,
                rank: Some(1),
                decoded_token: Some("a".to_string()),
            },
        )]))];
        let mut first = valid_prefill_output();
        first.data.as_mut().unwrap().index = Some(0);
        first.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
            prompt_logprobs: Some(prompt_logprobs.clone()),
            routed_experts: Some("choice-zero-prefill".to_string()),
            prefill_routed_experts: None,
            kv_transfer_params: None,
        });
        let mut second = valid_prefill_output();
        second.data.as_mut().unwrap().index = Some(1);
        second.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
            prompt_logprobs: Some(prompt_logprobs.clone()),
            routed_experts: Some("choice-one-prefill".to_string()),
            prefill_routed_experts: None,
            kv_transfer_params: None,
        });

        let completion =
            PrefillRouter::consume_prefill_stream(prefill_stream(vec![first, second]), None)
                .await
                .unwrap();
        let metadata = completion
            .result
            .generate_metadata
            .expect("typed Generate prefill metadata");

        assert_eq!(metadata.prompt_logprobs, Some(prompt_logprobs));
        assert_eq!(
            metadata
                .routed_experts_by_choice
                .get(&0)
                .map(String::as_str),
            Some("choice-zero-prefill")
        );
        assert_eq!(
            metadata
                .routed_experts_by_choice
                .get(&1)
                .map(String::as_str),
            Some("choice-one-prefill")
        );
    }

    #[tokio::test]
    async fn bootstrap_prefill_metadata_is_attached_to_matching_decode_choices() {
        let prompt_logprobs = vec![Some(HashMap::from([(
            11,
            GenerateLogprob {
                logprob: -0.25,
                rank: Some(1),
                decoded_token: None,
            },
        )]))];
        let prefill_metadata = crate::protocols::inference::generate::GeneratePrefillMetadata {
            prompt_logprobs: Some(prompt_logprobs.clone()),
            routed_experts_by_choice: std::collections::BTreeMap::from([
                (0, "prefill-zero".to_string()),
                (1, "prefill-one".to_string()),
            ]),
        };
        let completion = tokio::spawn(async move {
            Ok(PrefillCompletion {
                result: crate::protocols::common::preprocessor::PrefillResult {
                    disaggregated_params: json!({}),
                    prompt_tokens_details: None,
                    generate_metadata: Some(prefill_metadata),
                },
                worker_link: None,
            })
        });
        let terminal = |index| {
            Annotated::from_data(LLMEngineOutput {
                index: Some(index),
                finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                generate_metadata: Some(GenerateBackendMetadata {
                    routed_experts: Some(format!("decode-{index}")),
                    ..Default::default()
                }),
                ..Default::default()
            })
        };
        let stream = prefill_stream(vec![terminal(0), terminal(1)]);
        let outputs = PrefillRouter::attach_bootstrap_generate_metadata(stream, completion)
            .collect::<Vec<_>>()
            .await;

        assert_eq!(outputs.len(), 2);
        for (index, output) in outputs.into_iter().enumerate() {
            let metadata = output.data.unwrap().generate_metadata.unwrap();
            assert_eq!(metadata.prompt_logprobs, Some(prompt_logprobs.clone()));
            assert_eq!(
                metadata.prefill_routed_experts.as_deref(),
                Some(if index == 0 {
                    "prefill-zero"
                } else {
                    "prefill-one"
                })
            );
            assert_eq!(
                metadata.routed_experts.as_deref(),
                Some(if index == 0 { "decode-0" } else { "decode-1" })
            );
        }
    }

    #[tokio::test]
    async fn bootstrap_without_optional_metadata_handles_multiple_terminals() {
        let completion = tokio::spawn(async move {
            Ok(PrefillCompletion {
                result: crate::protocols::common::preprocessor::PrefillResult {
                    disaggregated_params: json!({}),
                    prompt_tokens_details: None,
                    generate_metadata: None,
                },
                worker_link: None,
            })
        });
        let terminal = |index| {
            Annotated::from_data(LLMEngineOutput {
                index: Some(index),
                finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                ..Default::default()
            })
        };
        let outputs = PrefillRouter::attach_bootstrap_generate_metadata(
            prefill_stream(vec![terminal(0), terminal(1)]),
            completion,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| output.error.is_none()));
    }
}
