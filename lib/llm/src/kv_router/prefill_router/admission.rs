// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_runtime::{
    pipeline::{ManyOut, PushRouter, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::{
    kv_router::KvPushRouter,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::RequestTracker,
    },
};

pub(super) enum InnerPrefillRouter {
    KvRouter(Arc<KvPushRouter>),
    SimpleRouter(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

impl InnerPrefillRouter {
    pub(super) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        pinned_worker: Option<u64>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>)>
    where
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M>,
    {
        match self {
            InnerPrefillRouter::KvRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                router
                    .select_and_dispatch_exact(request, pinned_worker, |request, worker_id| {
                        prepare(request, worker_id, None)
                    })
                    .await
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
}
