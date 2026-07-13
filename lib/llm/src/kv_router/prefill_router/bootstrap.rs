// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;

use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, ManyOut, ResponseStream},
    protocols::annotated::Annotated,
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::protocols::common::llm_backend::LLMEngineOutput;

pub(super) const BOOTSTRAP_PREFILL_COMPLETION_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

/// A spawned prefill task is part of the decode response lifetime. Tokio's raw
/// `JoinHandle` detaches on drop, which would let a disconnected decode leave a
/// prefill stream running forever. Keep abort-on-drop ownership in the wrapper.
pub(super) struct AbortOnDrop<T> {
    handle: tokio::task::JoinHandle<T>,
}

impl<T> AbortOnDrop<T> {
    pub(super) fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self { handle }
    }

    fn handle_mut(&mut self) -> &mut tokio::task::JoinHandle<T> {
        &mut self.handle
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl PrefillRouter {
    pub(super) fn attach_bootstrap_generate_metadata(
        mut decode_stream: ManyOut<Annotated<LLMEngineOutput>>,
        mut completion: AbortOnDrop<Result<PrefillCompletion, PrefillError>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        let context = decode_stream.context();
        let request_context = context.clone();
        let stream = async_stream::stream! {
            let mut completion_resolved = false;
            let mut metadata = None;
            while let Some(mut annotated) = decode_stream.next().await {
                let terminal = annotated
                    .data
                    .as_ref()
                    .is_some_and(|output| output.finish_reason.is_some());
                if terminal && !completion_resolved {
                    let result = tokio::select! {
                        biased;
                        _ = request_context.stopped() => None,
                        result = tokio::time::timeout(
                            BOOTSTRAP_PREFILL_COMPLETION_TIMEOUT,
                            completion.handle_mut(),
                        ) => Some(result),
                    };
                    let Some(result) = result else {
                        yield Annotated::from_error(
                            "asynchronous prefill metadata wait cancelled with its request",
                        );
                        break;
                    };
                    match result {
                        Ok(Ok(Ok(prefill))) => {
                            completion_resolved = true;
                            metadata = prefill.result.generate_metadata;
                        }
                        Ok(Ok(Err(error))) => {
                            yield Annotated::from_error(format!(
                                "asynchronous prefill metadata failed: {error}"
                            ));
                            break;
                        }
                        Ok(Err(error)) => {
                            yield Annotated::from_error(format!(
                                "asynchronous prefill metadata task failed: {error}"
                            ));
                            break;
                        }
                        Err(_) => {
                            yield Annotated::from_error(format!(
                                "asynchronous prefill metadata timed out after {} seconds",
                                BOOTSTRAP_PREFILL_COMPLETION_TIMEOUT.as_secs()
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
