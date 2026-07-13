// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::StreamExt;

use dynamo_runtime::{
    pipeline::ManyOut,
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::protocols::{
    common::{llm_backend::LLMEngineOutput, timing::RequestTracker},
    inference::generate::MAX_GENERATE_CHOICES,
};

const MAX_PREFILL_METADATA_FRAMES: usize = MAX_GENERATE_CHOICES as usize * 2 + 4;
struct PrefillMetadataAccumulator {
    metadata: crate::protocols::inference::generate::GeneratePrefillMetadata,
    expected_choices: u32,
    frame_count: usize,
    frame_limit: usize,
    routed_expert_budget: crate::protocols::inference::routed_experts::RoutedExpertResponseBudget,
}

impl PrefillMetadataAccumulator {
    fn new(expected_choices: u32) -> Result<Self, PrefillError> {
        if expected_choices == 0 || expected_choices > MAX_GENERATE_CHOICES {
            return Err(PrefillError::PrefillError(
                format!(
                    "Generate prefill choice count must be between 1 and {MAX_GENERATE_CHOICES}, got {expected_choices}"
                ),
                None,
            ));
        }
        let frame_limit = usize::try_from(expected_choices)
            .unwrap_or(MAX_PREFILL_METADATA_FRAMES)
            .saturating_mul(2)
            .saturating_add(4)
            .min(MAX_PREFILL_METADATA_FRAMES);
        Ok(Self {
            metadata: Default::default(),
            expected_choices,
            frame_count: 0,
            frame_limit,
            routed_expert_budget: Default::default(),
        })
    }

    fn merge(&mut self, output: &LLMEngineOutput) -> Result<(), PrefillError> {
        let Some(metadata) = output.generate_metadata.as_ref() else {
            return Ok(());
        };
        self.frame_count = self.frame_count.saturating_add(1);
        if self.frame_count > self.frame_limit {
            return Err(PrefillError::PrefillError(
                format!(
                    "Generate prefill metadata exceeded the {}-frame limit",
                    self.frame_limit
                ),
                None,
            ));
        }

        if let Some(prompt_logprobs) = metadata.prompt_logprobs.as_ref() {
            if self
                .metadata
                .prompt_logprobs
                .as_ref()
                .is_some_and(|current| current != prompt_logprobs)
            {
                return Err(PrefillError::PrefillError(
                    "Prefill workers returned inconsistent prompt logprobs".to_string(),
                    None,
                ));
            }
            if self.metadata.prompt_logprobs.is_none() {
                self.metadata.prompt_logprobs = Some(prompt_logprobs.clone());
            }
        }
        if let Some(routed_experts) = metadata.routed_experts.as_ref() {
            let choice = match output.index {
                Some(choice) => choice,
                None if self.expected_choices == 1 => 0,
                None => {
                    return Err(PrefillError::PrefillError(
                        "Generate prefill metadata requires an explicit choice index when n > 1"
                            .to_string(),
                        None,
                    ));
                }
            };
            if choice >= self.expected_choices {
                return Err(PrefillError::PrefillError(
                    format!(
                        "Generate prefill metadata choice {choice} is outside request n={}",
                        self.expected_choices
                    ),
                    None,
                ));
            }
            let stats =
                crate::protocols::inference::routed_experts::validate_routed_expert_payload(
                    "prefill",
                    routed_experts,
                )
                .map_err(|error| PrefillError::PrefillError(error.to_string(), None))?;
            self.routed_expert_budget
                .record(stats)
                .map_err(|error| PrefillError::PrefillError(error.to_string(), None))?;
            if self
                .metadata
                .routed_experts_by_choice
                .get(&choice)
                .is_some_and(|current| current != routed_experts)
            {
                return Err(PrefillError::PrefillError(
                    format!(
                        "Prefill workers returned inconsistent routed experts for choice {choice}"
                    ),
                    None,
                ));
            }
            self.metadata
                .routed_experts_by_choice
                .entry(choice)
                .or_insert_with(|| routed_experts.clone());
        }
        Ok(())
    }

    fn finish(
        self,
    ) -> Result<Option<crate::protocols::inference::generate::GeneratePrefillMetadata>, PrefillError>
    {
        if !self.metadata.routed_experts_by_choice.is_empty()
            && self.metadata.routed_experts_by_choice.len()
                != usize::try_from(self.expected_choices).unwrap_or(usize::MAX)
        {
            return Err(PrefillError::PrefillError(
                format!(
                    "Generate prefill returned routed experts for {} of {} choices",
                    self.metadata.routed_experts_by_choice.len(),
                    self.expected_choices
                ),
                None,
            ));
        }
        Ok((self.metadata.prompt_logprobs.is_some()
            || !self.metadata.routed_experts_by_choice.is_empty())
        .then_some(self.metadata))
    }
}

impl PrefillRouter {
    pub(super) async fn consume_prefill_stream(
        mut prefill_response: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        expected_choices: Option<u32>,
    ) -> Result<PrefillCompletion, PrefillError> {
        let mut generate_metadata = expected_choices
            .map(PrefillMetadataAccumulator::new)
            .transpose()?;
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
        let first_disaggregated_params = first_output
            .data
            .as_ref()
            .and_then(|output| output.disaggregated_params.clone());
        if let (Some(accumulator), Some(output)) =
            (generate_metadata.as_mut(), first_output.data.as_ref())
        {
            accumulator.merge(output)?;
        }

        while let Some(next) = prefill_response.next().await {
            if let Some(error) = next.err() {
                return Err(PrefillError::PrefillError(
                    "Prefill router returned error in output stream".to_string(),
                    Some(Box::new(error)),
                ));
            }
            if let Some(output) = next.data.as_ref() {
                if output.disaggregated_params != first_disaggregated_params {
                    return Err(PrefillError::PrefillError(
                        "Prefill frames returned inconsistent disaggregated_params".to_string(),
                        None,
                    ));
                }
                if prompt_tokens_details.is_none() {
                    prompt_tokens_details = output
                        .completion_usage
                        .as_ref()
                        .and_then(|usage| usage.prompt_tokens_details.clone());
                }
                if let Some(accumulator) = generate_metadata.as_mut() {
                    accumulator.merge(output)?;
                }
            }
        }

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };
        let Some(disaggregated_params) = first_disaggregated_params else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };

        let generate_metadata = match generate_metadata {
            Some(accumulator) => accumulator.finish()?,
            None => None,
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
}
