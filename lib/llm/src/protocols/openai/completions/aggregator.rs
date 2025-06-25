// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use anyhow::Result;
use futures::StreamExt;

use super::CompletionResponse;
use crate::protocols::{
    codec::{Message, SseCodecError},
    common::FinishReason,
    convert_sse_stream, Annotated, DataStream,
};

/// Aggregates a stream of [`CompletionResponse`]s into a single [`CompletionResponse`].
pub struct DeltaAggregator {
    id: String,
    model: String,
    created: u64,
    usage: Option<async_openai::types::CompletionUsage>,
    system_fingerprint: Option<String>,
    choices: HashMap<u64, DeltaChoice>,
    error: Option<String>,
}

struct DeltaChoice {
    index: u64,
    text: String,
    finish_reason: Option<FinishReason>,
    logprobs: Option<async_openai::types::Logprobs>,
}

impl Default for DeltaAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
        }
    }

    /// Aggregates a stream of [`Annotated<CompletionResponse>`]s into a single [`CompletionResponse`].
    pub async fn apply(
        stream: DataStream<Annotated<CompletionResponse>>,
    ) -> Result<CompletionResponse> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // note: we could extract annotations here and add them to the aggregator
                    // to be return as part of the NIM Response Extension
                    // TODO(#14) - Aggregate Annotation

                    // these are cheap to move so we do it every time since we are consuming the delta
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.id;
                    aggregator.model = delta.model;
                    aggregator.created = delta.created;
                    if let Some(usage) = delta.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // handle the choices
                    for choice in delta.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index as u64)
                                .or_insert(DeltaChoice {
                                    index: choice.index as u64,
                                    text: "".to_string(),
                                    finish_reason: None,
                                    logprobs: choice.logprobs,
                                });

                        state_choice.text.push_str(&choice.text);

                        // TODO - handle logprobs

                        // Handle CompletionFinishReason -> FinishReason conversation
                        state_choice.finish_reason = match choice.finish_reason {
                            Some(async_openai::types::CompletionFinishReason::Stop) => {
                                Some(FinishReason::Stop)
                            }
                            Some(async_openai::types::CompletionFinishReason::Length) => {
                                Some(FinishReason::Length)
                            }
                            Some(async_openai::types::CompletionFinishReason::ContentFilter) => {
                                Some(FinishReason::ContentFilter)
                            }
                            None => None,
                        };
                    }
                }
                aggregator
            })
            .await;

        // If we have an error, return it
        let aggregator = if let Some(error) = aggregator.error {
            return Err(anyhow::anyhow!(error));
        } else {
            aggregator
        };

        // extra the aggregated deltas and sort by index
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(async_openai::types::Choice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        Ok(CompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "text_completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
        })
    }
}

impl From<DeltaChoice> for async_openai::types::Choice {
    fn from(delta: DeltaChoice) -> Self {
        let finish_reason = delta.finish_reason.map(Into::into);

        async_openai::types::Choice {
            index: delta.index as u32,
            text: delta.text,
            finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

impl CompletionResponse {
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<CompletionResponse> {
        let stream = convert_sse_stream::<CompletionResponse>(stream);
        CompletionResponse::from_annotated_stream(stream).await
    }

    pub async fn from_annotated_stream(
        stream: DataStream<Annotated<CompletionResponse>>,
    ) -> Result<CompletionResponse> {
        DeltaAggregator::apply(stream).await
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use futures::stream;

    use super::*;
    use crate::protocols::openai::completions::CompletionResponse;

    fn create_test_delta(
        index: u64,
        text: &str,
        finish_reason: Option<String>,
    ) -> Annotated<CompletionResponse> {
        // This will silently discard invalid_finish reason values and fall back
        // to None - totally fine since this is test code
        let finish_reason = finish_reason
            .as_deref()
            .and_then(|s| FinishReason::from_str(s).ok())
            .map(Into::into);

        Annotated {
            data: Some(CompletionResponse {
                id: "test_id".to_string(),
                model: "meta/llama-3.1-8b".to_string(),
                created: 1234567890,
                usage: None,
                system_fingerprint: None,
                choices: vec![async_openai::types::Choice {
                    index: index as u32,
                    text: text.to_string(),
                    finish_reason,
                    logprobs: None,
                }],
                object: "text_completion".to_string(),
            }),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<CompletionResponse>> = Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.created, 0);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 0);
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta = create_test_delta(0, "Hello,", Some("length".to_string()));

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.id, "test_id");
        assert_eq!(response.model, "meta/llama-3.1-8b");
        assert_eq!(response.created, 1234567890);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello,".to_string());
        assert_eq!(
            choice.finish_reason,
            Some(async_openai::types::CompletionFinishReason::Length)
        );
        assert!(choice.logprobs.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 = create_test_delta(0, "Hello,", None);
        let annotated_delta2 = create_test_delta(0, " world!", Some("stop".to_string()));

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello, world!".to_string());
        assert_eq!(
            choice.finish_reason,
            Some(async_openai::types::CompletionFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        let annotated_delta = Annotated {
            data: Some(CompletionResponse {
                id: "test_id".to_string(),
                model: "meta/llama-3.1-8b".to_string(),
                created: 1234567890,
                usage: None,
                system_fingerprint: None,
                choices: vec![
                    async_openai::types::Choice {
                        index: 0,
                        text: "Choice 0".to_string(),
                        finish_reason: Some(async_openai::types::CompletionFinishReason::Stop),
                        logprobs: None,
                    },
                    async_openai::types::Choice {
                        index: 1,
                        text: "Choice 1".to_string(),
                        finish_reason: Some(async_openai::types::CompletionFinishReason::Stop),
                        logprobs: None,
                    },
                ],
                object: "text_completion".to_string(),
            }),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 2);
        response.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.text, "Choice 0".to_string());
        assert_eq!(
            choice0.finish_reason,
            Some(async_openai::types::CompletionFinishReason::Stop)
        );

        let choice1 = &response.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.text, "Choice 1".to_string());
        assert_eq!(
            choice1.finish_reason,
            Some(async_openai::types::CompletionFinishReason::Stop)
        );
    }
}
