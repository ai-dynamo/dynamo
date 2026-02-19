// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Result;
use futures::{Stream, StreamExt};

use super::NvCreateCompletionResponse;
use crate::protocols::{
    Annotated, DataStream,
    codec::{Message, SseCodecError},
    common::FinishReason,
    convert_sse_stream,
    openai::ParsingOptions,
};

/// Aggregates a stream of [`CompletionResponse`]s into a single [`CompletionResponse`].
pub struct DeltaAggregator {
    id: String,
    model: String,
    created: u32,
    usage: Option<dynamo_async_openai::types::CompletionUsage>,
    system_fingerprint: Option<String>,
    choices: HashMap<u32, DeltaChoice>,
    error: Option<String>,
    nvext: Option<serde_json::Value>,
}

struct DeltaChoice {
    index: u32,
    text: String,
    finish_reason: Option<FinishReason>,
    logprobs: Option<dynamo_async_openai::types::Logprobs>,
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
            nvext: None,
        }
    }

    /// Aggregates a stream of [`Annotated<CompletionResponse>`]s into a single [`CompletionResponse`].
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvCreateCompletionResponse>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateCompletionResponse> {
        tracing::debug!("Tool Call Parser: {:?}", parsing_options.tool_call_parser); // TODO: remove this once completion has tool call support
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
                    // TODO(#14) - Aggregate Annotation

                    // these are cheap to move so we do it every time since we are consuming the delta
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.inner.id;
                    aggregator.model = delta.inner.model;
                    aggregator.created = delta.inner.created;
                    if let Some(usage) = delta.inner.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.inner.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }
                    // Aggregate nvext field (take the last non-None value)
                    if delta.inner.nvext.is_some() {
                        aggregator.nvext = delta.inner.nvext;
                    }

                    // handle the choices
                    for choice in delta.inner.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert(DeltaChoice {
                                    index: choice.index,
                                    text: "".to_string(),
                                    finish_reason: None,
                                    logprobs: None,
                                });

                        state_choice.text.push_str(&choice.text);

                        // TODO - handle logprobs

                        // Handle CompletionFinishReason -> FinishReason conversation
                        state_choice.finish_reason = match choice.finish_reason {
                            Some(dynamo_async_openai::types::CompletionFinishReason::Stop) => {
                                Some(FinishReason::Stop)
                            }
                            Some(dynamo_async_openai::types::CompletionFinishReason::Length) => {
                                Some(FinishReason::Length)
                            }
                            Some(
                                dynamo_async_openai::types::CompletionFinishReason::ContentFilter,
                            ) => Some(FinishReason::ContentFilter),
                            None => None,
                        };

                        // Update logprobs
                        if let Some(logprobs) = &choice.logprobs {
                            let state_lps = state_choice.logprobs.get_or_insert(
                                dynamo_async_openai::types::Logprobs {
                                    tokens: Vec::new(),
                                    token_logprobs: Vec::new(),
                                    top_logprobs: Vec::new(),
                                    text_offset: Vec::new(),
                                },
                            );
                            state_lps.tokens.extend(logprobs.tokens.clone());
                            state_lps
                                .token_logprobs
                                .extend(logprobs.token_logprobs.clone());
                            state_lps.top_logprobs.extend(logprobs.top_logprobs.clone());
                            state_lps.text_offset.extend(logprobs.text_offset.clone());
                        }
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
            .map(dynamo_async_openai::types::Choice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "text_completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            nvext: aggregator.nvext,
        };

        let response = NvCreateCompletionResponse { inner };

        Ok(response)
    }
}

impl From<DeltaChoice> for dynamo_async_openai::types::Choice {
    fn from(delta: DeltaChoice) -> Self {
        let finish_reason = delta.finish_reason.map(Into::into);

        dynamo_async_openai::types::Choice {
            index: delta.index,
            text: delta.text,
            finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

impl NvCreateCompletionResponse {
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateCompletionResponse> {
        let stream = convert_sse_stream::<NvCreateCompletionResponse>(stream);
        NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options).await
    }

    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateCompletionResponse>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateCompletionResponse> {
        DeltaAggregator::apply(stream, parsing_options).await
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use futures::stream;

    use super::*;
    use crate::protocols::openai::completions::NvCreateCompletionResponse;

    fn create_test_delta(
        index: u32,
        text: &str,
        finish_reason: Option<String>,
        logprob: Option<f32>,
    ) -> Annotated<NvCreateCompletionResponse> {
        // This will silently discard invalid_finish reason values and fall back
        // to None - totally fine since this is test code
        let finish_reason = finish_reason
            .as_deref()
            .and_then(|s| FinishReason::from_str(s).ok())
            .map(Into::into);

        let logprobs = logprob.map(|lp| dynamo_async_openai::types::Logprobs {
            tokens: vec![text.to_string()],
            token_logprobs: vec![Some(lp)],
            top_logprobs: vec![
                serde_json::to_value(dynamo_async_openai::types::TopLogprobs {
                    token: text.to_string(),
                    logprob: lp,
                    bytes: None,
                })
                .unwrap(),
            ],
            text_offset: vec![0],
        });

        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b".to_string(),
            created: 1234567890,
            usage: None,
            system_fingerprint: None,
            choices: vec![dynamo_async_openai::types::Choice {
                index,
                text: text.to_string(),
                finish_reason,
                logprobs,
            }],
            object: "text_completion".to_string(),
            nvext: None,
        };

        let response = NvCreateCompletionResponse { inner };

        Annotated {
            data: Some(response),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<NvCreateCompletionResponse>> = Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.inner.id, "");
        assert_eq!(response.inner.model, "");
        assert_eq!(response.inner.created, 0);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 0);
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta = create_test_delta(0, "Hello,", Some("length".to_string()), None);

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.id, "test_id");
        assert_eq!(response.inner.model, "meta/llama-3.1-8b");
        assert_eq!(response.inner.created, 1234567890);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello,".to_string());
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Length)
        );
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Length)
        );
        assert!(choice.logprobs.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 = create_test_delta(0, "Hello,", None, Some(-0.1));
        let annotated_delta2 =
            create_test_delta(0, " world!", Some("stop".to_string()), Some(-0.2));

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello, world!".to_string());
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Stop)
        );
        assert_eq!(choice.logprobs.as_ref().unwrap().tokens.len(), 2);
        assert_eq!(
            choice.logprobs.as_ref().unwrap().token_logprobs,
            vec![Some(-0.1), Some(-0.2)]
        );
    }

    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b".to_string(),
            created: 1234567890,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                dynamo_async_openai::types::Choice {
                    index: 0,
                    text: "Choice 0".to_string(),
                    finish_reason: Some(dynamo_async_openai::types::CompletionFinishReason::Stop),
                    logprobs: None,
                },
                dynamo_async_openai::types::Choice {
                    index: 1,
                    text: "Choice 1".to_string(),
                    finish_reason: Some(dynamo_async_openai::types::CompletionFinishReason::Stop),
                    logprobs: None,
                },
            ],
            object: "text_completion".to_string(),
            nvext: None,
        };

        let response = NvCreateCompletionResponse { inner };

        let annotated_delta = Annotated {
            data: Some(response),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 2);
        response.inner.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.inner.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.text, "Choice 0".to_string());
        assert_eq!(
            choice0.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Stop)
        );
        assert_eq!(
            choice0.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Stop)
        );

        let choice1 = &response.inner.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.text, "Choice 1".to_string());
        assert_eq!(
            choice1.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Stop)
        );
        assert_eq!(
            choice1.finish_reason,
            Some(dynamo_async_openai::types::CompletionFinishReason::Stop)
        );
    }
}
