// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::NvCreateEmbeddingResponse;
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
};

use dynamo_runtime::engine::DataStream;
use futures::{Stream, StreamExt};

/// Aggregates a stream of [`NvCreateEmbeddingResponse`]s into a single
/// [`NvCreateEmbeddingResponse`]. For embeddings, this is typically simpler
/// than text generation as embeddings are usually returned as a complete response.
pub struct DeltaAggregator {
    /// The accumulated embeddings response.
    response: Option<NvCreateEmbeddingResponse>,
    /// Optional error message if an error occurs during aggregation.
    error: Option<String>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    /// Creates a new, empty [`DeltaAggregator`] instance.
    pub fn new() -> Self {
        Self {
            response: None,
            error: None,
        }
    }

    /// Aggregates a stream of [`NvCreateEmbeddingResponse`]s into a single
    /// [`NvCreateEmbeddingResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated embedding responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateEmbeddingResponse)` if aggregation is successful.
    /// * `Err(String)` if an error occurs during processing.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvCreateEmbeddingResponse>>,
    ) -> Result<NvCreateEmbeddingResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none()
                    && let Some(response) = delta.data
                {
                    // For embeddings, we typically expect a single complete response
                    // or we accumulate data from multiple responses
                    match &mut aggregator.response {
                        Some(existing) => {
                            // Merge embedding data if we have multiple responses
                            existing.inner.data.extend(response.inner.data);

                            // Update usage statistics
                            existing.inner.usage.prompt_tokens +=
                                response.inner.usage.prompt_tokens;
                            existing.inner.usage.total_tokens += response.inner.usage.total_tokens;
                        }
                        None => {
                            aggregator.response = Some(response);
                        }
                    }
                }
                aggregator
            })
            .await;

        // Return early if an error was encountered.
        if let Some(error) = aggregator.error {
            return Err(error);
        }

        // Return the aggregated response or an empty response if none was found.
        Ok(aggregator
            .response
            .unwrap_or_else(NvCreateEmbeddingResponse::empty))
    }
}

impl NvCreateEmbeddingResponse {
    /// Converts an SSE stream into a [`NvCreateEmbeddingResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of SSE messages containing embedding responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateEmbeddingResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreateEmbeddingResponse, String> {
        let stream = convert_sse_stream::<NvCreateEmbeddingResponse>(stream);
        NvCreateEmbeddingResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of embedding responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated embedding responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateEmbeddingResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateEmbeddingResponse>>,
    ) -> Result<NvCreateEmbeddingResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn create_test_embedding_response(
        embeddings: Vec<dynamo_async_openai::types::Embedding>,
        prompt_tokens: u32,
        total_tokens: u32,
    ) -> Annotated<NvCreateEmbeddingResponse> {
        let response = NvCreateEmbeddingResponse {
            inner: dynamo_async_openai::types::CreateEmbeddingResponse {
                object: "list".to_string(),
                model: "test-model".to_string(),
                data: embeddings,
                usage: dynamo_async_openai::types::EmbeddingUsage {
                    prompt_tokens,
                    total_tokens,
                },
            },
        };

        Annotated::from_data(response)
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let stream = stream::empty();
        let result = DeltaAggregator::apply(Box::pin(stream)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.data.len(), 0);
        assert_eq!(response.inner.object, "list");
        assert_eq!(response.inner.model, "embedding");
    }

    #[tokio::test]
    async fn test_single_embedding() {
        let embedding = dynamo_async_openai::types::Embedding {
            index: 0,
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };

        let annotated = create_test_embedding_response(vec![embedding.clone()], 10, 10);
        let stream = stream::iter(vec![annotated]);

        let result = DeltaAggregator::apply(Box::pin(stream)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.data.len(), 1);
        assert_eq!(response.inner.data[0].index, 0);
        assert_eq!(response.inner.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.inner.usage.prompt_tokens, 10);
        assert_eq!(response.inner.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn test_multiple_embeddings() {
        let embedding1 = dynamo_async_openai::types::Embedding {
            index: 0,
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };

        let embedding2 = dynamo_async_openai::types::Embedding {
            index: 1,
            object: "embedding".to_string(),
            embedding: vec![0.4, 0.5, 0.6],
        };

        let annotated1 = create_test_embedding_response(vec![embedding1.clone()], 5, 5);
        let annotated2 = create_test_embedding_response(vec![embedding2.clone()], 7, 7);
        let stream = stream::iter(vec![annotated1, annotated2]);

        let result = DeltaAggregator::apply(Box::pin(stream)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.data.len(), 2);
        assert_eq!(response.inner.data[0].index, 0);
        assert_eq!(response.inner.data[1].index, 1);
        assert_eq!(response.inner.usage.prompt_tokens, 12); // sum of 5 and 7
        assert_eq!(response.inner.usage.total_tokens, 12); // sum of 5 and 7
    }

    #[tokio::test]
    async fn test_error_in_stream() {
        let error_annotated =
            Annotated::<NvCreateEmbeddingResponse>::from_error("Test error".to_string());
        let stream = stream::iter(vec![error_annotated]);

        let result = DeltaAggregator::apply(Box::pin(stream)).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Test error"));
    }
}
