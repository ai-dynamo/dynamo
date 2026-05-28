// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::NvCreateEmbeddingResponse;
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::stream_aggregator::{StreamAggregable, aggregate_stream},
};

use dynamo_runtime::engine::DataStream;
use futures::Stream;

impl StreamAggregable for NvCreateEmbeddingResponse {
    fn empty() -> Self {
        Self::empty()
    }

    fn merge(&mut self, next: Self) {
        self.inner.data.extend(next.inner.data);
        self.inner.usage.prompt_tokens += next.inner.usage.prompt_tokens;
        self.inner.usage.total_tokens += next.inner.usage.total_tokens;
    }
}

impl NvCreateEmbeddingResponse {
    /// Converts an SSE stream into a [`NvCreateEmbeddingResponse`].
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreateEmbeddingResponse, String> {
        let stream = convert_sse_stream::<NvCreateEmbeddingResponse>(stream);
        NvCreateEmbeddingResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of embedding responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateEmbeddingResponse>>,
    ) -> Result<NvCreateEmbeddingResponse, String> {
        aggregate_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn create_test_embedding_response(
        embeddings: Vec<dynamo_protocols::types::Embedding>,
        prompt_tokens: u32,
        total_tokens: u32,
    ) -> Annotated<NvCreateEmbeddingResponse> {
        let response = NvCreateEmbeddingResponse {
            inner: dynamo_protocols::types::CreateEmbeddingResponse {
                object: "list".to_string(),
                model: "test-model".to_string(),
                data: embeddings,
                usage: dynamo_protocols::types::EmbeddingUsage {
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
        let result = NvCreateEmbeddingResponse::from_annotated_stream(Box::pin(stream)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.data.len(), 0);
        assert_eq!(response.inner.object, "list");
        assert_eq!(response.inner.model, "embedding");
    }

    #[tokio::test]
    async fn test_single_embedding() {
        let embedding = dynamo_protocols::types::Embedding {
            index: 0,
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };

        let annotated = create_test_embedding_response(vec![embedding.clone()], 10, 10);
        let stream = stream::iter(vec![annotated]);

        let result = NvCreateEmbeddingResponse::from_annotated_stream(Box::pin(stream)).await;

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
        let embedding1 = dynamo_protocols::types::Embedding {
            index: 0,
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };

        let embedding2 = dynamo_protocols::types::Embedding {
            index: 1,
            object: "embedding".to_string(),
            embedding: vec![0.4, 0.5, 0.6],
        };

        let annotated1 = create_test_embedding_response(vec![embedding1.clone()], 5, 5);
        let annotated2 = create_test_embedding_response(vec![embedding2.clone()], 7, 7);
        let stream = stream::iter(vec![annotated1, annotated2]);

        let result = NvCreateEmbeddingResponse::from_annotated_stream(Box::pin(stream)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.data.len(), 2);
        assert_eq!(response.inner.data[0].index, 0);
        assert_eq!(response.inner.data[1].index, 1);
        assert_eq!(response.inner.usage.prompt_tokens, 12);
        assert_eq!(response.inner.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn test_error_in_stream() {
        let error_annotated =
            Annotated::<NvCreateEmbeddingResponse>::from_error("Test error".to_string());
        let stream = stream::iter(vec![error_annotated]);

        let result = NvCreateEmbeddingResponse::from_annotated_stream(Box::pin(stream)).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Test error"));
    }
}
