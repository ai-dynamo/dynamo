// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::NvCreatePoolingResponse;
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::stream_aggregator::{StreamAggregable, aggregate_stream},
};

use dynamo_runtime::{engine::DataStream, error::DynamoError};
use futures::Stream;

impl StreamAggregable for NvCreatePoolingResponse {
    fn empty() -> Self {
        Self::empty()
    }

    fn merge(&mut self, next: Self) {
        // Preserve identity/model from the first non-empty response.
        if self.id.is_empty() {
            self.id = next.id;
        }
        if self.created == 0 {
            self.created = next.created;
        }
        if self.model == "pooling" && next.model != "pooling" {
            self.model = next.model;
        }
        self.data.extend(next.data);
        self.usage.prompt_tokens += next.usage.prompt_tokens;
        self.usage.total_tokens += next.usage.total_tokens;
        self.usage.completion_tokens += next.usage.completion_tokens;
    }
}

impl NvCreatePoolingResponse {
    /// Converts an SSE stream into a [`NvCreatePoolingResponse`].
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreatePoolingResponse, DynamoError> {
        let stream = convert_sse_stream::<NvCreatePoolingResponse>(stream);
        NvCreatePoolingResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of pooling responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreatePoolingResponse>>,
    ) -> Result<NvCreatePoolingResponse, DynamoError> {
        aggregate_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::pooling::{PoolingData, PoolingOutput, PoolingUsage};
    use futures::stream;

    fn make_response(
        index: u32,
        data: PoolingOutput,
        prompt_tokens: u32,
    ) -> Annotated<NvCreatePoolingResponse> {
        let response = NvCreatePoolingResponse {
            id: "pool-test".to_string(),
            object: "list".to_string(),
            created: 1,
            model: "test-model".to_string(),
            data: vec![PoolingData {
                index,
                object: "pooling".to_string(),
                data,
            }],
            usage: PoolingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
                completion_tokens: 0,
            },
        };
        Annotated::from_data(response)
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let stream = stream::empty();
        let result = NvCreatePoolingResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.data.len(), 0);
        assert_eq!(response.object, "list");
    }

    #[tokio::test]
    async fn test_single_pooling_output() {
        let annotated = make_response(
            0,
            PoolingOutput::Matrix(vec![vec![0.1, 0.7], vec![0.2, 0.3]]),
            5,
        );
        let stream = stream::iter(vec![annotated]);
        let result = NvCreatePoolingResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(matches!(response.data[0].data, PoolingOutput::Matrix(_)));
        assert_eq!(response.usage.prompt_tokens, 5);
        assert_eq!(response.model, "test-model");
    }

    #[tokio::test]
    async fn test_multiple_pooling_outputs() {
        let a = make_response(0, PoolingOutput::Vector(vec![0.8, 0.2]), 4);
        let b = make_response(1, PoolingOutput::Vector(vec![0.3, 0.7]), 6);
        let stream = stream::iter(vec![a, b]);
        let result = NvCreatePoolingResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].index, 0);
        assert_eq!(response.data[1].index, 1);
        assert_eq!(response.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn test_error_in_stream() {
        let error_annotated =
            Annotated::<NvCreatePoolingResponse>::from_error("Test error".to_string());
        let stream = stream::iter(vec![error_annotated]);
        let result = NvCreatePoolingResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Test error"));
    }
}
