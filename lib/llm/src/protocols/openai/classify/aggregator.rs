// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::NvCreateClassifyResponse;
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::stream_aggregator::{StreamAggregable, aggregate_stream},
};

use dynamo_runtime::engine::DataStream;
use futures::Stream;

impl StreamAggregable for NvCreateClassifyResponse {
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
        if self.model == "classify" && next.model != "classify" {
            self.model = next.model;
        }
        self.data.extend(next.data);
        self.usage.prompt_tokens += next.usage.prompt_tokens;
        self.usage.total_tokens += next.usage.total_tokens;
    }
}

impl NvCreateClassifyResponse {
    /// Converts an SSE stream into a [`NvCreateClassifyResponse`].
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreateClassifyResponse, String> {
        let stream = convert_sse_stream::<NvCreateClassifyResponse>(stream);
        NvCreateClassifyResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of classification responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateClassifyResponse>>,
    ) -> Result<NvCreateClassifyResponse, String> {
        aggregate_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::classify::ClassificationData;
    use futures::stream;

    fn make_response(
        index: u32,
        label: &str,
        probs: Vec<f32>,
        prompt_tokens: u32,
    ) -> Annotated<NvCreateClassifyResponse> {
        let response = NvCreateClassifyResponse {
            id: "classify-test".to_string(),
            object: "list".to_string(),
            created: 1,
            model: "test-model".to_string(),
            data: vec![ClassificationData {
                index,
                label: Some(label.to_string()),
                num_classes: probs.len() as u32,
                probs,
            }],
            usage: super::super::ClassificationUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
        };
        Annotated::from_data(response)
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let stream = stream::empty();
        let result = NvCreateClassifyResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.data.len(), 0);
        assert_eq!(response.object, "list");
    }

    #[tokio::test]
    async fn test_single_classification() {
        let annotated = make_response(0, "entailment", vec![0.1, 0.7, 0.2], 5);
        let stream = stream::iter(vec![annotated]);
        let result = NvCreateClassifyResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].label.as_deref(), Some("entailment"));
        assert_eq!(response.data[0].num_classes, 3);
        assert_eq!(response.usage.prompt_tokens, 5);
        assert_eq!(response.model, "test-model");
    }

    #[tokio::test]
    async fn test_multiple_classifications() {
        let a = make_response(0, "entailment", vec![0.8, 0.2], 4);
        let b = make_response(1, "contradiction", vec![0.3, 0.7], 6);
        let stream = stream::iter(vec![a, b]);
        let result = NvCreateClassifyResponse::from_annotated_stream(Box::pin(stream)).await;
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
            Annotated::<NvCreateClassifyResponse>::from_error("Test error".to_string());
        let stream = stream::iter(vec![error_annotated]);
        let result = NvCreateClassifyResponse::from_annotated_stream(Box::pin(stream)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Test error"));
    }
}
