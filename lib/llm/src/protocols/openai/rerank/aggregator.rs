// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::NvCreateRerankResponse;
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::stream_aggregator::{StreamAggregable, aggregate_stream},
};
use dynamo_runtime::{engine::DataStream, error::DynamoError};
use futures::Stream;

impl StreamAggregable for NvCreateRerankResponse {
    fn empty() -> Self {
        Self::default()
    }

    fn merge(&mut self, next: Self) {
        self.0.extend(next.0);
    }
}

impl NvCreateRerankResponse {
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<Self, DynamoError> {
        Self::from_annotated_stream(convert_sse_stream::<Self>(stream)).await
    }

    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<Self>>,
    ) -> Result<Self, DynamoError> {
        aggregate_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::rerank::RerankResult;
    use futures::stream;

    #[tokio::test]
    async fn aggregates_worker_chunks() {
        let item = |index, score| {
            Annotated::from_data(NvCreateRerankResponse(vec![RerankResult {
                score,
                index,
                document: None,
                meta_info: None,
            }]))
        };
        let response = NvCreateRerankResponse::from_annotated_stream(stream::iter(vec![
            item(1, 0.9),
            item(0, 0.2),
        ]))
        .await
        .unwrap();
        assert_eq!(response.0.len(), 2);
        assert_eq!(response.0[0].index, 1);
    }
}
