// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;

use crate::protocols::openai::stream_aggregator::{StreamAggregable, aggregate_stream};
use crate::types::Annotated;
use dynamo_runtime::error::DynamoError;

use super::NvImagesResponse;

fn merge_if_consistent<T: PartialEq>(current: &mut Option<T>, next: Option<T>) {
    if current.as_ref() != next.as_ref() {
        *current = None;
    }
}

impl StreamAggregable for NvImagesResponse {
    fn empty() -> Self {
        Self::empty()
    }

    fn merge(&mut self, next: Self) {
        if next.inner.data.is_empty() {
            return;
        }

        if self.inner.data.is_empty() {
            self.inner.created = next.inner.created;
            self.inner.output_format = next.inner.output_format;
            self.inner.size = next.inner.size;
        } else {
            merge_if_consistent(&mut self.inner.output_format, next.inner.output_format);
            merge_if_consistent(&mut self.inner.size, next.inner.size);
        }

        self.inner.data.extend(next.inner.data);
    }
}

impl NvImagesResponse {
    /// Aggregates an annotated stream of image responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvImagesResponse>>,
    ) -> Result<NvImagesResponse, DynamoError> {
        aggregate_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn response(
        created: i64,
        output_format: Option<&str>,
        size: Option<&str>,
        image: &str,
    ) -> NvImagesResponse {
        serde_json::from_value(serde_json::json!({
            "created": created,
            "data": [{"b64_json": image}],
            "output_format": output_format,
            "size": size,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn aggregates_chunk_metadata_and_image_data() {
        let result = NvImagesResponse::from_annotated_stream(stream::iter(vec![
            Annotated::from_data(response(100, Some("jpeg"), Some("37x19"), "first")),
            Annotated::from_data(response(101, Some("jpeg"), Some("37x19"), "second")),
        ]))
        .await
        .unwrap();
        let value = serde_json::to_value(result).unwrap();

        assert_eq!(value["created"], 100);
        assert_eq!(value["output_format"], "jpeg");
        assert_eq!(value["size"], "37x19");
        assert_eq!(value["data"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn keeps_metadata_from_first_nonempty_chunk_after_empty_chunk() {
        let result = NvImagesResponse::from_annotated_stream(stream::iter(vec![
            Annotated::from_data(NvImagesResponse::empty()),
            Annotated::from_data(response(100, Some("webp"), Some("41x23"), "image")),
        ]))
        .await
        .unwrap();
        let value = serde_json::to_value(result).unwrap();

        assert_eq!(value["created"], 100);
        assert_eq!(value["output_format"], "webp");
        assert_eq!(value["size"], "41x23");
        assert_eq!(value["data"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn clears_size_when_chunk_dimensions_differ_or_are_missing() {
        for second_size in [Some("41x23"), None] {
            let result = NvImagesResponse::from_annotated_stream(stream::iter(vec![
                Annotated::from_data(response(100, Some("png"), Some("37x19"), "first")),
                Annotated::from_data(response(101, Some("png"), second_size, "second")),
            ]))
            .await
            .unwrap();
            let value = serde_json::to_value(result).unwrap();

            assert_eq!(value["size"], serde_json::Value::Null);
            assert_eq!(value["output_format"], "png");
            assert_eq!(value["data"].as_array().unwrap().len(), 2);
        }
    }

    #[tokio::test]
    async fn clears_output_format_when_chunks_disagree() {
        let result = NvImagesResponse::from_annotated_stream(stream::iter(vec![
            Annotated::from_data(response(100, Some("jpeg"), Some("37x19"), "first")),
            Annotated::from_data(response(101, Some("webp"), Some("37x19"), "second")),
        ]))
        .await
        .unwrap();
        let value = serde_json::to_value(result).unwrap();

        assert_eq!(value["output_format"], serde_json::Value::Null);
        assert_eq!(value["size"], "37x19");
        assert_eq!(value["data"].as_array().unwrap().len(), 2);
    }
}
