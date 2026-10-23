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

    /// Fold `next` into `self`.
    ///
    /// Token `usage` keeps the first chunk's value: no image backend reports
    /// it today, and summing would double count a backend that repeats
    /// running totals. Image accounting folds per
    /// [`ImagesResponseNvExt::merge`](super::ImagesResponseNvExt::merge).
    /// `data` extends. The request-level fields come from the first chunk
    /// that carries images: `created` stays, while `size` and `output_format`
    /// are cleared when a later chunk disagrees, so the response never claims
    /// a value that does not hold for every image.
    fn merge(&mut self, next: Self) {
        if self.inner.usage.is_none() {
            self.inner.usage = next.inner.usage;
        }
        if let Some(next) = next.nvext {
            self.nvext.get_or_insert_with(Default::default).merge(next);
        }

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
    use crate::protocols::openai::images::{ImageAccounting, ImagesResponseNvExt};
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

    fn accounting(output_images: u32, steps: Option<u8>) -> ImagesResponseNvExt {
        ImagesResponseNvExt {
            usage: Some(ImageAccounting {
                output_images,
                input_images: 1,
                num_inference_steps: steps,
            }),
        }
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
        // No chunk reported accounting.
        assert_eq!(value["usage"], serde_json::Value::Null);
        assert!(value.get("nvext").is_none());
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

    #[test]
    fn merge_folds_accounting_across_chunks() {
        let token_usage: dynamo_protocols::types::ImageGenUsage =
            serde_json::from_value(serde_json::json!({
                "input_tokens": 3, "output_tokens": 4, "total_tokens": 7,
                "input_tokens_details": {"text_tokens": 3, "image_tokens": 0}
            }))
            .unwrap();

        // Chunks that agree: output counts add up, the request-side fields hold.
        let mut first = response(100, Some("png"), Some("37x19"), "first");
        first.nvext = Some(accounting(1, Some(4)));
        first.inner.usage = Some(token_usage.clone());
        let mut second = response(101, Some("png"), Some("37x19"), "second");
        second.nvext = Some(accounting(2, Some(4)));
        second.inner.usage = Some(token_usage.clone());
        first.merge(second);
        assert_eq!(first.nvext, Some(accounting(3, Some(4))));
        assert_eq!(first.inner.data.len(), 2);
        // Token usage is the first reported value, not a sum.
        assert_eq!(first.inner.usage, Some(token_usage));

        // A chunk without accounting leaves it unchanged.
        first.merge(NvImagesResponse::empty());
        assert_eq!(first.nvext, Some(accounting(3, Some(4))));

        // A chunk that disagrees on the step count, or does not report it,
        // clears the step count; output counts still add up.
        let mut third = response(102, Some("png"), Some("37x19"), "third");
        third.nvext = Some(accounting(1, None));
        first.merge(third);
        assert_eq!(first.nvext, Some(accounting(4, None)));

        // An empty aggregate adopts the first accounting it sees, even from a
        // chunk that carries no images.
        let mut empty = NvImagesResponse::empty();
        let mut terminal = NvImagesResponse::empty();
        terminal.nvext = Some(accounting(2, Some(4)));
        empty.merge(terminal);
        assert_eq!(empty.nvext, Some(accounting(2, Some(4))));
        assert!(empty.inner.data.is_empty());
        assert!(empty.inner.usage.is_none());
    }
}
