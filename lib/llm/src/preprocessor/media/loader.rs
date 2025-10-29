// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;
use super::decoders::{DecodedMediaData, Decoder, MediaDecoder};

// TODO: make this configurable
const HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";

pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    // TODO: NIXL agent
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .user_agent(HTTP_USER_AGENT)
            .build()?;

        Ok(Self {
            media_decoder,
            http_client,
        })
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        // TODO: request-level options
    ) -> Result<DecodedMediaData> {
        // fetch the media
        // TODO: decode and NIXL-register
        let decoded = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let url = &image_part.image_url.url;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.image_decoder.decode_async(data).await?
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                EncodedMediaData::from_url(url, &self.http_client).await?;
                anyhow::bail!("Video decoding is not supported yet");
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        };

        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::{ChatCompletionRequestMessageContentPartImage, ImageUrl};

    // warning: non-airgap test
    #[tokio::test]
    async fn test_fetch_and_decode() {
        let media_decoder = MediaDecoder::default();
        let loader = MediaLoader::new(media_decoder).unwrap();

        let image_url = ImageUrl::from(
            "https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/llm-optimize-deploy-graphic.png",
        );
        let content_part = ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage { image_url },
        );

        let result = loader.fetch_and_decode_media_part(&content_part).await;
        assert!(
            result.is_ok(),
            "Failed to fetch and decode image: {:?}",
            result.err()
        );

        let data = result.unwrap();
        assert_eq!(data.dtype, "uint8");

        // Verify image dimensions: 1,999px × 1,125px (width × height)
        // Shape format is [height, width, channels]
        assert!(!data.shape.is_empty(), "Shape should not be empty");
        assert_eq!(data.shape[0], 1125, "Height should be 1125");
        assert_eq!(data.shape[1], 1999, "Width should be 1999");
        assert_eq!(data.shape[2], 4, "RGBA channels should be 4");
    }
}
