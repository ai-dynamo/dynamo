// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;
use super::decoders::{Decoder, MediaDecoder};
use super::rdma::{RdmaMediaDataDescriptor, get_nixl_agent};
use nixl_sys::Agent as NixlAgent;

// TODO: make this configurable
const HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";

pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    nixl_agent: NixlAgent,
    nixl_metadata: String,
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .user_agent(HTTP_USER_AGENT)
            .build()?;

        let (nixl_agent, nixl_metadata) = get_nixl_agent()?;

        Ok(Self {
            media_decoder,
            http_client,
            nixl_agent,
            nixl_metadata,
        })
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        // TODO: request-level options
    ) -> Result<RdmaMediaDataDescriptor> {
        // fetch, decode, and NIXL-register the media
        let decoded = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let url = &image_part.image_url.url;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.image_decoder.decode_async(data).await
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.video_decoder.decode_async(data).await
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        }?;

        let rdma_descriptor =
            decoded.into_rdma_descriptor(&self.nixl_agent, self.nixl_metadata.clone())?;
        Ok(rdma_descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::storage::nixl::NixlRegisterableStorage;
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

        let descriptor = result.unwrap();
        assert_eq!(descriptor.dtype, "uint8");

        // Verify image dimensions: 1,999px × 1,125px (width × height)
        // Shape format is [height, width, channels]
        assert!(!descriptor.shape.is_empty(), "Shape should not be empty");
        assert_eq!(descriptor.shape[0], 1125, "Height should be 1125");
        assert_eq!(descriptor.shape[1], 1999, "Width should be 1999");
        assert_eq!(descriptor.shape[2], 4, "RGBA channels should be 4");

        assert!(
            descriptor.source_storage.is_some(),
            "Source storage should be present"
        );
        assert!(
            descriptor.source_storage.unwrap().is_nixl_registered(),
            "Source storage should be registered with NIXL"
        );
    }
}
