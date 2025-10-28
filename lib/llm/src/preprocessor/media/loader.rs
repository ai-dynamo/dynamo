// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;
use super::decoders::{Decoder, MediaDecoder};
use super::rdma::RdmaMediaDataDescriptor;
use nixl_sys::Agent as NixlAgent;

pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    nixl_agent: NixlAgent,
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .user_agent(
                "dynamo-ai/dynamo", // TODO: use a proper user agent
            )
            .build()?;

        let uuid = uuid::Uuid::new_v4();
        let nixl_agent = NixlAgent::new(&format!("media-loader-{}", uuid))?;
        let (_, ucx_params) = nixl_agent.get_plugin_params("UCX")?;
        nixl_agent.create_backend("UCX", &ucx_params)?;

        Ok(Self {
            media_decoder,
            http_client,
            nixl_agent,
        })
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
    ) -> Result<RdmaMediaDataDescriptor> {
        // TODO: request-level options
        // fetch and decode the media
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

        let rdma_descriptor = decoded.into_rdma_descriptor(&self.nixl_agent)?;
        Ok(rdma_descriptor)
    }
}
