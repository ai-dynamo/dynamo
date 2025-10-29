// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;

// TODO: make this configurable
const HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";

pub struct MediaLoader {
    http_client: reqwest::Client,
    // TODO: decoders, NIXL agent
}

impl MediaLoader {
    pub fn new() -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .user_agent(HTTP_USER_AGENT)
            .build()?;

        Ok(Self { http_client })
    }

    pub async fn fetch_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        // TODO: request-level options
    ) -> Result<EncodedMediaData> {
        // fetch the media
        // TODO: decode and NIXL-register
        let data = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let url = &image_part.image_url.url;
                EncodedMediaData::from_url(url, &self.http_client).await?
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                EncodedMediaData::from_url(url, &self.http_client).await?
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        };

        Ok(data)
    }
}
