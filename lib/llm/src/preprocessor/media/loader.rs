// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;
use super::decoders::{DecodedMediaData, Decoder, MediaDecoder};

const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MediaFetcher {
    pub user_agent: String,
    pub allow_direct_ip: bool,
    pub allow_direct_port: bool,
    pub allowed_media_domains: Option<HashSet<String>>,
    pub timeout: Option<Duration>,
}

impl Default for MediaFetcher {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_HTTP_USER_AGENT.to_string(),
            allow_direct_ip: false,
            allow_direct_port: false,
            allowed_media_domains: None,
            timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    media_fetcher: MediaFetcher,
    // TODO: NIXL agent
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder, media_fetcher: MediaFetcher) -> Result<Self> {
        let mut http_client_builder =
            reqwest::Client::builder().user_agent(&media_fetcher.user_agent);

        if let Some(timeout) = media_fetcher.timeout {
            http_client_builder = http_client_builder.timeout(timeout);
        }

        let http_client = http_client_builder.build()?;

        Ok(Self {
            media_decoder,
            http_client,
            media_fetcher,
        })
    }

    pub fn check_if_url_allowed(&self, url: &url::Url) -> Result<()> {
        if !matches!(url.scheme(), "http" | "https" | "data") {
            anyhow::bail!("Only HTTP(S) and data URLs are allowed");
        }

        if url.scheme() == "data" {
            return Ok(());
        }

        if !self.media_fetcher.allow_direct_ip && !matches!(url.host(), Some(url::Host::Domain(_)))
        {
            anyhow::bail!("Direct IP access is not allowed");
        }
        if !self.media_fetcher.allow_direct_port && url.port().is_some() {
            anyhow::bail!("Direct port access is not allowed");
        }
        if let Some(allowed_domains) = &self.media_fetcher.allowed_media_domains
            && let Some(host) = url.host_str()
            && !allowed_domains.contains(host)
        {
            anyhow::bail!("Domain '{host}' is not in allowed list");
        }

        Ok(())
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
                self.check_if_url_allowed(url)?;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.image_decoder.decode_async(data).await?
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                self.check_if_url_allowed(url)?;
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
    use super::super::decoders::DataType;
    use super::*;
    use dynamo_async_openai::types::{ChatCompletionRequestMessageContentPartImage, ImageUrl};

    #[tokio::test]
    async fn test_fetch_and_decode() {
        let test_image_bytes =
            include_bytes!("../../../tests/data/media/llm-optimize-deploy-graphic.png");

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/llm-optimize-deploy-graphic.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .create_async()
            .await;

        let media_decoder = MediaDecoder::default();
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            ..Default::default()
        };

        let loader = MediaLoader::new(media_decoder, fetcher).unwrap();

        let image_url = ImageUrl::from(format!("{}/llm-optimize-deploy-graphic.png", server.url()));
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
        assert_eq!(data.dtype, DataType::UINT8);

        // Verify image dimensions: 1,999px × 1,125px (width × height)
        // Shape format is [height, width, channels]
        assert_eq!(data.shape.len(), 3);
        assert_eq!(data.shape[0], 1125, "Height should be 1125");
        assert_eq!(data.shape[1], 1999, "Width should be 1999");
        assert_eq!(data.shape[2], 4, "RGBA channels should be 4");

        mock.assert_async().await;
    }

    #[test]
    fn test_direct_ip_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_ip: false,
            ..Default::default()
        };
        let loader = MediaLoader::new(MediaDecoder::default(), fetcher).unwrap();

        let url = url::Url::parse("http://192.168.1.1/image.jpg").unwrap();
        let result = loader.check_if_url_allowed(&url);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Direct IP access is not allowed")
        );
    }

    #[test]
    fn test_direct_port_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_port: false,
            ..Default::default()
        };
        let loader = MediaLoader::new(MediaDecoder::default(), fetcher).unwrap();

        let url = url::Url::parse("http://example.com:8080/image.jpg").unwrap();
        let result = loader.check_if_url_allowed(&url);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Direct port access is not allowed")
        );
    }

    #[test]
    fn test_domain_allowlist() {
        let mut allowed_domains = HashSet::new();
        allowed_domains.insert("trusted.com".to_string());
        allowed_domains.insert("example.com".to_string());

        let fetcher = MediaFetcher {
            allowed_media_domains: Some(allowed_domains),
            ..Default::default()
        };
        let loader = MediaLoader::new(MediaDecoder::default(), fetcher).unwrap();

        // Allowed domain should pass
        let url = url::Url::parse("https://trusted.com/image.jpg").unwrap();
        assert!(loader.check_if_url_allowed(&url).is_ok());

        // Disallowed domain should fail
        let url = url::Url::parse("https://untrusted.com/image.jpg").unwrap();
        let result = loader.check_if_url_allowed(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in allowed list")
        );
    }
}
