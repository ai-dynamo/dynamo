// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::decoders::MediaDecoder;
use super::rdma::RdmaMediaDataDescriptor;

#[cfg(feature = "media-nixl")]
use {
    super::common::EncodedMediaData, super::decoders::Decoder, super::rdma::get_nixl_agent,
    dynamo_memory::nixl::NixlAgent,
};

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

impl MediaFetcher {
    pub fn check_if_url_allowed(&self, url: &url::Url) -> Result<()> {
        if !matches!(url.scheme(), "http" | "https" | "data") {
            anyhow::bail!("Only HTTP(S) and data URLs are allowed");
        }

        if url.scheme() == "data" {
            return Ok(());
        }

        if !self.allow_direct_ip && !matches!(url.host(), Some(url::Host::Domain(_))) {
            anyhow::bail!("Direct IP access is not allowed");
        }
        if !self.allow_direct_port && url.port().is_some() {
            anyhow::bail!("Direct port access is not allowed");
        }
        if let Some(allowed_domains) = &self.allowed_media_domains
            && let Some(host) = url.host_str()
            && !allowed_domains.contains(host)
        {
            anyhow::bail!("Domain '{host}' is not in allowed list");
        }

        Ok(())
    }
}

pub struct MediaLoader {
    #[allow(dead_code)]
    media_decoder: MediaDecoder,
    #[allow(dead_code)]
    http_client: reqwest::Client,
    media_fetcher: MediaFetcher,
    #[cfg(feature = "media-nixl")]
    nixl_agent: NixlAgent,
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder, media_fetcher: Option<MediaFetcher>) -> Result<Self> {
        let media_fetcher = media_fetcher.unwrap_or_default();
        let mut http_client_builder: reqwest::ClientBuilder =
            reqwest::Client::builder().user_agent(&media_fetcher.user_agent);

        if let Some(timeout) = media_fetcher.timeout {
            http_client_builder = http_client_builder.timeout(timeout);
        }

        let http_client = http_client_builder.build()?;

        #[cfg(feature = "media-nixl")]
        let nixl_agent = get_nixl_agent()?;

        Ok(Self {
            media_decoder,
            http_client,
            media_fetcher,
            #[cfg(feature = "media-nixl")]
            nixl_agent,
        })
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        media_io_kwargs: Option<&MediaDecoder>,
    ) -> Result<RdmaMediaDataDescriptor> {
        #[cfg(not(feature = "media-nixl"))]
        anyhow::bail!(
            "NIXL is not supported, cannot decode and register media data {oai_content_part:?} with media_io_kwargs {media_io_kwargs:?}"
        );

        #[cfg(feature = "media-nixl")]
        {
            // fetch the media, decode and NIXL-register
            let decoded = match oai_content_part {
                ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                    let mdc_decoder =
                        self.media_decoder.image.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Model does not support image inputs")
                        })?;

                    let url = &image_part.image_url.url;
                    self.media_fetcher.check_if_url_allowed(url)?;
                    let data = EncodedMediaData::from_url(url, &self.http_client).await?;

                    // Use runtime decoder if provided, with MDC limits enforced
                    let decoder =
                        mdc_decoder.with_runtime(media_io_kwargs.and_then(|k| k.image.as_ref()));
                    decoder.decode_async(data).await?
                }
                #[allow(unused_variables)]
                ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                    #[cfg(not(feature = "media-ffmpeg"))]
                    anyhow::bail!(
                        "Video decoding requires the 'media-ffmpeg' feature to be enabled"
                    );

                    #[cfg(feature = "media-ffmpeg")]
                    {
                        let mdc_decoder = self.media_decoder.video.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Model does not support video inputs")
                        })?;

                        let url = &video_part.video_url.url;
                        self.media_fetcher.check_if_url_allowed(url)?;
                        let data = EncodedMediaData::from_url(url, &self.http_client).await?;

                        // Use runtime decoder if provided, with MDC limits enforced
                        let decoder = mdc_decoder
                            .with_runtime(media_io_kwargs.and_then(|k| k.video.as_ref()));
                        decoder.decode_async(data).await?
                    }
                }
                ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                    anyhow::bail!("Audio decoding is not supported yet");
                }
                _ => anyhow::bail!("Unsupported media type"),
            };

            let rdma_descriptor = decoded.into_rdma_descriptor(&self.nixl_agent)?;
            Ok(rdma_descriptor)
        }
    }
}

#[cfg(all(test, feature = "media-nixl", feature = "testing-nixl"))]
mod tests {
    use super::super::decoders::ImageDecoder;
    use super::super::rdma::DataType;
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

        let media_decoder = MediaDecoder {
            image: Some(ImageDecoder::default()),
            #[cfg(feature = "media-ffmpeg")]
            video: None,
        };
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            ..Default::default()
        };

        let loader: MediaLoader = MediaLoader::new(media_decoder, Some(fetcher)).unwrap();

        let image_url = ImageUrl::from(format!("{}/llm-optimize-deploy-graphic.png", server.url()));
        let content_part = ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage { image_url },
        );

        let result = loader
            .fetch_and_decode_media_part(&content_part, None)
            .await;

        let descriptor = match result {
            Ok(descriptor) => descriptor,
            Err(e) if e.to_string().contains("NIXL agent is not available") => {
                println!("test test_fetch_and_decode ... ignored (NIXL agent not available)");
                return;
            }
            Err(e) => panic!("Failed to fetch and decode image: {}", e),
        };
        mock.assert_async().await;
        assert_eq!(descriptor.tensor_info.dtype, DataType::UINT8);

        // Verify image dimensions: 1,999px × 1,125px (width × height)
        // Shape format is [height, width, channels]
        assert_eq!(descriptor.tensor_info.shape.len(), 3);
        assert_eq!(
            descriptor.tensor_info.shape[0], 1125,
            "Height should be 1125"
        );
        assert_eq!(
            descriptor.tensor_info.shape[1], 1999,
            "Width should be 1999"
        );
        assert_eq!(
            descriptor.tensor_info.shape[2], 4,
            "RGBA channels should be 4"
        );

        assert!(
            descriptor.source_storage.is_some(),
            "Source storage should be present"
        );
        assert!(
            descriptor.source_storage.unwrap().is_registered(),
            "Source storage should be registered with NIXL"
        );
    }
}

#[cfg(test)]
mod tests_non_nixl {
    use super::*;

    #[test]
    fn test_direct_ip_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_ip: false,
            ..Default::default()
        };

        let url = url::Url::parse("http://192.168.1.1/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);

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

        let url = url::Url::parse("http://example.com:8080/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);

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

        // Allowed domain should pass
        let url = url::Url::parse("https://trusted.com/image.jpg").unwrap();
        assert!(fetcher.check_if_url_allowed(&url).is_ok());

        // Disallowed domain should fail
        let url = url::Url::parse("https://untrusted.com/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in allowed list")
        );
    }
}
