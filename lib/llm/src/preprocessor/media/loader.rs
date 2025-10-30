// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;

const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";

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
            timeout: None,
        }
    }
}

pub struct MediaLoader {
    http_client: reqwest::Client,
    media_fetcher: MediaFetcher,
    // TODO: decoders, NIXL agent
}

impl MediaLoader {
    pub fn new(media_fetcher: MediaFetcher) -> Result<Self> {
        let mut http_client_builder =
            reqwest::Client::builder().user_agent(&media_fetcher.user_agent);

        if let Some(timeout) = media_fetcher.timeout {
            http_client_builder = http_client_builder.timeout(timeout);
        }

        let http_client = http_client_builder.build()?;

        Ok(Self {
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
                self.check_if_url_allowed(url)?;
                EncodedMediaData::from_url(url, &self.http_client).await?
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                self.check_if_url_allowed(url)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_ip_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_ip: false,
            ..Default::default()
        };
        let loader = MediaLoader::new(fetcher).unwrap();

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
        let loader = MediaLoader::new(fetcher).unwrap();

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
        let loader = MediaLoader::new(fetcher).unwrap();

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
