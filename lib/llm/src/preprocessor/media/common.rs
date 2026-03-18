// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};

// Raw encoded media data (.png, .mp4, ...), optionally b64-encoded
#[derive(Debug)]
pub struct EncodedMediaData {
    pub(crate) bytes: Vec<u8>,
    pub(crate) b64_encoded: bool,
}

impl EncodedMediaData {
    // Handles both web URLs (will download the bytes) and data URLs (will keep b64-encoded)
    pub async fn from_url(url: &url::Url, client: &reqwest::Client) -> Result<Self> {
        let (bytes, b64_encoded) = match url.scheme() {
            "data" => {
                let base64_data = url
                    .as_str()
                    .split_once(',')
                    .ok_or_else(|| anyhow::anyhow!("Invalid media data URL format"))?
                    .1;
                anyhow::ensure!(!base64_data.is_empty(), "Media data URL is empty");
                (base64_data.as_bytes().to_vec(), true)
            }
            "http" | "https" => {
                let bytes = client
                    .get(url.to_string())
                    .send()
                    .await?
                    .error_for_status()?
                    .bytes()
                    .await?;
                anyhow::ensure!(!bytes.is_empty(), "Media URL is empty");
                (bytes.to_vec(), false)
            }
            scheme => anyhow::bail!("Unsupported media URL scheme: {scheme}"),
        };

        Ok(Self { bytes, b64_encoded })
    }

    // Potentially decodes b64 bytes
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        if self.b64_encoded {
            Ok(general_purpose::STANDARD.decode(self.bytes)?)
        } else {
            Ok(self.bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_from_base64() {
        // Simple base64 encoded "test" string: dGVzdA==
        let data_url = url::Url::parse("data:text/plain;base64,dGVzdA==").unwrap();
        let client = reqwest::Client::new();

        let result = EncodedMediaData::from_url(&data_url, &client)
            .await
            .unwrap();

        assert!(result.b64_encoded);
        assert_eq!(result.bytes, b"dGVzdA==");
        let decoded = result.into_bytes().unwrap();
        assert_eq!(decoded, b"test");
    }

    #[tokio::test]
    async fn test_from_empty_base64() {
        let data_url = url::Url::parse("data:text/plain;base64,").unwrap();
        let client = reqwest::Client::new();

        let result = EncodedMediaData::from_url(&data_url, &client).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_from_invalid_base64() {
        let data_url = url::Url::parse("data:invalid").unwrap();
        let client = reqwest::Client::new();

        let result = EncodedMediaData::from_url(&data_url, &client).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_from_url_http() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/image.png")
            .with_status(200)
            .with_body(b"test data")
            .create_async()
            .await;

        let url = url::Url::parse(&format!("{}/image.png", server.url())).unwrap();
        let client = reqwest::Client::new();

        let result = EncodedMediaData::from_url(&url, &client).await.unwrap();

        assert!(!result.b64_encoded);
        assert_eq!(result.bytes, b"test data");
        let decoded = result.into_bytes().unwrap();
        assert_eq!(decoded, b"test data");

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_from_url_http_404() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/image.png")
            .with_status(404)
            .create_async()
            .await;

        let url = url::Url::parse(&format!("{}/image.png", server.url())).unwrap();
        let client = reqwest::Client::new();
        let result = EncodedMediaData::from_url(&url, &client).await;
        assert!(result.is_err());

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_from_unsupported_scheme() {
        let ftp_url = url::Url::parse("ftp://example.com/image.png").unwrap();
        let client = reqwest::Client::new();

        let result = EncodedMediaData::from_url(&ftp_url, &client).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported media URL scheme")
        );
    }
}
