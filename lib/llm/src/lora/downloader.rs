// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{cache::LoRACache, source::LoRASourceTrait};
use anyhow::Result;
use std::{path::PathBuf, sync::Arc};

pub struct LoRADownloader {
    sources: Vec<Arc<dyn LoRASourceTrait>>,
    cache: LoRACache,
}

impl LoRADownloader {
    pub fn new(sources: Vec<Arc<dyn LoRASourceTrait>>, cache: LoRACache) -> Self {
        Self { sources, cache }
    }

    /// Download LoRA if not in cache, return local path
    ///
    /// For local file:// URIs, this will return the original path without copying.
    /// For remote URIs (s3://, etc.), this will download to cache.
    pub async fn download_if_needed(&self, lora_uri: &str) -> Result<PathBuf> {
        // For local file:// URIs, don't use cache - just validate and return
        if lora_uri.starts_with("file://") {
            for source in &self.sources {
                // Ignore errors from incompatible sources
                if let Ok(exists) = source.exists(lora_uri).await {
                    if exists {
                        // LocalLoRASource.download() returns the original path
                        return source.download(lora_uri, &PathBuf::new()).await;
                    }
                }
            }
            anyhow::bail!("Local LoRA not found: {}", lora_uri);
        }

        // For remote URIs, use the URI as the cache key
        let cache_key = self.uri_to_cache_key(lora_uri);

        // Check cache first
        if self.cache.is_cached(&cache_key) && self.cache.validate_cached(&cache_key)? {
            tracing::debug!("LoRA found in cache: {}", cache_key);
            return Ok(self.cache.get_cache_path(&cache_key));
        }

        // Try sources in order
        let dest_path = self.cache.get_cache_path(&cache_key);

        for source in &self.sources {
            // Ignore errors from incompatible sources (e.g., LocalLoRASource receiving S3 URI)
            if let Ok(exists) = source.exists(lora_uri).await {
                if exists {
                    let downloaded_path = source.download(lora_uri, &dest_path).await?;

                    // Validate downloaded files
                    if self.cache.validate_cached(&cache_key)? {
                        return Ok(downloaded_path);
                    } else {
                        tracing::warn!(
                            "Downloaded LoRA at {} failed validation",
                            downloaded_path.display()
                        );
                    }
                }
            }
        }

        anyhow::bail!("LoRA {} not found in any source", lora_uri)
    }

    /// Convert URI to cache key
    /// This extracts a reasonable path from the URI for caching
    fn uri_to_cache_key(&self, uri: &str) -> String {
        // For s3://bucket/path/to/lora -> path/to/lora
        // For gs://bucket/path/to/lora -> path/to/lora
        // For http://example.com/path/to/lora -> path/to/lora
        if let Ok(url) = url::Url::parse(uri) {
            let path = url.path().trim_start_matches('/');
            if !path.is_empty() {
                return path.to_string();
            }
        }

        // Fallback: use the whole URI (sanitized)
        uri.replace("://", "_")
            .replace('/', "_")
            .replace('\\', "_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_to_cache_key() {
        let cache = LoRACache::from_env().unwrap();
        let downloader = LoRADownloader::new(vec![], cache);

        assert_eq!(
            downloader.uri_to_cache_key("s3://bucket/path/to/lora"),
            "path/to/lora"
        );
        assert_eq!(
            downloader.uri_to_cache_key("gs://bucket/models/lora-1"),
            "models/lora-1"
        );
        assert_eq!(
            downloader.uri_to_cache_key("https://example.com/loras/my-lora"),
            "loras/my-lora"
        );
    }
}

