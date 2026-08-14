// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{cache::LoRACache, source::LoRASource};
use anyhow::Result;
use std::{path::PathBuf, sync::Arc};

pub struct LoRADownloader {
    sources: Vec<Arc<dyn LoRASource>>,
    cache: LoRACache,
}

impl LoRADownloader {
    pub fn new(sources: Vec<Arc<dyn LoRASource>>, cache: LoRACache) -> Self {
        Self { sources, cache }
    }

    /// Check source-owned caches and then the Dynamo LoRA cache for a URI.
    pub fn is_cached(&self, lora_uri: &str) -> Result<bool> {
        for source in &self.sources {
            if let Some(path) = source.cached_path(lora_uri)? {
                return LoRACache::validate_path(&path);
            }
        }

        let cache_key = self.uri_to_cache_key(lora_uri);
        self.cache.validate_cached(&cache_key)
    }

    /// Download LoRA if not in cache, return local path
    ///
    /// For local file:// URIs, this will return the original path without copying.
    /// For remote URIs (s3://, gcs://, etc.), this will download to cache.
    pub async fn download_if_needed(&self, lora_uri: &str) -> Result<PathBuf> {
        // For local file:// URIs, don't use cache - just validate and return
        if lora_uri.starts_with("file://") {
            for source in &self.sources {
                // Ignore errors from incompatible sources
                if let Ok(exists) = source.exists(lora_uri).await
                    && exists
                {
                    // LocalLoRASource.download() returns the original path
                    return source.download(lora_uri, &PathBuf::new()).await;
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
            if let Ok(exists) = source.exists(lora_uri).await
                && exists
            {
                let downloaded_path = source.download(lora_uri, &dest_path).await?;
                if LoRACache::validate_path(&downloaded_path)? {
                    return Ok(downloaded_path);
                } else {
                    tracing::warn!(
                        "Downloaded LoRA at {} failed validation",
                        downloaded_path.display()
                    );
                }
            }
        }

        anyhow::bail!("LoRA {} not found in any source", lora_uri)
    }

    /// Convert URI to cache key (delegates to LoRACache for consistency)
    fn uri_to_cache_key(&self, uri: &str) -> String {
        LoRACache::uri_to_cache_key(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::path::Path;
    use tempfile::TempDir;

    struct ExternalSnapshotSource {
        snapshot: PathBuf,
    }

    #[async_trait]
    impl LoRASource for ExternalSnapshotSource {
        async fn download(&self, _lora_uri: &str, _dest_path: &Path) -> Result<PathBuf> {
            Ok(self.snapshot.clone())
        }

        async fn exists(&self, _lora_uri: &str) -> Result<bool> {
            Ok(true)
        }

        fn cached_path(&self, lora_uri: &str) -> Result<Option<PathBuf>> {
            Ok(lora_uri.starts_with("hf://").then(|| self.snapshot.clone()))
        }
    }

    #[tokio::test]
    async fn accepts_valid_snapshot_returned_outside_dynamo_cache() {
        let dynamo_cache = TempDir::new().unwrap();
        let hf_cache = TempDir::new().unwrap();
        std::fs::write(hf_cache.path().join("adapter_config.json"), "{}").unwrap();
        std::fs::write(hf_cache.path().join("adapter_model.safetensors"), "").unwrap();

        let source = ExternalSnapshotSource {
            snapshot: hf_cache.path().to_path_buf(),
        };
        let downloader = LoRADownloader::new(
            vec![Arc::new(source)],
            LoRACache::new(dynamo_cache.path().to_path_buf()),
        );

        let result = downloader
            .download_if_needed("hf://org/adapter")
            .await
            .unwrap();

        assert_eq!(result, hf_cache.path());
    }

    #[test]
    fn is_cached_uses_source_owned_snapshot() {
        let dynamo_cache = TempDir::new().unwrap();
        let hf_cache = TempDir::new().unwrap();
        std::fs::write(hf_cache.path().join("adapter_config.json"), "{}").unwrap();
        std::fs::write(hf_cache.path().join("adapter_model.safetensors"), "").unwrap();

        let downloader = LoRADownloader::new(
            vec![Arc::new(ExternalSnapshotSource {
                snapshot: hf_cache.path().to_path_buf(),
            })],
            LoRACache::new(dynamo_cache.path().to_path_buf()),
        );

        assert!(downloader.is_cached("hf://org/adapter").unwrap());
    }
}
