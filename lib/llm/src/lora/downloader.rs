// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{cache::LoRACache, source::LoRASource};
use anyhow::Result;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

pub struct LoRADownloader {
    sources: Vec<Arc<dyn LoRASource>>,
    cache: LoRACache,
    /// Per-URI locks to prevent concurrent downloads to the same path.
    download_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl LoRADownloader {
    pub fn new(sources: Vec<Arc<dyn LoRASource>>, cache: LoRACache) -> Self {
        Self {
            sources,
            cache,
            download_locks: Mutex::new(HashMap::new()),
        }
    }

    /// Download LoRA if not in cache, return local path
    ///
    /// For local file:// URIs, this will return the original path without copying.
    /// For remote URIs (s3://, gcs://, etc.), this will download to cache.
    ///
    /// This method is safe to call concurrently for the same URI - only one download
    /// will happen at a time, and other callers will wait and reuse the cached result.
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

        // Fast path: check cache without holding the download lock
        if self.cache.is_cached(&cache_key) && self.cache.validate_cached(&cache_key)? {
            tracing::debug!("LoRA found in cache: {}", cache_key);
            return Ok(self.cache.get_cache_path(&cache_key));
        }

        // Acquire per-URI download lock to prevent concurrent downloads of the same URI.
        // The HashMap lock is held only briefly to get/create the per-URI lock, then released.
        // This allows different URIs to download in parallel while same-URI downloads are serialized.
        let lock = self
            .download_locks
            .lock()
            .await
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = lock.lock().await;

        // Double-check: another concurrent call may have completed the download while we waited
        if self.cache.is_cached(&cache_key) && self.cache.validate_cached(&cache_key)? {
            tracing::debug!(
                "LoRA found in cache after acquiring lock (downloaded by concurrent request): {}",
                cache_key
            );
            return Ok(self.cache.get_cache_path(&cache_key));
        }

        tracing::info!(
            "Downloading LoRA (no concurrent download in progress): {}",
            lora_uri
        );

        // Try sources in order
        let dest_path = self.cache.get_cache_path(&cache_key);

        for source in &self.sources {
            if let Ok(exists) = source.exists(lora_uri).await
                && exists
            {
                let downloaded_path = source.download(lora_uri, &dest_path).await?;
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

        anyhow::bail!("LoRA {} not found in any source", lora_uri)
    }

    /// Convert URI to cache key (delegates to LoRACache for consistency)
    fn uri_to_cache_key(&self, uri: &str) -> String {
        LoRACache::uri_to_cache_key(uri)
    }
}
