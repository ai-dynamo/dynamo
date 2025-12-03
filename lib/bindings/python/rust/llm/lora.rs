// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for LoRA downloading and caching
//!
//! Provides a single unified interface (LoRADownloader) for all LoRA operations.

use dynamo_llm::lora::{
    LoRACache as RsLoRACache, LoRADownloader as RsLoRADownloader, LocalLoRASource, S3LoRASource,
};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Unified Python interface for LoRA downloading and caching.
///
/// Handles local file:// URIs (zero-copy) and S3 s3:// URIs with automatic caching.
#[pyclass(name = "LoRADownloader")]
pub struct LoRADownloader {
    inner: Arc<RsLoRADownloader>,
    cache: RsLoRACache,
}

#[pymethods]
impl LoRADownloader {
    /// Create downloader with custom cache path.
    #[new]
    #[pyo3(signature = (cache_path=None))]
    fn new(cache_path: Option<String>) -> PyResult<Self> {
        let cache = match cache_path {
            Some(path) => RsLoRACache::new(PathBuf::from(path)),
            None => RsLoRACache::from_env().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create cache: {}", e))
            })?,
        };

        let mut sources: Vec<Arc<dyn dynamo_llm::lora::LoRASource>> =
            vec![Arc::new(LocalLoRASource::new())];

        if let Ok(s3_source) = S3LoRASource::from_env() {
            sources.push(Arc::new(s3_source));
        }

        let downloader = RsLoRADownloader::new(sources, cache.clone());
        Ok(Self {
            inner: Arc::new(downloader),
            cache,
        })
    }

    /// Download LoRA if not in cache, return local path.
    ///
    /// - file:// URIs: Returns original path (no copy)
    /// - s3:// URIs: Downloads to cache, returns cache path
    fn download_if_needed<'p>(
        &self,
        py: Python<'p>,
        lora_uri: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let downloader = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let path = downloader
                .download_if_needed(&lora_uri)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Download failed: {}", e))
                })?;
            Ok(path.display().to_string())
        })
    }

    /// Get local cache path for a cache key.
    fn get_cache_path(&self, cache_key: &str) -> String {
        self.cache.get_cache_path(cache_key).display().to_string()
    }

    /// Check if LoRA is cached (by cache key).
    fn is_cached(&self, cache_key: &str) -> bool {
        self.cache.is_cached(cache_key)
    }

    /// Validate cached LoRA has required files.
    fn validate_cached(&self, cache_key: &str) -> PyResult<bool> {
        self.cache.validate_cached(cache_key).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Validation failed: {}", e))
        })
    }

    /// Convert a LoRA URI to a cache key.
    /// This ensures consistent cache key generation across Rust and Python.
    #[staticmethod]
    fn uri_to_cache_key(uri: &str) -> String {
        RsLoRACache::uri_to_cache_key(uri)
    }
}
