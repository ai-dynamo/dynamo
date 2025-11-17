// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::lora::{LoRACache, LoRADownloader, LoRASourceTrait, LocalLoRASource, S3LoRASource};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct PyLoRACache {
    inner: LoRACache,
}

#[pymethods]
impl PyLoRACache {
    #[new]
    pub fn new(cache_root: String) -> PyResult<Self> {
        Ok(Self {
            inner: LoRACache::new(cache_root.into()),
        })
    }

    #[staticmethod]
    pub fn from_env() -> PyResult<Self> {
        let inner = LoRACache::from_env()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn is_cached(&self, lora_id: String) -> bool {
        self.inner.is_cached(&lora_id)
    }

    pub fn get_cache_path(&self, lora_id: String) -> String {
        self.inner.get_cache_path(&lora_id).display().to_string()
    }

    pub fn validate_cached(&self, lora_id: String) -> PyResult<bool> {
        self.inner
            .validate_cached(&lora_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pyclass]
pub struct PyLocalLoRASource {
    inner: Arc<LocalLoRASource>,
}

#[pymethods]
impl PyLocalLoRASource {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(LocalLoRASource::new()),
        }
    }

    pub fn exists(&self, py: Python, lora_uri: String) -> PyResult<bool> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            rt.block_on(async move {
                inner
                    .exists(&lora_uri)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })
    }
}

#[pyclass]
pub struct PyS3LoRASource {
    inner: Arc<S3LoRASource>,
}

#[pymethods]
impl PyS3LoRASource {
    #[staticmethod]
    pub fn from_env(py: Python) -> PyResult<Self> {
        py.allow_threads(|| {
            let inner = S3LoRASource::from_env()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(inner),
            })
        })
    }

    pub fn exists(&self, py: Python, s3_uri: String) -> PyResult<bool> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            rt.block_on(async move {
                inner
                    .exists(&s3_uri)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })
    }
}

#[pyclass]
pub struct PyLoRADownloader {
    inner: Arc<LoRADownloader>,
}

#[pymethods]
impl PyLoRADownloader {
    #[new]
    pub fn new(py: Python, cache: &PyLoRACache) -> PyResult<Self> {
        py.allow_threads(|| {
            // Create a downloader with local and S3 sources from env
            let mut rust_sources: Vec<Arc<dyn LoRASourceTrait>> = vec![];

            // Add LocalLoRASource
            rust_sources.push(Arc::new(LocalLoRASource::new()));

            // Try to add S3LoRASource if env vars are set
            if let Ok(s3_source) = S3LoRASource::from_env() {
                rust_sources.push(Arc::new(s3_source));
            }

            let downloader = LoRADownloader::new(rust_sources, cache.inner.clone());

            Ok(Self {
                inner: Arc::new(downloader),
            })
        })
    }

    #[staticmethod]
    pub fn create_default(py: Python) -> PyResult<Self> {
        py.allow_threads(|| {
            let cache = LoRACache::from_env()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut sources: Vec<Arc<dyn LoRASourceTrait>> = vec![];

            // Add LocalLoRASource
            sources.push(Arc::new(LocalLoRASource::new()));

            // Try to add S3LoRASource if env vars are set
            if let Ok(s3_source) = S3LoRASource::from_env() {
                sources.push(Arc::new(s3_source));
            }

            let downloader = LoRADownloader::new(sources, cache);

            Ok(Self {
                inner: Arc::new(downloader),
            })
        })
    }

    pub fn download_if_needed(&self, py: Python, lora_uri: String) -> PyResult<String> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let path = rt
                .block_on(inner.download_if_needed(&lora_uri))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(path.display().to_string())
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoRACache>()?;
    m.add_class::<PyLocalLoRASource>()?;
    m.add_class::<PyS3LoRASource>()?;
    m.add_class::<PyLoRADownloader>()?;
    Ok(())
}

