// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for vLLM configuration.
//!
//! This module provides Python-facing structs that implement the Rust
//! configuration traits, allowing Python code to pass vLLM configuration
//! data to Rust constructors.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use dynamo_kvbm::v2::ModelExecutorBackend;
use dynamo_kvbm::v2::integrations::config::{AttentionConfig, ParallelConfig};
use dynamo_kvbm::v2::integrations::vllm::{
    KvbmVllmConfig, VllmAttentionConfig, VllmParallelConfig,
};

/// Python-facing parallel configuration.
///
/// This struct holds the data extracted from vLLM's ParallelConfig and
/// implements the Rust ParallelConfig trait.
#[derive(Debug, Clone)]
pub struct PyParallelConfig {
    pub world_size: usize,
    pub rank: usize,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub data_parallel_size: usize,
    pub data_parallel_rank: usize,
    pub backend: String,
}

impl PyParallelConfig {
    /// Create from Python dictionary.
    ///
    /// Expected keys match the output of `extract_vllm_config_for_kvbm()`:
    /// - world_size: int
    /// - rank: int
    /// - tensor_parallel_size: int
    /// - pipeline_parallel_size: int
    /// - data_parallel_size: int
    /// - data_parallel_rank: int
    /// - backend: str
    pub fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            world_size: dict
                .get_item("world_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'world_size'")
                })?
                .extract()?,
            rank: dict
                .get_item("rank")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'rank'"))?
                .extract()?,
            tensor_parallel_size: dict
                .get_item("tensor_parallel_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'tensor_parallel_size'")
                })?
                .extract()?,
            pipeline_parallel_size: dict
                .get_item("pipeline_parallel_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'pipeline_parallel_size'",
                    )
                })?
                .extract()?,
            data_parallel_size: dict
                .get_item("data_parallel_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'data_parallel_size'")
                })?
                .extract()?,
            data_parallel_rank: dict
                .get_item("data_parallel_rank")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'data_parallel_rank'")
                })?
                .extract()?,
            backend: dict
                .get_item("backend")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'backend'"))?
                .extract()?,
        })
    }
}

// Implement the core ParallelConfig trait
impl ParallelConfig for PyParallelConfig {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    fn pipeline_parallel_size(&self) -> usize {
        self.pipeline_parallel_size
    }

    fn data_parallel_size(&self) -> usize {
        self.data_parallel_size
    }

    fn data_parallel_rank(&self) -> usize {
        self.data_parallel_rank
    }

    fn backend(&self) -> ModelExecutorBackend {
        ModelExecutorBackend::parse(&self.backend)
    }
}

// Implement the vLLM marker trait
impl VllmParallelConfig for PyParallelConfig {}

/// Python-facing attention configuration.
///
/// This struct holds the data extracted from vLLM's CacheConfig and ModelConfig
/// and implements the Rust AttentionConfig trait.
#[derive(Debug, Clone)]
pub struct PyAttentionConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub num_cpu_blocks: usize,
    pub cache_dtype_bytes: usize,
    pub kv_cache_layout: String,
    pub head_size: usize,
    pub num_heads: usize,
    #[allow(dead_code)]
    pub device_id: usize,
}

impl PyAttentionConfig {
    /// Create from Python dictionary.
    ///
    /// Expected keys match the output of `extract_vllm_config_for_kvbm()`:
    /// - block_size: int
    /// - num_gpu_blocks: int
    /// - num_cpu_blocks: int
    /// - cache_dtype_bytes: int
    /// - kv_cache_layout: str
    /// - head_size: int
    /// - num_heads: int
    ///
    /// The device_id is computed from the parallel config rank.
    pub fn from_dict(dict: &Bound<'_, PyDict>, device_id: usize) -> PyResult<Self> {
        Ok(Self {
            block_size: dict
                .get_item("block_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'block_size'")
                })?
                .extract()?,
            num_gpu_blocks: dict
                .get_item("num_gpu_blocks")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'num_gpu_blocks'")
                })?
                .extract()?,
            num_cpu_blocks: dict
                .get_item("num_cpu_blocks")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'num_cpu_blocks'")
                })?
                .extract()?,
            cache_dtype_bytes: dict
                .get_item("cache_dtype_bytes")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'cache_dtype_bytes'")
                })?
                .extract()?,
            kv_cache_layout: dict
                .get_item("kv_cache_layout")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'kv_cache_layout'")
                })?
                .extract()?,
            head_size: dict
                .get_item("head_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'head_size'")
                })?
                .extract()?,
            num_heads: dict
                .get_item("num_heads")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'num_heads'")
                })?
                .extract()?,
            device_id,
        })
    }
}

// Implement the core AttentionConfig trait
impl AttentionConfig for PyAttentionConfig {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    fn num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }

    fn cache_dtype_bytes(&self) -> usize {
        self.cache_dtype_bytes
    }

    fn kv_cache_layout(&self) -> &str {
        &self.kv_cache_layout
    }

    fn head_size(&self) -> usize {
        self.head_size
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// Implement the vLLM marker trait
impl VllmAttentionConfig for PyAttentionConfig {}

/// Python-facing KvbmVllmConfig wrapper.
///
/// This is the main class exposed to Python. It takes the dictionaries
/// from `extract_vllm_config_for_kvbm()` and creates a Rust KvbmVllmConfig
/// that can be passed to Rust constructors.
#[pyclass(name = "KvbmVllmConfig")]
pub struct PyKvbmVllmConfig {
    inner: KvbmVllmConfig,
}

#[pymethods]
impl PyKvbmVllmConfig {
    /// Create a new KvbmVllmConfig from Python dictionaries.
    ///
    /// # Arguments
    /// * `parallel_dict` - Dictionary with parallel configuration (from vLLM)
    /// * `attention_dict` - Dictionary with attention/cache configuration (from vLLM)
    ///
    /// # Example
    /// ```python
    /// from kvbm.contrib.vllm.config import extract_vllm_config_for_kvbm
    /// from kvbm.contrib.vllm import KvbmVllmConfig
    ///
    /// config_dict = extract_vllm_config_for_kvbm(vllm_config)
    /// vllm_config = KvbmVllmConfig(config_dict["parallel"], config_dict["attention"])
    /// ```
    #[new]
    pub fn new(
        parallel_dict: &Bound<'_, PyDict>,
        attention_dict: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // Parse parallel config
        let parallel_config = PyParallelConfig::from_dict(parallel_dict)?;

        // Compute device_id from rank (could be more sophisticated)
        let device_id = parallel_config.rank();

        // Parse attention config
        let attention_config = PyAttentionConfig::from_dict(attention_dict, device_id)?;

        // Wrap in Arc<dyn Trait>
        let parallel: Arc<dyn VllmParallelConfig> = Arc::new(parallel_config);
        let attention: Arc<dyn VllmAttentionConfig> = Arc::new(attention_config);

        // Create KvbmVllmConfig
        let inner = KvbmVllmConfig::new(parallel, attention);

        Ok(Self { inner })
    }

    /// Get the block size from attention configuration.
    pub fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    // === Parallel Configuration Getters ===

    /// Get the worker ID (rank) from parallel configuration.
    pub fn worker_id(&self) -> usize {
        self.inner.parallel.rank()
    }

    /// Get the global rank of this process.
    pub fn rank(&self) -> usize {
        self.inner.parallel.rank()
    }

    /// Get the total world size (number of processes).
    pub fn world_size(&self) -> usize {
        self.inner.parallel.world_size()
    }

    /// Get the tensor parallel size.
    pub fn tensor_parallel_size(&self) -> usize {
        self.inner.parallel.tensor_parallel_size()
    }

    /// Get the pipeline parallel size.
    pub fn pipeline_parallel_size(&self) -> usize {
        self.inner.parallel.pipeline_parallel_size()
    }

    /// Get the data parallel size.
    pub fn data_parallel_size(&self) -> usize {
        self.inner.parallel.data_parallel_size()
    }

    /// Get the data parallel rank.
    pub fn data_parallel_rank(&self) -> usize {
        self.inner.parallel.data_parallel_rank()
    }

    /// Get the backend type as a string.
    pub fn backend(&self) -> String {
        format!("{:?}", self.inner.parallel.backend())
    }

    /// Get the device ID (same as rank in standard configuration).
    pub fn device_id(&self) -> usize {
        // Device ID is computed from rank during construction
        self.inner.parallel.rank()
    }

    // === Attention Configuration Getters ===

    /// Get the number of GPU blocks allocated for KV cache.
    pub fn num_gpu_blocks(&self) -> usize {
        self.inner.attention.num_gpu_blocks()
    }

    /// Get the number of CPU blocks allocated for KV cache offloading.
    pub fn num_cpu_blocks(&self) -> usize {
        self.inner.attention.num_cpu_blocks()
    }

    /// Get the cache dtype size in bytes.
    pub fn cache_dtype_bytes(&self) -> usize {
        self.inner.attention.cache_dtype_bytes()
    }

    /// Get the KV cache memory layout as a string (e.g., "NHD", "HND").
    pub fn kv_cache_layout(&self) -> String {
        self.inner.attention.kv_cache_layout().to_string()
    }

    /// Get the head size (dimension per attention head).
    pub fn head_size(&self) -> usize {
        self.inner.attention.head_size()
    }

    /// Get the number of key-value heads.
    pub fn num_heads(&self) -> usize {
        self.inner.attention.num_heads()
    }

    /// Get a detailed debug string showing full configuration.
    ///
    /// This method uses Rust's Debug trait formatting to show all
    /// configuration values in a structured format.
    pub fn debug_string(&self) -> String {
        format!("{:#?}", self.inner)
    }

    /// Get parallel configuration as a Python dictionary.
    ///
    /// Returns a dictionary with all parallel configuration values:
    /// - world_size
    /// - rank
    /// - tensor_parallel_size
    /// - pipeline_parallel_size
    /// - data_parallel_size
    /// - data_parallel_rank
    /// - backend
    pub fn get_parallel_config<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("world_size", self.inner.parallel.world_size())?;
        dict.set_item("rank", self.inner.parallel.rank())?;
        dict.set_item(
            "tensor_parallel_size",
            self.inner.parallel.tensor_parallel_size(),
        )?;
        dict.set_item(
            "pipeline_parallel_size",
            self.inner.parallel.pipeline_parallel_size(),
        )?;
        dict.set_item(
            "data_parallel_size",
            self.inner.parallel.data_parallel_size(),
        )?;
        dict.set_item(
            "data_parallel_rank",
            self.inner.parallel.data_parallel_rank(),
        )?;
        dict.set_item("backend", format!("{:?}", self.inner.parallel.backend()))?;
        Ok(dict)
    }

    /// Get attention configuration as a Python dictionary.
    ///
    /// Returns a dictionary with all attention/cache configuration values:
    /// - block_size
    /// - num_gpu_blocks
    /// - num_cpu_blocks
    /// - cache_dtype_bytes
    /// - kv_cache_layout
    /// - head_size
    /// - num_heads
    pub fn get_attention_config<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("block_size", self.inner.attention.block_size())?;
        dict.set_item("num_gpu_blocks", self.inner.attention.num_gpu_blocks())?;
        dict.set_item("num_cpu_blocks", self.inner.attention.num_cpu_blocks())?;
        dict.set_item(
            "cache_dtype_bytes",
            self.inner.attention.cache_dtype_bytes(),
        )?;
        dict.set_item("kv_cache_layout", self.inner.attention.kv_cache_layout())?;
        dict.set_item("head_size", self.inner.attention.head_size())?;
        dict.set_item("num_heads", self.inner.attention.num_heads())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "KvbmVllmConfig(worker_id={}, device_id={}, block_size={}, world_size={}, tp={}, pp={}, dp={})",
            self.worker_id(),
            self.device_id(),
            self.block_size(),
            self.inner.parallel.world_size(),
            self.inner.parallel.tensor_parallel_size(),
            self.inner.parallel.pipeline_parallel_size(),
            self.inner.parallel.data_parallel_size(),
        )
    }

    fn __str__(&self) -> String {
        format!(
            "KvbmVllmConfig(rank={}/{}, block_size={})",
            self.inner.parallel.rank(),
            self.inner.parallel.world_size(),
            self.block_size()
        )
    }
}

impl PyKvbmVllmConfig {
    /// Get the inner KvbmVllmConfig for passing to Rust constructors.
    pub fn inner(&self) -> &KvbmVllmConfig {
        &self.inner
    }
}
