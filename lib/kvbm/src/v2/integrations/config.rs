// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core configuration traits for framework integrations.
//!
//! This module defines framework-agnostic configuration interfaces based on
//! the actual data extracted from serving frameworks like vLLM.
//!
//! These traits match the structure of configuration dictionaries extracted
//! from vLLM 0.11.x via `wheels/kvbm/src/kvbm/contrib/vllm/config.py`.

use std::sync::Arc;

use crate::v2::{CacheLayout, ModelExecutorBackend};

/// Trait for parallel execution configuration.
pub trait ParallelConfig: Send + Sync + std::fmt::Debug {
    /// Total world size (total number of processes).
    fn world_size(&self) -> usize;

    /// Global rank of this process.
    fn rank(&self) -> usize;

    /// Tensor parallel size (number of GPUs for tensor parallelism).
    fn tensor_parallel_size(&self) -> usize;

    /// Pipeline parallel size (number of stages in pipeline).
    fn pipeline_parallel_size(&self) -> usize;

    /// Data parallel size (for multi-instance serving).
    fn data_parallel_size(&self) -> usize;

    /// Data parallel rank (rank within data parallel group).
    fn data_parallel_rank(&self) -> usize;

    /// Distributed backend type.
    ///
    /// This parses the vLLM backend string and returns a typed enum:
    /// - "ray" → ModelExecutorBackend::Ray
    /// - "mp" → ModelExecutorBackend::MultiProcessor
    /// - "uni", "external_launcher", etc. → ModelExecutorBackend::Unknown
    fn backend(&self) -> ModelExecutorBackend;
}

/// Trait for attention mechanism and cache configuration.
pub trait AttentionConfig: Send + Sync + std::fmt::Debug {
    /// Block size (tokens per block/page).
    fn block_size(&self) -> usize;

    /// Number of GPU blocks allocated for KV cache.
    fn num_gpu_blocks(&self) -> usize;

    /// Number of CPU blocks allocated for KV cache offloading.
    fn num_cpu_blocks(&self) -> usize;

    /// Cache dtype size in bytes (1, 2, or 4).
    ///
    /// This is the raw byte size extracted from vLLM configuration.
    /// Use `cache_dtype()` helper to get a typed CacheDtype enum.
    fn cache_dtype_bytes(&self) -> usize;

    /// KV cache memory layout as string (e.g., "NHD", "HND").
    ///
    /// This is the raw layout string from vLLM's `get_kv_cache_layout()`.
    /// Use `cache_layout()` helper to get a typed CacheLayout enum.
    fn kv_cache_layout(&self) -> &str;

    /// Head size (dimension per attention head).
    fn head_size(&self) -> usize;

    /// Number of key-value heads.
    fn num_heads(&self) -> usize;

    // === Typed helper methods ===

    /// Get the cache layout as a typed enum.
    ///
    /// Parses the raw `kv_cache_layout()` string into a CacheLayout enum.
    /// Returns CacheLayout::Unknown for unrecognized strings.
    fn cache_layout(&self) -> CacheLayout {
        CacheLayout::parse(self.kv_cache_layout())
    }
}

/// Generic KVBM configuration container.
///
/// Holds trait objects for parallel and attention configuration from any
/// framework (vLLM, TensorRT-LLM, etc.). This allows framework-agnostic
/// code to work with configuration data.
#[derive(Clone)]
pub struct IntegrationsConfig {
    /// Parallel execution configuration
    pub parallel: Arc<dyn ParallelConfig>,

    /// Attention and cache configuration
    pub attention: Arc<dyn AttentionConfig>,

    /// Optional host cache (G2 tier) configuration
    pub host_cache: Option<dynamo_kvbm_config::HostCacheConfig>,

    /// Optional disk cache (G3 tier) configuration
    pub disk_cache: Option<dynamo_kvbm_config::DiskCacheConfig>,
}

impl std::fmt::Debug for IntegrationsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntegrationsConfig")
            .field("parallel", &self.parallel)
            .field("attention", &self.attention)
            .field("host_cache", &self.host_cache)
            .field("disk_cache", &self.disk_cache)
            .finish()
    }
}

impl IntegrationsConfig {
    /// Create a new IntegrationsConfig from trait implementations.
    pub fn new(parallel: Arc<dyn ParallelConfig>, attention: Arc<dyn AttentionConfig>) -> Self {
        Self {
            parallel,
            attention,
            host_cache: None,
            disk_cache: None,
        }
    }

    /// Set the host cache configuration.
    pub fn with_host_cache(mut self, config: dynamo_kvbm_config::HostCacheConfig) -> Self {
        self.host_cache = Some(config);
        self
    }

    /// Set the disk cache configuration.
    pub fn with_disk_cache(mut self, config: dynamo_kvbm_config::DiskCacheConfig) -> Self {
        self.disk_cache = Some(config);
        self
    }

    /// Get the block size from attention configuration.
    pub fn block_size(&self) -> usize {
        self.attention.block_size()
    }

    /// Get the rank from parallel configuration.
    pub fn rank(&self) -> usize {
        self.parallel.rank()
    }

    /// Get the world size from parallel configuration.
    pub fn world_size(&self) -> usize {
        self.parallel.world_size()
    }
}
