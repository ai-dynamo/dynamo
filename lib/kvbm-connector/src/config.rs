// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core configuration traits for framework integrations.
//!
//! This module defines framework-agnostic configuration interfaces based on
//! the actual data extracted from serving frameworks like vLLM.
//!
//! These traits match the structure of configuration dictionaries extracted
//! from vLLM 0.11.x via `wheels/kvbm/src/kvbm/contrib/vllm/config.py`.

use std::sync::Arc;

use kvbm_physical::layout::KvBlockLayout;

/// KV cache memory layout.
///
/// Parsed from vLLM's `get_kv_cache_layout()` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLayout {
    NHD,
    HND,
    Unknown,
}

impl CacheLayout {
    /// Parse a layout string into a CacheLayout enum.
    pub fn parse(s: &str) -> Self {
        match s {
            "NHD" => Self::NHD,
            "HND" => Self::HND,
            _ => Self::Unknown,
        }
    }
}

/// Map the framework-agnostic [`CacheLayout`] (parsed from vLLM's
/// `get_kv_cache_layout()`) onto the physical-layer [`KvBlockLayout`] used by
/// transfer-time `requires_transform()` checks.
///
/// `CacheLayout` only describes the per-block dimension ordering reported by
/// the engine, so the conversion is total only into the operational variants
/// (`OperationalNHD` / `OperationalHND`) plus `Unknown`. Universal/Custom
/// layouts are constructed elsewhere and never enter through this path.
impl From<CacheLayout> for KvBlockLayout {
    fn from(layout: CacheLayout) -> Self {
        match layout {
            CacheLayout::NHD => KvBlockLayout::OperationalNHD,
            CacheLayout::HND => KvBlockLayout::OperationalHND,
            CacheLayout::Unknown => KvBlockLayout::Unknown,
        }
    }
}

/// Distributed execution backend type.
///
/// Parsed from vLLM's backend string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelExecutorBackend {
    Ray,
    MultiProcessor,
    Unknown,
}

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
    pub host_cache: Option<kvbm_config::HostCacheConfig>,

    /// Optional disk cache (G3 tier) configuration
    pub disk_cache: Option<kvbm_config::DiskCacheConfig>,
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
    pub fn with_host_cache(mut self, config: kvbm_config::HostCacheConfig) -> Self {
        self.host_cache = Some(config);
        self
    }

    /// Set the disk cache configuration.
    pub fn with_disk_cache(mut self, config: kvbm_config::DiskCacheConfig) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_layout_into_kv_block_layout() {
        assert_eq!(
            KvBlockLayout::from(CacheLayout::NHD),
            KvBlockLayout::OperationalNHD,
        );
        assert_eq!(
            KvBlockLayout::from(CacheLayout::HND),
            KvBlockLayout::OperationalHND,
        );
        assert_eq!(
            KvBlockLayout::from(CacheLayout::Unknown),
            KvBlockLayout::Unknown,
        );
    }

    /// The full production path: vLLM reports a layout string, we parse it
    /// into [`CacheLayout`] and convert to [`KvBlockLayout`]. A prefill
    /// worker reporting `"HND"` and a decode worker reporting `"NHD"` MUST
    /// compare as `requires_transform == true` rather than falling into the
    /// silent `Unknown -> Unknown` soft-pass branch.
    #[test]
    fn parse_then_convert_round_trip() {
        let prefill: KvBlockLayout = CacheLayout::parse("HND").into();
        let decode: KvBlockLayout = CacheLayout::parse("NHD").into();
        assert_eq!(prefill, KvBlockLayout::OperationalHND);
        assert_eq!(decode, KvBlockLayout::OperationalNHD);
        assert!(
            prefill.requires_transform(&decode),
            "HND prefill -> NHD decode must require a transform kernel",
        );

        // Same layout (the GB10 dev box case) must not require a transform.
        let prefill: KvBlockLayout = CacheLayout::parse("NHD").into();
        let decode: KvBlockLayout = CacheLayout::parse("NHD").into();
        assert!(!prefill.requires_transform(&decode));

        // Empty / bogus / unrecognized strings fall back to Unknown — and
        // the resulting KvBlockLayout::Unknown is the soft-pass default we
        // are explicitly NOT pinning to a specific operational variant.
        let parsed: KvBlockLayout = CacheLayout::parse("").into();
        assert_eq!(parsed, KvBlockLayout::Unknown);
        let parsed: KvBlockLayout = CacheLayout::parse("totally-bogus").into();
        assert_eq!(parsed, KvBlockLayout::Unknown);
    }
}
