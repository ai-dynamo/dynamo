// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM configuration traits and types.
//!
//! This module defines trait-based interfaces for vLLM configuration, allowing
//! the Python bindings to implement these traits while keeping pure Rust code
//! independent of Python-specific types.

use std::sync::Arc;

use crate::v2::integrations::config::{AttentionConfig, IntegrationsConfig, ParallelConfig};

/// Trait for vLLM parallel configuration access.
///
/// This extends the core `ParallelConfig` trait with vLLM-specific methods.
/// Most functionality is provided by the core trait.
pub trait VllmParallelConfig: ParallelConfig {
    // All core methods are inherited from ParallelConfig
    // Add vLLM-specific methods here if needed in the future
}

/// Trait for vLLM attention and cache configuration access.
///
/// This extends the core `AttentionConfig` trait with vLLM-specific methods.
/// Most functionality is provided by the core trait.
pub trait VllmAttentionConfig: AttentionConfig {
    // All core methods are inherited from AttentionConfig
    // Add vLLM-specific methods here if needed in the future
}

/// Combined vLLM configuration container.
///
/// Holds trait objects for parallel and attention configuration,
/// allowing flexible implementation while maintaining type safety.
#[derive(Clone)]
pub struct KvbmVllmConfig {
    /// Parallel execution configuration
    pub parallel: Arc<dyn VllmParallelConfig>,

    /// Attention and cache configuration
    pub attention: Arc<dyn VllmAttentionConfig>,
}

impl std::fmt::Debug for KvbmVllmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvbmVllmConfig")
            .field("parallel", &self.parallel)
            .field("attention", &self.attention)
            .finish()
    }
}

impl KvbmVllmConfig {
    /// Create a new KvbmVllmConfig from trait implementations.
    pub fn new(
        parallel: Arc<dyn VllmParallelConfig>,
        attention: Arc<dyn VllmAttentionConfig>,
    ) -> Self {
        Self {
            parallel,
            attention,
        }
    }

    /// Get the block size from attention configuration.
    pub fn block_size(&self) -> usize {
        self.attention.block_size()
    }

    /// Convert to a generic IntegrationsConfig.
    ///
    /// This allows vLLM-specific configuration to be used with framework-agnostic
    /// code by upcasting the trait objects to the base traits.
    pub fn as_generic(&self) -> IntegrationsConfig {
        IntegrationsConfig::new(
            self.parallel.clone() as Arc<dyn ParallelConfig>,
            self.attention.clone() as Arc<dyn AttentionConfig>,
        )
    }
}
