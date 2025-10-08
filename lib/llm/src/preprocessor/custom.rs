// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom preprocessors with external implementations
//!
//! This module provides trait-based interfaces for custom preprocessors that can be
//! implemented externally (e.g., in the Python bindings crate). The core LLM library
//! remains Python-free while allowing custom implementations to be injected.

use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;

use super::prompt::OAIPromptFormatter;
use crate::tokenizers::traits::Tokenizer;

/// Factory trait for creating custom preprocessors
///
/// This trait allows external crates (like bindings) to provide factories
/// that create custom formatters and tokenizers without the core library
/// needing to know about the implementation details.
pub trait CustomPreprocessorFactory: Send + Sync + 'static {
    /// Create a custom formatter by name
    fn create_formatter(&self, name: &str) -> Result<Option<Arc<dyn OAIPromptFormatter>>>;

    /// Create a custom tokenizer by name
    fn create_tokenizer(&self, name: &str) -> Result<Option<Arc<dyn Tokenizer>>>;

    /// Check if a formatter is available
    fn has_formatter(&self, name: &str) -> bool;

    /// Check if a tokenizer is available
    fn has_tokenizer(&self, name: &str) -> bool;
}

/// Configuration for custom preprocessors in ModelDeploymentCard
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomPreprocessorConfig {
    /// Name of the custom formatter (if any)
    pub formatter_name: Option<String>,

    /// Name of the custom tokenizer (if any)
    pub tokenizer_name: Option<String>,

    /// Additional configuration parameters
    pub config: HashMap<String, serde_json::Value>,
}

impl Default for CustomPreprocessorConfig {
    fn default() -> Self {
        Self {
            formatter_name: None,
            tokenizer_name: None,
            config: HashMap::new(),
        }
    }
}

/// Global factory registry
static FACTORY_REGISTRY: once_cell::sync::Lazy<parking_lot::RwLock<Option<Arc<dyn CustomPreprocessorFactory>>>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Register a global custom preprocessor factory
///
/// This should be called once at startup by the bindings crate to register
/// its factory implementation.
pub fn register_factory(factory: Arc<dyn CustomPreprocessorFactory>) {
    let mut registry = FACTORY_REGISTRY.write();
    *registry = Some(factory);
}

/// Get the registered factory
pub fn get_factory() -> Option<Arc<dyn CustomPreprocessorFactory>> {
    FACTORY_REGISTRY.read().clone()
}

/// Create a custom formatter if available
pub fn create_custom_formatter(name: &str) -> Result<Option<Arc<dyn OAIPromptFormatter>>> {
    if let Some(factory) = get_factory() {
        factory.create_formatter(name)
    } else {
        Ok(None)
    }
}

/// Create a custom tokenizer if available
pub fn create_custom_tokenizer(name: &str) -> Result<Option<Arc<dyn Tokenizer>>> {
    if let Some(factory) = get_factory() {
        factory.create_tokenizer(name)
    } else {
        Ok(None)
    }
}

/// Check if a custom formatter is available
pub fn has_custom_formatter(name: &str) -> bool {
    if let Some(factory) = get_factory() {
        factory.has_formatter(name)
    } else {
        false
    }
}

/// Check if a custom tokenizer is available
pub fn has_custom_tokenizer(name: &str) -> bool {
    if let Some(factory) = get_factory() {
        factory.has_tokenizer(name)
    } else {
        false
    }
}