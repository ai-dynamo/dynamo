// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Classifier factory for creating instances from configuration
//!
//! Provides a unified way to instantiate classifiers without backend-specific
//! code in the HTTP layer. The factory reads configuration and returns a
//! trait object that can be used anywhere.

use super::{Classifier, MockClassifier};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::sync::Arc;

/// Classifier backend type
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierBackend {
    /// Mock classifier for testing
    Mock,
    /// FastText classifier
    Fasttext,
    /// ONNX Runtime classifier
    Onnx,
    /// Candle/HuggingFace classifier
    Candle,
    /// HTTP remote classifier
    Http,
}

/// Classifier configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ClassifierConfig {
    /// Backend type
    pub backend: ClassifierBackend,

    /// Model path (for file-based classifiers)
    #[serde(default)]
    pub path: Option<String>,

    /// Vocabulary path (for some classifiers)
    #[serde(default)]
    pub vocab: Option<String>,

    /// Confidence threshold
    #[serde(default = "default_threshold")]
    pub threshold: f32,

    /// HTTP endpoint (for remote classifiers)
    #[serde(default)]
    pub endpoint: Option<String>,

    /// Device (cpu, cuda:0, etc.)
    #[serde(default = "default_device")]
    pub device: String,

    /// Model ID (for HuggingFace models)
    #[serde(default)]
    pub model_id: Option<String>,

    /// Max sequence length
    #[serde(default = "default_max_length")]
    pub max_length: usize,
}

fn default_threshold() -> f32 {
    0.6
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_max_length() -> usize {
    256
}

/// Create a classifier from configuration
///
/// # Arguments
/// * `config` - Classifier configuration
///
/// # Returns
/// * `Arc<dyn Classifier>` - Classifier instance
///
/// # Errors
/// * Feature not enabled for requested backend
/// * Invalid configuration
/// * Model loading failure
pub fn create_classifier(config: &ClassifierConfig) -> Result<Arc<dyn Classifier>> {
    match config.backend {
        ClassifierBackend::Mock => {
            tracing::info!("Creating MockClassifier");
            Ok(Arc::new(MockClassifier::new()))
        }

        #[cfg(any(feature = "clf-fasttext", feature = "fasttext-classifier"))]
        ClassifierBackend::Fasttext => {
            use super::FasttextClassifier;

            let path = config.path.as_ref()
                .ok_or_else(|| anyhow!("FastText classifier requires 'path' config"))?;

            tracing::info!("Creating FasttextClassifier from: {}", path);
            let classifier = FasttextClassifier::new(path)?;
            Ok(Arc::new(classifier))
        }

        #[cfg(not(any(feature = "clf-fasttext", feature = "fasttext-classifier")))]
        ClassifierBackend::Fasttext => {
            Err(anyhow!("FastText classifier requires 'clf-fasttext' or 'fasttext-classifier' feature"))
        }

        #[cfg(any(feature = "clf-candle", feature = "candle-classifier"))]
        ClassifierBackend::Candle => {
            use super::CandleClassifier;

            let model_id = config.model_id.as_ref()
                .ok_or_else(|| anyhow!("Candle classifier requires 'model_id' config"))?;

            let device = parse_device(&config.device)?;

            tracing::info!("Creating CandleClassifier: model={}, device={:?}", model_id, device);
            let classifier = CandleClassifier::from_pretrained(
                model_id,
                config.max_length,
                device,
            )?;
            Ok(Arc::new(classifier))
        }

        #[cfg(not(any(feature = "clf-candle", feature = "candle-classifier")))]
        ClassifierBackend::Candle => {
            Err(anyhow!("Candle classifier requires 'clf-candle' or 'candle-classifier' feature"))
        }

        ClassifierBackend::Onnx => {
            // TODO: Implement ONNX classifier
            Err(anyhow!("ONNX classifier not yet implemented"))
        }

        ClassifierBackend::Http => {
            // TODO: Implement HTTP classifier
            Err(anyhow!("HTTP classifier not yet implemented"))
        }
    }
}

#[cfg(any(feature = "clf-candle", feature = "candle-classifier"))]
fn parse_device(device_str: &str) -> Result<candle_core::Device> {
    if device_str.starts_with("cuda") {
        #[cfg(feature = "cuda")]
        {
            let device_id = device_str
                .strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Ok(candle_core::Device::new_cuda(device_id)?)
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!("CUDA requested but not available, using CPU");
            Ok(candle_core::Device::Cpu)
        }
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mock_classifier() {
        let config = ClassifierConfig {
            backend: ClassifierBackend::Mock,
            path: None,
            vocab: None,
            threshold: 0.6,
            endpoint: None,
            device: "cpu".to_string(),
            model_id: None,
            max_length: 256,
        };

        let classifier = create_classifier(&config).unwrap();
        assert_eq!(classifier.name(), "mock");
    }
}

