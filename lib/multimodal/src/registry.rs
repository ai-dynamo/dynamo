// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry boundary for model-specific processors.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::processed::ProcessedMedia;
use crate::types::RgbFrameRef;
use crate::vision::TransformError;
use crate::{Qwen3VlVideoConfig, Qwen3VlVideoPreprocessor, VideoTiming};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "processor", rename_all = "snake_case")]
pub enum VideoProcessorConfig {
    Qwen3Vl { config: Qwen3VlVideoConfig },
}

impl VideoProcessorConfig {
    pub fn from_hf(
        model_type: &str,
        preprocessor_config_json: &str,
    ) -> Result<Self, TransformError> {
        match model_type {
            "qwen3_vl" => Ok(Self::Qwen3Vl {
                config: Qwen3VlVideoConfig::from_preprocessor_json(preprocessor_config_json)?,
            }),
            _ => Err(TransformError::UnsupportedModelType(model_type.to_string())),
        }
    }
}

pub trait VideoProcessor: Send + Sync {
    fn process(
        &self,
        frames: &[RgbFrameRef<'_>],
        timing: &VideoTiming,
    ) -> Result<ProcessedMedia, TransformError>;
}

pub struct VideoProcessorRegistry;

impl VideoProcessorRegistry {
    pub fn build(config: VideoProcessorConfig) -> Result<Arc<dyn VideoProcessor>, TransformError> {
        match config {
            VideoProcessorConfig::Qwen3Vl { config } => {
                Ok(Arc::new(Qwen3VlVideoPreprocessor::new(config)?))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_hf_model_type_to_processor_config() {
        let config = VideoProcessorConfig::from_hf("qwen3_vl", "{}").unwrap();
        assert!(matches!(config, VideoProcessorConfig::Qwen3Vl { .. }));
    }

    #[test]
    fn rejects_unsupported_hf_model_type() {
        let error = VideoProcessorConfig::from_hf("other", "{}").unwrap_err();
        assert!(error.to_string().contains("model type: other"));
    }
}
