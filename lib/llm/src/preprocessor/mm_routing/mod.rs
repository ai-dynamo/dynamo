// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight model-visible video token expansion for MM-aware routing.

// The facade is instantiated only when FFmpeg-backed frontend video decoding
// is enabled. Its model-specific unit tests remain available without FFmpeg.
#![cfg_attr(not(feature = "media-ffmpeg"), allow(dead_code))]

mod config;
mod qwen3;

use std::{path::Path, sync::Arc};

use anyhow::Result;
use serde::Deserialize;

use crate::{protocols::TokenIdType, tokenizers::traits::Tokenizer};

/// Which token sequence the running vLLM Qwen3 processor replaces for video.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QwenVideoPlaceholderTarget {
    BareVideoToken,
    VisionWrappedVideoToken,
}

/// Temporal rounding used by the running Transformers video processor.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QwenVideoResizeMode {
    LegacyCeil,
    RoundTiesEven,
}

/// Worker-reported Qwen video prompt-expansion behavior.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub(crate) struct QwenVideoProcessorContract {
    pub placeholder_target: QwenVideoPlaceholderTarget,
    pub resize_mode: QwenVideoResizeMode,
}

/// Geometry and temporal metadata visible to a model's video processor.
pub(crate) struct VideoRoutingInput<'a> {
    pub frame_count: usize,
    pub width: u32,
    pub height: u32,
    pub source_fps: f64,
    pub sampled_timestamps: &'a [f64],
}

pub(crate) struct VideoRoutingReplacement {
    pub placeholder_token_id: TokenIdType,
    /// Exact chat-template token sequence replaced by the model processor.
    pub target_tokens: Vec<TokenIdType>,
    pub replacement_tokens: Vec<TokenIdType>,
}

enum SupportedVideoModel {
    Qwen3(qwen3::Qwen3VideoRoutingSpec),
    #[cfg(test)]
    TestStub,
}

pub(crate) struct VideoRoutingProcessor {
    model: SupportedVideoModel,
}

impl VideoRoutingProcessor {
    #[cfg(test)]
    pub(crate) fn test_stub() -> Self {
        Self {
            model: SupportedVideoModel::TestStub,
        }
    }

    pub(crate) fn try_new(
        model_id: &str,
        model_type: &str,
        model_dir: &Path,
        tokenizer: Arc<dyn Tokenizer>,
        qwen_video_contract: QwenVideoProcessorContract,
    ) -> Result<Option<Self>> {
        let model = if qwen3::supports_model_type(model_type) {
            SupportedVideoModel::Qwen3(qwen3::Qwen3VideoRoutingSpec::from_model_dir(
                model_id,
                model_type,
                model_dir,
                tokenizer,
                qwen_video_contract,
            )?)
        } else {
            return Ok(None);
        };

        Ok(Some(Self { model }))
    }

    pub(crate) fn build_replacement(
        &self,
        input: &VideoRoutingInput<'_>,
    ) -> Result<VideoRoutingReplacement> {
        match &self.model {
            SupportedVideoModel::Qwen3(spec) => spec.build_replacement(input),
            #[cfg(test)]
            SupportedVideoModel::TestStub => anyhow::bail!("test video routing processor stub"),
        }
    }
}
