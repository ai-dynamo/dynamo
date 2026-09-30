// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight model-visible media token expansion for MM-aware routing.

// The video facade is instantiated only when FFmpeg-backed frontend decoding
// is enabled. Image-only model helpers and unit tests remain available without
// FFmpeg.
#![cfg_attr(not(feature = "media-ffmpeg"), allow(dead_code))]

mod config;
pub mod image;
pub(super) mod nemotron;
mod qwen3;

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
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

/// Worker-reported Nemotron video prompt-expansion behavior.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub(crate) struct NemotronVideoProcessorContract {
    pub video_pruning_rate: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VideoProcessorContracts {
    pub qwen: Option<QwenVideoProcessorContract>,
    pub nemotron: Option<NemotronVideoProcessorContract>,
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
    /// Token ID passed to the worker KV-event normalizer for video runs.
    /// Nemotron uses the image placeholder for both modalities, so it keeps
    /// the worker's existing image-run normalization instead.
    pub event_video_token_id: Option<TokenIdType>,
    /// Exact chat-template token sequence replaced by the model processor.
    pub target_tokens: Vec<TokenIdType>,
    pub replacement_tokens: Vec<TokenIdType>,
}

enum SupportedVideoModel {
    Qwen3(qwen3::Qwen3VideoRoutingSpec),
    Nemotron(nemotron::NemotronVideoRoutingSpec),
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
        contracts: VideoProcessorContracts,
    ) -> Result<Option<Self>> {
        let model = if qwen3::supports_model_type(model_type) {
            SupportedVideoModel::Qwen3(qwen3::Qwen3VideoRoutingSpec::from_model_dir(
                model_id,
                model_type,
                model_dir,
                tokenizer,
                contracts
                    .qwen
                    .context("mm-routing: Qwen video worker contract is missing")?,
            )?)
        } else if nemotron::supports_model_type(Some(model_type)) {
            SupportedVideoModel::Nemotron(nemotron::NemotronVideoRoutingSpec::from_model_dir(
                model_id,
                model_type,
                model_dir,
                tokenizer,
                contracts
                    .nemotron
                    .context("mm-routing: Nemotron video worker contract is missing")?,
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
            SupportedVideoModel::Nemotron(spec) => spec.build_replacement(input),
            #[cfg(test)]
            SupportedVideoModel::TestStub => anyhow::bail!("test video routing processor stub"),
        }
    }
}
