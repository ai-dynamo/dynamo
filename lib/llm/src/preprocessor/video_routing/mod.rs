// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight model-visible video token expansion for MM-aware routing.

mod config;
mod qwen3;

use std::{path::Path, sync::Arc};

use anyhow::Result;

use crate::{protocols::TokenIdType, tokenizers::traits::Tokenizer};

/// Geometry and temporal metadata visible to a model's video processor.
pub(crate) struct VideoRoutingInput<'a> {
    pub frame_count: usize,
    pub width: u32,
    pub height: u32,
    pub source_fps: f64,
    pub sampled_timestamps: &'a [f64],
}

/// Router-side replacement for one video placeholder.
pub(crate) struct VideoRoutingReplacement {
    pub placeholder_token_id: TokenIdType,
    /// Exact chat-template token sequence replaced by the model processor.
    pub target_tokens: Vec<TokenIdType>,
    pub replacement_tokens: Vec<TokenIdType>,
}

enum SupportedVideoModel {
    Qwen3(qwen3::Qwen3VideoRoutingSpec),
}

/// Model-specific video routing facade constructed once at frontend startup.
pub(crate) struct VideoRoutingProcessor {
    model: SupportedVideoModel,
}

impl VideoRoutingProcessor {
    pub(crate) fn try_new(
        model_id: &str,
        model_type: &str,
        model_dir: &Path,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Option<Self>> {
        let model = if qwen3::supports_model_type(model_type) {
            SupportedVideoModel::Qwen3(qwen3::Qwen3VideoRoutingSpec::from_model_dir(
                model_id, model_type, model_dir, tokenizer,
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
        }
    }
}
