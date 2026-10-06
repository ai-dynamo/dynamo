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

/// How the worker hashes a block that intersects a video expansion but has no
/// video placeholder run. vLLM carries MM metadata in its KV event for these
/// boundary blocks; SGLang emits only the block's token IDs.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QwenVideoRunlessBoundaryHash {
    MmMetadata,
    TokensOnly,
}

/// Engine-independent Qwen video prompt-expansion behavior used internally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QwenVideoProcessorContract {
    pub placeholder_target: QwenVideoPlaceholderTarget,
    pub resize_mode: QwenVideoResizeMode,
    pub runless_boundary_hash: QwenVideoRunlessBoundaryHash,
    pub sglang_preprocess: Option<SglangQwenVideoPreprocessContract>,
}

/// Qwen video contract accepted only under the vLLM runtime key.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub(crate) struct VllmQwenVideoProcessorContract {
    pub placeholder_target: QwenVideoPlaceholderTarget,
    pub resize_mode: QwenVideoResizeMode,
    #[serde(default)]
    runless_boundary_hash: Option<QwenVideoRunlessBoundaryHash>,
    #[serde(default)]
    sglang_preprocess: Option<SglangQwenVideoPreprocessContract>,
}

impl TryFrom<VllmQwenVideoProcessorContract> for QwenVideoProcessorContract {
    type Error = anyhow::Error;

    fn try_from(contract: VllmQwenVideoProcessorContract) -> Result<Self> {
        anyhow::ensure!(
            contract.sglang_preprocess.is_none(),
            "mm-routing: vLLM Qwen contract must not publish SGLang preprocessing"
        );
        anyhow::ensure!(
            contract
                .runless_boundary_hash
                .unwrap_or(QwenVideoRunlessBoundaryHash::MmMetadata)
                == QwenVideoRunlessBoundaryHash::MmMetadata,
            "mm-routing: vLLM Qwen contract must use MM metadata KV-event identity"
        );
        Ok(Self {
            placeholder_target: contract.placeholder_target,
            resize_mode: contract.resize_mode,
            runless_boundary_hash: QwenVideoRunlessBoundaryHash::MmMetadata,
            sglang_preprocess: None,
        })
    }
}

/// Qwen video contract accepted only under the SGLang runtime key.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub(crate) struct SglangQwenVideoProcessorContract {
    pub placeholder_target: QwenVideoPlaceholderTarget,
    pub resize_mode: QwenVideoResizeMode,
    pub runless_boundary_hash: QwenVideoRunlessBoundaryHash,
    pub sglang_preprocess: SglangQwenVideoPreprocessContract,
}

impl TryFrom<SglangQwenVideoProcessorContract> for QwenVideoProcessorContract {
    type Error = anyhow::Error;

    fn try_from(contract: SglangQwenVideoProcessorContract) -> Result<Self> {
        anyhow::ensure!(
            contract.runless_boundary_hash == QwenVideoRunlessBoundaryHash::TokensOnly,
            "mm-routing: SGLang Qwen contract must use pad-value token KV-event identity"
        );
        Ok(Self {
            placeholder_target: contract.placeholder_target,
            resize_mode: contract.resize_mode,
            runless_boundary_hash: contract.runless_boundary_hash,
            sglang_preprocess: Some(contract.sglang_preprocess),
        })
    }
}

/// SGLang's Qwen video preprocessing stage before Transformers runs.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub(crate) struct SglangQwenVideoPreprocessContract {
    pub image_factor: usize,
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
    pub video_total_pixels: usize,
    pub frame_factor: usize,
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
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

/// How a worker represents multimodal identity in published KV-event blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvEventMmIdentity {
    /// Ambiguous blocks retain worker token IDs and carry media hashes as
    /// separate block metadata.
    MmMetadata,
    /// Media placeholder runs are replaced with hash-derived pad values before
    /// the worker publishes the block; no separate MM metadata is attached.
    PadValueTokens,
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

    pub(crate) fn kv_event_mm_identity(&self) -> KvEventMmIdentity {
        match &self.model {
            SupportedVideoModel::Qwen3(spec) => spec.kv_event_mm_identity(),
            SupportedVideoModel::Nemotron(_) => KvEventMmIdentity::MmMetadata,
            #[cfg(test)]
            SupportedVideoModel::TestStub => KvEventMmIdentity::MmMetadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUBLISHED_SGLANG_QWEN_CONTRACT: &str = r#"{
        "placeholder_target": "bare_video_token",
        "resize_mode": "legacy_ceil",
        "runless_boundary_hash": "tokens_only",
        "sglang_preprocess": {
            "image_factor": 28,
            "video_min_pixels": 100352,
            "video_max_pixels": 602112,
            "video_total_pixels": 90316800,
            "frame_factor": 2,
            "fps": 2.0,
            "min_frames": 4,
            "max_frames": 768
        }
    }"#;

    #[test]
    fn parses_published_sglang_qwen_contract() {
        let published: SglangQwenVideoProcessorContract =
            serde_json::from_str(PUBLISHED_SGLANG_QWEN_CONTRACT).unwrap();
        let contract = QwenVideoProcessorContract::try_from(published).unwrap();

        assert_eq!(
            contract,
            QwenVideoProcessorContract {
                placeholder_target: QwenVideoPlaceholderTarget::BareVideoToken,
                resize_mode: QwenVideoResizeMode::LegacyCeil,
                runless_boundary_hash: QwenVideoRunlessBoundaryHash::TokensOnly,
                sglang_preprocess: Some(SglangQwenVideoPreprocessContract {
                    image_factor: 28,
                    video_min_pixels: 100_352,
                    video_max_pixels: 602_112,
                    video_total_pixels: 90_316_800,
                    frame_factor: 2,
                    fps: 2.0,
                    min_frames: 4,
                    max_frames: 768,
                }),
            }
        );
    }

    #[test]
    fn rejects_incomplete_sglang_qwen_contract() {
        let missing_preprocess = r#"{
            "placeholder_target": "bare_video_token",
            "resize_mode": "legacy_ceil",
            "runless_boundary_hash": "tokens_only"
        }"#;
        let missing_identity = r#"{
            "placeholder_target": "bare_video_token",
            "resize_mode": "legacy_ceil",
            "sglang_preprocess": {
                "image_factor": 28,
                "video_min_pixels": 100352,
                "video_max_pixels": 602112,
                "video_total_pixels": 90316800,
                "frame_factor": 2,
                "fps": 2.0,
                "min_frames": 4,
                "max_frames": 768
            }
        }"#;

        assert!(
            serde_json::from_str::<SglangQwenVideoProcessorContract>(missing_preprocess).is_err()
        );
        assert!(
            serde_json::from_str::<SglangQwenVideoProcessorContract>(missing_identity).is_err()
        );
    }

    #[test]
    fn rejects_cross_engine_qwen_contract_semantics() {
        let vllm_with_sglang_preprocess: VllmQwenVideoProcessorContract =
            serde_json::from_str(PUBLISHED_SGLANG_QWEN_CONTRACT).unwrap();
        assert!(QwenVideoProcessorContract::try_from(vllm_with_sglang_preprocess).is_err());

        let sglang_with_vllm_identity =
            PUBLISHED_SGLANG_QWEN_CONTRACT.replace("\"tokens_only\"", "\"mm_metadata\"");
        let sglang_with_vllm_identity: SglangQwenVideoProcessorContract =
            serde_json::from_str(&sglang_with_vllm_identity).unwrap();
        assert!(QwenVideoProcessorContract::try_from(sglang_with_vllm_identity).is_err());
    }
}
