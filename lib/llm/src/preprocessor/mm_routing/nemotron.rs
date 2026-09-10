// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM-compatible image and video prompt accounting for Nemotron 3 Nano Omni.

use std::{path::Path, sync::Arc};

use anyhow::{Result, anyhow, bail, ensure};
use llm_multimodal::vision::PreProcessorConfig;

use super::{
    NemotronVideoProcessorContract, VideoRoutingInput, VideoRoutingReplacement,
    config::read_model_config,
};
use crate::{protocols::TokenIdType, tokenizers::traits::Tokenizer};

pub(in crate::preprocessor) const MODEL_TYPE: &str = "NemotronH_Nano_Omni_Reasoning_V3";
pub(in crate::preprocessor) const IMAGE_START: &str = "<img>";
pub(in crate::preprocessor) const IMAGE_END: &str = "</img>";
pub(in crate::preprocessor) const IMAGE_CONTEXT: &str = "<image>";
const VIDEO_CONTEXT: &str = "<video>";
// Transformers' multimodal chat normalization emits each media marker followed
// by a newline. `<video>` is not an atomic Nemotron token, so that boundary
// changes its final token ID and must be part of the replacement target.
const VIDEO_CONTEXT_WITH_SEPARATOR: &str = "<video>\n";
const VIDEO_TRAILING_SEPARATOR: &str = "\n";

/// Routing-only implementation of vLLM's `DynamicResolutionImageTiler`.
///
/// It performs geometry and token accounting only. Pixel resize and
/// normalization remain worker-owned.
pub(in crate::preprocessor) struct NemotronImageTokenCounter {
    patch_size: usize,
    min_num_patches: usize,
    max_num_patches: usize,
}

impl NemotronImageTokenCounter {
    pub(in crate::preprocessor) fn try_from_configs(
        processor_config: &PreProcessorConfig,
        model_config: &serde_json::Value,
    ) -> Result<Self> {
        ensure!(
            is_nemotron_config(model_config),
            "unsupported Nemotron model architecture"
        );

        let image_size = json_usize(model_config, &["force_image_size"])
            .ok_or_else(|| anyhow!("Nemotron Nano Omni force_image_size is missing"))?;
        let patch_size = json_usize(model_config, &["patch_size"])
            .ok_or_else(|| anyhow!("Nemotron Nano Omni patch_size is missing"))?;
        let downsample_ratio = model_config
            .get("downsample_ratio")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("Nemotron Nano Omni downsample_ratio is missing"))?;
        let min_num_patches =
            json_usize(model_config, &["vision_config", "args", "min_num_patches"])
                .ok_or_else(|| anyhow!("Nemotron Nano Omni min_num_patches is missing"))?;
        let max_num_patches =
            json_usize(model_config, &["vision_config", "args", "max_num_patches"])
                .ok_or_else(|| anyhow!("Nemotron Nano Omni max_num_patches is missing"))?;

        ensure!(image_size > 0, "force_image_size must be positive");
        ensure!(patch_size > 0, "patch_size must be positive");
        ensure!(
            image_size % patch_size == 0,
            "force_image_size must be divisible by patch_size"
        );
        // vLLM's Nano Nemotron dynamic tiler asserts one 2x pixel-shuffle
        // reduction. Supporting another value here would silently produce a
        // different model-visible sequence.
        ensure!(
            (downsample_ratio - 0.5).abs() < f64::EPSILON,
            "downsample_ratio must be exactly 0.5"
        );
        ensure!(min_num_patches > 0, "min_num_patches must be positive");
        ensure!(
            max_num_patches >= min_num_patches,
            "max_num_patches must be at least min_num_patches"
        );

        // These duplicate values are not used by vLLM's dynamic path, but a
        // disagreement is a useful indication that the checkpoint's processor
        // contract changed. Fail closed rather than route approximately.
        if processor_config.patch_size.is_some() {
            ensure!(
                processor_config.get_patch_size(0) == patch_size,
                "preprocessor and model patch_size values disagree"
            );
        }
        if let Some(configured_image_size) = processor_config.get_extra::<usize>("image_size") {
            ensure!(
                configured_image_size == image_size,
                "preprocessor and model image_size values disagree"
            );
        }
        if let Some(configured_ratio) = processor_config.get_extra::<f64>("downsample_ratio") {
            ensure!(
                (configured_ratio - downsample_ratio).abs() < f64::EPSILON,
                "preprocessor and model downsample_ratio values disagree"
            );
        }

        Ok(Self {
            patch_size,
            min_num_patches,
            max_num_patches,
        })
    }

    /// Count one image without a request context constraint. This supports
    /// per-image diagnostics; exact routing and aggregate metrics use
    /// [`Self::count_tokens_for_images`].
    pub(in crate::preprocessor) fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.process_image(width, height, self.max_num_patches)
            .map(|(_, embeddings)| embeddings)
            .unwrap_or(0)
    }

    /// Match vLLM's request-wide dynamic-resolution budget.
    pub(in crate::preprocessor) fn count_tokens_for_images(
        &self,
        dimensions: &[(u32, u32)],
        max_model_len: usize,
        text_prompt_len: usize,
    ) -> Result<Vec<usize>> {
        if dimensions.is_empty() {
            return Ok(Vec::new());
        }

        let post_shuffle_budget = max_model_len
            .checked_sub(text_prompt_len)
            .and_then(|remaining| remaining.checked_sub(4))
            .ok_or_else(|| anyhow!("text prompt leaves no room for Nemotron image tokens"))?;
        let mut total_patch_budget = post_shuffle_budget
            .checked_mul(4)
            .ok_or_else(|| anyhow!("Nemotron image patch budget overflowed"))?;
        let minimum_batch_budget = self
            .min_num_patches
            .checked_mul(dimensions.len())
            .ok_or_else(|| anyhow!("Nemotron minimum image patch budget overflowed"))?;
        total_patch_budget = total_patch_budget.max(minimum_batch_budget);

        let initial_budget = total_patch_budget.clamp(self.min_num_patches, self.max_num_patches);
        let mut per_image_budgets = vec![initial_budget; dimensions.len()];

        for _ in 0..10 {
            let processed: Result<Vec<(usize, usize)>> = dimensions
                .iter()
                .zip(&per_image_budgets)
                .map(|(&(width, height), &budget)| self.process_image(width, height, budget))
                .collect();
            let processed = processed?;
            let total_patches = processed.iter().try_fold(0usize, |sum, (patches, _)| {
                sum.checked_add(*patches)
                    .ok_or_else(|| anyhow!("Nemotron image patch count overflowed"))
            })?;
            if total_patches <= total_patch_budget {
                return Ok(processed
                    .into_iter()
                    .map(|(_, embeddings)| embeddings)
                    .collect());
            }

            let scale = total_patch_budget as f64 / total_patches as f64;
            let scaled: Vec<usize> = processed
                .iter()
                .map(|(patches, _)| self.min_num_patches.max((*patches as f64 * scale) as usize))
                .collect();
            let scaled_down = scaled
                .iter()
                .zip(&per_image_budgets)
                .any(|(scaled, previous)| scaled < previous);
            per_image_budgets = if scaled_down {
                scaled
            } else {
                vec![self.min_num_patches; dimensions.len()]
            };
        }

        bail!("Nemotron dynamic image budgeting did not converge")
    }

    /// Return `(pre-shuffle patches, post-shuffle image tokens)`.
    fn process_image(
        &self,
        width: u32,
        height: u32,
        patch_budget: usize,
    ) -> Result<(usize, usize)> {
        ensure!(width > 0 && height > 0, "image dimensions must be positive");
        ensure!(patch_budget > 0, "image patch budget must be positive");

        // Python's round() is ties-to-even. vLLM deliberately adds 0.5 first.
        let closest_patch_height =
            (f64::from(height) / self.patch_size as f64 + 0.5).round_ties_even() as usize;
        let closest_patch_width =
            (f64::from(width) / self.patch_size as f64 + 0.5).round_ties_even() as usize;
        ensure!(
            closest_patch_height > 0 && closest_patch_width > 0,
            "image dimensions produced an empty patch grid"
        );
        let patches = closest_patch_height
            .checked_mul(closest_patch_width)
            .ok_or_else(|| anyhow!("image patch grid overflowed"))?;
        let factor = (patch_budget as f64 / patches as f64).sqrt().min(1.0);
        let mut target_height = (factor * closest_patch_height as f64).floor() as usize;
        let mut target_width = (factor * closest_patch_width as f64).floor() as usize;
        ensure!(
            target_height > 0 && target_width > 0,
            "image aspect ratio produced an empty target patch grid"
        );

        let target_patches = target_height
            .checked_mul(target_width)
            .ok_or_else(|| anyhow!("target image patch grid overflowed"))?;
        if patch_budget > self.min_num_patches && target_patches < self.min_num_patches {
            let up_factor = (self.min_num_patches as f64 / target_patches as f64).sqrt();
            target_height = (up_factor * target_height as f64).ceil() as usize;
            target_width = (up_factor * target_width as f64).ceil() as usize;
        }

        // Nano Nemotron applies one 2x pixel-shuffle reduction.
        round_patch_dimension(&mut target_height, target_width, patch_budget, 2);
        round_patch_dimension(&mut target_width, target_height, patch_budget, 2);

        let raw_patches = target_height
            .checked_mul(target_width)
            .ok_or_else(|| anyhow!("rounded image patch grid overflowed"))?;
        Ok((raw_patches, raw_patches / 4))
    }
}

/// Routing-only implementation of vLLM's Nemotron video prompt expansion.
///
/// Pixel resize and EVS embedding selection remain worker-owned. This adapter
/// reproduces only the model-visible placeholder sequence used for KV hashing.
pub(super) struct NemotronVideoRoutingSpec {
    patch_size: usize,
    video_target_num_patches: usize,
    video_maintain_aspect_ratio: bool,
    video_temporal_patch_size: usize,
    video_pruning_rate: f64,
    image_start_token_id: TokenIdType,
    image_end_token_id: TokenIdType,
    image_context_token_id: TokenIdType,
    video_target_tokens: Vec<TokenIdType>,
    video_trailing_tokens: Vec<TokenIdType>,
    tokenizer: Arc<dyn Tokenizer>,
}

impl NemotronVideoRoutingSpec {
    pub(super) fn from_model_dir(
        model_id: &str,
        expected_model_type: &str,
        model_dir: &Path,
        tokenizer: Arc<dyn Tokenizer>,
        processor_contract: NemotronVideoProcessorContract,
    ) -> Result<Self> {
        let config = read_model_config(
            model_id,
            expected_model_type,
            MODEL_TYPE,
            "Nemotron",
            model_dir,
        )?;
        let vision_config = config
            .get("vision_config")
            .ok_or_else(|| anyhow!("mm-routing: Nemotron vision_config is missing"))?;

        let patch_size = json_usize(&config, &["patch_size"])
            .ok_or_else(|| anyhow!("mm-routing: Nemotron patch_size is missing"))?;
        let downsample_ratio = config
            .get("downsample_ratio")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("mm-routing: Nemotron downsample_ratio is missing"))?;
        let video_target_num_patches =
            json_usize(&config, &["vision_config", "video_target_num_patches"]).ok_or_else(
                || anyhow!("mm-routing: Nemotron video_target_num_patches is missing"),
            )?;
        let video_maintain_aspect_ratio = vision_config
            .get("video_maintain_aspect_ratio")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                anyhow!("mm-routing: Nemotron video_maintain_aspect_ratio is missing")
            })?;
        let video_temporal_patch_size =
            json_usize(&config, &["vision_config", "video_temporal_patch_size"]).ok_or_else(
                || anyhow!("mm-routing: Nemotron video_temporal_patch_size is missing"),
            )?;

        ensure!(
            patch_size > 0,
            "mm-routing: Nemotron patch_size must be positive"
        );
        ensure!(
            (downsample_ratio - 0.5).abs() < f64::EPSILON,
            "mm-routing: Nemotron video routing requires downsample_ratio=0.5"
        );
        ensure!(
            video_target_num_patches > 0,
            "mm-routing: Nemotron video_target_num_patches must be positive"
        );
        ensure!(
            video_temporal_patch_size > 0,
            "mm-routing: Nemotron video_temporal_patch_size must be positive"
        );
        ensure!(
            vision_config
                .get("video_target_img_size")
                .is_none_or(serde_json::Value::is_null),
            "mm-routing: Nemotron video_target_img_size is unsupported"
        );
        ensure!(
            processor_contract.video_pruning_rate.is_finite()
                && (0.0..1.0).contains(&processor_contract.video_pruning_rate),
            "mm-routing: Nemotron video_pruning_rate must be in [0, 1)"
        );

        ensure_config_string(&config, "img_start_token", IMAGE_START)?;
        ensure_config_string(&config, "img_end_token", IMAGE_END)?;
        ensure_config_string(&config, "img_context_token", IMAGE_CONTEXT)?;
        ensure_config_string(&config, "video_context_token", VIDEO_CONTEXT)?;

        let expected_image_context_id = image_context_token_id(&config)?;
        let image_start_token_id = atomic_token_id(tokenizer.as_ref(), IMAGE_START, "image-start")?;
        let image_end_token_id = atomic_token_id(tokenizer.as_ref(), IMAGE_END, "image-end")?;
        let resolved_image_context_id =
            atomic_token_id(tokenizer.as_ref(), IMAGE_CONTEXT, "image-context")?;
        ensure!(
            resolved_image_context_id == expected_image_context_id,
            "mm-routing: Nemotron tokenizer image-context id {resolved_image_context_id} does not match config id {expected_image_context_id}"
        );
        let video_target_tokens = tokenizer
            .encode(VIDEO_CONTEXT_WITH_SEPARATOR)?
            .token_ids()
            .to_vec();
        ensure!(
            !video_target_tokens.is_empty(),
            "mm-routing: Nemotron video target tokenized to an empty sequence"
        );
        let video_trailing_tokens = tokenizer
            .encode(VIDEO_TRAILING_SEPARATOR)?
            .token_ids()
            .to_vec();
        ensure!(
            !video_trailing_tokens.is_empty(),
            "mm-routing: Nemotron video separator tokenized to an empty sequence"
        );

        Ok(Self {
            patch_size,
            video_target_num_patches,
            video_maintain_aspect_ratio,
            video_temporal_patch_size,
            video_pruning_rate: processor_contract.video_pruning_rate,
            image_start_token_id,
            image_end_token_id,
            image_context_token_id: resolved_image_context_id,
            video_target_tokens,
            video_trailing_tokens,
            tokenizer,
        })
    }

    pub(super) fn build_replacement(
        &self,
        input: &VideoRoutingInput<'_>,
    ) -> Result<VideoRoutingReplacement> {
        self.validate_input(input)?;
        let tokens_per_tubelet = self.tokens_per_tubelet(input)?;
        let frame_duration_ms = (1000.0 / input.source_fps) as usize;
        let frame_indices = input
            .sampled_timestamps
            .iter()
            .map(|timestamp| {
                let index = (timestamp * input.source_fps).round_ties_even();
                ensure!(
                    index.is_finite() && index >= 0.0 && index <= usize::MAX as f64,
                    "mm-routing: Nemotron sampled frame index is out of range"
                );
                Ok(index as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        let context_tokens = tokens_per_tubelet.iter().try_fold(0usize, |sum, count| {
            sum.checked_add(*count)
                .ok_or_else(|| anyhow!("mm-routing: Nemotron video token count overflow"))
        })?;
        let mut replacement_tokens = Vec::with_capacity(
            context_tokens
                .checked_add(tokens_per_tubelet.len().saturating_mul(32))
                .ok_or_else(|| anyhow!("mm-routing: Nemotron replacement capacity overflow"))?,
        );
        for (group_index, (frame_group, &num_tokens)) in frame_indices
            .chunks(self.video_temporal_patch_size)
            .zip(&tokens_per_tubelet)
            .enumerate()
        {
            let separator = frame_separator(
                frame_group,
                group_index,
                self.video_temporal_patch_size,
                frame_duration_ms,
            );
            let encoded = self.tokenizer.encode(&separator).map_err(|error| {
                anyhow!(
                    "mm-routing: failed to tokenize Nemotron video separator {separator:?}: {error}"
                )
            })?;
            replacement_tokens.extend_from_slice(encoded.token_ids());
            replacement_tokens.push(self.image_start_token_id);
            replacement_tokens.extend(std::iter::repeat_n(self.image_context_token_id, num_tokens));
            replacement_tokens.push(self.image_end_token_id);
        }
        replacement_tokens.extend_from_slice(&self.video_trailing_tokens);

        Ok(VideoRoutingReplacement {
            placeholder_token_id: self.image_context_token_id,
            // vLLM expands both Nemotron images and videos with <image>. Keep
            // the existing image-run event normalizer for mixed-version safety.
            event_video_token_id: None,
            target_tokens: self.video_target_tokens.clone(),
            replacement_tokens,
        })
    }

    fn validate_input(&self, input: &VideoRoutingInput<'_>) -> Result<()> {
        ensure!(input.frame_count > 0, "mm-routing: Nemotron video is empty");
        ensure!(
            input.width > 0 && input.height > 0,
            "mm-routing: Nemotron video dimensions must be positive"
        );
        ensure!(
            input.sampled_timestamps.len() == input.frame_count,
            "mm-routing: Nemotron sampled timestamp count does not match frame count"
        );
        ensure!(
            input.source_fps.is_finite() && input.source_fps > 0.0,
            "mm-routing: Nemotron source fps must be finite and positive"
        );
        ensure!(
            input
                .sampled_timestamps
                .iter()
                .all(|timestamp| timestamp.is_finite() && *timestamp >= 0.0),
            "mm-routing: Nemotron sampled timestamps must be finite and non-negative"
        );
        ensure!(
            input
                .sampled_timestamps
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "mm-routing: Nemotron sampled timestamps must be non-decreasing"
        );
        Ok(())
    }

    fn tokens_per_tubelet(&self, input: &VideoRoutingInput<'_>) -> Result<Vec<usize>> {
        let (patch_width, patch_height) = self.target_patch_grid(input.width, input.height)?;
        let feature_size = (patch_height / 2)
            .checked_mul(patch_width / 2)
            .ok_or_else(|| anyhow!("mm-routing: Nemotron video feature size overflow"))?;
        let tubelets = input.frame_count.div_ceil(self.video_temporal_patch_size);
        if self.video_pruning_rate > 0.0 {
            let total = feature_size
                .checked_mul(tubelets)
                .ok_or_else(|| anyhow!("mm-routing: Nemotron video token count overflow"))?;
            let retained = ((total as f64) * (1.0 - self.video_pruning_rate)) as usize;
            let mut counts = vec![0; tubelets];
            counts[0] = feature_size.max(retained);
            Ok(counts)
        } else {
            Ok(vec![feature_size; tubelets])
        }
    }

    /// Match vLLM's `_compute_aspect_preserving_size` in patch-grid space.
    fn target_patch_grid(&self, width: u32, height: u32) -> Result<(usize, usize)> {
        let target = self.video_target_num_patches;
        let (mut patch_width, mut patch_height) = if self.video_maintain_aspect_ratio {
            let aspect = f64::from(width) / f64::from(height.max(1));
            let patch_height = ((target as f64 / aspect).sqrt()).round_ties_even() as usize;
            let patch_width = ((target as f64 * aspect).sqrt()).round_ties_even() as usize;
            (patch_width.max(1), patch_height.max(1))
        } else {
            let side = (target as f64).sqrt() as usize;
            let side = 2.max(side / 2 * 2);
            (side, side)
        };

        if self.video_maintain_aspect_ratio {
            let up = |value: usize| value + ((2 - value % 2) % 2);
            let down = |value: usize| value - value % 2;
            let width_up = up(patch_width);
            let height_up = up(patch_height);
            if width_up
                .checked_mul(height_up)
                .is_some_and(|area| area <= target)
            {
                patch_width = width_up;
                patch_height = height_up;
            } else {
                patch_width = 2.max(down(patch_width));
                patch_height = 2.max(down(patch_height));
            }
        }

        // Preserve the pixel-space overflow checks made by the worker path.
        patch_width
            .checked_mul(self.patch_size)
            .and_then(|_| patch_height.checked_mul(self.patch_size))
            .ok_or_else(|| anyhow!("mm-routing: Nemotron video target size overflow"))?;
        Ok((patch_width, patch_height))
    }
}

fn atomic_token_id(tokenizer: &dyn Tokenizer, token: &str, label: &str) -> Result<TokenIdType> {
    let ids = tokenizer.encode(token)?.token_ids().to_vec();
    ensure!(
        ids.len() == 1,
        "mm-routing: Nemotron {label} token {token:?} encoded to {} ids ({ids:?})",
        ids.len()
    );
    Ok(ids[0])
}

fn frame_separator(
    frame_indices: &[usize],
    group_index: usize,
    temporal_patch_size: usize,
    frame_duration_ms: usize,
) -> String {
    let mut parts = Vec::with_capacity(frame_indices.len());
    for (index_in_group, frame_index) in frame_indices.iter().enumerate() {
        let label = if index_in_group == 0 {
            "Frame"
        } else {
            "frame"
        };
        let ordinal = group_index
            .saturating_mul(temporal_patch_size)
            .saturating_add(index_in_group)
            .saturating_add(1);
        let timestamp = *frame_index as f64 * frame_duration_ms as f64 / 1000.0;
        parts.push(format!(
            "{label} {ordinal} sampled at {timestamp:.2} seconds"
        ));
    }
    let prefix = if group_index == 0 { "" } else { "\n" };
    format!("{prefix}{}: ", parts.join(" and "))
}

pub(in crate::preprocessor) fn supports_model_type(model_type: Option<&str>) -> bool {
    model_type.is_some_and(|value| value.eq_ignore_ascii_case(MODEL_TYPE))
}

pub(in crate::preprocessor) fn is_nemotron_config(config: &serde_json::Value) -> bool {
    supports_model_type(config.get("model_type").and_then(serde_json::Value::as_str))
        || config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures.iter().any(|architecture| {
                    architecture
                        .as_str()
                        .is_some_and(|value| value.eq_ignore_ascii_case(MODEL_TYPE))
                })
            })
}

/// Validate the static prompt replacement contract used by vLLM.
pub(in crate::preprocessor) fn image_context_token_id(config: &serde_json::Value) -> Result<u32> {
    ensure!(
        is_nemotron_config(config),
        "not a Nemotron Nano Omni config"
    );
    ensure_config_string(config, "img_start_token", IMAGE_START)?;
    ensure_config_string(config, "img_end_token", IMAGE_END)?;
    ensure_config_string(config, "img_context_token", IMAGE_CONTEXT)?;
    config
        .get("img_context_token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| anyhow!("Nemotron img_context_token_id is missing or invalid"))
}

fn ensure_config_string(config: &serde_json::Value, field: &str, expected: &str) -> Result<()> {
    let actual = config
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("Nemotron {field} is missing"))?;
    ensure!(
        actual == expected,
        "Nemotron {field}={actual:?} does not match vLLM's {expected:?}"
    );
    Ok(())
}

fn round_patch_dimension(
    dimension: &mut usize,
    other_dimension: usize,
    patch_budget: usize,
    divisor: usize,
) {
    let remainder = *dimension % divisor;
    if remainder == 0 {
        return;
    }
    let increase = divisor - remainder;
    if dimension
        .checked_add(increase)
        .and_then(|value| value.checked_mul(other_dimension))
        .is_some_and(|patches| patches <= patch_budget)
    {
        *dimension += increase;
    } else {
        *dimension = divisor.max(*dimension - remainder);
    }
}

fn json_usize(config: &serde_json::Value, path: &[&str]) -> Option<usize> {
    path.iter()
        .try_fold(config, |value, key| value.get(*key))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use crate::tokenizers::{Encoding, traits::DecodeResult};

    use super::*;

    struct VideoTokenizer;

    impl crate::tokenizers::traits::Encoder for VideoTokenizer {
        fn encode(&self, input: &str) -> anyhow::Result<Encoding> {
            let ids = match input {
                VIDEO_CONTEXT_WITH_SEPARATOR => vec![1060, 24073, 1561],
                VIDEO_TRAILING_SEPARATOR => vec![102],
                IMAGE_START => vec![19],
                IMAGE_END => vec![20],
                IMAGE_CONTEXT => vec![18],
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.03 seconds: " => {
                    vec![100]
                }
                "\nFrame 3 sampled at 0.07 seconds: " => vec![101],
                _ => anyhow::bail!("unexpected tokenization input {input:?}"),
            };
            Ok(Encoding::Sp(ids))
        }

        fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
            inputs.iter().map(|input| self.encode(input)).collect()
        }
    }

    impl crate::tokenizers::traits::Decoder for VideoTokenizer {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<DecodeResult> {
            Ok(DecodeResult::Complete(String::new()))
        }
    }

    impl Tokenizer for VideoTokenizer {}

    fn video_spec(pruning_rate: f64) -> NemotronVideoRoutingSpec {
        NemotronVideoRoutingSpec {
            patch_size: 16,
            video_target_num_patches: 1024,
            video_maintain_aspect_ratio: true,
            video_temporal_patch_size: 2,
            video_pruning_rate: pruning_rate,
            image_start_token_id: 19,
            image_end_token_id: 20,
            image_context_token_id: 18,
            video_target_tokens: vec![1060, 24073, 1561],
            video_trailing_tokens: vec![102],
            tokenizer: Arc::new(VideoTokenizer),
        }
    }

    fn model_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": [MODEL_TYPE],
            "model_type": MODEL_TYPE,
            "force_image_size": 512,
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "img_context_token": IMAGE_CONTEXT,
            "img_context_token_id": 18,
            "img_start_token": IMAGE_START,
            "img_end_token": IMAGE_END,
            "video_context_token": VIDEO_CONTEXT,
            "vision_config": {
                "video_target_num_patches": 1024,
                "video_maintain_aspect_ratio": true,
                "video_temporal_patch_size": 2,
                "args": {
                    "min_num_patches": 1024,
                    "max_num_patches": 13312
                }
            }
        })
    }

    fn processor_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(
            &serde_json::json!({
                "image_processor_type": "NemotronH_Nano_Omni_Reasoning_V3ImageProcessor",
                "image_size": 512,
                "patch_size": 16,
                "downsample_ratio": 0.5
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn counts_match_vllm_0_28_dynamic_resolution_goldens() {
        let counter =
            NemotronImageTokenCounter::try_from_configs(&processor_config(), &model_config())
                .unwrap();

        for (dimensions, expected) in [
            ((224, 224), 256),
            ((64, 32), 276),
            ((512, 512), 256),
            ((1000, 500), 512),
            ((500, 1000), 512),
            ((1024, 1024), 1024),
            ((1536, 1536), 2304),
            ((1920, 1080), 2040),
        ] {
            assert_eq!(counter.count_tokens(dimensions.0, dimensions.1), expected);
        }

        assert_eq!(
            counter
                .count_tokens_for_images(&[(1920, 1080); 3], 4096, 10)
                .unwrap(),
            vec![1344, 1344, 1344]
        );
    }

    #[test]
    fn rejects_processor_contract_drift() {
        let mut config = model_config();
        config["downsample_ratio"] = serde_json::json!(0.25);
        assert!(NemotronImageTokenCounter::try_from_configs(&processor_config(), &config).is_err());

        let mut config = model_config();
        config["img_end_token"] = serde_json::json!("<different>");
        assert!(image_context_token_id(&config).is_err());
    }

    #[test]
    fn video_geometry_matches_vllm_0_28_goldens() {
        let spec = video_spec(0.0);
        assert_eq!(spec.target_patch_grid(512, 512).unwrap(), (32, 32));
        assert_eq!(spec.target_patch_grid(640, 360).unwrap(), (42, 24));
        assert_eq!(spec.target_patch_grid(3760, 1120).unwrap(), (58, 16));
    }

    #[test]
    fn video_replacement_matches_vllm_0_28_evs_layout() {
        let input = VideoRoutingInput {
            frame_count: 3,
            width: 512,
            height: 512,
            source_fps: 30.0,
            sampled_timestamps: &[0.0, 1.0 / 30.0, 2.0 / 30.0],
        };
        let replacement = video_spec(0.5).build_replacement(&input).unwrap();

        assert_eq!(replacement.placeholder_token_id, 18);
        assert_eq!(replacement.event_video_token_id, None);
        assert_eq!(replacement.target_tokens, [1060, 24073, 1561]);
        assert_eq!(replacement.replacement_tokens.len(), 263);
        assert_eq!(replacement.replacement_tokens[0], 100);
        assert_eq!(replacement.replacement_tokens[1], 19);
        assert!(
            replacement.replacement_tokens[2..258]
                .iter()
                .all(|id| *id == 18)
        );
        assert_eq!(
            &replacement.replacement_tokens[258..],
            &[20, 101, 19, 20, 102]
        );
    }

    #[test]
    fn video_replacement_without_pruning_keeps_each_tubelet() {
        let input = VideoRoutingInput {
            frame_count: 3,
            width: 512,
            height: 512,
            source_fps: 30.0,
            sampled_timestamps: &[0.0, 1.0 / 30.0, 2.0 / 30.0],
        };
        let replacement = video_spec(0.0).build_replacement(&input).unwrap();

        assert_eq!(replacement.replacement_tokens.len(), 519);
        assert_eq!(
            replacement
                .replacement_tokens
                .iter()
                .filter(|id| **id == 18)
                .count(),
            512
        );
    }

    #[test]
    fn loads_video_contract_from_checkpoint_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), model_config().to_string()).unwrap();

        let spec = NemotronVideoRoutingSpec::from_model_dir(
            "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            MODEL_TYPE,
            dir.path(),
            Arc::new(VideoTokenizer),
            NemotronVideoProcessorContract {
                video_pruning_rate: 0.5,
            },
        )
        .unwrap();

        assert_eq!(spec.video_target_num_patches, 1024);
        assert_eq!(spec.video_temporal_patch_size, 2);
        assert_eq!(spec.video_target_tokens, [1060, 24073, 1561]);
    }

    #[test]
    fn rejects_unsupported_video_contract() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = model_config();
        config["vision_config"]["video_target_img_size"] = serde_json::json!(512);
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        assert!(
            NemotronVideoRoutingSpec::from_model_dir(
                "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
                MODEL_TYPE,
                dir.path(),
                Arc::new(VideoTokenizer),
                NemotronVideoProcessorContract {
                    video_pruning_rate: 0.5,
                },
            )
            .is_err()
        );
    }
}
