// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM-compatible image token accounting for Nemotron 3 Nano Omni.

use anyhow::{Result, anyhow, bail, ensure};
use llm_multimodal::vision::PreProcessorConfig;

use super::image::{ImagePromptKind, ImageRoutingBackend, ImageRoutingRuntime};

pub(in crate::preprocessor) const MODEL_TYPE: &str = "NemotronH_Nano_Omni_Reasoning_V3";
pub(in crate::preprocessor) const IMAGE_START: &str = "<img>";
pub(in crate::preprocessor) const IMAGE_END: &str = "</img>";
pub(in crate::preprocessor) const IMAGE_CONTEXT: &str = "<image>";

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

impl ImageRoutingBackend for NemotronImageTokenCounter {
    fn count_tokens(&self, width: u32, height: u32) -> usize {
        NemotronImageTokenCounter::count_tokens(self, width, height)
    }

    fn count_tokens_for_images(
        &self,
        dimensions: &[(u32, u32)],
        max_model_len: usize,
        text_prompt_len: usize,
    ) -> Result<Vec<usize>> {
        NemotronImageTokenCounter::count_tokens_for_images(
            self,
            dimensions,
            max_model_len,
            text_prompt_len,
        )
    }

    fn uses_request_context_budget(&self) -> bool {
        true
    }

    fn validate_runtime(&self, runtime: ImageRoutingRuntime) -> Result<()> {
        ensure!(
            runtime == ImageRoutingRuntime::VllmNativeGenerate,
            "Nemotron image routing requires the vLLM native Generate runtime"
        );
        Ok(())
    }

    fn routing_prompt_kind(&self) -> Option<ImagePromptKind> {
        Some(ImagePromptKind::Nemotron)
    }

    fn context_budget_prompt(&self, formatted_prompt: &str, image_count: usize) -> Result<String> {
        prompt_without_images(formatted_prompt, image_count)
    }
}

fn prompt_without_images(formatted_prompt: &str, image_count: usize) -> Result<String> {
    let placeholder_count = formatted_prompt.match_indices(IMAGE_CONTEXT).count();
    ensure!(
        placeholder_count == image_count,
        "Nemotron rendered prompt contains {placeholder_count} image placeholders for {image_count} images"
    );
    Ok(formatted_prompt.replace(IMAGE_CONTEXT, ""))
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
    use super::*;

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
            "vision_config": {
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
    fn budget_prompt_removes_all_image_placeholders() {
        assert_eq!(
            prompt_without_images("before<image>middle<image>after", 2).unwrap(),
            "beforemiddleafter"
        );
        assert!(prompt_without_images("before<image>after", 2).is_err());
    }
}
