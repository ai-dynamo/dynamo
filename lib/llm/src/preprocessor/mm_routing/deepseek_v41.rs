// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4.1 image-token accounting backed by `dynamo-multimodal`.
//!
//! The chat template emits one image token per image, and the backend expands
//! it into `rows * (columns + 1) + 2` copies of that same id. Routing keeps the
//! repeated-pad prompt shape and takes only the span length from the planner.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use dynamo_multimodal::models::deepseek_v41::DeepseekV41Processor;
use dynamo_multimodal::registry::{ProcessorSpec, spec_from_hf_configs};

use super::image::{ImagePromptKind, ImageRoutingBackend};

const MODEL_TYPE: &str = "deepseek_v41";

/// DeepSeek V4 checkpoints share the architecture but not this image layout,
/// so only the V4.1 model type selects the adapter.
pub(super) fn supports_model_type(model_type: Option<&str>) -> bool {
    model_type == Some(MODEL_TYPE)
}

/// Pixel-free DSV4.1 planner resolved from the checkpoint's own configuration.
pub(super) struct DeepseekV41ImageTokenCounter {
    processor: DeepseekV41Processor,
}

impl DeepseekV41ImageTokenCounter {
    /// When `preprocessor_config.json` is present, the planner rejects any
    /// knob it cannot honor instead of approximating it.
    pub(super) fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let config = read_optional(model_dir, "config.json")?.ok_or_else(|| {
            anyhow!(
                "mm-routing: DeepSeek V4.1 config.json is missing at {}",
                model_dir.display()
            )
        })?;
        let preprocessor_config = read_optional(model_dir, "preprocessor_config.json")?;
        let ProcessorSpec::DeepseekV41(spec) =
            spec_from_hf_configs(&config, preprocessor_config.as_deref().unwrap_or("{}"))?
        else {
            bail!("mm-routing: config.json does not describe a DeepSeek V4.1 image processor");
        };
        Ok(Self {
            processor: DeepseekV41Processor::new(spec)?,
        })
    }
}

impl ImageRoutingBackend for DeepseekV41ImageTokenCounter {
    fn count_tokens(&self, width: u32, height: u32) -> Result<usize> {
        let plan = self.processor.plan_image(width, height)?;
        Ok(usize::try_from(plan.num_image_tokens)?)
    }

    fn is_orientation_sensitive(&self) -> bool {
        true
    }

    fn routing_prompt_kind(&self) -> Option<ImagePromptKind> {
        Some(ImagePromptKind::RepeatedPad)
    }
}

fn read_optional(model_dir: &Path, filename: &str) -> Result<Option<String>> {
    let path = model_dir.join(filename);
    match std::fs::read_to_string(&path) {
        Ok(contents) => Ok(Some(contents)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("mm-routing: failed to read {}", path.display()))
        }
    }
}

/// The DeepSeek-V4.1-Flash `config.json` fields that image routing reads.
#[cfg(test)]
pub(in crate::preprocessor) fn write_test_config(model_dir: &Path) {
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": MODEL_TYPE,
            "image_token_id": 129264,
            "vision_config": {
                "patch_size": 14,
                "downsample_ratio": 3,
                "max_image_tokens": 1024,
                "min_pixels": 295936,
                "max_wh_ratio": null
            }
        })
        .to_string(),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_engine_geometry_without_preprocessor_config() {
        let model_dir = tempfile::tempdir().unwrap();
        write_test_config(model_dir.path());
        let counter = DeepseekV41ImageTokenCounter::from_model_dir(model_dir.path()).unwrap();

        // SGLang's and vLLM's DSV4.1 image preprocessing agree on these counts
        // for the published checkpoint: minimum-pixel upscale, plain grid,
        // token-budget shrink, and the orientation pair.
        for (width, height, tokens) in [
            (300, 300, 184),
            (1024, 1024, 652),
            (8000, 6000, 1001),
            (1920, 1080, 968),
            (1080, 1920, 986),
        ] {
            assert_eq!(
                counter.count_tokens(width, height).unwrap(),
                tokens,
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn rejects_preprocessor_knobs_the_planner_cannot_honor() {
        let model_dir = tempfile::tempdir().unwrap();
        write_test_config(model_dir.path());
        std::fs::write(
            model_dir.path().join("preprocessor_config.json"),
            serde_json::json!({ "max_pixels": 1_000_000 }).to_string(),
        )
        .unwrap();

        assert!(DeepseekV41ImageTokenCounter::from_model_dir(model_dir.path()).is_err());
    }
}
