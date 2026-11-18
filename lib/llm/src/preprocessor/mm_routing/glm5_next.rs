// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge the shared GLM image planner into Dynamo's routing-only token view.

use std::path::Path;

use anyhow::{Context, Result, bail};
use dynamo_multimodal::models::glm5_next::Glm5NextProcessor;
use dynamo_multimodal::processor::{MediaMetadata, MmFamilyProcessor};
use dynamo_multimodal::registry::{ProcessorSpec, spec_from_model_dir};

use crate::protocols::TokenIdType;

use super::image::{ImagePromptKind, ImageRoutingBackend};

pub(super) struct Glm5NextImageRoutingBackend {
    processor: Glm5NextProcessor,
}

impl Glm5NextImageRoutingBackend {
    pub(super) fn try_new(model_dir: &Path) -> Result<Self> {
        let ProcessorSpec::Glm5Next(spec) = spec_from_model_dir(model_dir)
            .context("mm-routing: failed to resolve GLM image planner configuration")?
        else {
            bail!("mm-routing: expected glm5_next checkpoint configuration");
        };
        Ok(Self {
            processor: Glm5NextProcessor::new(spec)?,
        })
    }
}

impl ImageRoutingBackend for Glm5NextImageRoutingBackend {
    fn count_tokens(&self, width: u32, height: u32) -> Result<usize> {
        Ok(self
            .processor
            .num_media_tokens(&MediaMetadata::Image { width, height })?)
    }

    fn validate_image_prompt(
        &self,
        token_ids: &[TokenIdType],
        dimensions: &[(u32, u32)],
    ) -> Result<()> {
        let input_ids = token_ids
            .iter()
            .map(|&id| i32::try_from(id))
            .collect::<Result<Vec<_>, _>>()
            .context("mm-routing: GLM prompt token ID exceeds the planner's i32 range")?;
        let media: Vec<_> = dimensions
            .iter()
            .map(|&(width, height)| MediaMetadata::Image { width, height })
            .collect();
        // In particular, refuse the video placeholder: GLM expands video
        // frames with the same image token, which this adapter cannot model.
        self.processor.layout_from_metadata(&input_ids, &media)?;
        Ok(())
    }

    fn routing_prompt_kind(&self) -> Option<ImagePromptKind> {
        // The template already supplies begin/end delimiters. The shared
        // planner expands only the image placeholder, matching RepeatedPad.
        Some(ImagePromptKind::RepeatedPad)
    }
}

#[cfg(test)]
pub(in crate::preprocessor) mod tests {
    use super::*;
    use crate::preprocessor::mm_routing::image::{
        ImageRoutingProcessor, resolve_exact_routing_image_token_id,
    };

    // Trimmed checkpoint metadata, shared with the Python binding tests. No
    // tokenizer or weights are needed to resolve the GLM planner.
    pub(in crate::preprocessor) fn model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, contents) in [
            (
                "config.json",
                include_str!("../../../tests/data/glm5-next-routing/config.json"),
            ),
            (
                "processor_config.json",
                include_str!("../../../tests/data/glm5-next-routing/processor_config.json"),
            ),
        ] {
            std::fs::write(dir.path().join(name), contents).unwrap();
        }
        dir
    }

    fn counter(dir: &Path) -> ImageRoutingProcessor {
        ImageRoutingProcessor::try_new("custom-checkpoint", Some("glm5_next"), dir).unwrap()
    }

    #[test]
    fn glm_checkpoint_drives_counts_and_placeholder_resolution() {
        let dir = model_dir();
        let processor = counter(dir.path());
        // Pinned HF geometry from frontend-crates' GLM fixture: portrait,
        // landscape, minimum-size upscaling, and an alignment boundary.
        for ((w, h), expected) in [
            ((448, 448), 256),
            ((1024, 768), 1036),
            ((768, 1024), 1036),
            ((1, 1), 16),
            ((13, 15), 20),
        ] {
            assert_eq!(processor.count_tokens(w, h).unwrap(), expected);
        }
        assert_eq!(
            processor.routing_prompt_kind(),
            Some(ImagePromptKind::RepeatedPad)
        );
        assert_eq!(
            resolve_exact_routing_image_token_id("custom-checkpoint", dir.path()),
            Some(154854)
        );

        // Counts must use checkpoint settings, not hardcoded GLM defaults.
        let path = dir.path().join("processor_config.json");
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        config["image_processor"]["max_image_tokens"] = 64.into();
        std::fs::write(path, config.to_string()).unwrap();
        assert_eq!(counter(dir.path()).count_tokens(448, 448).unwrap(), 64);
    }

    #[test]
    fn glm_count_failures_propagate_through_batch_accounting() {
        let dir = model_dir();
        let processor = counter(dir.path());
        for dimensions in [(0, 448), (448, 0), (u32::MAX, u32::MAX)] {
            assert!(processor.count_tokens(dimensions.0, dimensions.1).is_err());
            assert!(
                processor
                    .count_tokens_for_images(&[(448, 448), dimensions], 4096, 7)
                    .is_err()
            );
        }
    }

    #[test]
    fn glm_layout_rejects_video_mismatches_and_out_of_range_ids() {
        let dir = model_dir();
        let processor = counter(dir.path());
        let dimensions = [(448, 448), (13, 15)];
        processor
            .validate_image_prompt(
                &[7, 154830, 154854, 154831, 8, 154830, 154854, 154831, 9],
                &dimensions,
            )
            .unwrap();
        for ids in [
            vec![154854],
            vec![154854, 154854, 154854],
            vec![154854, 154855, 154854],
            vec![u32::MAX, 154854, 154854],
        ] {
            assert!(processor.validate_image_prompt(&ids, &dimensions).is_err());
        }
    }

    #[test]
    fn glm_unsupported_config_disables_the_worker_placeholder() {
        let dir = model_dir();
        let path = dir.path().join("processor_config.json");
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        config["image_processor"]["do_resize"] = false.into();
        std::fs::write(path, config.to_string()).unwrap();
        assert!(
            ImageRoutingProcessor::try_new("custom-checkpoint", Some("glm5_next"), dir.path())
                .is_err()
        );
        assert_eq!(
            resolve_exact_routing_image_token_id("custom-checkpoint", dir.path()),
            None
        );
    }
}
