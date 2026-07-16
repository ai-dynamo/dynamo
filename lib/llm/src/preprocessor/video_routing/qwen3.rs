// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use serde_json::Value;

use super::{
    VideoRoutingInput, VideoRoutingReplacement,
    config::{read_json, read_model_config, required_token_id, required_usize},
};
use crate::{protocols::TokenIdType, tokenizers::traits::Tokenizer};

const SUPPORTED_MODEL_TYPES: &[&str] = &["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"];

fn expected_architecture(model_type: &str) -> Option<&'static str> {
    match model_type {
        "qwen3_vl" => Some("Qwen3VLForConditionalGeneration"),
        "qwen3_vl_moe" => Some("Qwen3VLMoeForConditionalGeneration"),
        "qwen3_5" => Some("Qwen3_5ForConditionalGeneration"),
        "qwen3_5_moe" => Some("Qwen3_5MoeForConditionalGeneration"),
        _ => None,
    }
}

pub(super) fn supports_model_type(model_type: &str) -> bool {
    SUPPORTED_MODEL_TYPES.contains(&model_type)
}

pub(super) struct Qwen3VideoRoutingSpec {
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    video_min_pixels: usize,
    video_max_pixels: usize,
    video_token_id: TokenIdType,
    vision_start_token_id: TokenIdType,
    vision_end_token_id: TokenIdType,
    tokenizer: Arc<dyn Tokenizer>,
}

impl Qwen3VideoRoutingSpec {
    pub(super) fn from_model_dir(
        model_id: &str,
        expected_model_type: &str,
        model_dir: &Path,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Self> {
        let expected_architecture = expected_architecture(expected_model_type)
            .context("mm-routing: Qwen video model_type has no registered architecture")?;
        let model_config = read_model_config(
            model_id,
            expected_model_type,
            expected_architecture,
            "Qwen",
            model_dir,
        )?;

        let vision_config = model_config
            .get("vision_config")
            .context("mm-routing: Qwen vision_config is missing")?;
        let patch_size = required_usize(vision_config, "patch_size", "Qwen")?;
        let spatial_merge_size = required_usize(vision_config, "spatial_merge_size", "Qwen")?;
        let temporal_patch_size = required_usize(vision_config, "temporal_patch_size", "Qwen")?;
        anyhow::ensure!(
            patch_size > 0 && spatial_merge_size > 0 && temporal_patch_size > 0,
            "mm-routing: Qwen video patch and merge sizes must be positive"
        );

        let video_config = read_json(model_dir, "video_preprocessor_config.json")?;
        anyhow::ensure!(
            video_config
                .get("video_processor_type")
                .and_then(Value::as_str)
                == Some("Qwen3VLVideoProcessor"),
            "mm-routing: unsupported Qwen video_processor_type"
        );
        anyhow::ensure!(
            video_config
                .get("do_resize")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            "mm-routing: Qwen video routing does not support do_resize=false"
        );
        ensure_matching_value(&video_config, "patch_size", patch_size)?;
        ensure_matching_value(&video_config, "temporal_patch_size", temporal_patch_size)?;
        ensure_matching_value(&video_config, "merge_size", spatial_merge_size)?;

        let size = video_config
            .get("size")
            .context("mm-routing: Qwen video processor size is missing")?;
        let video_min_pixels = required_usize(size, "shortest_edge", "Qwen")?;
        let video_max_pixels = required_usize(size, "longest_edge", "Qwen")?;
        anyhow::ensure!(
            video_min_pixels > 0 && video_max_pixels >= video_min_pixels,
            "mm-routing: invalid Qwen video pixel bounds"
        );

        Ok(Self {
            patch_size,
            spatial_merge_size,
            temporal_patch_size,
            video_min_pixels,
            video_max_pixels,
            video_token_id: required_token_id(&model_config, "video_token_id", "Qwen")?,
            vision_start_token_id: required_token_id(
                &model_config,
                "vision_start_token_id",
                "Qwen",
            )?,
            vision_end_token_id: required_token_id(&model_config, "vision_end_token_id", "Qwen")?,
            tokenizer,
        })
    }

    pub(super) fn build_replacement(
        &self,
        input: &VideoRoutingInput<'_>,
    ) -> Result<VideoRoutingReplacement> {
        self.validate_input(input)?;
        let (grid_t, grid_h, grid_w) = self.video_grid(input)?;
        let merge_area = self
            .spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .context("mm-routing: Qwen spatial merge area overflow")?;
        let spatial_patches = grid_h
            .checked_mul(grid_w)
            .context("mm-routing: Qwen video spatial grid overflow")?;
        anyhow::ensure!(
            spatial_patches.is_multiple_of(merge_area),
            "mm-routing: Qwen video grid is not divisible by the spatial merge area"
        );
        let tokens_per_grid = spatial_patches / merge_area;
        let base_video_tokens = grid_t
            .checked_mul(tokens_per_grid)
            .context("mm-routing: Qwen video token count overflow")?;

        let grid_timestamps = self.grid_timestamps(input, grid_t)?;
        let mut replacement_tokens = Vec::with_capacity(
            base_video_tokens
                .checked_add(grid_t.saturating_mul(8))
                .context("mm-routing: Qwen video replacement capacity overflow")?,
        );
        for timestamp in grid_timestamps {
            let timestamp_text = format!("<{timestamp:.1} seconds>");
            let timestamp_tokens = self.tokenizer.encode(&timestamp_text).with_context(|| {
                format!("mm-routing: failed to tokenize Qwen video timestamp {timestamp_text:?}")
            })?;
            replacement_tokens.extend_from_slice(timestamp_tokens.token_ids());
            replacement_tokens.push(self.vision_start_token_id);
            replacement_tokens.extend(std::iter::repeat_n(self.video_token_id, tokens_per_grid));
            replacement_tokens.push(self.vision_end_token_id);
        }

        Ok(VideoRoutingReplacement {
            placeholder_token_id: self.video_token_id,
            target_tokens: vec![
                self.vision_start_token_id,
                self.video_token_id,
                self.vision_end_token_id,
            ],
            replacement_tokens,
        })
    }

    fn validate_input(&self, input: &VideoRoutingInput<'_>) -> Result<()> {
        anyhow::ensure!(
            input.frame_count > 0,
            "mm-routing: Qwen video requires at least one sampled frame"
        );
        anyhow::ensure!(
            input.sampled_timestamps.len() == input.frame_count,
            "mm-routing: sampled timestamp count {} does not match frame count {}",
            input.sampled_timestamps.len(),
            input.frame_count
        );
        anyhow::ensure!(
            input.source_fps.is_finite() && input.source_fps > 0.0,
            "mm-routing: Qwen video source fps must be finite and positive"
        );
        anyhow::ensure!(
            input
                .sampled_timestamps
                .iter()
                .all(|timestamp| timestamp.is_finite() && *timestamp >= 0.0),
            "mm-routing: Qwen sampled timestamps must be finite and non-negative"
        );
        anyhow::ensure!(
            input
                .sampled_timestamps
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "mm-routing: Qwen sampled timestamps must be non-decreasing"
        );
        Ok(())
    }

    fn video_grid(&self, input: &VideoRoutingInput<'_>) -> Result<(usize, usize, usize)> {
        let (resized_height, resized_width) = self.smart_resize(
            input.frame_count,
            usize::try_from(input.height).context("mm-routing: video height exceeds usize")?,
            usize::try_from(input.width).context("mm-routing: video width exceeds usize")?,
        )?;
        let padded_frames = self.padded_frame_count(input.frame_count)?;
        Ok((
            padded_frames / self.temporal_patch_size,
            resized_height / self.patch_size,
            resized_width / self.patch_size,
        ))
    }

    fn padded_frame_count(&self, frame_count: usize) -> Result<usize> {
        let temporal_groups = frame_count
            .checked_add(self.temporal_patch_size - 1)
            .context("mm-routing: Qwen temporal padding overflow")?
            / self.temporal_patch_size;
        temporal_groups
            .checked_mul(self.temporal_patch_size)
            .context("mm-routing: Qwen temporal padding overflow")
    }

    /// Match Transformers' Qwen3VLVideoProcessor.smart_resize.
    fn smart_resize(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize)> {
        let factor = self
            .patch_size
            .checked_mul(self.spatial_merge_size)
            .context("mm-routing: Qwen resize factor overflow")?;
        anyhow::ensure!(
            height >= factor && width >= factor,
            "mm-routing: Qwen video dimensions {width}x{height} are smaller than resize factor {factor}"
        );
        let aspect_ratio = height.max(width) as f64 / height.min(width) as f64;
        anyhow::ensure!(
            aspect_ratio <= 200.0,
            "mm-routing: Qwen video aspect ratio exceeds 200:1"
        );

        let mut resized_height =
            (height as f64 / factor as f64).round_ties_even() as usize * factor;
        let mut resized_width = (width as f64 / factor as f64).round_ties_even() as usize * factor;
        let padded_frames = self.padded_frame_count(num_frames)?;

        let resized_volume = padded_frames as f64 * resized_height as f64 * resized_width as f64;
        let source_volume = num_frames as f64 * height as f64 * width as f64;
        if resized_volume > self.video_max_pixels as f64 {
            let beta = (source_volume / self.video_max_pixels as f64).sqrt();
            resized_height =
                ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
            resized_width =
                ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        } else if resized_volume < self.video_min_pixels as f64 {
            let beta = (self.video_min_pixels as f64 / source_volume).sqrt();
            resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
            resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
        }

        Ok((resized_height, resized_width))
    }

    fn grid_timestamps(&self, input: &VideoRoutingInput<'_>, grid_t: usize) -> Result<Vec<f64>> {
        let mut frame_timestamps = Vec::with_capacity(
            grid_t
                .checked_mul(self.temporal_patch_size)
                .context("mm-routing: Qwen timestamp padding overflow")?,
        );
        for timestamp in input.sampled_timestamps {
            let frame_index = (timestamp * input.source_fps).round_ties_even();
            anyhow::ensure!(
                frame_index.is_finite() && frame_index <= u64::MAX as f64,
                "mm-routing: Qwen sampled frame index is out of range"
            );
            frame_timestamps.push(frame_index / input.source_fps);
        }
        let last = *frame_timestamps
            .last()
            .context("mm-routing: Qwen sampled timestamps are empty")?;
        frame_timestamps.resize(grid_t * self.temporal_patch_size, last);

        Ok(frame_timestamps
            .chunks_exact(self.temporal_patch_size)
            .map(|timestamps| (timestamps[0] + timestamps[self.temporal_patch_size - 1]) / 2.0)
            .collect())
    }
}

fn ensure_matching_value(config: &Value, field: &str, expected: usize) -> Result<()> {
    let actual = required_usize(config, field, "Qwen")?;
    anyhow::ensure!(
        actual == expected,
        "mm-routing: Qwen {field} differs between config.json ({expected}) and video_preprocessor_config.json ({actual})"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tokenizers::{Encoding, traits::DecodeResult};

    use super::*;

    struct TimestampTokenizer;

    impl crate::tokenizers::traits::Encoder for TimestampTokenizer {
        fn encode(&self, input: &str) -> anyhow::Result<Encoding> {
            let ids = match input {
                "<0.5 seconds>" => vec![10, 11, 12, 13],
                "<1.0 seconds>" => vec![30, 31, 32, 33],
                "<2.5 seconds>" => vec![20, 21, 22, 23],
                _ => anyhow::bail!("unexpected timestamp {input:?}"),
            };
            Ok(Encoding::Sp(ids))
        }

        fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
            inputs.iter().map(|input| self.encode(input)).collect()
        }
    }

    impl crate::tokenizers::traits::Decoder for TimestampTokenizer {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<DecodeResult> {
            Ok(DecodeResult::Complete(String::new()))
        }
    }

    impl Tokenizer for TimestampTokenizer {}

    fn spec() -> Qwen3VideoRoutingSpec {
        Qwen3VideoRoutingSpec {
            patch_size: 8,
            spatial_merge_size: 1,
            temporal_patch_size: 2,
            video_min_pixels: 1,
            video_max_pixels: 4096 * 4,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            tokenizer: Arc::new(TimestampTokenizer),
        }
    }

    fn production_geometry_spec() -> Qwen3VideoRoutingSpec {
        Qwen3VideoRoutingSpec {
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            video_min_pixels: 4096,
            video_max_pixels: 25_165_824,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            tokenizer: Arc::new(TimestampTokenizer),
        }
    }

    #[test]
    fn builds_timestamped_qwen_replacement_without_preprocessing_pixels() {
        let timestamps = [0.0, 1.0, 2.0, 3.0];
        let input = VideoRoutingInput {
            frame_count: 4,
            width: 32,
            height: 32,
            source_fps: 1.0,
            sampled_timestamps: &timestamps,
        };

        let replacement = spec().build_replacement(&input).unwrap();

        assert_eq!(replacement.placeholder_token_id, 151656);
        assert_eq!(replacement.target_tokens, [151652, 151656, 151653]);
        assert_eq!(replacement.replacement_tokens.len(), 44);
        assert_eq!(
            &replacement.replacement_tokens[..6],
            &[10, 11, 12, 13, 151652, 151656]
        );
        assert_eq!(replacement.replacement_tokens[20], 151656);
        assert_eq!(replacement.replacement_tokens[21], 151653);
        assert_eq!(
            &replacement.replacement_tokens[22..28],
            &[20, 21, 22, 23, 151652, 151656]
        );
        assert_eq!(replacement.replacement_tokens[42], 151656);
        assert_eq!(replacement.replacement_tokens[43], 151653);
    }

    #[test]
    fn timestamp_frame_indices_use_ties_to_even_like_worker() {
        let timestamps = [1.25];
        let input = VideoRoutingInput {
            frame_count: 1,
            width: 32,
            height: 32,
            source_fps: 2.0,
            sampled_timestamps: &timestamps,
        };

        assert_eq!(spec().grid_timestamps(&input, 1).unwrap(), vec![1.0]);

        let replacement = spec().build_replacement(&input).unwrap();
        assert_eq!(&replacement.replacement_tokens[..4], &[30, 31, 32, 33]);
    }

    #[test]
    fn temporal_padding_repeats_the_last_sampled_frame() {
        struct PaddingTokenizer;
        impl crate::tokenizers::traits::Encoder for PaddingTokenizer {
            fn encode(&self, input: &str) -> anyhow::Result<Encoding> {
                let id = match input {
                    "<0.5 seconds>" => 1,
                    "<2.5 seconds>" => 2,
                    "<4.0 seconds>" => 3,
                    _ => anyhow::bail!("unexpected timestamp {input:?}"),
                };
                Ok(Encoding::Sp(vec![id]))
            }
            fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
                inputs.iter().map(|input| self.encode(input)).collect()
            }
        }
        impl crate::tokenizers::traits::Decoder for PaddingTokenizer {
            fn decode(
                &self,
                _token_ids: &[TokenIdType],
                _skip_special_tokens: bool,
            ) -> anyhow::Result<DecodeResult> {
                Ok(DecodeResult::Complete(String::new()))
            }
        }
        impl Tokenizer for PaddingTokenizer {}

        let mut spec = spec();
        spec.tokenizer = Arc::new(PaddingTokenizer);
        let timestamps = [0.0, 1.0, 2.0, 3.0, 4.0];
        let input = VideoRoutingInput {
            frame_count: 5,
            width: 32,
            height: 32,
            source_fps: 1.0,
            sampled_timestamps: &timestamps,
        };

        let replacement = spec.build_replacement(&input).unwrap();
        let timestamp_positions: Vec<_> = replacement
            .replacement_tokens
            .iter()
            .copied()
            .filter(|token| (1..=3).contains(token))
            .collect();
        assert_eq!(timestamp_positions, vec![1, 2, 3]);
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        let timestamps = [0.0, 2.0, 1.0, 3.0];
        let input = VideoRoutingInput {
            frame_count: 4,
            width: 32,
            height: 32,
            source_fps: 1.0,
            sampled_timestamps: &timestamps,
        };
        assert!(spec().build_replacement(&input).is_err());
    }

    #[test]
    fn video_grid_matches_transformers_golden_cases() {
        // Generated with Qwen3VLVideoProcessor.smart_resize using checkpoint defaults.
        let cases = [
            // frames, width, height, expected [T, H, W]
            (4, 426, 240, (2, 16, 26)),
            (30, 1280, 720, (15, 42, 76)),
            (5, 641, 359, (3, 22, 40)),
            (2, 32, 32, (1, 4, 4)),
            (31, 1920, 1080, (16, 42, 74)),
            (32, 1920, 1080, (16, 40, 72)),
            (3, 224, 224, (2, 14, 14)),
            (1, 224, 224, (1, 14, 14)),
            (5, 3760, 1120, (3, 76, 256)),
        ];
        let spec = production_geometry_spec();
        for (frame_count, width, height, expected) in cases {
            let timestamps = vec![0.0; frame_count];
            let input = VideoRoutingInput {
                frame_count,
                width,
                height,
                source_fps: 24.0,
                sampled_timestamps: &timestamps,
            };
            assert_eq!(
                spec.video_grid(&input).unwrap(),
                expected,
                "geometry mismatch for {frame_count} frames at {width}x{height}"
            );
        }
    }

    #[test]
    fn parses_qwen35_video_routing_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "video_token_id": 248057,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
                "vision_config": {
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("video_preprocessor_config.json"),
            r#"{
                "video_processor_type": "Qwen3VLVideoProcessor",
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}
            }"#,
        )
        .unwrap();

        let parsed = Qwen3VideoRoutingSpec::from_model_dir(
            "Qwen/Qwen3.5-4B",
            "qwen3_5",
            dir.path(),
            Arc::new(TimestampTokenizer),
        )
        .unwrap();

        assert_eq!(parsed.video_token_id, 248057);
        assert_eq!(parsed.vision_start_token_id, 248053);
        assert_eq!(parsed.vision_end_token_id, 248054);
        assert_eq!(parsed.video_min_pixels, 4096);
        assert_eq!(parsed.video_max_pixels, 25_165_824);
    }

    #[test]
    fn rejects_video_processor_geometry_that_differs_from_model_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "video_token_id": 151656,
                "vision_start_token_id": 151652,
                "vision_end_token_id": 151653,
                "vision_config": {
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("video_preprocessor_config.json"),
            r#"{
                "video_processor_type": "Qwen3VLVideoProcessor",
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}
            }"#,
        )
        .unwrap();

        assert!(
            Qwen3VideoRoutingSpec::from_model_dir(
                "Qwen/Qwen3-VL-2B-Instruct",
                "qwen3_vl",
                dir.path(),
                Arc::new(TimestampTokenizer),
            )
            .is_err()
        );
    }

    #[test]
    fn supports_only_explicit_qwen3_video_model_types() {
        for model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"] {
            assert!(supports_model_type(model_type));
        }
        assert!(!supports_model_type("qwen2_5_vl"));
        assert!(!supports_model_type("my_qwen3_vl_finetune"));
    }

    #[test]
    fn model_type_requires_its_multimodal_architecture() {
        assert_eq!(
            expected_architecture("qwen3_5"),
            Some("Qwen3_5ForConditionalGeneration")
        );
        assert_eq!(expected_architecture("qwen2_5_vl"), None);
    }
}
