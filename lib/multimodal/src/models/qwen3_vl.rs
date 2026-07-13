// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL video preprocessing and Hugging Face configuration mapping.
//!
//! The initial implementation provides Hugging Face compatible Qwen3-VL
//! preprocessing for already-decoded RGB video frames. Fetching, decoding,
//! transport, and inference-engine adaptation remain outside this crate.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::processed::{ProcessedField, ProcessedMedia, ProcessedValue};
use crate::registry::VideoProcessor;
use crate::types::{FieldLayout, Modality, RgbFrameRef};
use crate::vision::{ModelSpecificValue, PreProcessorConfig, Qwen3VLProcessor, TransformError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoTiming {
    pub source_fps: f64,
    pub source_duration: f64,
    pub sampled_timestamps: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VlVideoConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub do_resize: bool,
    pub do_normalize: bool,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub resampling: usize,
}

impl Default for Qwen3VlVideoConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            do_resize: true,
            do_normalize: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            resampling: 3,
        }
    }
}

impl Qwen3VlVideoConfig {
    pub fn from_preprocessor_json(json: &str) -> Result<Self, TransformError> {
        let parsed = PreProcessorConfig::from_json(json)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        let defaults = Self::default();
        let patch_size = parsed
            .patch_size
            .as_ref()
            .and_then(|value| value.height.or(value.width))
            .map_or(defaults.patch_size, |value| value as usize);
        let size = parsed.size.as_ref();
        let min_pixels = parsed
            .min_pixels
            .or_else(|| {
                size.and_then(|value| value.get("shortest_edge").copied().map(|v| v as usize))
            })
            .unwrap_or(defaults.min_pixels);
        let max_pixels = parsed
            .max_pixels
            .or_else(|| {
                size.and_then(|value| value.get("longest_edge").copied().map(|v| v as usize))
            })
            .unwrap_or(defaults.max_pixels);
        let mean = parsed.get_image_mean();
        let std = parsed.get_image_std();
        let config = Self {
            patch_size,
            merge_size: parsed.merge_size.unwrap_or(defaults.merge_size),
            temporal_patch_size: parsed
                .temporal_patch_size
                .unwrap_or(defaults.temporal_patch_size),
            min_pixels,
            max_pixels,
            do_resize: parsed.do_resize.unwrap_or(defaults.do_resize),
            do_normalize: parsed.do_normalize.unwrap_or(defaults.do_normalize),
            image_mean: mean,
            image_std: std,
            resampling: parsed.resampling.unwrap_or(defaults.resampling),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), TransformError> {
        if self.patch_size == 0 || self.merge_size == 0 || self.temporal_patch_size == 0 {
            return Err(TransformError::ShapeError(
                "patch_size, merge_size, and temporal_patch_size must be greater than zero"
                    .to_string(),
            ));
        }
        if self.min_pixels == 0 || self.min_pixels > self.max_pixels {
            return Err(TransformError::ShapeError(
                "min_pixels must be positive and no greater than max_pixels".to_string(),
            ));
        }
        if self.image_mean.iter().any(|value| !value.is_finite())
            || self
                .image_std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(TransformError::ShapeError(
                "image mean must be finite and image std must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }

    fn processor_config(&self) -> PreProcessorConfig {
        PreProcessorConfig {
            do_resize: Some(self.do_resize),
            do_normalize: Some(self.do_normalize),
            image_mean: Some(self.image_mean.to_vec()),
            image_std: Some(self.image_std.to_vec()),
            resampling: Some(self.resampling),
            min_pixels: Some(self.min_pixels),
            max_pixels: Some(self.max_pixels),
            temporal_patch_size: Some(self.temporal_patch_size),
            merge_size: Some(self.merge_size),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVideoPreprocessor {
    config: Qwen3VlVideoConfig,
    processor: Qwen3VLProcessor,
    processor_config: PreProcessorConfig,
}

impl Qwen3VlVideoPreprocessor {
    pub fn new(config: Qwen3VlVideoConfig) -> Result<Self, TransformError> {
        config.validate()?;
        let processor = Qwen3VLProcessor::with_config(
            config.patch_size,
            config.merge_size,
            config.min_pixels,
            config.max_pixels,
            config.temporal_patch_size,
        );
        let processor_config = config.processor_config();
        Ok(Self {
            config,
            processor,
            processor_config,
        })
    }

    pub fn preprocess(
        &self,
        frames: &[RgbFrameRef<'_>],
        timing: &VideoTiming,
    ) -> Result<ProcessedMedia, TransformError> {
        if timing.sampled_timestamps.len() != frames.len() {
            return Err(TransformError::InvalidShape {
                expected: format!("{} sampled timestamps", frames.len()),
                actual: vec![timing.sampled_timestamps.len()],
            });
        }
        let output = self
            .processor
            .preprocess_configured_video_rgb(frames, &self.processor_config)?;
        let grid = match output.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape }) if shape == &[1, 3] => [
                u32::try_from(data[0])
                    .map_err(|_| TransformError::ShapeError("invalid grid_t".into()))?,
                u32::try_from(data[1])
                    .map_err(|_| TransformError::ShapeError("invalid grid_h".into()))?,
                u32::try_from(data[2])
                    .map_err(|_| TransformError::ShapeError("invalid grid_w".into()))?,
            ],
            value => {
                return Err(TransformError::ShapeError(format!(
                    "Qwen3-VL processor returned invalid video_grid_thw: {value:?}"
                )));
            }
        };
        let output_shape = output.encoder_input.shape().to_vec();
        if output_shape.len() != 2 {
            return Err(TransformError::InvalidShape {
                expected: "two-dimensional Qwen3-VL video tensor".to_string(),
                actual: output_shape,
            });
        }
        let patch_count = output_shape[0];
        let timestamps = timing
            .sampled_timestamps
            .chunks(self.config.temporal_patch_size)
            .map(|chunk| {
                let first = chunk[0];
                let last = chunk.last().copied().unwrap_or(first);
                (first + last) / 2.0
            })
            .collect::<Vec<_>>();
        if timestamps.len() != grid[0] as usize {
            return Err(TransformError::InvalidShape {
                expected: format!("{} temporal grid timestamps", grid[0]),
                actual: vec![timestamps.len()],
            });
        }
        let mut fields = BTreeMap::new();
        fields.insert(
            "pixel_values_videos".to_string(),
            ProcessedField {
                value: ProcessedValue::F32Tensor {
                    data: output.encoder_input.into_raw_vec_and_offset().0,
                    shape: output_shape,
                },
                layout: FieldLayout::flat("patches_per_video"),
                keep_on_host: false,
                forward: true,
            },
        );
        fields.insert(
            "video_grid_thw".to_string(),
            ProcessedField {
                value: ProcessedValue::I64Tensor {
                    data: grid.iter().map(|value| i64::from(*value)).collect(),
                    shape: vec![1, 3],
                },
                layout: FieldLayout::Batched,
                keep_on_host: true,
                forward: true,
            },
        );
        fields.insert(
            "timestamps".to_string(),
            ProcessedField {
                value: ProcessedValue::F64Tensor {
                    shape: vec![1, timestamps.len()],
                    data: timestamps,
                },
                layout: FieldLayout::Batched,
                keep_on_host: true,
                forward: true,
            },
        );
        fields.insert(
            "patches_per_video".to_string(),
            ProcessedField {
                value: ProcessedValue::I64Tensor {
                    data: vec![i64::try_from(patch_count).map_err(|_| {
                        TransformError::ShapeError("video patch count exceeds i64".to_string())
                    })?],
                    shape: vec![1],
                },
                layout: FieldLayout::Batched,
                keep_on_host: true,
                forward: false,
            },
        );
        let processed = ProcessedMedia {
            modality: Modality::Video,
            fields,
            feature_token_counts: output.feature_token_counts,
            original_sizes: output.item_sizes,
        };
        processed
            .validate()
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        Ok(processed)
    }
}

impl VideoProcessor for Qwen3VlVideoPreprocessor {
    fn process(
        &self,
        frames: &[RgbFrameRef<'_>],
        timing: &VideoTiming,
    ) -> Result<ProcessedMedia, TransformError> {
        self.preprocess(frames, timing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, seed: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                bytes.extend_from_slice(&[
                    seed.wrapping_add((x * 7 + y * 3) as u8),
                    seed.wrapping_add((x * 5 + y * 11) as u8),
                    seed.wrapping_add((x + y * 2) as u8),
                ]);
            }
        }
        bytes
    }

    #[test]
    fn qwen3_video_output_has_named_fields_and_layouts() {
        let buffers = [frame(32, 32, 3), frame(32, 32, 101), frame(32, 32, 177)];
        let frames = buffers
            .iter()
            .map(|data| RgbFrameRef {
                width: 32,
                height: 32,
                data,
            })
            .collect::<Vec<_>>();
        let config = Qwen3VlVideoConfig {
            min_pixels: 1024,
            max_pixels: 1024,
            do_resize: false,
            ..Default::default()
        };
        let result = Qwen3VlVideoPreprocessor::new(config)
            .unwrap()
            .preprocess(
                &frames,
                &VideoTiming {
                    source_fps: 30.0,
                    source_duration: 0.1,
                    sampled_timestamps: vec![0.0, 1.0 / 30.0, 2.0 / 30.0],
                },
            )
            .unwrap();

        assert_eq!(result.modality, Modality::Video);
        let ProcessedValue::F32Tensor { data, shape } =
            &result.field("pixel_values_videos").unwrap().value
        else {
            panic!("pixel_values_videos must be FP32");
        };
        assert_eq!(shape, &[8, 1536]);
        assert_eq!(data.len(), 8 * 1536);
        assert_eq!(
            result.field("pixel_values_videos").unwrap().layout,
            FieldLayout::flat("patches_per_video")
        );
        let ProcessedValue::I64Tensor { data: grid, shape } =
            &result.field("video_grid_thw").unwrap().value
        else {
            panic!("video_grid_thw must be I64");
        };
        assert_eq!(shape, &[1, 3]);
        assert_eq!(grid, &[2, 2, 2]);
        let ProcessedValue::F64Tensor {
            data: timestamps, ..
        } = &result.field("timestamps").unwrap().value
        else {
            panic!("timestamps must be F64");
        };
        assert_eq!(timestamps, &[1.0 / 60.0, 2.0 / 30.0]);
    }

    #[test]
    fn timestamp_count_must_match_decoded_frames() {
        let bytes = frame(32, 32, 0);
        let error = Qwen3VlVideoPreprocessor::new(Qwen3VlVideoConfig::default())
            .unwrap()
            .preprocess(
                &[RgbFrameRef {
                    width: 32,
                    height: 32,
                    data: &bytes,
                }],
                &VideoTiming {
                    source_fps: 30.0,
                    source_duration: 1.0,
                    sampled_timestamps: vec![],
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("sampled timestamps"));
    }

    #[test]
    fn qwen3_video_is_bit_exact_with_hugging_face_4_57_1_x86_fp32() {
        // Generated by scripts/generate_qwen3_vl_video_fingerprint.py on x86:
        // transformers 4.57.1, Pillow 12.2.0. The FP32 fingerprint enforces
        // bit parity; the pixel fingerprint isolates resize/layout failures.
        let buffers = [frame(37, 35, 3), frame(37, 35, 101), frame(37, 35, 177)];
        let frames = buffers
            .iter()
            .map(|data| RgbFrameRef {
                width: 37,
                height: 35,
                data,
            })
            .collect::<Vec<_>>();
        let result = Qwen3VlVideoPreprocessor::new(Qwen3VlVideoConfig::default())
            .unwrap()
            .preprocess(
                &frames,
                &VideoTiming {
                    source_fps: 2.0,
                    source_duration: 1.5,
                    sampled_timestamps: vec![0.0, 0.5, 1.0],
                },
            )
            .unwrap();

        let ProcessedValue::F32Tensor { data, shape } =
            &result.field("pixel_values_videos").unwrap().value
        else {
            panic!("pixel_values_videos must be FP32");
        };
        assert_eq!(shape, &[200, 1536]);
        let ProcessedValue::I64Tensor { data: grid, .. } =
            &result.field("video_grid_thw").unwrap().value
        else {
            panic!("video_grid_thw must be I64");
        };
        assert_eq!(grid, &[2, 10, 10]);
        let fp32_fingerprint = data
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |mut hash, value| {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
                hash
            });
        assert_eq!(fp32_fingerprint, 0xbcb1_32ea_87ca_1806);

        let fingerprint = data.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            let byte = ((value * 0.5 + 0.5) * 255.0)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8;
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        assert_eq!(fingerprint, 0x3f34_d7b3_e45e_d9d7);
    }
}
