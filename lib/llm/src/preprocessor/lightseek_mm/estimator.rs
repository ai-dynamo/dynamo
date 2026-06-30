// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free image-token estimators used by MM-aware routing.
//!
//! The compatibility target is the image-counting behavior exposed by
//! `llm-multimodal` 1.7.0. Only the six algorithms Dynamo actually calls are
//! retained; image decoding, tensor construction, video, and model prompt
//! rewriting deliberately remain outside this module.

use std::collections::HashMap;

use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct PreprocessorConfig {
    do_convert_rgb: Option<bool>,
    do_normalize: Option<bool>,
    do_pad: Option<bool>,
    do_rescale: Option<bool>,
    do_resize: Option<bool>,
    do_center_crop: Option<bool>,
    image_processor_type: Option<String>,
    im_start_token: Option<String>,
    im_end_token: Option<String>,
    slice_start_token: Option<String>,
    slice_end_token: Option<String>,
    vision_start_token: Option<String>,
    vision_end_token: Option<String>,
    #[serde(alias = "norm_mean")]
    image_mean: Option<Vec<f64>>,
    #[serde(alias = "norm_std")]
    image_std: Option<Vec<f64>>,
    rescale_factor: Option<f64>,
    #[serde(alias = "resample")]
    resampling: Option<usize>,
    patch_size: Option<PatchSize>,
    merge_size: Option<usize>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    temporal_patch_size: Option<usize>,
    num_crops: Option<usize>,
    dynamic_hd: Option<usize>,
    max_image_tiles: Option<usize>,
    num_img_tokens: Option<usize>,
    size: Option<HashMap<String, u32>>,
    crop_size: Option<HashMap<String, u32>>,
    #[serde(default, deserialize_with = "deserialize_media_proc_config")]
    media_proc_cfg: Option<MediaProcConfig>,
}

impl PreprocessorConfig {
    fn patch_size(&self, default: usize) -> usize {
        match self.patch_size.as_ref() {
            Some(patch_size) => patch_size.height().unwrap_or(default),
            None => self
                .media_proc_cfg
                .as_ref()
                .and_then(|config| config.patch_size)
                .map(|value| value as usize)
                .unwrap_or(default),
        }
    }

    fn merge_size(&self) -> Option<usize> {
        self.merge_size.or_else(|| {
            self.media_proc_cfg
                .as_ref()
                .and_then(|config| config.merge_kernel_size)
        })
    }

    fn size_field(&self, field: &str) -> Option<usize> {
        self.size
            .as_ref()?
            .get(field)
            .copied()
            .map(|value| value as usize)
    }

    fn target_height(&self) -> Option<u32> {
        let size = self.size.as_ref()?;
        Some(
            size.get("height")
                .or_else(|| size.get("shortest_edge"))
                .copied()
                .unwrap_or(224),
        )
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct MediaProcConfig {
    #[serde(default, deserialize_with = "deserialize_lenient_option")]
    patch_size: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_lenient_option")]
    merge_kernel_size: Option<usize>,
}

fn deserialize_media_proc_config<'de, D>(
    deserializer: D,
) -> Result<Option<MediaProcConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    if !value.is_object() {
        return Ok(None);
    }
    Ok(serde_json::from_value(value).ok())
}

fn deserialize_lenient_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PatchSize {
    Scalar(u32),
    Dimensions(PatchDimensions),
}

impl PatchSize {
    fn height(&self) -> Option<usize> {
        match self {
            Self::Scalar(value) => Some(*value as usize),
            Self::Dimensions(dimensions) => dimensions.height.map(|value| value as usize),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct PatchDimensions {
    #[serde(deserialize_with = "deserialize_present")]
    height: Option<u32>,
    #[serde(deserialize_with = "deserialize_present")]
    width: Option<u32>,
}

fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelFamily {
    Qwen2,
    Qwen3,
    Llava,
    LlavaNext,
    Llama4,
    KimiK2,
}

impl ModelFamily {
    /// Resolve exact architecture aliases first, then stable model-ID
    /// patterns. Keeping this ordered avoids the old hash-map iteration
    /// ambiguity (notably LLaVA-NeXT versus LLaVA).
    pub(super) fn identify(model_id: &str, model_type: Option<&str>) -> Option<Self> {
        if let Some(family) = model_type.and_then(Self::from_model_type) {
            return Some(family);
        }
        Self::from_model_id(model_id)
    }

    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type.to_ascii_lowercase().as_str() {
            "qwen2_vl" | "qwen2_5_vl" => Some(Self::Qwen2),
            "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe" | "qwen3_6" | "qwen3_6_moe" => {
                Some(Self::Qwen3)
            }
            "llava_next" => Some(Self::LlavaNext),
            "llava" => Some(Self::Llava),
            "llama4" => Some(Self::Llama4),
            "kimi_k25" | "kimi_k2_5" | "kimi_k2_6" => Some(Self::KimiK2),
            _ => None,
        }
    }

    fn from_model_id(model_id: &str) -> Option<Self> {
        let id = model_id.to_ascii_lowercase();

        if id.contains("qwen3.5")
            || id.contains("qwen3_5")
            || id.contains("qwen3.6")
            || id.contains("qwen3_6")
            || (id.contains("qwen3-vl") || id.contains("qwen3_vl"))
        {
            return Some(Self::Qwen3);
        }
        if id.contains("qwen2-vl")
            || id.contains("qwen2_vl")
            || id.contains("qwen2.5-vl")
            || id.contains("qwen2_5-vl")
            || id.contains("qwen2_5_vl")
        {
            return Some(Self::Qwen2);
        }
        if id.contains("llava-next") || id.contains("llava_next") || id.contains("llava-v1.6") {
            return Some(Self::LlavaNext);
        }
        if id.contains("llava-1.5") || id.contains("llava-v1.5") {
            return Some(Self::Llava);
        }
        if id.contains("llama-4") || id.contains("llama4") {
            return Some(Self::Llama4);
        }
        if id.contains("kimi") && id.contains("k2") {
            return Some(Self::KimiK2);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub(super) enum ImageTokenEstimator {
    Qwen(QwenEstimator),
    Llava(LlavaEstimator),
    LlavaNext(LlavaNextEstimator),
    Llama4(Llama4Estimator),
    KimiK2(KimiK2Estimator),
}

impl ImageTokenEstimator {
    pub(super) fn from_config(
        family: ModelFamily,
        config: &PreprocessorConfig,
    ) -> anyhow::Result<Self> {
        let estimator = match family {
            ModelFamily::Qwen2 => Self::Qwen(QwenEstimator::qwen2(config)),
            ModelFamily::Qwen3 => Self::Qwen(QwenEstimator::qwen3(config)),
            ModelFamily::Llava => Self::Llava(LlavaEstimator::from_config(config)),
            ModelFamily::LlavaNext => Self::LlavaNext(LlavaNextEstimator::from_config(config)),
            ModelFamily::Llama4 => Self::Llama4(Llama4Estimator::from_config(config)),
            ModelFamily::KimiK2 => Self::KimiK2(KimiK2Estimator::from_config(config)),
        };
        estimator.validate()?;
        Ok(estimator)
    }

    fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Qwen(estimator) => estimator.validate(),
            Self::Llava(estimator) => estimator.validate(),
            Self::LlavaNext(estimator) => estimator.validate(),
            Self::Llama4(estimator) => estimator.validate(),
            Self::KimiK2(_) => Ok(()),
        }
    }

    #[inline]
    pub(super) fn count_tokens(&self, width: u32, height: u32) -> usize {
        match self {
            Self::Qwen(estimator) => estimator.count_tokens(width, height),
            Self::Llava(estimator) => estimator.count_tokens(),
            Self::LlavaNext(estimator) => estimator.count_tokens(width, height),
            Self::Llama4(estimator) => estimator.count_tokens(width, height),
            Self::KimiK2(estimator) => estimator.count_tokens(width, height),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct QwenEstimator {
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
    temporal_patch_size: usize,
}

impl QwenEstimator {
    const QWEN2_MIN_PIXELS: usize = 256 * 28 * 28;
    const QWEN2_MAX_PIXELS: usize = 1280 * 28 * 28;
    const QWEN3_MIN_PIXELS: usize = 65_536;
    const QWEN3_MAX_PIXELS: usize = 16_777_216;

    fn qwen2(config: &PreprocessorConfig) -> Self {
        Self {
            patch_size: config.patch_size(14),
            merge_size: config.merge_size().unwrap_or(2),
            min_pixels: config.min_pixels.unwrap_or(Self::QWEN2_MIN_PIXELS),
            max_pixels: config.max_pixels.unwrap_or(Self::QWEN2_MAX_PIXELS),
            temporal_patch_size: config.temporal_patch_size.unwrap_or(2),
        }
    }

    fn qwen3(config: &PreprocessorConfig) -> Self {
        Self {
            patch_size: config.patch_size(16),
            merge_size: config.merge_size().unwrap_or(2),
            min_pixels: config
                .min_pixels
                .or_else(|| config.size_field("shortest_edge"))
                .unwrap_or(Self::QWEN3_MIN_PIXELS),
            max_pixels: config
                .max_pixels
                .or_else(|| config.size_field("longest_edge"))
                .unwrap_or(Self::QWEN3_MAX_PIXELS),
            temporal_patch_size: config.temporal_patch_size.unwrap_or(2),
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.patch_size > 0,
            "Qwen patch_size must be greater than zero"
        );
        anyhow::ensure!(
            self.merge_size > 0,
            "Qwen merge_size must be greater than zero"
        );
        anyhow::ensure!(
            self.temporal_patch_size > 0,
            "Qwen temporal_patch_size must be greater than zero"
        );
        anyhow::ensure!(
            self.min_pixels > 0,
            "Qwen min_pixels must be greater than zero"
        );
        anyhow::ensure!(
            self.max_pixels > 0,
            "Qwen max_pixels must be greater than zero"
        );
        anyhow::ensure!(
            self.min_pixels <= u32::MAX as usize && self.max_pixels <= u32::MAX as usize,
            "Qwen pixel budgets exceed the supported u32 image area"
        );
        anyhow::ensure!(
            self.min_pixels <= self.max_pixels,
            "Qwen min_pixels must not exceed max_pixels"
        );
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .ok_or_else(|| anyhow::anyhow!("Qwen patch_size * merge_size overflowed"))?;
        anyhow::ensure!(
            factor <= u32::MAX as usize,
            "Qwen alignment factor exceeds the supported image dimension range"
        );
        Ok(())
    }

    #[inline]
    fn count_tokens(&self, width: u32, height: u32) -> usize {
        let factor = self.patch_size * self.merge_size;
        let (resized_h, resized_w) = self
            .smart_resize(height as usize, width as usize)
            .unwrap_or((factor, factor));
        let grid_t = 1_usize.max(self.temporal_patch_size) / self.temporal_patch_size;
        let grid_h = resized_h / self.patch_size;
        let grid_w = resized_w / self.patch_size;
        (grid_t * grid_h * grid_w) / (self.merge_size * self.merge_size)
    }

    fn smart_resize(&self, height: usize, width: usize) -> Option<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;
        if height == 0 || width == 0 {
            return None;
        }
        if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
            return None;
        }

        let mut resized_h = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut resized_w = round_half_to_even(width as f64 / factor as f64) as usize * factor;
        resized_h = resized_h.max(factor);
        resized_w = resized_w.max(factor);

        let resized_pixels = resized_h.saturating_mul(resized_w);
        let original_pixels = height as f64 * width as f64;
        if resized_pixels > self.max_pixels {
            let beta = (original_pixels / self.max_pixels as f64).sqrt();
            resized_h = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            resized_w = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            resized_h = resized_h.max(factor);
            resized_w = resized_w.max(factor);
        } else if resized_pixels < self.min_pixels {
            let beta = (self.min_pixels as f64 / original_pixels).sqrt();
            resized_h = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            resized_w = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Some((resized_h, resized_w))
    }
}

/// Python-compatible round-half-to-even, matching the pinned implementation.
#[inline]
fn round_half_to_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value - value.floor() - 0.5).abs() < 1e-9 && rounded as i64 % 2 != 0 {
        rounded - 1.0
    } else {
        rounded
    }
}

#[derive(Debug, Clone)]
pub(super) struct LlavaEstimator {
    patch_size: usize,
    image_size: usize,
}

impl LlavaEstimator {
    fn from_config(config: &PreprocessorConfig) -> Self {
        Self {
            patch_size: config.patch_size(14),
            image_size: config.target_height().unwrap_or(336) as usize,
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.patch_size > 0,
            "LLaVA patch_size must be greater than zero"
        );
        anyhow::ensure!(
            self.image_size > 0,
            "LLaVA image size must be greater than zero"
        );
        Ok(())
    }

    #[inline]
    fn count_tokens(&self) -> usize {
        let patches_per_side = self.image_size / self.patch_size;
        patches_per_side * patches_per_side
    }
}

#[derive(Debug, Clone)]
pub(super) struct LlavaNextEstimator {
    image_size: usize,
}

impl LlavaNextEstimator {
    const BASE_IMAGE_SIZE: u32 = 336;
    const PATCH_SIZE: u32 = 14;
    const GRID_PINPOINTS: [(u32, u32); 5] =
        [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];

    fn from_config(config: &PreprocessorConfig) -> Self {
        Self {
            image_size: config.target_height().unwrap_or(Self::BASE_IMAGE_SIZE) as usize,
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.image_size > 0,
            "LLaVA-NeXT image size must be greater than zero"
        );
        Ok(())
    }

    fn count_tokens(&self, width: u32, height: u32) -> usize {
        let npatches = self.image_size / Self::PATCH_SIZE as usize;
        let base_features = npatches * npatches;
        let (best_w, best_h) = select_best_resolution((width, height), &Self::GRID_PINPOINTS);
        // The pinned processor uses its 336px base here even when the
        // preprocessor config overrides `size`; retain that quirk.
        let grid_w = best_w / Self::BASE_IMAGE_SIZE;
        let grid_h = best_h / Self::BASE_IMAGE_SIZE;

        let current_w = (npatches * grid_w as usize) as f32;
        let current_h = (npatches * grid_h as usize) as f32;
        let aspect_ratio = width as f32 / height as f32;
        let current_aspect = current_w / current_h;

        let (feature_h, feature_w) = if aspect_ratio > current_aspect {
            let new_h = (height as f32 * (current_w / width as f32)).round();
            let padding = ((current_h - new_h) / 2.0).floor() as usize;
            (current_h as usize - 2 * padding, current_w as usize)
        } else {
            let new_w = (width as f32 * (current_h / height as f32)).round();
            let padding = ((current_w - new_w) / 2.0).floor() as usize;
            (current_h as usize, current_w as usize - 2 * padding)
        };

        feature_h * feature_w + feature_h + base_features
    }
}

fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    let mut max_effective_resolution = 0_u64;
    let mut min_wasted_resolution = u64::MAX;

    for &(width, height) in possible_resolutions {
        let scale =
            (width as f32 / original_width as f32).min(height as f32 / original_height as f32);
        let downscaled_width = (original_width as f32 * scale) as u32;
        let downscaled_height = (original_height as f32 * scale) as u32;
        let effective_resolution = (downscaled_width as u64 * downscaled_height as u64)
            .min(original_width as u64 * original_height as u64);
        let wasted_resolution = width as u64 * height as u64 - effective_resolution;

        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (width, height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}

#[derive(Debug, Clone)]
pub(super) struct Llama4Estimator {
    tile_size: usize,
    max_patches: usize,
}

#[derive(Clone, Copy)]
struct TileCandidate {
    scale: f64,
    tiles: usize,
}

impl Llama4Estimator {
    const PATCH_SIZE: usize = 14;

    fn from_config(config: &PreprocessorConfig) -> Self {
        Self {
            tile_size: config.size_field("height").unwrap_or(336),
            max_patches: config.max_image_tiles.unwrap_or(16),
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.tile_size >= Self::PATCH_SIZE,
            "Llama4 tile size must be at least {}",
            Self::PATCH_SIZE
        );
        anyhow::ensure!(
            self.tile_size <= u32::MAX as usize,
            "Llama4 tile size exceeds the supported image dimension range"
        );
        anyhow::ensure!(
            self.max_patches > 0,
            "Llama4 max_image_tiles must be greater than zero"
        );
        let tokens_per_tile = (self.tile_size / Self::PATCH_SIZE)
            .checked_pow(2)
            .ok_or_else(|| anyhow::anyhow!("Llama4 tokens per tile overflowed"))?;
        self.max_patches
            .checked_add(1)
            .and_then(|tiles| tiles.checked_mul(tokens_per_tile))
            .ok_or_else(|| anyhow::anyhow!("Llama4 maximum token count overflowed"))?;
        Ok(())
    }

    fn count_tokens(&self, width: u32, height: u32) -> usize {
        let local_tiles = self.best_fit_tile_count(height, width);
        let total_tiles = if local_tiles > 1 {
            local_tiles + 1
        } else {
            local_tiles
        };

        // TODO: Llama4's backend prompt uses 144 patch positions per tile
        // after pixel shuffle plus structural tokens. This intentionally
        // preserves Dynamo's current llm-multimodal 1.7.0 scalar count (576
        // positions for a 336px tile) until routing can represent that exact
        // structured sequence.
        total_tiles * (self.tile_size / Self::PATCH_SIZE).pow(2)
    }

    fn best_fit_tile_count(&self, original_h: u32, original_w: u32) -> usize {
        if original_h == 0 && original_w == 0 {
            return 0;
        }

        let min_h_tiles = (original_h as usize).div_ceil(self.tile_size).max(1);
        let min_w_tiles = (original_w as usize).div_ceil(self.tile_size).max(1);
        if min_h_tiles
            .checked_mul(min_w_tiles)
            .is_some_and(|tiles| tiles <= self.max_patches)
        {
            let tiles = min_h_tiles * min_w_tiles;
            let selected_scale =
                self.canvas_scale(original_h, original_w, min_h_tiles, min_w_tiles);
            return self
                .min_tiles_within_scale_epsilon(original_h, original_w, selected_scale)
                .unwrap_or(tiles);
        }

        // Below the first upscaling canvas, height scale increases with
        // `h_tiles` while width scale decreases, so the optimum brackets
        // their monotonic crossing.
        let scale_is_height_limited = |h_tiles: usize| {
            if original_h == 0 {
                return false;
            }
            if original_w == 0 {
                return true;
            }
            let w_tiles = self.max_patches / h_tiles;
            h_tiles as u128 * original_w as u128 <= w_tiles as u128 * original_h as u128
        };

        let mut lower = 1usize;
        let mut upper = self.max_patches;
        let mut crossing = 0usize;
        while lower <= upper {
            let middle = lower + (upper - lower) / 2;
            if scale_is_height_limited(middle) {
                crossing = middle;
                lower = middle + 1;
            } else {
                upper = middle - 1;
            }
        }

        let candidates = [
            1,
            self.max_patches,
            crossing.max(1),
            crossing.saturating_add(1).min(self.max_patches),
        ];
        let mut best: Option<TileCandidate> = None;
        for h_tiles in candidates {
            let candidate = self.downscale_candidate(original_h, original_w, h_tiles);
            let replace = best.is_none_or(|current| {
                candidate.scale > current.scale
                    || (candidate.scale == current.scale && candidate.tiles < current.tiles)
            });
            if replace {
                best = Some(candidate);
            }
        }
        let best = best.expect("Llama4 downscale search has at least one candidate");
        self.min_tiles_within_scale_epsilon(original_h, original_w, best.scale)
            .unwrap_or(best.tiles)
    }

    fn downscale_candidate(
        &self,
        original_h: u32,
        original_w: u32,
        h_tiles: usize,
    ) -> TileCandidate {
        let w_tiles = self.max_patches / h_tiles;
        if original_h == 0 {
            return TileCandidate {
                scale: self.canvas_scale(original_h, original_w, 1, w_tiles),
                tiles: w_tiles,
            };
        }
        if original_w == 0 {
            return TileCandidate {
                scale: self.canvas_scale(original_h, original_w, h_tiles, 1),
                tiles: h_tiles,
            };
        }

        if h_tiles as u128 * original_w as u128 <= w_tiles as u128 * original_h as u128 {
            let min_w_tiles = (h_tiles as u128 * original_w as u128)
                .div_ceil(original_h as u128)
                .max(1) as usize;
            TileCandidate {
                scale: self.canvas_scale(original_h, original_w, h_tiles, min_w_tiles),
                tiles: h_tiles * min_w_tiles,
            }
        } else {
            let min_h_tiles = (w_tiles as u128 * original_h as u128)
                .div_ceil(original_w as u128)
                .max(1) as usize;
            TileCandidate {
                scale: self.canvas_scale(original_h, original_w, min_h_tiles, w_tiles),
                tiles: min_h_tiles * w_tiles,
            }
        }
    }

    fn canvas_scale(
        &self,
        original_h: u32,
        original_w: u32,
        h_tiles: usize,
        w_tiles: usize,
    ) -> f64 {
        let height_scale = if original_h == 0 {
            f64::INFINITY
        } else {
            h_tiles as f64 * self.tile_size as f64 / original_h as f64
        };
        let width_scale = if original_w == 0 {
            f64::INFINITY
        } else {
            w_tiles as f64 * self.tile_size as f64 / original_w as f64
        };
        height_scale.min(width_scale)
    }

    fn min_tiles_within_scale_epsilon(
        &self,
        original_h: u32,
        original_w: u32,
        selected_scale: f64,
    ) -> Option<usize> {
        const SCALE_EPSILON: f64 = 1e-9;

        let lower_bound = (selected_scale - SCALE_EPSILON).max(0.0);
        let h_tiles = if original_h == 0 {
            1
        } else {
            ((lower_bound * original_h as f64 / self.tile_size as f64).floor() as usize)
                .saturating_add(1)
        };
        let w_tiles = if original_w == 0 {
            1
        } else {
            ((lower_bound * original_w as f64 / self.tile_size as f64).floor() as usize)
                .saturating_add(1)
        };
        let tiles = h_tiles.checked_mul(w_tiles)?;
        if tiles > self.max_patches {
            return None;
        }
        let scale = self.canvas_scale(original_h, original_w, h_tiles, w_tiles);
        ((scale - selected_scale).abs() < SCALE_EPSILON).then_some(tiles)
    }
}

#[derive(Debug, Clone)]
pub(super) struct KimiK2Estimator {
    patch_size: usize,
    merge_size: usize,
    in_patch_limit: usize,
    patch_limit_on_one_side: usize,
}

impl KimiK2Estimator {
    fn from_config(_config: &PreprocessorConfig) -> Self {
        // The 1.7.0 registry constructs this counter with defaults and its
        // count method ignores `media_proc_cfg`. Preserve that quirk even
        // though the full image-preprocessing path applies those overrides.
        Self {
            patch_size: 14,
            merge_size: 2,
            in_patch_limit: 16_384,
            patch_limit_on_one_side: 512,
        }
    }

    #[inline(always)]
    fn count_tokens(&self, width: u32, height: u32) -> usize {
        let width = width as usize;
        let height = height as usize;
        let patches_w = (width / self.patch_size).max(1) as f64;
        let patches_h = (height / self.patch_size).max(1) as f64;

        let total_patch_scale = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let max_width_scale =
            (self.patch_limit_on_one_side * self.patch_size) as f64 / width as f64;
        let max_height_scale =
            (self.patch_limit_on_one_side * self.patch_size) as f64 / height as f64;
        let scale = f64::min(
            1.0,
            f64::min(
                total_patch_scale,
                f64::min(max_width_scale, max_height_scale),
            ),
        );

        let new_width = ((width as f64 * scale) as usize)
            .max(1)
            .min(self.patch_limit_on_one_side * self.patch_size);
        let new_height = ((height as f64 * scale) as usize)
            .max(1)
            .min(self.patch_limit_on_one_side * self.patch_size);
        let factor = self.patch_size * self.merge_size;
        let padded_width = new_width + (factor - new_width % factor) % factor;
        let padded_height = new_height + (factor - new_height % factor) % factor;

        (padded_height / factor) * (padded_width / factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama4_tile_count_reference(
        tile_size: usize,
        max_patches: usize,
        original_h: u32,
        original_w: u32,
    ) -> usize {
        let mut scales = Vec::new();
        for tiles in 1..=max_patches {
            for h_tiles in 1..=tiles {
                if tiles.is_multiple_of(h_tiles) {
                    let w_tiles = tiles / h_tiles;
                    let scale_h = h_tiles as f64 * tile_size as f64 / original_h as f64;
                    let scale_w = w_tiles as f64 * tile_size as f64 / original_w as f64;
                    scales.push((scale_h.min(scale_w), tiles));
                }
            }
        }
        let selected_scale = scales
            .iter()
            .filter(|(scale, _)| *scale >= 1.0)
            .map(|(scale, _)| *scale)
            .reduce(f64::min)
            .unwrap_or_else(|| {
                scales
                    .iter()
                    .map(|(scale, _)| *scale)
                    .fold(f64::NEG_INFINITY, f64::max)
            });
        scales
            .into_iter()
            .filter(|(scale, _)| (*scale - selected_scale).abs() < 1e-9)
            .map(|(_, tiles)| tiles)
            .min()
            .unwrap()
    }

    #[test]
    fn model_type_has_priority_over_model_id() {
        assert_eq!(
            ModelFamily::identify("Qwen/Qwen3-VL-2B", Some("llava_next")),
            Some(ModelFamily::LlavaNext)
        );
    }

    #[test]
    fn phi3_is_not_incidentally_supported() {
        assert_eq!(
            ModelFamily::identify("microsoft/Phi-3-vision-128k-instruct", Some("phi3_v")),
            None
        );
    }

    #[test]
    fn half_to_even_matches_python() {
        assert_eq!(round_half_to_even(12.5), 12.0);
        assert_eq!(round_half_to_even(13.5), 14.0);
    }

    #[test]
    fn llama4_tile_search_matches_reference_without_artificial_cap() {
        const DIMENSIONS: &[(u32, u32)] = &[
            (0, 1),
            (1, 0),
            (1, 1),
            (336, 336),
            (337, 511),
            (1_000, 100),
            (100, 1_000),
            (21_840, 336),
            (2_529_745_434, 1_689_440_957),
            (u32::MAX, 1),
        ];

        for max_patches in 1..=128 {
            let estimator = Llama4Estimator {
                tile_size: 336,
                max_patches,
            };
            for &(width, height) in DIMENSIONS {
                assert_eq!(
                    estimator.best_fit_tile_count(height, width),
                    llama4_tile_count_reference(336, max_patches, height, width),
                    "{width}x{height}, max_patches={max_patches}"
                );
            }
        }

        let estimator = Llama4Estimator {
            tile_size: 336,
            max_patches: 65,
        };
        assert_eq!(estimator.count_tokens(21_840, 336), 38_016);

        let epsilon_tie = Llama4Estimator {
            tile_size: 15,
            max_patches: 50,
        };
        assert_eq!(
            epsilon_tie.best_fit_tile_count(4_271_543_601, 2_305_564_504),
            45
        );
        assert_eq!(epsilon_tie.best_fit_tile_count(0, 0), 0);
    }

    #[test]
    fn typed_config_preserves_permissive_fields() {
        let config: PreprocessorConfig = serde_json::from_str(
            r#"{
                "patch_size": null,
                "merge_size": null,
                "size": null,
                "unknown_field": {"malformed": "ignored"}
            }"#,
        )
        .unwrap();

        assert_eq!(config.patch_size(14), 14);
        assert_eq!(config.merge_size, None);
        assert_eq!(config.target_height(), None);
    }

    #[test]
    fn typed_config_accepts_aliases_and_rejects_duplicates() {
        let config: PreprocessorConfig =
            serde_json::from_str(r#"{"norm_mean":[0.5],"norm_std":[0.25],"resample":3}"#).unwrap();
        assert_eq!(config.image_mean, Some(vec![0.5]));
        assert_eq!(config.image_std, Some(vec![0.25]));
        assert_eq!(config.resampling, Some(3));

        for duplicate in [
            r#"{"image_mean":[0.5],"norm_mean":[0.5]}"#,
            r#"{"image_std":[0.5],"norm_std":[0.5]}"#,
            r#"{"resampling":3,"resample":3}"#,
        ] {
            assert!(serde_json::from_str::<PreprocessorConfig>(duplicate).is_err());
        }
    }

    #[test]
    fn typed_patch_size_rejects_malformed_known_components() {
        for malformed in [
            r#"{"patch_size":{"height":null}}"#,
            r#"{"patch_size":{"height":"14"}}"#,
            r#"{"patch_size":{"width":4294967296}}"#,
        ] {
            assert!(serde_json::from_str::<PreprocessorConfig>(malformed).is_err());
        }

        let config: PreprocessorConfig =
            serde_json::from_str(r#"{"patch_size":{"height":16,"unknown":null}}"#).unwrap();
        assert_eq!(config.patch_size(14), 16);
    }

    #[test]
    fn nested_media_proc_config_preserves_1_7_0_overlay() {
        let nested: PreprocessorConfig = serde_json::from_str(
            r#"{
                "min_pixels": 65536,
                "media_proc_cfg": {"patch_size": 16, "merge_kernel_size": 2}
            }"#,
        )
        .unwrap();
        assert_eq!(QwenEstimator::qwen2(&nested).count_tokens(400, 400), 144);
        assert_eq!(LlavaEstimator::from_config(&nested).count_tokens(), 441);

        let top_level: PreprocessorConfig = serde_json::from_str(
            r#"{
                "patch_size": 14,
                "merge_size": 2,
                "min_pixels": 65536,
                "media_proc_cfg": {"patch_size": 16, "merge_kernel_size": 4}
            }"#,
        )
        .unwrap();
        assert_eq!(QwenEstimator::qwen2(&top_level).count_tokens(400, 400), 196);

        let malformed_nested: PreprocessorConfig = serde_json::from_str(
            r#"{"media_proc_cfg":{"patch_size":"16","merge_kernel_size":"2"}}"#,
        )
        .unwrap();
        assert_eq!(malformed_nested.patch_size(14), 14);
        assert_eq!(malformed_nested.merge_size(), None);
    }
}
