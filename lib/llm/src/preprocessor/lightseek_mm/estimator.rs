// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free image-token estimators used by MM-aware routing.
//!
//! The compatibility target is the image-counting behavior exposed by
//! `llm-multimodal` 1.7.0. Only the six algorithms Dynamo actually calls are
//! retained; image decoding, tensor construction, video, and model prompt
//! rewriting deliberately remain outside this module.

use serde::de::DeserializeOwned;
use serde_json::Value;

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
            "qwen3_vl" | "qwen3_5" | "qwen3_5_moe" | "qwen3_6" | "qwen3_6_moe" => Some(Self::Qwen3),
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
    pub(super) fn from_config(family: ModelFamily, config: &Value) -> anyhow::Result<Self> {
        validate_config_field_types(config)?;
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

    fn qwen2(config: &Value) -> Self {
        Self {
            patch_size: patch_size(config, 14),
            merge_size: usize_field(config, "merge_size").unwrap_or(2),
            min_pixels: usize_field(config, "min_pixels").unwrap_or(Self::QWEN2_MIN_PIXELS),
            max_pixels: usize_field(config, "max_pixels").unwrap_or(Self::QWEN2_MAX_PIXELS),
            temporal_patch_size: usize_field(config, "temporal_patch_size").unwrap_or(2),
        }
    }

    fn qwen3(config: &Value) -> Self {
        Self {
            patch_size: patch_size(config, 16),
            merge_size: usize_field(config, "merge_size").unwrap_or(2),
            min_pixels: usize_field(config, "min_pixels")
                .or_else(|| size_field(config, "shortest_edge"))
                .unwrap_or(Self::QWEN3_MIN_PIXELS),
            max_pixels: usize_field(config, "max_pixels")
                .or_else(|| size_field(config, "longest_edge"))
                .unwrap_or(Self::QWEN3_MAX_PIXELS),
            temporal_patch_size: usize_field(config, "temporal_patch_size").unwrap_or(2),
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
    fn from_config(config: &Value) -> Self {
        Self {
            patch_size: patch_size(config, 14),
            image_size: target_height(config).unwrap_or(336) as usize,
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

    fn from_config(config: &Value) -> Self {
        Self {
            image_size: target_height(config).unwrap_or(Self::BASE_IMAGE_SIZE) as usize,
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

impl Llama4Estimator {
    const PATCH_SIZE: usize = 14;
    const MAX_IMAGE_TILES: usize = 64;

    fn from_config(config: &Value) -> Self {
        Self {
            tile_size: size_field(config, "height").unwrap_or(336),
            max_patches: usize_field(config, "max_image_tiles").unwrap_or(16),
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
            (1..=Self::MAX_IMAGE_TILES).contains(&self.max_patches),
            "Llama4 max_image_tiles must be between 1 and {}",
            Self::MAX_IMAGE_TILES
        );
        self.tile_size
            .checked_mul(self.max_patches)
            .ok_or_else(|| anyhow::anyhow!("Llama4 tile canvas overflowed"))?;
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

    /// Allocation-free equivalent of the pinned processor's supported-canvas
    /// search. Only the tile product is needed by routing.
    fn best_fit_tile_count(&self, original_h: u32, original_w: u32) -> usize {
        let mut has_upscaling = false;
        let mut selected_scale = f64::NEG_INFINITY;

        for local_tiles in 1..=self.max_patches {
            for h_tiles in 1..=local_tiles {
                if !local_tiles.is_multiple_of(h_tiles) {
                    continue;
                }
                let w_tiles = local_tiles / h_tiles;
                let scale_w = (w_tiles * self.tile_size) as f64 / original_w as f64;
                let scale_h = (h_tiles * self.tile_size) as f64 / original_h as f64;
                let scale = scale_w.min(scale_h);
                if scale >= 1.0 {
                    if !has_upscaling || scale < selected_scale {
                        has_upscaling = true;
                        selected_scale = scale;
                    }
                } else if !has_upscaling && scale > selected_scale {
                    selected_scale = scale;
                }
            }
        }

        let mut best_tiles = 0;
        for local_tiles in 1..=self.max_patches {
            for h_tiles in 1..=local_tiles {
                if !local_tiles.is_multiple_of(h_tiles) {
                    continue;
                }
                let w_tiles = local_tiles / h_tiles;
                let scale_w = (w_tiles * self.tile_size) as f64 / original_w as f64;
                let scale_h = (h_tiles * self.tile_size) as f64 / original_h as f64;
                let scale = scale_w.min(scale_h);
                if (scale - selected_scale).abs() < 1e-9
                    && (best_tiles == 0 || local_tiles < best_tiles)
                {
                    best_tiles = local_tiles;
                }
            }
        }
        best_tiles
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
    fn from_config(_config: &Value) -> Self {
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

fn usize_field(config: &Value, field: &str) -> Option<usize> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

/// Validate every typed field recognized by the former 1.7.0 config parser.
/// Missing, null, and unknown fields preserve its permissive behavior, but a
/// malformed known field must disable MM routing instead of silently selecting
/// a different count.
fn validate_config_field_types(config: &Value) -> anyhow::Result<()> {
    let object = config
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("preprocessor config must be a JSON object"))?;

    for (canonical, alias) in [
        ("image_mean", "norm_mean"),
        ("image_std", "norm_std"),
        ("resampling", "resample"),
    ] {
        anyhow::ensure!(
            !(object.contains_key(canonical) && object.contains_key(alias)),
            "preprocessor config contains both {canonical} and its alias {alias}"
        );
    }

    for field in [
        "do_convert_rgb",
        "do_normalize",
        "do_pad",
        "do_rescale",
        "do_resize",
        "do_center_crop",
    ] {
        validate_optional_json_type::<bool>(config.get(field), field)?;
    }
    for field in [
        "image_processor_type",
        "im_start_token",
        "im_end_token",
        "slice_start_token",
        "slice_end_token",
        "vision_start_token",
        "vision_end_token",
    ] {
        validate_optional_json_type::<String>(config.get(field), field)?;
    }
    for field in ["image_mean", "norm_mean", "image_std", "norm_std"] {
        validate_optional_json_type::<Vec<f64>>(config.get(field), field)?;
    }
    validate_optional_json_type::<f64>(config.get("rescale_factor"), "rescale_factor")?;
    for field in [
        "resampling",
        "resample",
        "merge_size",
        "min_pixels",
        "max_pixels",
        "temporal_patch_size",
        "num_crops",
        "dynamic_hd",
        "max_image_tiles",
        "num_img_tokens",
    ] {
        validate_optional_usize(config.get(field), field)?;
    }
    validate_patch_size(config.get("patch_size"))?;
    validate_u32_map(config.get("size"), "size")?;
    validate_u32_map(config.get("crop_size"), "crop_size")?;
    Ok(())
}

fn validate_optional_json_type<T>(value: Option<&Value>, field: &str) -> anyhow::Result<()>
where
    T: DeserializeOwned,
{
    let Some(value) = value else {
        return Ok(());
    };
    serde_json::from_value::<Option<T>>(value.clone())
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("invalid {field}: {error}"))
}

fn validate_optional_usize(value: Option<&Value>, field: &str) -> anyhow::Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    let raw = value
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("{field} must be a non-negative integer or null"))?;
    usize::try_from(raw)
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("{field} exceeds usize"))
}

fn validate_patch_size(value: Option<&Value>) -> anyhow::Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    if let Some(raw) = value.as_u64() {
        anyhow::ensure!(raw <= u32::MAX as u64, "patch_size exceeds u32");
        return Ok(());
    }
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("patch_size must be an integer, object, or null"))?;
    for field in ["height", "width"] {
        let Some(component) = object.get(field) else {
            continue;
        };
        let raw = component
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("patch_size.{field} must be a non-negative integer"))?;
        anyhow::ensure!(raw <= u32::MAX as u64, "patch_size.{field} exceeds u32");
    }
    Ok(())
}

fn validate_u32_map(value: Option<&Value>, field: &str) -> anyhow::Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{field} must be an object or null"))?;
    for (key, component) in object {
        let raw = component
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("{field}.{key} must be a non-negative integer"))?;
        anyhow::ensure!(raw <= u32::MAX as u64, "{field}.{key} exceeds u32");
    }
    Ok(())
}

fn size_field(config: &Value, field: &str) -> Option<usize> {
    config.get("size").and_then(|size| usize_field(size, field))
}

fn patch_size(config: &Value, default: usize) -> usize {
    patch_size_optional(config).unwrap_or(default)
}

fn patch_size_optional(config: &Value) -> Option<usize> {
    let value = config.get("patch_size")?;
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| {
            value
                .get("height")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
        })
}

fn target_height(config: &Value) -> Option<u32> {
    let size = config.get("size")?;
    size.get("height")
        .or_else(|| size.get("shortest_edge"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .or(Some(224))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
