//! Shared base implementation for Qwen VL family image processors.
//!
//! This module provides a generic processor that handles the common logic
//! for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models. The specific variants
//! differ only in their default parameters (patch_size, normalization values).
//!
//! # Processing Pipeline
//!
//! 1. Validate aspect ratio (must be < 200:1)
//! 2. Smart resize to fit within min/max pixel bounds
//! 3. Align dimensions to (patch_size * merge_size) boundary
//! 4. Convert to tensor and normalize
//! 5. Reshape into patches for the vision encoder
//!
//! # Token Calculation
//!
//! ```text
//! grid_t = 1  (for images, temporal dimension is 1)
//! grid_h = resized_height / patch_size
//! grid_w = resized_width / patch_size
//! num_tokens = (grid_t * grid_h * grid_w) / merge_size²
//! ```

use std::borrow::Cow;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array2, Array3};

use crate::{
    types::RgbFrameRef,
    vision::{
        execution::{scope as parallel_scope, task_count},
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{
            par_threads, pil_to_filter, resize, resize_bicubic_pil, resize_bicubic_pil_rgb,
            resize_rgb_bytes, rgb_bytes, TransformError,
        },
    },
};

/// Python-compatible rounding (banker's rounding / round half to even).
///
/// This matches Python's `round()` behavior where 0.5 is rounded to the nearest
/// even number (e.g. `12.5 -> 12`, `13.5 -> 14`), unlike Rust's `f64::round()`
/// which rounds half away from zero.
#[inline]
fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    // Check if we're exactly at a .5 case
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        // Round to nearest even
        if rounded as i64 % 2 != 0 {
            return rounded - 1.0;
        }
    }
    rounded
}

/// Configuration for a Qwen VL processor variant.
#[derive(Debug, Clone)]
pub struct QwenVLConfig {
    /// Vision encoder patch size
    pub patch_size: usize,
    /// Merge size for token reduction
    pub merge_size: usize,
    /// Minimum total pixels allowed
    pub min_pixels: usize,
    /// Maximum total pixels allowed
    pub max_pixels: usize,
    /// Minimum video pixels, interpreted according to `video_resize_mode`.
    pub video_min_pixels: usize,
    /// Maximum video pixels, interpreted according to `video_resize_mode`.
    pub video_max_pixels: usize,
    /// Whether the video budget applies per frame or to the sampled volume.
    pub video_resize_mode: QwenVideoResizeMode,
    /// Temporal patch size for video
    pub temporal_patch_size: usize,
    /// Normalization mean values
    pub mean: [f64; 3],
    /// Normalization std values
    pub std: [f64; 3],
    /// Model name for identification
    pub model_name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenVideoResizeMode {
    TotalVolume,
    PerFrame,
}

#[derive(Clone)]
struct VideoFrameRgb<'a> {
    width: usize,
    height: usize,
    data: Cow<'a, [u8]>,
}

struct QwenImagePlan {
    target_width: u32,
    target_height: u32,
    needs_resize: bool,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    num_patches: usize,
    patch_values: usize,
    tokens: usize,
}

struct QwenVideoPlan {
    original_size: (u32, u32),
    target_width: u32,
    target_height: u32,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    patch_features: usize,
    num_patches: usize,
    output_values: usize,
    tokens: usize,
    second_per_grid: f32,
    filter: FilterType,
    do_resize: bool,
    lut: [[f32; 256]; 3],
}

fn normalization_lut(
    config: &PreProcessorConfig,
    default_mean: [f64; 3],
    default_std: [f64; 3],
) -> [[f32; 256]; 3] {
    let mean = config
        .image_mean
        .as_ref()
        .filter(|values| values.len() >= 3)
        .map(|values| [values[0], values[1], values[2]])
        .unwrap_or(default_mean);
    let std = config
        .image_std
        .as_ref()
        .filter(|values| values.len() >= 3)
        .map(|values| [values[0], values[1], values[2]])
        .unwrap_or(default_std);
    let do_normalize = config.do_normalize.unwrap_or(true);
    let scale: [f32; 3] = if do_normalize {
        std::array::from_fn(|channel| 1.0 / (255.0 * std[channel] as f32))
    } else {
        [1.0 / 255.0; 3]
    };
    let bias: [f32; 3] = if do_normalize {
        std::array::from_fn(|channel| -(mean[channel] as f32) / std[channel] as f32)
    } else {
        [0.0; 3]
    };
    std::array::from_fn(|channel| {
        std::array::from_fn(|value| value as f32 * scale[channel] + bias[channel])
    })
}

fn dispatch_patch_blocks(
    region: &mut [f32],
    n_blocks: usize,
    block_out: usize,
    write_blocks: impl Fn(usize, &mut [f32]) + Sync,
) {
    let nthreads = par_threads(size_of_val(region), n_blocks);
    if nthreads <= 1 {
        write_blocks(0, region);
        return;
    }

    let chunk_blocks = n_blocks.div_ceil(nthreads);
    parallel_scope(|scope| {
        let write_blocks = &write_blocks;
        let mut rest = region;
        let mut block_start = 0usize;
        while block_start < n_blocks {
            let blocks = chunk_blocks.min(n_blocks - block_start);
            let (band, tail) = rest.split_at_mut(blocks * block_out);
            rest = tail;
            let start = block_start;
            scope.spawn(move |_| write_blocks(start, band));
            block_start += blocks;
        }
    });
}

fn resize_rgb_frame_to_raw(
    frame: RgbFrameRef<'_>,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<Vec<u8>, TransformError> {
    // BICUBIC (Qwen default) uses the PIL-compatible path, same as the image
    // path; other filters keep the SIMD resizer.
    let resized = if filter == FilterType::CatmullRom {
        resize_bicubic_pil_rgb(
            frame.data,
            frame.width,
            frame.height,
            target_width,
            target_height,
        )?
    } else {
        resize_rgb_bytes(
            frame.data,
            frame.width,
            frame.height,
            target_width,
            target_height,
            filter,
        )?
    };
    Ok(resized.into_raw())
}

fn prepare_video_rgb_frame<'a>(
    frame: RgbFrameRef<'a>,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
    do_resize: bool,
) -> Result<VideoFrameRgb<'a>, TransformError> {
    if do_resize && (frame.width != target_width || frame.height != target_height) {
        Ok(VideoFrameRgb {
            width: target_width as usize,
            height: target_height as usize,
            data: Cow::Owned(resize_rgb_frame_to_raw(
                frame,
                target_width,
                target_height,
                filter,
            )?),
        })
    } else {
        Ok(VideoFrameRgb {
            width: frame.width as usize,
            height: frame.height as usize,
            data: Cow::Borrowed(frame.data),
        })
    }
}

fn prepare_video_rgb_chunk<'a>(
    frames: &[RgbFrameRef<'a>],
    temporal_index: usize,
    temporal_patch_size: usize,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
    do_resize: bool,
) -> Result<Vec<VideoFrameRgb<'a>>, TransformError> {
    let mut prepared = Vec::with_capacity(temporal_patch_size);
    prepare_video_frame_chunk(
        frames.len(),
        temporal_index,
        temporal_patch_size,
        &mut prepared,
        |frame_index| {
            prepare_video_rgb_frame(
                frames[frame_index],
                target_width,
                target_height,
                filter,
                do_resize,
            )
        },
    )?;
    Ok(prepared)
}

fn prepare_video_frame_chunk<'a>(
    frame_count: usize,
    temporal_index: usize,
    temporal_patch_size: usize,
    prepared: &mut Vec<VideoFrameRgb<'a>>,
    mut prepare_frame: impl FnMut(usize) -> Result<VideoFrameRgb<'a>, TransformError>,
) -> Result<(), TransformError> {
    prepared.clear();
    let mut previous_frame_index = None;
    for temporal_offset in 0..temporal_patch_size {
        let frame_index =
            (temporal_index * temporal_patch_size + temporal_offset).min(frame_count - 1);
        if previous_frame_index == Some(frame_index) {
            if let Some(previous) = prepared.last().cloned() {
                prepared.push(previous);
                continue;
            }
        }
        prepared.push(prepare_frame(frame_index)?);
        previous_frame_index = Some(frame_index);
    }
    Ok(())
}

fn resize_dynamic_frame_to_raw(
    frame: &DynamicImage,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> (usize, usize, Vec<u8>) {
    let resized = if filter == FilterType::CatmullRom {
        resize_bicubic_pil(frame, target_width, target_height)
    } else {
        resize(frame, target_width, target_height, filter)
    };
    let (width, height, data) = rgb_bytes(&resized);
    (width, height, data.into_owned())
}

/// Generic Qwen VL image processor.
///
/// This struct implements the shared preprocessing logic for all Qwen VL
/// model variants. Each variant (Qwen2-VL, Qwen3-VL, etc.) uses this with
/// different configuration values.
#[derive(Debug, Clone)]
pub struct QwenVLProcessorBase {
    config: QwenVLConfig,
}

impl QwenVLProcessorBase {
    /// Create a new processor with the given configuration.
    pub fn new(config: QwenVLConfig) -> Self {
        Self { config }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.config.merge_size
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.config.min_pixels
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.config.max_pixels
    }

    pub fn video_min_pixels(&self) -> usize {
        self.config.video_min_pixels
    }

    pub fn video_max_pixels(&self) -> usize {
        self.config.video_max_pixels
    }

    pub fn video_resize_mode(&self) -> QwenVideoResizeMode {
        self.config.video_resize_mode
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.config.temporal_patch_size
    }

    fn plan_video(
        &self,
        frame_count: usize,
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<QwenVideoPlan, TransformError> {
        let temporal_patch_size = self.config.temporal_patch_size;
        let padded_frames = frame_count.div_ceil(temporal_patch_size) * temporal_patch_size;
        let (target_height, target_width) =
            self.smart_resize_video(frame_count, height as usize, width as usize)?;
        let (grid_t, grid_h, grid_w) =
            self.calculate_grid_thw(target_height, target_width, padded_frames);
        let patch_features =
            3 * temporal_patch_size * self.config.patch_size * self.config.patch_size;
        let num_patches = grid_t
            .checked_mul(grid_h)
            .and_then(|value| value.checked_mul(grid_w))
            .ok_or_else(|| {
                TransformError::ShapeError(format!(
                    "Qwen video patch count overflow: grid=({grid_t}, {grid_h}, {grid_w})"
                ))
            })?;
        let output_values = num_patches.checked_mul(patch_features).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "Qwen video patch buffer size overflow: patches={num_patches}, features={patch_features}"
            ))
        })?;
        // MediaConnector samples video at 2 fps by default. A checkpoint may
        // override that value in video_preprocessor_config.json.
        let sample_fps = config.get_extra::<f32>("fps").unwrap_or(2.0);
        if !sample_fps.is_finite() || sample_fps <= 0.0 {
            return Err(TransformError::ShapeError(format!(
                "Qwen video fps must be finite and positive, got {sample_fps}"
            )));
        }

        Ok(QwenVideoPlan {
            original_size: (width, height),
            target_width: target_width as u32,
            target_height: target_height as u32,
            grid_t,
            grid_h,
            grid_w,
            patch_features,
            num_patches,
            output_values,
            tokens: self.calculate_tokens_from_grid(grid_t, grid_h, grid_w),
            second_per_grid: temporal_patch_size as f32 / sample_fps,
            filter: pil_to_filter(config.resampling.or(Some(3))),
            do_resize: config.do_resize.unwrap_or(true),
            lut: normalization_lut(config, self.config.mean, self.config.std),
        })
    }

    fn finish_video(
        plan: QwenVideoPlan,
        patches: Vec<f32>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let encoder_input =
            Array2::from_shape_vec((plan.num_patches, plan.patch_features), patches).map_err(
                |error| {
                    TransformError::ShapeError(format!(
                        "Failed to create video encoder_input [{}, {}]: {error}",
                        plan.num_patches, plan.patch_features
                    ))
                },
            )?;

        Ok(PreprocessedEncoderInputs::new(
            encoder_input,
            vec![plan.tokens],
            vec![plan.original_size],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(
                vec![plan.grid_t as i64, plan.grid_h as i64, plan.grid_w as i64],
                1,
                3,
            ),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![plan.num_patches as i64]),
        )
        .with_extra(
            "patches_per_image",
            ModelSpecificValue::int_1d(vec![plan.num_patches as i64]),
        )
        .with_extra(
            "video_second_per_grid",
            ModelSpecificValue::Tensor {
                data: vec![plan.second_per_grid],
                shape: vec![1],
            },
        ))
    }

    /// Get the factor for dimension alignment.
    ///
    /// Dimensions must be divisible by (patch_size * merge_size).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.config.patch_size * self.config.merge_size
    }

    /// Smart resize algorithm for Qwen VL models.
    ///
    /// Resizes image dimensions to fit within min/max pixel bounds while:
    /// - Preserving aspect ratio
    /// - Aligning to (patch_size * merge_size) boundaries
    ///
    /// # Arguments
    /// * `height` - Original image height
    /// * `width` - Original image width
    ///
    /// # Returns
    /// (new_height, new_width) or error if aspect ratio is too extreme
    ///
    /// # Errors
    /// - If height or width is zero
    /// - If aspect ratio exceeds 200:1
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.get_factor();

        // Validate non-zero dimensions
        if height == 0 || width == 0 {
            return Err(TransformError::InvalidShape {
                expected: "non-zero dimensions".to_string(),
                actual: vec![height, width],
            });
        }

        // Validate aspect ratio
        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        let aspect_ratio = max_dim / min_dim;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        // Round to nearest factor multiple using Python-compatible rounding
        // Python uses banker's rounding (round half to even), which affects
        // edge cases like 400/32 = 12.5 -> 12 (not 13)
        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;

        // Ensure minimum size
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        // Scale down if exceeding max_pixels
        if h_bar * w_bar > self.config.max_pixels {
            let beta = ((height * width) as f64 / self.config.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            // Ensure minimum size after scaling down
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        }
        // Scale up if below min_pixels
        else if h_bar * w_bar < self.config.min_pixels {
            let beta = (self.config.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    /// Smart resize for Qwen3-style video processors.
    ///
    /// `TotalVolume` applies the pixel budget to the padded sampled video
    /// volume (`T * H * W`), while `PerFrame` applies it to each frame's
    /// spatial area (`H * W`).
    pub fn smart_resize_video(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.get_factor();

        if num_frames == 0 {
            return Err(TransformError::InvalidShape {
                expected: "num_frames > 0".to_string(),
                actual: vec![num_frames],
            });
        }

        if height < factor || width < factor {
            return Err(TransformError::InvalidShape {
                expected: format!("height and width >= factor ({factor})"),
                actual: vec![height, width],
            });
        }

        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        let aspect_ratio = max_dim / min_dim;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        let (budget_scale, resized_pixels) = match self.config.video_resize_mode {
            QwenVideoResizeMode::TotalVolume => {
                let padded_frames = num_frames.div_ceil(self.config.temporal_patch_size)
                    * self.config.temporal_patch_size;
                (
                    num_frames as f64,
                    padded_frames as f64 * h_bar as f64 * w_bar as f64,
                )
            }
            QwenVideoResizeMode::PerFrame => (1.0, h_bar as f64 * w_bar as f64),
        };
        let source_pixels = budget_scale * height as f64 * width as f64;
        if resized_pixels > self.config.video_max_pixels as f64 {
            let beta = (source_pixels / self.config.video_max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        } else if resized_pixels < self.config.video_min_pixels as f64 {
            let beta = (self.config.video_min_pixels as f64 / source_pixels).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    ///
    /// For single images, T=1. For video, T = num_frames / temporal_patch_size.
    ///
    /// # Arguments
    /// * `height` - Resized image height
    /// * `width` - Resized image width
    /// * `num_frames` - Number of frames (1 for images)
    ///
    /// # Returns
    /// (grid_t, grid_h, grid_w)
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        let grid_t =
            num_frames.max(self.config.temporal_patch_size) / self.config.temporal_patch_size;
        let grid_h = height / self.config.patch_size;
        let grid_w = width / self.config.patch_size;
        (grid_t, grid_h, grid_w)
    }

    /// Calculate the number of image tokens after merge.
    ///
    /// tokens = (grid_t * grid_h * grid_w) / merge_size²
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        (grid_t * grid_h * grid_w) / (self.config.merge_size * self.config.merge_size)
    }

    /// Patchify tensor directly into an output buffer (avoids intermediate Vec allocation).
    /// Patchify a [C, H, W] tensor and append the patches to `output`.
    ///
    /// Output layout per image:
    ///   `[grid_t, patch_rows, patch_cols, merge_h, merge_w, C, temporal, patch_h, patch_w]`
    ///
    /// Each "merged patch" covers a `(merge_size * patch_size)²` spatial region.
    /// Within it, `merge_size²` sub-patches are emitted, each containing all channels.
    pub fn patchify_into(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        let channel = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(
            height,
            grid_h * patch_size,
            "Height must match grid_h * patch_size"
        );
        debug_assert_eq!(
            width,
            grid_w * patch_size,
            "Width must match grid_w * patch_size"
        );

        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;
        let base_idx = output.len();
        output.resize(base_idx + num_patches * patch_features, 0.0);

        let data = tensor.as_standard_layout();
        let flat = data.as_slice().ok_or_else(|| {
            TransformError::ShapeError("tensor not contiguous after as_standard_layout".to_string())
        })?;
        let planes: Vec<&[f32]> = (0..channel)
            .map(|c| &flat[c * height * width..(c + 1) * height * width])
            .collect();

        let merged_patch = merge_size * patch_size;
        let pr_blocks = grid_h / merge_size;
        let pc_blocks = grid_w / merge_size;
        let n_blocks = grid_t * pr_blocks * pc_blocks;
        let block_out = merge_size * merge_size * patch_features;
        // Each (gt,pr,pc) block writes a contiguous, deterministic output
        // region of pure copies, so banding blocks across threads is
        // BIT-IDENTICAL. Small grids stay serial.
        let region = &mut output[base_idx..base_idx + n_blocks * block_out];
        dispatch_patch_blocks(region, n_blocks, block_out, |block_start, band| {
            Self::patchify_block_band(
                &planes,
                width,
                patch_size,
                merge_size,
                temporal_patch_size,
                merged_patch,
                pr_blocks,
                pc_blocks,
                block_start,
                band,
            );
        });

        Ok(())
    }

    /// Fill `band` with the patchified output for blocks
    /// `[block_start, block_start + band.len()/block_out)` in (gt, pr, pc)
    /// row-major order. Pure gather/copy from `planes`; deterministic and
    /// independent per block (safe to call concurrently on disjoint bands).
    #[expect(
        clippy::too_many_arguments,
        reason = "block-band patchifier: planes + grid dims + output band"
    )]
    fn patchify_block_band(
        planes: &[&[f32]],
        width: usize,
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        merged_patch: usize,
        pr_blocks: usize,
        pc_blocks: usize,
        block_start: usize,
        band: &mut [f32],
    ) {
        let block_out =
            merge_size * merge_size * planes.len() * temporal_patch_size * patch_size * patch_size;
        let per_t = pr_blocks * pc_blocks;
        for (bi, chunk) in band.chunks_mut(block_out).enumerate() {
            let blk = block_start + bi;
            let rem = blk % per_t;
            let pr = rem / pc_blocks;
            let pc = rem % pc_blocks;
            let y0 = pr * merged_patch;
            let x0 = pc * merged_patch;
            let mut o = 0usize;
            for mh in 0..merge_size {
                for mw in 0..merge_size {
                    for plane in planes {
                        for _tp in 0..temporal_patch_size {
                            for py in 0..patch_size {
                                let row =
                                    (y0 + mh * patch_size + py) * width + x0 + mw * patch_size;
                                chunk[o..o + patch_size]
                                    .copy_from_slice(&plane[row..row + patch_size]);
                                o += patch_size;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Patchify a sequence of frame tensors into Qwen's video patch layout.
    ///
    /// `tensors` must already be resized and normalized to the same spatial
    /// shape. If the frame count is not divisible by `temporal_patch_size`,
    /// the caller should pad by repeating the final frame before calling this.
    pub fn patchify_video_into(
        &self,
        tensors: &[Array3<f32>],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        if tensors.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let channel = tensors[0].shape()[0];
        let height = tensors[0].shape()[1];
        let width = tensors[0].shape()[2];
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(height, grid_h * patch_size);
        debug_assert_eq!(width, grid_w * patch_size);
        debug_assert_eq!(tensors.len(), grid_t * temporal_patch_size);

        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;
        let base_idx = output.len();
        output.resize(base_idx + num_patches * patch_features, 0.0);

        let frame_planes: Vec<Vec<&[f32]>> = tensors
            .iter()
            .map(|tensor| {
                let flat = tensor.as_slice().ok_or_else(|| {
                    TransformError::ShapeError("video frame tensor is not contiguous".to_string())
                })?;
                Ok((0..channel)
                    .map(|c| &flat[c * height * width..(c + 1) * height * width])
                    .collect::<Vec<_>>())
            })
            .collect::<Result<_, TransformError>>()?;

        let merged_patch = merge_size * patch_size;
        let mut out_idx = base_idx;

        for gt in 0..grid_t {
            let frame_start = gt * temporal_patch_size;
            for pr in 0..grid_h / merge_size {
                for pc in 0..grid_w / merge_size {
                    let y0 = pr * merged_patch;
                    let x0 = pc * merged_patch;

                    for mh in 0..merge_size {
                        for mw in 0..merge_size {
                            let frame_window =
                                &frame_planes[frame_start..frame_start + temporal_patch_size];
                            for channel_frames in (0..channel).map(|channel_idx| {
                                frame_window.iter().map(move |planes| planes[channel_idx])
                            }) {
                                for plane in channel_frames {
                                    for py in 0..patch_size {
                                        let row = (y0 + mh * patch_size + py) * width
                                            + x0
                                            + mw * patch_size;
                                        output[out_idx..out_idx + patch_size]
                                            .copy_from_slice(&plane[row..row + patch_size]);
                                        out_idx += patch_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn patchify_video_rgb_chunk_into(
        &self,
        frames: &[VideoFrameRgb<'_>],
        grid_h: usize,
        grid_w: usize,
        output: &mut [f32],
        out_idx: &mut usize,
        lut: &[[f32; 256]; 3],
    ) -> Result<(), TransformError> {
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;
        if frames.len() != temporal_patch_size {
            return Err(TransformError::InvalidShape {
                expected: format!("{temporal_patch_size} video frames in temporal patch"),
                actual: vec![frames.len()],
            });
        }

        let height = grid_h * patch_size;
        let width = grid_w * patch_size;
        for frame in frames {
            if frame.height != height || frame.width != width {
                return Err(TransformError::InvalidShape {
                    expected: format!("video frame size {width}x{height}"),
                    actual: vec![frame.width, frame.height],
                });
            }
        }

        let merged_patch = merge_size * patch_size;
        let pr_blocks = grid_h / merge_size;
        let pc_blocks = grid_w / merge_size;
        let n_blocks = pr_blocks.checked_mul(pc_blocks).ok_or_else(|| {
            TransformError::ShapeError("Qwen video patch block count overflow".to_string())
        })?;
        let block_out = merge_size * merge_size * 3 * temporal_patch_size * patch_size * patch_size;
        let base_idx = *out_idx;
        let patch_values = n_blocks.checked_mul(block_out).ok_or_else(|| {
            TransformError::ShapeError("Qwen video patch output size overflow".to_string())
        })?;
        let end_idx = base_idx.checked_add(patch_values).ok_or_else(|| {
            TransformError::ShapeError("Qwen video patch output range overflow".to_string())
        })?;
        let region = output.get_mut(base_idx..end_idx).ok_or_else(|| {
            TransformError::ShapeError("Qwen video patch output range out of bounds".to_string())
        })?;
        dispatch_patch_blocks(region, n_blocks, block_out, |block_start, band| {
            Self::patchify_video_rgb_block_band(
                frames,
                width,
                patch_size,
                merge_size,
                merged_patch,
                pc_blocks,
                block_start,
                band,
                lut,
            );
        });
        *out_idx = end_idx;

        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "RGB video patchifier: frame window + grid dims + output band"
    )]
    fn patchify_video_rgb_block_band(
        frames: &[VideoFrameRgb<'_>],
        width: usize,
        patch_size: usize,
        merge_size: usize,
        merged_patch: usize,
        pc_blocks: usize,
        block_start: usize,
        band: &mut [f32],
        lut: &[[f32; 256]; 3],
    ) {
        let block_out = merge_size * merge_size * 3 * frames.len() * patch_size * patch_size;
        for (bi, chunk) in band.chunks_mut(block_out).enumerate() {
            let blk = block_start + bi;
            let pr = blk / pc_blocks;
            let pc = blk % pc_blocks;
            let y0 = pr * merged_patch;
            let x0 = pc * merged_patch;
            let mut o = 0usize;

            for mh in 0..merge_size {
                for mw in 0..merge_size {
                    for (c, lut_c) in lut.iter().enumerate().take(3) {
                        for frame in frames {
                            let raw = frame.data.as_ref();
                            for py in 0..patch_size {
                                let row =
                                    (y0 + mh * patch_size + py) * width + x0 + mw * patch_size;
                                let source_start = row * 3;
                                let source_end = (row + patch_size) * 3;
                                for (dst, pixel) in chunk[o..o + patch_size]
                                    .iter_mut()
                                    .zip(raw[source_start..source_end].chunks_exact(3))
                                {
                                    *dst = lut_c[pixel[c] as usize];
                                }
                                o += patch_size;
                            }
                        }
                    }
                }
            }
        }
    }

    fn patchify_image_rgb_into(
        &self,
        image: &DynamicImage,
        grid_h: usize,
        grid_w: usize,
        output: &mut [f32],
        out_idx: &mut usize,
        lut: &[[f32; 256]; 3],
    ) -> Result<(), TransformError> {
        let (width, height, data) = rgb_bytes(image);
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;
        let expected_height = grid_h * patch_size;
        let expected_width = grid_w * patch_size;
        if height != expected_height || width != expected_width {
            return Err(TransformError::InvalidShape {
                expected: format!("image size {expected_width}x{expected_height}"),
                actual: vec![width, height],
            });
        }

        let raw = data.as_ref();
        let merged_patch = merge_size * patch_size;
        let pr_blocks = grid_h / merge_size;
        let pc_blocks = grid_w / merge_size;
        let n_blocks = pr_blocks.checked_mul(pc_blocks).ok_or_else(|| {
            TransformError::ShapeError("Qwen image patch block count overflow".to_string())
        })?;
        let block_out = merge_size * merge_size * 3 * temporal_patch_size * patch_size * patch_size;
        let base_idx = *out_idx;
        let patch_values = n_blocks.checked_mul(block_out).ok_or_else(|| {
            TransformError::ShapeError("Qwen image patch output size overflow".to_string())
        })?;
        let end_idx = base_idx.checked_add(patch_values).ok_or_else(|| {
            TransformError::ShapeError("Qwen image patch output range overflow".to_string())
        })?;
        let region = output.get_mut(base_idx..end_idx).ok_or_else(|| {
            TransformError::ShapeError("Qwen image patch output range out of bounds".to_string())
        })?;
        dispatch_patch_blocks(region, n_blocks, block_out, |block_start, band| {
            Self::patchify_image_rgb_block_band(
                raw,
                width,
                patch_size,
                merge_size,
                temporal_patch_size,
                merged_patch,
                pc_blocks,
                block_start,
                band,
                lut,
            );
        });
        *out_idx = end_idx;

        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "RGB image patchifier: raw bytes + grid dims + output band"
    )]
    fn patchify_image_rgb_block_band(
        raw: &[u8],
        width: usize,
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        merged_patch: usize,
        pc_blocks: usize,
        block_start: usize,
        band: &mut [f32],
        lut: &[[f32; 256]; 3],
    ) {
        let block_out = merge_size * merge_size * 3 * temporal_patch_size * patch_size * patch_size;
        for (bi, chunk) in band.chunks_mut(block_out).enumerate() {
            let blk = block_start + bi;
            let pr = blk / pc_blocks;
            let pc = blk % pc_blocks;
            let y0 = pr * merged_patch;
            let x0 = pc * merged_patch;
            let mut o = 0usize;

            for mh in 0..merge_size {
                for mw in 0..merge_size {
                    for (c, lut_c) in lut.iter().enumerate().take(3) {
                        for _tp in 0..temporal_patch_size {
                            for py in 0..patch_size {
                                let row =
                                    (y0 + mh * patch_size + py) * width + x0 + mw * patch_size;
                                let mut src_idx = row * 3 + c;
                                for dst in &mut chunk[o..o + patch_size] {
                                    *dst = lut_c[raw[src_idx] as usize];
                                    src_idx += 3;
                                }
                                o += patch_size;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl VisionPreProcessor for QwenVLProcessorBase {
    fn default_mean(&self) -> [f64; 3] {
        self.config.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.config.std
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        // Qwen2VL/Qwen3VL image processors default to BICUBIC (PIL resample=3)
        // when the preprocessor config omits `resample`. The global pil_to_filter
        // fallback is bilinear, which yields smoother features and measurably
        // degrades VLM accuracy, so pin the HF-correct default here.
        let filter = pil_to_filter(config.resampling.or(Some(3)));

        let patch_size = self.config.patch_size;
        let temporal_patch_size = self.config.temporal_patch_size;
        let patch_features = 3 * temporal_patch_size * patch_size * patch_size;
        let do_resize = config.do_resize.unwrap_or(true);
        let lut = normalization_lut(config, self.config.mean, self.config.std);

        let mut image_plans = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut total_patch_values = 0usize;
        let mut total_patches = 0usize;
        for image in images {
            let (w, h) = image.dimensions();
            item_sizes.push((w, h));
            let (target_h, target_w) = self.smart_resize(h as usize, w as usize)?;
            let (tw32, th32) = (target_w as u32, target_h as u32);
            let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, 1);
            let num_patches = grid_t
                .checked_mul(grid_h)
                .and_then(|value| value.checked_mul(grid_w))
                .ok_or_else(|| {
                    TransformError::ShapeError(format!(
                        "Qwen image patch count overflow: grid=({grid_t}, {grid_h}, {grid_w})"
                    ))
                })?;
            total_patches = total_patches.checked_add(num_patches).ok_or_else(|| {
                TransformError::ShapeError("Qwen image total patch count overflow".to_string())
            })?;
            let patch_values = num_patches.checked_mul(patch_features).ok_or_else(|| {
                TransformError::ShapeError(format!(
                    "Qwen image patch buffer size overflow: patches={num_patches}, features={patch_features}"
                ))
            })?;
            total_patch_values = total_patch_values
                .checked_add(patch_values)
                .ok_or_else(|| {
                    TransformError::ShapeError(
                        "Qwen image patch buffer total size overflow".to_string(),
                    )
                })?;
            image_plans.push(QwenImagePlan {
                target_width: tw32,
                target_height: th32,
                needs_resize: do_resize && (w != tw32 || h != th32),
                grid_t,
                grid_h,
                grid_w,
                num_patches,
                patch_values,
                tokens: self.calculate_tokens_from_grid(grid_t, grid_h, grid_w),
            });
        }

        let mut all_patches: Vec<f32> = Vec::with_capacity(total_patch_values);
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut feature_token_counts = Vec::with_capacity(images.len());

        for (image, plan) in images.iter().zip(image_plans) {
            // Resize to the image's own target size (skip if dimensions match)
            let resized;
            let img_ref = if plan.needs_resize {
                // BICUBIC (Qwen default) uses the PIL-compatible path; other
                // filters keep the SIMD path.
                resized = if filter == FilterType::CatmullRom {
                    resize_bicubic_pil(image, plan.target_width, plan.target_height)
                } else {
                    resize(image, plan.target_width, plan.target_height, filter)
                };
                &resized
            } else {
                image
            };

            grid_thw_data.push(plan.grid_t as i64);
            grid_thw_data.push(plan.grid_h as i64);
            grid_thw_data.push(plan.grid_w as i64);

            feature_token_counts.push(plan.tokens);

            // Patchify directly from RGB bytes to avoid the intermediate
            // [C,H,W] tensor allocation. This matches the tensor path's
            // channel/temporal/spatial order.
            let base_idx = all_patches.len();
            all_patches.resize(base_idx + plan.patch_values, 0.0);
            let mut out_idx = base_idx;
            self.patchify_image_rgb_into(
                img_ref,
                plan.grid_h,
                plan.grid_w,
                &mut all_patches,
                &mut out_idx,
                &lut,
            )?;
            debug_assert_eq!(out_idx, all_patches.len());
            patches_per_image.push(plan.num_patches as i64);
        }

        let encoder_input =
            Array2::from_shape_vec((total_patches, patch_features), all_patches).map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create patchified encoder_input [{total_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result =
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "image_grid_thw",
                    ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                );

        Ok(result)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let (width, height) = frames[0].dimensions();
        let plan = self.plan_video(frames.len(), width, height, config)?;
        let temporal_patch_size = self.config.temporal_patch_size;
        let mut all_patches = vec![0.0; plan.output_values];
        let mut out_idx = 0;
        let mut frame_rgbs = Vec::with_capacity(temporal_patch_size);
        for gt in 0..plan.grid_t {
            prepare_video_frame_chunk(
                frames.len(),
                gt,
                temporal_patch_size,
                &mut frame_rgbs,
                |frame_index| {
                    let frame = &frames[frame_index];
                    let needs_resize = plan.do_resize
                        && (frame.width() != plan.target_width
                            || frame.height() != plan.target_height);
                    if needs_resize {
                        let (width, height, data) = resize_dynamic_frame_to_raw(
                            frame,
                            plan.target_width,
                            plan.target_height,
                            plan.filter,
                        );
                        Ok(VideoFrameRgb {
                            width,
                            height,
                            data: Cow::Owned(data),
                        })
                    } else {
                        let (width, height, data) = rgb_bytes(frame);
                        Ok(VideoFrameRgb {
                            width,
                            height,
                            data,
                        })
                    }
                },
            )?;

            self.patchify_video_rgb_chunk_into(
                &frame_rgbs,
                plan.grid_h,
                plan.grid_w,
                &mut all_patches,
                &mut out_idx,
                &plan.lut,
            )?;
        }
        debug_assert_eq!(out_idx, all_patches.len());
        Self::finish_video(plan, all_patches)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let plan = self.plan_video(frames.len(), frames[0].width, frames[0].height, config)?;
        let temporal_patch_size = self.config.temporal_patch_size;
        let mut all_patches = vec![0.0; plan.output_values];
        for frame in frames {
            let expected_len = (frame.width as usize)
                .checked_mul(frame.height as usize)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| {
                    TransformError::ShapeError(format!(
                        "video frame dimensions are too large: {}x{}",
                        frame.width, frame.height
                    ))
                })?;
            if frame.data.len() != expected_len {
                return Err(TransformError::InvalidShape {
                    expected: format!(
                        "RGB frame byte length {expected_len} for {}x{}",
                        frame.width, frame.height
                    ),
                    actual: vec![frame.data.len()],
                });
            }
        }

        let values_per_group = plan.grid_h * plan.grid_w * plan.patch_features;
        let parallel_tasks = task_count(all_patches.len() * size_of::<f32>(), plan.grid_t, 1);
        let groups_per_task = plan.grid_t.div_ceil(parallel_tasks);
        let mut errors = (0..parallel_tasks).map(|_| None).collect::<Vec<_>>();
        parallel_scope(|scope| {
            for (task_index, (output_band, error_slot)) in all_patches
                .chunks_mut(groups_per_task * values_per_group)
                .zip(errors.iter_mut())
                .enumerate()
            {
                let first_group = task_index * groups_per_task;
                scope.spawn(move |_| {
                    let outcome = (|| {
                        for (group_offset, group_output) in
                            output_band.chunks_mut(values_per_group).enumerate()
                        {
                            let temporal_index = first_group + group_offset;
                            let prepared = prepare_video_rgb_chunk(
                                frames,
                                temporal_index,
                                temporal_patch_size,
                                plan.target_width,
                                plan.target_height,
                                plan.filter,
                                plan.do_resize,
                            )?;
                            let mut output_index = 0;
                            self.patchify_video_rgb_chunk_into(
                                &prepared,
                                plan.grid_h,
                                plan.grid_w,
                                group_output,
                                &mut output_index,
                                &plan.lut,
                            )?;
                            debug_assert_eq!(output_index, group_output.len());
                        }
                        Ok::<_, TransformError>(())
                    })();
                    *error_slot = outcome.err();
                });
            }
        });
        if let Some(error) = errors.into_iter().flatten().next() {
            return Err(error);
        }

        Self::finish_video(plan, all_patches)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        // Calculate resized dimensions
        let (new_height, new_width) = match self.smart_resize(height as usize, width as usize) {
            Ok((h, w)) => (h, w),
            Err(_) => {
                // Fallback: use minimum size
                let factor = self.get_factor();
                (factor, factor)
            }
        };

        // Calculate grid and tokens
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(new_height, new_width, 1);
        self.calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }

    fn model_name(&self) -> &'static str {
        self.config.model_name
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Qwen VL models have dynamic sizing, no fixed output size
        None
    }
}

#[cfg(test)]
mod tests {
    use image::RgbImage;

    use super::*;
    use crate::vision::transforms::to_tensor_and_normalize;

    fn create_test_config() -> QwenVLConfig {
        QwenVLConfig {
            patch_size: 14,
            merge_size: 2,
            min_pixels: 256 * 28 * 28,
            max_pixels: 1280 * 28 * 28,
            video_min_pixels: 256 * 28 * 28,
            video_max_pixels: 1280 * 28 * 28,
            video_resize_mode: QwenVideoResizeMode::TotalVolume,
            temporal_patch_size: 2,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            model_name: "test-qwen-vl",
        }
    }

    fn create_video_test_config() -> QwenVLConfig {
        QwenVLConfig {
            patch_size: 2,
            merge_size: 1,
            min_pixels: 1,
            max_pixels: 1024 * 1024,
            video_min_pixels: 1,
            video_max_pixels: 1024 * 1024,
            video_resize_mode: QwenVideoResizeMode::TotalVolume,
            temporal_patch_size: 2,
            mean: [0.5, 0.25, 0.75],
            std: [0.5, 0.25, 0.5],
            model_name: "test-qwen-vl-video",
        }
    }

    fn create_pattern_frame(seed: u8) -> DynamicImage {
        let mut image = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                image.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        seed.wrapping_add((x * 3 + y) as u8),
                        seed.wrapping_add((x + y * 5) as u8),
                        seed.wrapping_add((x * 7 + y * 11) as u8),
                    ]),
                );
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    fn create_sized_pattern_frame(w: u32, h: u32, seed: u8) -> DynamicImage {
        let mut image = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                image.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        seed.wrapping_add((x * 3 + y) as u8),
                        seed.wrapping_add((x + y * 5) as u8),
                        seed.wrapping_add((x * 7 + y * 11) as u8),
                    ]),
                );
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn test_prepare_video_frame_chunk_reuses_temporal_padding() {
        let frames = [[1_u8, 2, 3], [4, 5, 6], [7, 8, 9]];
        let prepare_calls = std::cell::Cell::new(0);
        let mut prepared = Vec::new();

        prepare_video_frame_chunk(3, 1, 2, &mut prepared, |frame_index| {
            prepare_calls.set(prepare_calls.get() + 1);
            Ok(VideoFrameRgb {
                width: 1,
                height: 1,
                data: Cow::Borrowed(&frames[frame_index]),
            })
        })
        .unwrap();

        assert_eq!(prepare_calls.get(), 1);
        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].data.as_ref(), prepared[1].data.as_ref());
    }

    #[test]
    fn test_preprocess_image_matches_tensor_patchify_with_resize() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let image = create_sized_pattern_frame(7, 9, 3);
        let (target_h, target_w) = processor.smart_resize(9, 7).unwrap();
        assert!(
            (target_w as u32, target_h as u32) != (7u32, 9u32),
            "test must force a resize; target {target_w}x{target_h} should differ from 7x9"
        );

        let result = processor
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();
        let actual = result.encoder_input.as_slice_memory_order().unwrap();

        let resized = resize_bicubic_pil(&image, target_w as u32, target_h as u32);
        let tensor = to_tensor_and_normalize(
            &resized,
            &processor.default_mean(),
            &processor.default_std(),
        );
        let (grid_t, grid_h, grid_w) = processor.calculate_grid_thw(target_h, target_w, 1);
        let mut expected = Vec::new();
        processor
            .patchify_into(&tensor, grid_t, grid_h, grid_w, &mut expected)
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "image patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_patchify_image_rgb_block_band_matches_tensor_patchify_with_resize() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let image = create_sized_pattern_frame(7, 9, 3);
        let (target_h, target_w) = processor.smart_resize(9, 7).unwrap();
        let resized = resize_bicubic_pil(&image, target_w as u32, target_h as u32);
        let tensor = to_tensor_and_normalize(
            &resized,
            &processor.default_mean(),
            &processor.default_std(),
        );
        let (grid_t, grid_h, grid_w) = processor.calculate_grid_thw(target_h, target_w, 1);
        let mut expected = Vec::new();
        processor
            .patchify_into(&tensor, grid_t, grid_h, grid_w, &mut expected)
            .unwrap();

        let (width, height, raw) = rgb_bytes(&resized);
        assert_eq!((width, height), (target_w, target_h));
        let patch_size = processor.config.patch_size;
        let merge_size = processor.config.merge_size;
        let temporal_patch_size = processor.config.temporal_patch_size;
        let merged_patch = merge_size * patch_size;
        let pr_blocks = grid_h / merge_size;
        let pc_blocks = grid_w / merge_size;
        let n_blocks = pr_blocks * pc_blocks;
        assert!(
            n_blocks > 1,
            "test must exercise multiple image patch blocks"
        );
        let block_out = merge_size * merge_size * 3 * temporal_patch_size * patch_size * patch_size;
        let mean = processor.default_mean();
        let std = processor.default_std();
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));
        let lut: [[f32; 256]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|v| v as f32 * scale[c] + bias[c]));

        let mut actual = vec![0.0; expected.len()];
        let split_blocks = n_blocks / 2;
        let split_at = split_blocks * block_out;
        let (first, second) = actual.split_at_mut(split_at);
        QwenVLProcessorBase::patchify_image_rgb_block_band(
            raw.as_ref(),
            width,
            patch_size,
            merge_size,
            temporal_patch_size,
            merged_patch,
            pc_blocks,
            0,
            first,
            &lut,
        );
        QwenVLProcessorBase::patchify_image_rgb_block_band(
            raw.as_ref(),
            width,
            patch_size,
            merge_size,
            temporal_patch_size,
            merged_patch,
            pc_blocks,
            split_blocks,
            second,
            &lut,
        );

        assert_eq!(actual.len(), expected.len());
        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "image block-band patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }

    /// When a video frame actually needs resizing, the DynamicImage path
    /// (`preprocess_video` → `resize_bicubic_pil`) and the raw-RGB path
    /// (`preprocess_video_rgb` → `resize_bicubic_pil_rgb`) must produce
    /// byte-for-byte identical encoder inputs. The other video tests use 4x4
    /// frames that need no resize, so this is the one exercising the
    /// default-bicubic resize branch.
    #[test]
    fn test_preprocess_video_rgb_matches_dynamic_with_resize() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        // Odd dimensions force smart_resize_video to a different factor-aligned
        // target, guaranteeing the resize branch runs.
        let frames = vec![
            create_sized_pattern_frame(7, 9, 3),
            create_sized_pattern_frame(7, 9, 101),
        ];
        let (target_h, target_w) = processor.smart_resize_video(frames.len(), 9, 7).unwrap();
        assert!(
            (target_w as u32, target_h as u32) != (7u32, 9u32),
            "test must force a resize; target {target_w}x{target_h} should differ from 7x9"
        );

        let rgb_frames = frames
            .iter()
            .map(|frame| {
                let DynamicImage::ImageRgb8(rgb) = frame else {
                    panic!("test frame is not RGB8");
                };
                RgbFrameRef {
                    width: rgb.width(),
                    height: rgb.height(),
                    data: rgb.as_raw(),
                }
            })
            .collect::<Vec<_>>();

        let dynamic = processor.preprocess_video(&frames, &config).unwrap();
        let rgb = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        let a = dynamic.encoder_input.as_slice_memory_order().unwrap();
        let b = rgb.encoder_input.as_slice_memory_order().unwrap();
        assert_eq!(
            a.len(),
            b.len(),
            "resized video encoder input length differs"
        );
        for (idx, (&got, &want)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "resized video path diverges at index {idx}: dynamic {got} vs rgb {want}"
            );
        }
    }

    #[test]
    fn test_preprocess_video_rgb_matches_dynamic_with_resize_and_padding() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![
            create_sized_pattern_frame(7, 9, 3),
            create_sized_pattern_frame(7, 9, 101),
            create_sized_pattern_frame(7, 9, 177),
        ];
        assert_ne!(
            frames.len() % processor.temporal_patch_size(),
            0,
            "test must force temporal padding"
        );
        let (target_h, target_w) = processor.smart_resize_video(frames.len(), 9, 7).unwrap();
        assert!(
            (target_w as u32, target_h as u32) != (7u32, 9u32),
            "test must force a resize; target {target_w}x{target_h} should differ from 7x9"
        );

        let rgb_frames = frames
            .iter()
            .map(|frame| {
                let DynamicImage::ImageRgb8(rgb) = frame else {
                    panic!("test frame is not RGB8");
                };
                RgbFrameRef {
                    width: rgb.width(),
                    height: rgb.height(),
                    data: rgb.as_raw(),
                }
            })
            .collect::<Vec<_>>();

        let dynamic = processor.preprocess_video(&frames, &config).unwrap();
        let rgb = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        let a = dynamic.encoder_input.as_slice_memory_order().unwrap();
        let b = rgb.encoder_input.as_slice_memory_order().unwrap();
        assert_eq!(a.len(), b.len());
        for (idx, (&got, &want)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "resized padded video path diverges at index {idx}: dynamic {got} vs rgb {want}"
            );
        }
    }

    #[test]
    fn test_qwen_vl_base_factor() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_video_matches_hf_actual_frame_beta() {
        let processor = QwenVLProcessorBase::new(QwenVLConfig {
            patch_size: 16,
            merge_size: 2,
            min_pixels: 1,
            max_pixels: 16_777_216,
            video_min_pixels: 1,
            video_max_pixels: 16_777_216,
            video_resize_mode: QwenVideoResizeMode::TotalVolume,
            temporal_patch_size: 2,
            mean: [0.5; 3],
            std: [0.5; 3],
            model_name: "test-qwen-vl-video-budget",
        });

        let (height, width) = processor.smart_resize_video(1, 3000, 3000).unwrap();

        // Hugging Face uses the padded frame count for the threshold, but the
        // actual frame count for beta. Preserve that behavior for parity.
        assert_eq!((height, width), (4096, 4096));
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_grid_thw() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1);
        assert_eq!(h, 448 / 14);
        assert_eq!(w, 448 / 14);
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4);
    }

    #[test]
    fn test_preprocess_video_matches_tensor_patchify() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![create_pattern_frame(3), create_pattern_frame(101)];

        let result = processor.preprocess_video(&frames, &config).unwrap();
        let actual = result.encoder_input.as_slice_memory_order().unwrap();

        let tensors = frames
            .iter()
            .map(|frame| {
                to_tensor_and_normalize(frame, &processor.default_mean(), &processor.default_std())
            })
            .collect::<Vec<_>>();
        let (grid_t, grid_h, grid_w) = processor.calculate_grid_thw(4, 4, frames.len());
        let mut expected = Vec::new();
        processor
            .patchify_video_into(&tensors, grid_t, grid_h, grid_w, &mut expected)
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "video patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_preprocess_video_rgb_matches_dynamic_video() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![
            create_pattern_frame(3),
            create_pattern_frame(101),
            create_pattern_frame(177),
        ];

        let rgb_frames = frames
            .iter()
            .map(|frame| {
                let DynamicImage::ImageRgb8(rgb) = frame else {
                    panic!("test frame is not RGB8");
                };
                RgbFrameRef {
                    width: rgb.width(),
                    height: rgb.height(),
                    data: rgb.as_raw(),
                }
            })
            .collect::<Vec<_>>();

        let dynamic_result = processor.preprocess_video(&frames, &config).unwrap();
        let rgb_result = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        assert_eq!(
            dynamic_result.encoder_input.shape(),
            rgb_result.encoder_input.shape()
        );
        let mut dynamic_keys = dynamic_result.model_specific.keys().collect::<Vec<_>>();
        let mut rgb_keys = rgb_result.model_specific.keys().collect::<Vec<_>>();
        dynamic_keys.sort();
        rgb_keys.sort();
        assert_eq!(dynamic_keys, rgb_keys);

        let dynamic_values = dynamic_result
            .encoder_input
            .as_slice_memory_order()
            .unwrap();
        let rgb_values = rgb_result.encoder_input.as_slice_memory_order().unwrap();
        for (idx, (&got, &want)) in rgb_values.iter().zip(dynamic_values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "RGB video patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_preprocess_video_rgb_matches_dynamic_video_parallel_blocks() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![
            create_sized_pattern_frame(280, 280, 3),
            create_sized_pattern_frame(280, 280, 101),
        ];

        let rgb_frames = frames
            .iter()
            .map(|frame| {
                let DynamicImage::ImageRgb8(rgb) = frame else {
                    panic!("test frame is not RGB8");
                };
                RgbFrameRef {
                    width: rgb.width(),
                    height: rgb.height(),
                    data: rgb.as_raw(),
                }
            })
            .collect::<Vec<_>>();

        let dynamic_result = processor.preprocess_video(&frames, &config).unwrap();
        let rgb_result = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        assert_eq!(
            dynamic_result.encoder_input.shape(),
            rgb_result.encoder_input.shape()
        );
        let dynamic_values = dynamic_result
            .encoder_input
            .as_slice_memory_order()
            .unwrap();
        let rgb_values = rgb_result.encoder_input.as_slice_memory_order().unwrap();
        for (idx, (&got, &want)) in rgb_values.iter().zip(dynamic_values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "parallel RGB video patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }
}
