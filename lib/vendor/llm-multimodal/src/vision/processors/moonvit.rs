//! Shared MoonViT preprocessing core for the Kimi vision family.
//!
//! Kimi-K2.5 and Kimi-K3 run the same MoonViT stack:
//!
//! 1. Compute scale to fit within patch limits (never upscale)
//! 2. Resize with BICUBIC interpolation
//! 3. Zero-pad to make dimensions divisible by factor (patch_size * merge_size)
//! 4. Normalize with the configured mean/std
//! 5. Extract patches as [N, C, patch_size, patch_size]
//!
//! Kimi resizes then zero-pads to make dimensions divisible by the alignment
//! factor. The model was trained with zero-padded images, so using direct
//! resize-to-aligned would degrade image quality.
//!
//! The reference `navit_resize_image`, `navit_patchify`, and `normalize` are
//! byte-identical between the two releases. The models diverge only in their
//! configured patch budget and in whether transparent pixels are composited
//! over a background before patchify — see [`MoonVitParams`] and the
//! `transparent_bg` argument to [`preprocess`].

use std::borrow::Cow;

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs},
    scratch,
    transforms::{self, TransformError, TransparentBg, TransparentBgFillStage},
};

/// MoonViT resize/patchify parameters for a single model.
#[derive(Debug, Clone, Copy)]
pub struct MoonVitParams {
    pub patch_size: usize,
    pub merge_size: usize,
    /// Maximum total patches before merge (`in_patch_limit`).
    pub in_patch_limit: usize,
    /// Maximum patches along one spatial dimension.
    pub patch_limit_on_one_side: usize,
}

impl MoonVitParams {
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Overlay whatever the model's `preprocessor_config.json` supplies.
    ///
    /// The registry hands out one shared processor instance built from
    /// compiled-in defaults, so this is the only point where a model's real
    /// limits can be applied. K2.5 and K3 ship different `in_patch_limit`
    /// values (16384 vs 65536), which is exactly the kind of divergence that
    /// silently caps resolution if the defaults win.
    pub fn resolved(self, config: &PreProcessorConfig) -> Self {
        Self {
            // A zero patch or merge size would divide by zero downstream.
            patch_size: config.get_patch_size(self.patch_size).max(1),
            merge_size: config.merge_size.unwrap_or(self.merge_size).max(1),
            in_patch_limit: config
                .get_extra::<usize>("in_patch_limit")
                .unwrap_or(self.in_patch_limit)
                .max(1),
            patch_limit_on_one_side: config
                .get_extra::<usize>("patch_limit_on_one_side")
                .unwrap_or(self.patch_limit_on_one_side)
                .max(1),
        }
    }

    /// Compute resize dimensions and padding, matching HF `navit_resize_image`.
    ///
    /// Never upscales (scale capped at 1.0). Pads with zeros to align to factor.
    pub fn compute_resize_config(&self, width: usize, height: usize) -> ResizeConfig {
        let ps = self.patch_size;
        let patches_w = (width / ps).max(1) as f64;
        let patches_h = (height / ps).max(1) as f64;

        let s1 = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let s2 = (self.patch_limit_on_one_side * ps) as f64 / width as f64;
        let s3 = (self.patch_limit_on_one_side * ps) as f64 / height as f64;
        let scale = f64::min(1.0, f64::min(s1, f64::min(s2, s3)));

        let new_w = ((width as f64 * scale) as usize).max(1);
        let new_h = ((height as f64 * scale) as usize).max(1);
        let new_w = new_w.min(self.patch_limit_on_one_side * ps);
        let new_h = new_h.min(self.patch_limit_on_one_side * ps);

        let factor = self.factor();
        let pad_width = (factor - new_w % factor) % factor;
        let pad_height = (factor - new_h % factor) % factor;

        let token_height = (new_h + pad_height) / factor;
        let token_width = (new_w + pad_width) / factor;
        let num_tokens = token_height * token_width;

        ResizeConfig {
            new_width: new_w,
            new_height: new_h,
            pad_width,
            pad_height,
            num_tokens,
        }
    }
}

/// MoonViT resize configuration for a single image.
pub struct ResizeConfig {
    pub new_width: usize,
    pub new_height: usize,
    pub pad_width: usize,
    pub pad_height: usize,
    pub num_tokens: usize,
}

/// Fused resize + zero-pad + normalize into a single [C, H_padded, W_padded] tensor.
///
/// Avoids intermediate allocations by:
/// 1. Allocating the final padded canvas directly
/// 2. Pre-filling with normalized black (bias value)
/// 3. Deinterleaving + normalizing the image region in one pass
///
/// Alpha handling follows the reference. With a `transparent_bg` the image is
/// composited at the configured stage; K3 ships `"after_resize"`, and the
/// ordering is load-bearing because a chessboard is generated at the resolution
/// of whatever it is painted onto. With `None` alpha is dropped *before* the
/// resize, matching the reference's `.convert("RGB")` at load time.
fn resize_pad_and_normalize(
    image: &DynamicImage,
    cfg: &ResizeConfig,
    mean: &[f64; 3],
    std: &[f64; 3],
    transparent_bg: Option<TransparentBg>,
) -> Array3<f32> {
    let canvas_h = cfg.new_height + cfg.pad_height;
    let canvas_w = cfg.new_width + cfg.pad_width;

    // Nothing to composite over if the image has no alpha to begin with.
    let bg = transparent_bg.filter(|_| image.color().has_alpha());

    // Reduce to RGB up front unless the background is painted after the resize,
    // in which case alpha has to survive the convolution.
    let pre_flattened = match bg {
        Some(b) if b.stage == TransparentBgFillStage::BeforeResize => Some(
            DynamicImage::ImageRgb8(transforms::fill_transparent_bg(image, b.config)),
        ),
        Some(_) => None,
        None if image.color().has_alpha() => Some(DynamicImage::ImageRgb8(image.to_rgb8())),
        None => None,
    };
    let source = pre_flattened.as_ref().unwrap_or(image);

    // SIMD-accelerated BICUBIC (fast_image_resize). Surviving alpha is
    // premultiplied, as `PIL.Image.resize` does for RGBA.
    let after_resize = bg.is_some_and(|b| b.stage == TransparentBgFillStage::AfterResize);
    let resized = transforms::resize(
        source,
        cfg.new_width as u32,
        cfg.new_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let post_filled = bg
        .filter(|_| after_resize)
        .map(|b| transforms::fill_transparent_bg(&resized, b.config));
    let (img_w, img_h, raw): (usize, usize, Cow<'_, [u8]>) = match &post_filled {
        Some(rgb) => (
            rgb.width() as usize,
            rgb.height() as usize,
            Cow::Borrowed(rgb.as_raw().as_slice()),
        ),
        None => transforms::rgb_bytes(&resized),
    };
    let canvas_pixels = canvas_h * canvas_w;

    // Precompute fused scale/bias: pixel/255 → normalized
    // output[c][i] = raw[i*3+c] / 255.0 * (1/std[c]) + (-mean[c]/std[c])
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
    let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));

    // Pooled: this per-image CHW buffer (tens of MB) is recycled by the
    // caller after patch extraction, keeping its pages mapped and hot.
    let mut data = scratch::take_f32(3 * canvas_pixels);
    let (r_plane, rest) = data.split_at_mut(canvas_pixels);
    let (g_plane, b_plane) = rest.split_at_mut(canvas_pixels);

    // Pre-fill with normalized black: (0/255 - mean) / std = bias
    r_plane.fill(bias[0]);
    g_plane.fill(bias[1]);
    b_plane.fill(bias[2]);

    // Overwrite image region row-by-row using vectorized deinterleave
    let rw = img_w.min(canvas_w);
    let rh = img_h.min(canvas_h);
    for y in 0..rh {
        let src_row = &raw[y * img_w * 3..y * img_w * 3 + rw * 3];
        let dst_offset = y * canvas_w;
        transforms::deinterleave_rgb_to_planes(
            src_row,
            &mut r_plane[dst_offset..dst_offset + rw],
            &mut g_plane[dst_offset..dst_offset + rw],
            &mut b_plane[dst_offset..dst_offset + rw],
            scale,
            bias,
        );
    }

    #[expect(
        clippy::expect_used,
        reason = "data has exactly 3*canvas_h*canvas_w elements by construction"
    )]
    Array3::from_shape_vec((3, canvas_h, canvas_w), data)
        .expect("shape matches pre-allocated buffer")
}

/// Extract [C, patch_size, patch_size] patches from a contiguous [C, H, W] tensor.
///
/// Uses row-based `copy_from_slice` instead of per-element indexing so the
/// compiler can auto-vectorize the inner copy.
/// Append this image's patches directly into `out` (no per-image intermediate
/// Vec): `out` is the pooled batch buffer pre-sized for the whole request.
fn extract_patches_into(tensor: &Array3<f32>, patch_size: usize, out: &mut Vec<f32>) {
    let channels = tensor.shape()[0];
    let height = tensor.shape()[1];
    let width = tensor.shape()[2];

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;

    // Get contiguous slice for direct row addressing
    let flat = tensor.as_standard_layout();
    #[expect(
        clippy::expect_used,
        reason = "as_standard_layout guarantees contiguous C-order memory"
    )]
    let data = flat
        .as_slice()
        .expect("as_standard_layout guarantees contiguous memory");

    for gh in 0..grid_h {
        for gw in 0..grid_w {
            let h_start = gh * patch_size;
            let w_start = gw * patch_size;
            for c in 0..channels {
                let plane_offset = c * height * width;
                for ph in 0..patch_size {
                    let row_start = plane_offset + (h_start + ph) * width + w_start;
                    out.extend_from_slice(&data[row_start..row_start + patch_size]);
                }
            }
        }
    }
}

/// Run the full MoonViT pipeline over a batch of images.
pub fn preprocess(
    params: MoonVitParams,
    images: &[DynamicImage],
    config: &PreProcessorConfig,
    transparent_bg: Option<TransparentBg>,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    if images.is_empty() {
        return Err(TransformError::EmptyBatch);
    }

    let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
    let mean = config.get_image_mean();
    let std = config.get_image_std();

    // Pre-size the pooled batch buffer exactly (patch_features per patch =
    // 3 * patch_size^2; this is the data plane's hottest allocation).
    let patch_features = 3 * params.patch_size * params.patch_size;
    let mut estimated_total = 0usize;
    for image in images {
        let (w, h) = image.dimensions();
        let cfg = params.compute_resize_config(w as usize, h as usize);
        let grid_h = (cfg.new_height + cfg.pad_height) / params.patch_size;
        let grid_w = (cfg.new_width + cfg.pad_width) / params.patch_size;
        estimated_total += grid_h * grid_w * patch_features;
    }
    let mut all_patches: Vec<f32> = scratch::take_f32_cap(estimated_total);
    let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
    let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
    let mut feature_token_counts = Vec::with_capacity(images.len());

    for image in images {
        let (w, h) = image.dimensions();
        let cfg = params.compute_resize_config(w as usize, h as usize);

        // Fused resize + pad + normalize in one pass (avoids 2 extra allocations)
        let tensor = resize_pad_and_normalize(image, &cfg, &mean, &std, transparent_bg);

        let padded_h = cfg.new_height + cfg.pad_height;
        let padded_w = cfg.new_width + cfg.pad_width;
        let grid_h = padded_h / params.patch_size;
        let grid_w = padded_w / params.patch_size;
        let grid_t = 1usize;

        grid_thw_data.push(grid_t as i64);
        grid_thw_data.push(grid_h as i64);
        grid_thw_data.push(grid_w as i64);

        let num_patches = grid_h * grid_w;
        feature_token_counts.push(cfg.num_tokens);

        // Patchify directly into the pooled batch buffer, then recycle the
        // CHW tensor's storage (standard layout, offset 0) for the next image.
        extract_patches_into(&tensor, params.patch_size, &mut all_patches);
        let (storage, _offset) = tensor.into_raw_vec_and_offset();
        scratch::give_f32(storage);
        patches_per_image.push(num_patches as i64);
    }

    let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
    let encoder_input = ndarray::Array4::from_shape_vec(
        (total_patches, 3, params.patch_size, params.patch_size),
        all_patches,
    )
    .map_err(|e| {
        TransformError::ShapeError(format!(
            "Failed to create encoder_input [{total_patches}, 3, {}, {}]: {e}",
            params.patch_size, params.patch_size
        ))
    })?;

    Ok(
        PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
            .with_extra(
                "grid_thws",
                ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
            )
            .with_extra(
                "patches_per_image",
                ModelSpecificValue::int_1d(patches_per_image),
            ),
    )
}
