// SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Image transformation functions for vision preprocessing.
//!
//! This module provides composable transforms that match HuggingFace image processor
//! behavior, enabling pure Rust preprocessing without Python dependencies.

use std::cell::RefCell;

use fast_image_resize::{
    IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image as FirImage, ImageRef as FirImageRef},
};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use ndarray::{Array3, Array4, s};
use thiserror::Error;

use super::execution::{scope as parallel_scope, task_count};

/// Errors that can occur during image transformations.
#[derive(Error, Debug)]
pub enum TransformError {
    #[error("Invalid tensor shape: expected {expected}, got {actual:?}")]
    InvalidShape {
        expected: String,
        actual: Vec<usize>,
    },

    #[error("Image operation failed: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Empty batch: cannot stack zero tensors")]
    EmptyBatch,

    #[error("Inconsistent tensor shapes in batch")]
    InconsistentShapes,

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("Unsupported video processor model type: {0}")]
    UnsupportedModelType(String),
}

pub type Result<T> = std::result::Result<T, TransformError>;

/// Extract RGB pixel data from a DynamicImage, avoiding a copy when already RGB8.
/// Returns (width, height, raw_bytes) where raw_bytes is interleaved R,G,B,R,G,B,...
pub fn rgb_bytes(image: &DynamicImage) -> (usize, usize, std::borrow::Cow<'_, [u8]>) {
    match image {
        DynamicImage::ImageRgb8(rgb) => (
            rgb.width() as usize,
            rgb.height() as usize,
            std::borrow::Cow::Borrowed(rgb.as_raw()),
        ),
        _ => {
            let rgb = image.to_rgb8();
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            (w, h, std::borrow::Cow::Owned(rgb.into_raw()))
        }
    }
}

/// Deinterleave interleaved RGB bytes into separate R, G, B f32 planes with
/// per-channel `scale` and `bias`: `plane[c][i] = rgb[i*3 + c] * scale[c] + bias[c]`.
///
/// Processes 8 pixels at a time so the compiler can unroll and auto-vectorize
/// the stride-3 gather pattern.
pub fn deinterleave_rgb_to_planes(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    scale: [f32; 3],
    bias: [f32; 3],
) {
    let pixels = r_plane.len();
    debug_assert_eq!(pixels, g_plane.len());
    debug_assert_eq!(pixels, b_plane.len());
    debug_assert!(rgb.len() >= pixels * 3);

    // Each output element depends only on its own input byte, so banding the
    // pixel range across threads is BIT-IDENTICAL (elementwise f32, no
    // reduction). Small images stay serial.
    let nthreads = par_threads(pixels * 3 * 4, pixels);
    if nthreads <= 1 {
        deinterleave_contiguous(rgb, r_plane, g_plane, b_plane, scale, bias);
        return;
    }
    let chunk = pixels.div_ceil(nthreads);
    let (mut rr, mut gg, mut bb) = (r_plane, g_plane, b_plane);
    parallel_scope(|s| {
        let mut p0 = 0usize;
        while p0 < pixels {
            let n = chunk.min(pixels - p0);
            let (rb, rt) = rr.split_at_mut(n);
            let (gb, gt) = gg.split_at_mut(n);
            let (bbnd, bt) = bb.split_at_mut(n);
            rr = rt;
            gg = gt;
            bb = bt;
            let rgb_band = &rgb[p0 * 3..(p0 + n) * 3];
            s.spawn(move |_| deinterleave_contiguous(rgb_band, rb, gb, bbnd, scale, bias));
            p0 += n;
        }
    });
}

/// Deinterleave a contiguous pixel range (planes/rgb already sliced to the band).
fn deinterleave_contiguous(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    scale: [f32; 3],
    bias: [f32; 3],
) {
    let pixels = r_plane.len();
    let full_blocks = pixels / 8;
    let remainder = pixels % 8;

    for block in 0..full_blocks {
        let dst = block * 8;
        let src_base = dst * 3;
        let src = &rgb[src_base..src_base + 24];
        let rd = &mut r_plane[dst..dst + 8];
        let gd = &mut g_plane[dst..dst + 8];
        let bd = &mut b_plane[dst..dst + 8];

        for i in 0..8 {
            let s = i * 3;
            rd[i] = src[s] as f32 * scale[0] + bias[0];
            gd[i] = src[s + 1] as f32 * scale[1] + bias[1];
            bd[i] = src[s + 2] as f32 * scale[2] + bias[2];
        }
    }

    let tail_dst = full_blocks * 8;
    let tail_src = tail_dst * 3;
    for i in 0..remainder {
        let s = tail_src + i * 3;
        r_plane[tail_dst + i] = rgb[s] as f32 * scale[0] + bias[0];
        g_plane[tail_dst + i] = rgb[s + 1] as f32 * scale[1] + bias[1];
        b_plane[tail_dst + i] = rgb[s + 2] as f32 * scale[2] + bias[2];
    }
}

/// Build a [C, H, W] f32 tensor from interleaved RGB bytes with per-channel
/// `scale` and `bias`: `output[c][i] = raw[i*3 + c] * scale[c] + bias[c]`.
fn build_planar_tensor(
    raw: &[u8],
    w: usize,
    h: usize,
    scale: [f32; 3],
    bias: [f32; 3],
) -> Array3<f32> {
    let pixels = h * w;
    let mut data = vec![0.0f32; 3 * pixels];
    let (r_plane, rest) = data.split_at_mut(pixels);
    let (g_plane, b_plane) = rest.split_at_mut(pixels);

    deinterleave_rgb_to_planes(raw, r_plane, g_plane, b_plane, scale, bias);

    #[expect(
        clippy::expect_used,
        reason = "data has exactly 3*h*w elements by construction"
    )]
    Array3::from_shape_vec((3, h, w), data).expect("shape matches pre-allocated buffer")
}

/// Convert image to tensor [C, H, W] normalized to [0, 1].
///
/// This matches the default behavior of `torchvision.transforms.ToTensor()`.
pub fn to_tensor(image: &DynamicImage) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    let s = 1.0 / 255.0;
    build_planar_tensor(&raw, w, h, [s, s, s], [0.0, 0.0, 0.0])
}

/// Convert image to tensor [C, H, W] without normalization (keeps [0, 255]).
#[cfg(test)]
pub fn to_tensor_no_norm(image: &DynamicImage) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    build_planar_tensor(&raw, w, h, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
}

/// Normalize tensor per channel: (x - mean) / std.
///
/// This matches `torchvision.transforms.Normalize(mean, std)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `mean` - Per-channel mean values
/// * `std` - Per-channel standard deviation values
pub fn normalize(tensor: &mut Array3<f32>, mean: &[f64; 3], std: &[f64; 3]) {
    let [h, w] = [tensor.shape()[1], tensor.shape()[2]];
    let pixels = h * w;

    if let Some(flat) = tensor.as_slice_mut() {
        // Fast path: contiguous memory, process channel planes directly
        for c in 0..3 {
            let mean_c = mean[c] as f32;
            let std_c = std[c] as f32;
            let plane = &mut flat[c * pixels..(c + 1) * pixels];
            for v in plane.iter_mut() {
                *v = (*v - mean_c) / std_c;
            }
        }
    } else {
        for c in 0..3 {
            let mean_c = mean[c] as f32;
            let std_c = std[c] as f32;
            tensor
                .slice_mut(s![c, .., ..])
                .mapv_inplace(|v| (v - mean_c) / std_c);
        }
    }
}

/// Convert an image to a tensor with the HF fast-processor normalization order.
pub fn to_tensor_and_normalize(
    image: &DynamicImage,
    mean: &[f64; 3],
    std: &[f64; 3],
) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    let mut tensor = build_planar_tensor(&raw, w, h, [1.0; 3], [0.0; 3]);
    for channel in 0..3 {
        let scaled_mean = mean[channel] as f32 * 255.0;
        let scaled_std = std[channel] as f32 * 255.0;
        tensor
            .slice_mut(s![channel, .., ..])
            .mapv_inplace(|value| (value - scaled_mean) / scaled_std);
    }
    tensor
}

/// Rescale tensor by a constant factor.
///
/// Used when `do_rescale=True` in HuggingFace configs (typically 1/255).
pub fn rescale(tensor: &mut Array3<f32>, factor: f64) {
    let factor = factor as f32;
    tensor.mapv_inplace(|v| v * factor);
}

/// Map `image` crate filter types to `fast_image_resize` algorithm.
fn to_fir_algorithm(filter: FilterType) -> ResizeAlg {
    use fast_image_resize::FilterType as FirFilter;
    match filter {
        FilterType::Nearest => ResizeAlg::Nearest,
        FilterType::Triangle => ResizeAlg::Convolution(FirFilter::Bilinear),
        FilterType::CatmullRom => ResizeAlg::Convolution(FirFilter::CatmullRom),
        FilterType::Gaussian => ResizeAlg::Convolution(FirFilter::Gaussian),
        FilterType::Lanczos3 => ResizeAlg::Convolution(FirFilter::Lanczos3),
    }
}

thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

/// Resize image to exact dimensions using SIMD-accelerated resizer.
///
/// # Arguments
/// * `image` - Input image
/// * `width` - Target width
/// * `height` - Target height
/// * `filter` - Interpolation filter (Nearest, Triangle/Bilinear, CatmullRom/Bicubic, Lanczos3)
pub fn resize(image: &DynamicImage, width: u32, height: u32, filter: FilterType) -> DynamicImage {
    let pixel_type = match image.pixel_type() {
        Some(pt) => pt,
        None => return image.resize_exact(width, height, filter),
    };
    let mut dst = FirImage::new(width, height, pixel_type);
    let options = ResizeOptions::new().resize_alg(to_fir_algorithm(filter));
    let ok = RESIZER.with(|r| r.borrow_mut().resize(image, &mut dst, &options).is_ok());
    if !ok {
        return image.resize_exact(width, height, filter);
    }
    fir_image_to_dynamic(dst, width, height, image, filter)
}

/// Resize borrowed interleaved RGB bytes without first materializing an
/// `image::RgbImage` over an owned input buffer.
pub fn resize_rgb_bytes(
    data: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<RgbImage> {
    let src = FirImageRef::new(width, height, data, PixelType::U8x3)
        .map_err(|e| TransformError::ShapeError(format!("invalid RGB source image: {e}")))?;
    let mut dst = FirImage::new(target_width, target_height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(to_fir_algorithm(filter));
    RESIZER
        .with(|r| r.borrow_mut().resize(&src, &mut dst, &options))
        .map_err(|e| TransformError::ShapeError(format!("RGB resize failed: {e}")))?;

    RgbImage::from_raw(target_width, target_height, dst.into_vec()).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build resized RGB image for {target_width}x{target_height}"
        ))
    })
}

/// Convert a `fast_image_resize::Image` back to a `DynamicImage`.
///
/// Falls back to the `image` crate resize for unhandled pixel formats.
fn fir_image_to_dynamic(
    img: FirImage<'_>,
    width: u32,
    height: u32,
    source: &DynamicImage,
    filter: FilterType,
) -> DynamicImage {
    let buf = img.into_vec();
    match source {
        DynamicImage::ImageRgb8(_) => {
            RgbImage::from_raw(width, height, buf).map(DynamicImage::ImageRgb8)
        }
        DynamicImage::ImageRgba8(_) => {
            image::RgbaImage::from_raw(width, height, buf).map(DynamicImage::ImageRgba8)
        }
        DynamicImage::ImageLuma8(_) => {
            image::GrayImage::from_raw(width, height, buf).map(DynamicImage::ImageLuma8)
        }
        _ => None,
    }
    .unwrap_or_else(|| source.resize_exact(width, height, filter))
}

// ---------------------------------------------------------------------------
// Pillow-exact bicubic resize.
//
// Qwen image processors resize via `PIL.Image.resize(size, BICUBIC)` on the
// uint8 image. The SIMD `fast_image_resize` path above is the same filter
// *family* (Catmull-Rom, a=-0.5) but diverges bit-wise on non-integer ratios
// (support scaling + fixed-point details), which the vision encoder amplifies
// into a large embedding shift. This routine replicates Pillow's `Resample.c`
// algorithm exactly, validated against Pillow.
const PIL_PRECISION_BITS: i64 = 32 - 8 - 2;
const PIL_BICUBIC_SUPPORT: f64 = 2.0;

#[inline]
fn pil_cubic(x: f64) -> f64 {
    // Keys cubic with a = -0.5 (Pillow's BICUBIC).
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow `precompute_coeffs` for one axis: integer (fixed-point) kernels plus
/// per-output bounds `(start, count)`.
fn pil_precompute_coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<i64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale >= 1.0 { scale } else { 1.0 };
    let support = PIL_BICUBIC_SUPPORT * filterscale;
    let inv = 1.0 / filterscale;
    let coeff_scale = (1_i64 << PIL_PRECISION_BITS) as f64;

    let mut bounds = Vec::with_capacity(out_size);
    let mut kernels = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xmin = xmin as usize;
        let xmax = (xmax as usize).saturating_sub(xmin);

        let mut w = vec![0.0_f64; xmax];
        let mut tot = 0.0;
        for (x, wx) in w.iter_mut().enumerate() {
            let v = pil_cubic(((x + xmin) as f64 - center + 0.5) * inv);
            *wx = v;
            tot += v;
        }
        if tot != 0.0 {
            for wx in &mut w {
                *wx /= tot;
            }
        }
        // Pillow normalize_coeffs_8bpc: round half away from zero into fixed point.
        let k: Vec<i64> = w
            .iter()
            .map(|&c| {
                if c < 0.0 {
                    (-0.5 + c * coeff_scale) as i64
                } else {
                    (0.5 + c * coeff_scale) as i64
                }
            })
            .collect();
        bounds.push((xmin, xmax));
        kernels.push(k);
    }
    (bounds, kernels)
}

#[inline]
fn pil_clip8(v: i64) -> u8 {
    let v = v >> PIL_PRECISION_BITS;
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Number of threads to split an elementwise or row-banded preprocessing pass
/// across. Each output row/element is independent, so banding work over threads
/// yields BIT-IDENTICAL output: no shared accumulation and no inner-loop order
/// changes. Small images run serial to avoid thread-spawn overhead.
pub(crate) fn par_threads(out_bytes: usize, out_rows: usize) -> usize {
    task_count(out_bytes, out_rows, 32)
}

/// Process output rows `[oy0, oy0 + out_band.len()/row_out)` of the horizontal
/// pass into `out_band`. Horizontal pass preserves row count, so output row i
/// reads input row `oy0 + i`.
#[expect(
    clippy::too_many_arguments,
    reason = "row-band resampler: precomputed coeffs + dims + output band"
)]
fn pil_h_band(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    in_w: usize,
    out_w: usize,
    channels: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = out_w * channels;
    for (i, orow) in out_band.chunks_mut(row_out).enumerate() {
        let y = oy0 + i;
        let row = &src[y * in_w * channels..(y + 1) * in_w * channels];
        for xx in 0..out_w {
            let (xmin, xmax) = bounds[xx];
            let k = &kernels[xx];
            for c in 0..channels {
                let mut ss = half;
                for x in 0..xmax {
                    ss += row[(xmin + x) * channels + c] as i64 * k[x];
                }
                orow[xx * channels + c] = pil_clip8(ss);
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "RGB row-band resampler: precomputed coeffs + dims + output band"
)]
fn pil_h_band_rgb(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    in_w: usize,
    out_w: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = out_w * 3;
    for (i, output_row) in out_band.chunks_mut(row_out).enumerate() {
        let y = oy0 + i;
        let row = &src[y * in_w * 3..(y + 1) * in_w * 3];
        for output_x in 0..out_w {
            let (source_x, source_columns) = bounds[output_x];
            let kernel = &kernels[output_x];
            let mut red = half;
            let mut green = half;
            let mut blue = half;
            let source_start = source_x * 3;
            let source_end = (source_x + source_columns) * 3;
            for (pixel, &coefficient) in row[source_start..source_end].chunks_exact(3).zip(kernel) {
                red += pixel[0] as i64 * coefficient;
                green += pixel[1] as i64 * coefficient;
                blue += pixel[2] as i64 * coefficient;
            }
            let output = output_x * 3;
            output_row[output] = pil_clip8(red);
            output_row[output + 1] = pil_clip8(green);
            output_row[output + 2] = pil_clip8(blue);
        }
    }
}

/// Resample interleaved `channels`-channel u8 data along the width axis.
/// `src` is `rows * in_w * channels`; returns `rows * out_w * channels`.
fn pil_resample_horizontal(
    src: &[u8],
    rows: usize,
    in_w: usize,
    out_w: usize,
    channels: usize,
) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_w, out_w);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = out_w * channels;
    let mut out = vec![0_u8; rows * row_out];
    let nthreads = par_threads(out.len(), rows);
    if nthreads <= 1 {
        pil_h_band(
            src, &bounds, &kernels, half, in_w, out_w, channels, 0, &mut out,
        );
    } else {
        let chunk_rows = rows.div_ceil(nthreads);
        parallel_scope(|s| {
            let (b, k) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut oy0 = 0usize;
            while oy0 < rows {
                let n = chunk_rows.min(rows - oy0);
                let (band, tail) = rest.split_at_mut(n * row_out);
                rest = tail;
                let start = oy0;
                s.spawn(move |_| {
                    pil_h_band(src, b, k, half, in_w, out_w, channels, start, band);
                });
                oy0 += n;
            }
        });
    }
    out
}

fn pil_resample_horizontal_rgb(src: &[u8], rows: usize, in_w: usize, out_w: usize) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_w, out_w);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = out_w * 3;
    let mut out = vec![0_u8; rows * row_out];
    let nthreads = par_threads(out.len(), rows);
    if nthreads <= 1 {
        pil_h_band_rgb(src, &bounds, &kernels, half, in_w, out_w, 0, &mut out);
    } else {
        let chunk_rows = rows.div_ceil(nthreads);
        parallel_scope(|scope| {
            let (bounds, kernels) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut output_y = 0;
            while output_y < rows {
                let band_rows = chunk_rows.min(rows - output_y);
                let (band, tail) = rest.split_at_mut(band_rows * row_out);
                rest = tail;
                let start = output_y;
                scope.spawn(move |_| {
                    pil_h_band_rgb(src, bounds, kernels, half, in_w, out_w, start, band);
                });
                output_y += band_rows;
            }
        });
    }
    out
}

/// Process output rows `[oy0, oy0 + out_band.len()/row_out)` of the vertical
/// pass into `out_band`.
#[expect(
    clippy::too_many_arguments,
    reason = "row-band resampler: precomputed coeffs + dims + output band"
)]
fn pil_v_band(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    width: usize,
    channels: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = width * channels;
    for (i, orow) in out_band.chunks_mut(row_out).enumerate() {
        let yy = oy0 + i;
        let (ymin, ymax) = bounds[yy];
        let k = &kernels[yy];
        for x in 0..width {
            for c in 0..channels {
                let mut ss = half;
                for y in 0..ymax {
                    ss += src[((ymin + y) * width + x) * channels + c] as i64 * k[y];
                }
                orow[x * channels + c] = pil_clip8(ss);
            }
        }
    }
}

fn pil_v_band_rgb(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    width: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = width * 3;
    for (i, output_row) in out_band.chunks_mut(row_out).enumerate() {
        let output_y = oy0 + i;
        let (source_y, source_rows) = bounds[output_y];
        let kernel = &kernels[output_y];
        let blocked_width = width / 4 * 4;
        for x in (0..blocked_width).step_by(4) {
            let mut sums = [[half; 3]; 4];
            for (y, &coefficient) in kernel.iter().take(source_rows).enumerate() {
                let row_start = (source_y + y) * row_out;
                // Present the four adjacent RGB pixels as one bounded slice.
                // This lets LLVM eliminate repeated bounds checks in the hot
                // vertical convolution without changing accumulation order.
                let pixels = &src[row_start + x * 3..row_start + (x + 4) * 3];
                for (pixel, sums) in pixels.chunks_exact(3).zip(&mut sums) {
                    sums[0] += pixel[0] as i64 * coefficient;
                    sums[1] += pixel[1] as i64 * coefficient;
                    sums[2] += pixel[2] as i64 * coefficient;
                }
            }
            let output = &mut output_row[x * 3..(x + 4) * 3];
            for (pixel, sums) in output.chunks_exact_mut(3).zip(&sums) {
                pixel[0] = pil_clip8(sums[0]);
                pixel[1] = pil_clip8(sums[1]);
                pixel[2] = pil_clip8(sums[2]);
            }
        }
        for x in blocked_width..width {
            let mut red = half;
            let mut green = half;
            let mut blue = half;
            for (y, &coefficient) in kernel.iter().take(source_rows).enumerate() {
                let source = (source_y + y) * row_out + x * 3;
                let pixel = &src[source..source + 3];
                red += pixel[0] as i64 * coefficient;
                green += pixel[1] as i64 * coefficient;
                blue += pixel[2] as i64 * coefficient;
            }
            let output = x * 3;
            output_row[output] = pil_clip8(red);
            output_row[output + 1] = pil_clip8(green);
            output_row[output + 2] = pil_clip8(blue);
        }
    }
}

/// Resample interleaved `channels`-channel u8 data along the height axis.
fn pil_resample_vertical(
    src: &[u8],
    in_h: usize,
    width: usize,
    out_h: usize,
    channels: usize,
) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_h, out_h);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = width * channels;
    let mut out = vec![0_u8; out_h * row_out];
    let nthreads = par_threads(out.len(), out_h);
    if nthreads <= 1 {
        pil_v_band(src, &bounds, &kernels, half, width, channels, 0, &mut out);
    } else {
        let chunk_rows = out_h.div_ceil(nthreads);
        parallel_scope(|s| {
            let (b, k) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut oy0 = 0usize;
            while oy0 < out_h {
                let n = chunk_rows.min(out_h - oy0);
                let (band, tail) = rest.split_at_mut(n * row_out);
                rest = tail;
                let start = oy0;
                s.spawn(move |_| pil_v_band(src, b, k, half, width, channels, start, band));
                oy0 += n;
            }
        });
    }
    out
}

fn pil_resample_vertical_rgb(src: &[u8], in_h: usize, width: usize, out_h: usize) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_h, out_h);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = width * 3;
    let mut out = vec![0_u8; out_h * row_out];
    let nthreads = par_threads(out.len(), out_h);
    if nthreads <= 1 {
        pil_v_band_rgb(src, &bounds, &kernels, half, width, 0, &mut out);
    } else {
        let chunk_rows = out_h.div_ceil(nthreads);
        parallel_scope(|scope| {
            let (bounds, kernels) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut output_y = 0;
            while output_y < out_h {
                let rows = chunk_rows.min(out_h - output_y);
                let (band, tail) = rest.split_at_mut(rows * row_out);
                rest = tail;
                let start = output_y;
                scope.spawn(move |_| {
                    pil_v_band_rgb(src, bounds, kernels, half, width, start, band);
                });
                output_y += rows;
            }
        });
    }
    out
}

/// Pillow-exact BICUBIC resize (RGB8), matching
/// `PIL.Image.resize(.., BICUBIC)`.
pub fn resize_bicubic_pil(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (in_w, in_h) = rgb.dimensions();
    let output = resize_bicubic_pil_bytes(rgb.as_raw(), in_w, in_h, out_w, out_h, false);
    #[expect(
        clippy::expect_used,
        reason = "output is exactly out_w*out_h*3 bytes by construction"
    )]
    DynamicImage::ImageRgb8(
        RgbImage::from_raw(out_w, out_h, output).expect("pil resize buffer size"),
    )
}

/// PIL-exact bicubic resize over borrowed interleaved RGB bytes.
///
/// Byte-for-byte equivalent of [`resize_bicubic_pil`] but for the raw-RGB video
/// frame path (`preprocess_video_rgb`). Returns an `RgbImage` to drop straight
/// into the existing [`resize_rgb_bytes`] call sites.
pub fn resize_bicubic_pil_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<RgbImage> {
    let (in_w, in_h) = (width as usize, height as usize);
    let expected = in_w.saturating_mul(in_h).saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "PIL bicubic RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    let output = resize_bicubic_pil_bytes(data, width, height, out_w, out_h, true);
    RgbImage::from_raw(out_w, out_h, output).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build PIL bicubic RGB image for {out_w}x{out_h}"
        ))
    })
}

fn resize_bicubic_pil_bytes(
    data: &[u8],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    joint_rgb: bool,
) -> Vec<u8> {
    let (in_w, in_h, out_w, out_h) = (in_w as usize, in_h as usize, out_w as usize, out_h as usize);
    if in_w == out_w && in_h == out_h {
        data.to_vec()
    } else if in_w == out_w {
        if joint_rgb {
            pil_resample_vertical_rgb(data, in_h, in_w, out_h)
        } else {
            pil_resample_vertical(data, in_h, in_w, out_h, 3)
        }
    } else {
        let horiz = if joint_rgb {
            pil_resample_horizontal_rgb(data, in_h, in_w, out_w)
        } else {
            pil_resample_horizontal(data, in_h, in_w, out_w, 3)
        };
        if in_h == out_h {
            horiz
        } else if joint_rgb {
            pil_resample_vertical_rgb(&horiz, in_h, out_w, out_h)
        } else {
            pil_resample_vertical(&horiz, in_h, out_w, out_h, 3)
        }
    }
}

/// Resize image preserving aspect ratio, fitting within max dimensions.
pub fn resize_to_fit(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    filter: FilterType,
) -> DynamicImage {
    let (w, h) = image.dimensions();
    let ratio = (max_width as f64 / w as f64).min(max_height as f64 / h as f64);
    if ratio >= 1.0 {
        return image.clone();
    }
    let new_w = ((w as f64 * ratio).round() as u32).max(1);
    let new_h = ((h as f64 * ratio).round() as u32).max(1);
    resize(image, new_w, new_h, filter)
}

/// Center crop image to specified dimensions.
///
/// If the crop size is larger than the image, the image is returned unchanged.
pub fn center_crop(image: &DynamicImage, crop_w: u32, crop_h: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    if crop_w >= w && crop_h >= h {
        return image.clone();
    }
    let left = (w.saturating_sub(crop_w)) / 2;
    let top = (h.saturating_sub(crop_h)) / 2;
    let actual_w = crop_w.min(w);
    let actual_h = crop_h.min(h);
    image.crop_imm(left, top, actual_w, actual_h)
}

/// Expand image to square by padding with background color.
///
/// This is used by LLaVA models which expect square inputs. The image is
/// centered and padded with the mean color on the shorter dimension.
pub fn expand_to_square(image: &DynamicImage, background: Rgb<u8>) -> DynamicImage {
    let (w, h) = image.dimensions();
    match w.cmp(&h) {
        std::cmp::Ordering::Equal => image.clone(),
        std::cmp::Ordering::Less => {
            // Height > Width: pad horizontally
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(h, h, background));
            image::imageops::overlay(&mut new_image, image, ((h - w) / 2) as i64, 0);
            new_image
        }
        std::cmp::Ordering::Greater => {
            // Width > Height: pad vertically
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(w, w, background));
            image::imageops::overlay(&mut new_image, image, 0, ((w - h) / 2) as i64);
            new_image
        }
    }
}

/// Stack multiple [C, H, W] tensors into [B, C, H, W].
///
/// All tensors must have the same shape.
pub fn stack_batch(tensors: &[Array3<f32>]) -> Result<Array4<f32>> {
    if tensors.is_empty() {
        return Err(TransformError::EmptyBatch);
    }

    let shape = tensors[0].shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);

    // Verify all tensors have the same shape
    for tensor in tensors.iter().skip(1) {
        if tensor.shape() != shape {
            return Err(TransformError::InvalidShape {
                expected: format!("[{c}, {h}, {w}]"),
                actual: tensor.shape().to_vec(),
            });
        }
    }

    let mut batch = Array4::<f32>::zeros((tensors.len(), c, h, w));
    for (i, tensor) in tensors.iter().enumerate() {
        batch.slice_mut(s![i, .., .., ..]).assign(tensor);
    }

    Ok(batch)
}

/// Convert PIL/HuggingFace resampling enum to image crate filter.
///
/// PIL resampling constants:
/// - 0: NEAREST
/// - 1: LANCZOS (also ANTIALIAS)
/// - 2: BILINEAR
/// - 3: BICUBIC
/// - 4: BOX
/// - 5: HAMMING
pub fn pil_to_filter(resampling: Option<usize>) -> FilterType {
    match resampling {
        Some(0) => FilterType::Nearest,
        Some(1) => FilterType::Lanczos3,
        Some(2) | None => FilterType::Triangle, // Bilinear (default)
        Some(3) => FilterType::CatmullRom,      // Bicubic
        // Box and Hamming don't have direct equivalents, use Triangle
        Some(4) | Some(5) => FilterType::Triangle,
        _ => FilterType::Triangle,
    }
}

/// Calculate mean color of an image as RGB.
pub fn calculate_mean_color(image: &DynamicImage) -> Rgb<u8> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as u64, rgb.height() as u64);
    let total_pixels = w * h;

    if total_pixels == 0 {
        return Rgb([128, 128, 128]);
    }

    let (mut r_sum, mut g_sum, mut b_sum) = (0u64, 0u64, 0u64);
    for pixel in rgb.pixels() {
        r_sum += pixel[0] as u64;
        g_sum += pixel[1] as u64;
        b_sum += pixel[2] as u64;
    }

    Rgb([
        (r_sum / total_pixels) as u8,
        (g_sum / total_pixels) as u8,
        (b_sum / total_pixels) as u8,
    ])
}

/// Convert normalized mean values [0, 1] to RGB bytes.
pub fn mean_to_rgb(mean: &[f64; 3]) -> Rgb<u8> {
    Rgb([
        (mean[0] * 255.0).round() as u8,
        (mean[1] * 255.0).round() as u8,
        (mean[2] * 255.0).round() as u8,
    ])
}

/// Cubic interpolation weight function (Keys bicubic kernel with a=-0.5).
///
/// This matches PyTorch's bicubic interpolation used in
/// `torch.nn.functional.interpolate(mode='bicubic')`.
#[inline]
pub fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Perform bicubic interpolation at a single point in a tensor.
///
/// Uses a 4x4 kernel with Keys bicubic weights (a=-0.5) to match PyTorch's
/// `torch.nn.functional.interpolate(mode='bicubic')`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `c` - Channel index
/// * `src_y` - Source Y coordinate (can be fractional)
/// * `src_x` - Source X coordinate (can be fractional)
/// * `h` - Height of the tensor
/// * `w` - Width of the tensor
///
/// # Returns
/// The interpolated value at the specified position.
pub fn bicubic_interpolate(
    tensor: &Array3<f32>,
    c: usize,
    src_y: f32,
    src_x: f32,
    h: usize,
    w: usize,
) -> f32 {
    let y_int = src_y.floor() as i32;
    let x_int = src_x.floor() as i32;
    let y_frac = src_y - y_int as f32;
    let x_frac = src_x - x_int as f32;

    let mut result = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in -1..=2 {
        let y_idx = (y_int + dy).clamp(0, h as i32 - 1) as usize;
        let y_weight = cubic_weight(y_frac - dy as f32);

        for dx in -1..=2 {
            let x_idx = (x_int + dx).clamp(0, w as i32 - 1) as usize;
            let x_weight = cubic_weight(x_frac - dx as f32);

            result += tensor[[c, y_idx, x_idx]] * y_weight * x_weight;
        }
    }

    result
}

/// Resize a tensor using bicubic interpolation.
///
/// This matches PyTorch's `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `target_h` - Target height
/// * `target_w` - Target width
///
/// # Returns
/// Resized tensor of shape [C, target_h, target_w].
pub fn bicubic_resize(tensor: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);

    if h == target_h && w == target_w {
        return tensor.clone();
    }

    let mut result = Array3::<f32>::zeros((c, target_h, target_w));

    // PyTorch align_corners=False coordinate mapping
    let scale_h = h as f32 / target_h as f32;
    let scale_w = w as f32 / target_w as f32;

    for ch in 0..c {
        for y in 0..target_h {
            for x in 0..target_w {
                // PyTorch align_corners=False: src = (dst + 0.5) * scale - 0.5
                let src_y = (y as f32 + 0.5) * scale_h - 0.5;
                let src_x = (x as f32 + 0.5) * scale_w - 0.5;

                result[[ch, y, x]] = bicubic_interpolate(tensor, ch, src_y, src_x, h, w);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    /// The raw-RGB video resizer must be byte-for-byte identical to the
    /// DynamicImage PIL-bicubic resizer used for images. Guards the video resize
    /// path used by `preprocess_video_rgb`.
    #[test]
    fn resize_bicubic_pil_rgb_matches_dynamic_path() {
        let (src_w, src_h) = (37u32, 23u32); // non-aligned source, non-trivial ratios
        let (out_w, out_h) = (16u32, 28u32); // downscale width, upscale height
        let mut img = RgbImage::new(src_w, src_h);
        for y in 0..src_h {
            for x in 0..src_w {
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        ((x * 7) ^ (y * 13)) as u8,
                        (x * 3 + y * 5) as u8,
                        (x + y * y) as u8,
                    ]),
                );
            }
        }
        let via_dynamic = resize_bicubic_pil(&DynamicImage::ImageRgb8(img.clone()), out_w, out_h);
        let via_bytes = resize_bicubic_pil_rgb(img.as_raw(), src_w, src_h, out_w, out_h).unwrap();
        assert_eq!(
            via_dynamic.to_rgb8().into_raw(),
            via_bytes.into_raw(),
            "raw-RGB PIL bicubic must equal DynamicImage PIL bicubic byte-for-byte"
        );
    }

    #[test]
    fn resize_bicubic_pil_rgb_skips_identity_axes_bit_exactly() {
        let (src_w, src_h) = (31u32, 23u32);
        let mut data = vec![0u8; src_w as usize * src_h as usize * 3];
        for (index, value) in data.iter_mut().enumerate() {
            *value = (index as u8).wrapping_mul(37).wrapping_add(11);
        }

        for (out_w, out_h) in [(src_w, 17), (19, src_h), (src_w, src_h)] {
            let horizontal =
                pil_resample_horizontal(&data, src_h as usize, src_w as usize, out_w as usize, 3);
            let expected = pil_resample_vertical(
                &horizontal,
                src_h as usize,
                out_w as usize,
                out_h as usize,
                3,
            );
            let actual = resize_bicubic_pil_rgb(&data, src_w, src_h, out_w, out_h)
                .unwrap()
                .into_raw();

            assert_eq!(actual, expected, "identity-axis fast path changed pixels");
        }
    }

    /// `resize_bicubic_pil_rgb` rejects a buffer whose length doesn't match the
    /// declared dimensions rather than reading out of bounds.
    #[test]
    fn resize_bicubic_pil_rgb_rejects_wrong_length() {
        assert!(
            resize_bicubic_pil_rgb(&[0u8; 10], 4, 4, 2, 2).is_err(),
            "wrong-length RGB buffer must error, not panic"
        );
    }

    #[test]
    fn test_to_tensor_shape() {
        let img = create_test_image(10, 20, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);
        assert_eq!(tensor.shape(), &[3, 20, 10]); // [C, H, W]
    }

    #[test]
    fn test_to_tensor_values() {
        let img = create_test_image(2, 2, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);

        // Check normalization to [0, 1]
        assert!((tensor[[0, 0, 0]] - 1.0).abs() < 1e-6); // R=255 -> 1.0
        assert!((tensor[[1, 0, 0]] - 0.502).abs() < 0.01); // G=128 -> ~0.5
        assert!((tensor[[2, 0, 0]] - 0.0).abs() < 1e-6); // B=0 -> 0.0
    }

    #[test]
    fn test_to_tensor_no_norm() {
        let img = create_test_image(2, 2, Rgb([255, 128, 64]));
        let tensor = to_tensor_no_norm(&img);

        assert!((tensor[[0, 0, 0]] - 255.0).abs() < 1e-6);
        assert!((tensor[[1, 0, 0]] - 128.0).abs() < 1e-6);
        assert!((tensor[[2, 0, 0]] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 0.5);
        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];

        normalize(&mut tensor, &mean, &std);

        // (0.5 - 0.5) / 0.5 = 0.0
        for val in &tensor {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_rescale() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 255.0);
        rescale(&mut tensor, 1.0 / 255.0);

        for val in &tensor {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resize() {
        let img = create_test_image(100, 50, Rgb([128, 128, 128]));
        let resized = resize(&img, 50, 25, FilterType::Triangle);

        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 25);
    }

    #[test]
    fn test_resize_rgb_bytes_matches_resize() {
        let mut rgb = RgbImage::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                rgb.put_pixel(x, y, Rgb([(x * 40) as u8, (y * 90) as u8, 128]));
            }
        }
        let img = DynamicImage::ImageRgb8(rgb.clone());
        let expected = resize(&img, 2, 2, FilterType::Triangle).to_rgb8();
        let actual = resize_rgb_bytes(rgb.as_raw(), 3, 2, 2, 2, FilterType::Triangle)
            .expect("resize_rgb_bytes should resize valid RGB input");

        assert_eq!(actual.as_raw(), expected.as_raw());
    }

    #[test]
    fn test_resize_rgb_bytes_rejects_invalid_length() {
        let result = resize_rgb_bytes(&[1, 2, 3, 4, 5], 2, 1, 1, 1, FilterType::Triangle);
        assert!(matches!(result, Err(TransformError::ShapeError(_))));
    }

    #[test]
    fn test_center_crop() {
        let img = create_test_image(100, 100, Rgb([128, 128, 128]));
        let cropped = center_crop(&img, 50, 50);

        assert_eq!(cropped.width(), 50);
        assert_eq!(cropped.height(), 50);
    }

    #[test]
    fn test_expand_to_square_horizontal() {
        let img = create_test_image(100, 50, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_vertical() {
        let img = create_test_image(50, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_already_square() {
        let img = create_test_image(100, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_stack_batch() {
        let t1 = Array3::<f32>::zeros((3, 10, 10));
        let t2 = Array3::<f32>::ones((3, 10, 10));

        let batch = stack_batch(&[t1, t2]).unwrap();

        assert_eq!(batch.shape(), &[2, 3, 10, 10]);
    }

    #[test]
    fn test_stack_batch_empty() {
        let result = stack_batch(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
    }

    #[test]
    fn test_pil_to_filter() {
        assert!(matches!(pil_to_filter(Some(0)), FilterType::Nearest));
        assert!(matches!(pil_to_filter(Some(1)), FilterType::Lanczos3));
        assert!(matches!(pil_to_filter(Some(2)), FilterType::Triangle));
        assert!(matches!(pil_to_filter(Some(3)), FilterType::CatmullRom));
        assert!(matches!(pil_to_filter(None), FilterType::Triangle));
    }

    #[test]
    fn test_mean_to_rgb() {
        let mean = [0.5, 0.25, 1.0];
        let rgb = mean_to_rgb(&mean);

        assert_eq!(rgb[0], 128);
        assert_eq!(rgb[1], 64);
        assert_eq!(rgb[2], 255);
    }
}
