// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::time::Instant;

use dynamo_multimodal::processed::ProcessedValue;
use dynamo_multimodal::types::RgbFrameRef;
use dynamo_multimodal::vision::{PreProcessorConfig, Qwen3VLProcessor};
use dynamo_multimodal::{Qwen3VlVideoConfig, Qwen3VlVideoPreprocessor, VideoTiming};

fn measure<T>(
    iterations: usize,
    mut operation: impl FnMut() -> Result<T, dynamo_multimodal::vision::TransformError>,
) -> Result<(f64, f64), dynamo_multimodal::vision::TransformError> {
    black_box(operation()?);
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(operation()?);
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2].as_secs_f64() * 1_000.0;
    let mean =
        samples.iter().sum::<std::time::Duration>().as_secs_f64() * 1_000.0 / iterations as f64;
    Ok((median, mean))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    let width = args
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(224);
    let height = args
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(width);
    let frame_count = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    anyhow::ensure!(iterations > 0, "iterations must be greater than zero");
    anyhow::ensure!(width > 0 && height > 0, "frame dimensions must be positive");
    anyhow::ensure!(frame_count > 0, "frame count must be greater than zero");
    anyhow::ensure!(args.next().is_none(), "too many arguments");

    let buffers = (0..frame_count)
        .map(|frame| {
            (0..width * height * 3)
                .map(|index| (index as u8).wrapping_add((frame as u8).wrapping_mul(17)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let frames = buffers
        .iter()
        .map(|data| RgbFrameRef {
            width,
            height,
            data,
        })
        .collect::<Vec<_>>();
    let timing = VideoTiming {
        source_fps: 2.0,
        source_duration: frame_count as f64 / 2.0,
        sampled_timestamps: (0..frame_count).map(|index| index as f64 / 2.0).collect(),
    };
    let model_config = Qwen3VlVideoConfig::default();
    let core_processor = Qwen3VLProcessor::with_config(
        model_config.patch_size,
        model_config.merge_size,
        model_config.min_pixels,
        model_config.max_pixels,
        model_config.temporal_patch_size,
    );
    let core_config = PreProcessorConfig {
        do_resize: Some(model_config.do_resize),
        do_normalize: Some(model_config.do_normalize),
        image_mean: Some(model_config.image_mean.to_vec()),
        image_std: Some(model_config.image_std.to_vec()),
        resampling: Some(model_config.resampling),
        min_pixels: Some(model_config.min_pixels),
        max_pixels: Some(model_config.max_pixels),
        temporal_patch_size: Some(model_config.temporal_patch_size),
        merge_size: Some(model_config.merge_size),
        ..Default::default()
    };
    let processor = Qwen3VlVideoPreprocessor::new(model_config)?;

    let warmup = black_box(processor.preprocess(&frames, &timing)?);
    let Some(pixel_values) = warmup.field("pixel_values_videos") else {
        anyhow::bail!("pixel_values_videos is missing");
    };
    let ProcessedValue::F32Tensor { data, shape } = &pixel_values.value else {
        anyhow::bail!("pixel_values_videos is not FP32");
    };
    let output_shape = shape.clone();
    let output_bytes = data.len() * size_of::<f32>();
    black_box(warmup);
    let (core_median_ms, core_mean_ms) = measure(iterations, || {
        core_processor.preprocess_configured_video_rgb(black_box(&frames), black_box(&core_config))
    })?;
    let (total_median_ms, total_mean_ms) = measure(iterations, || {
        processor.preprocess(black_box(&frames), black_box(&timing))
    })?;
    println!(
        "qwen3-vl video preprocess: frames={frame_count}, size={width}x{height}, \
         output_shape={output_shape:?}, output_bytes={output_bytes}, iterations={iterations}, \
         core_median_ms={core_median_ms:.3}, core_mean_ms={core_mean_ms:.3}, \
         total_median_ms={total_median_ms:.3}, total_mean_ms={total_mean_ms:.3}",
    );
    if std::env::var_os("DYNAMO_BENCH_FINGERPRINT").is_some() {
        let output = processor.preprocess(&frames, &timing)?;
        let Some(pixel_values) = output.field("pixel_values_videos") else {
            anyhow::bail!("pixel_values_videos is missing");
        };
        let ProcessedValue::F32Tensor { data, .. } = &pixel_values.value else {
            anyhow::bail!("pixel_values_videos is not FP32");
        };
        let fingerprint = data
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |mut hash, value| {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
                hash
            });
        println!("fnv1a_fp32={fingerprint:016x}");
    }
    Ok(())
}
