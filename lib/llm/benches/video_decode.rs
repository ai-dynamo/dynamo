// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;

use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use dynamo_llm::preprocessor::media::{Decoder, EncodedMediaData, MediaDecoder};
use rayon::{ThreadPoolBuilder, prelude::*};

const RUN_CONCURRENT_SWEEP_ENV: &str = "RUN_VIDEO_DECODE_SWEEP";
const BENCH_INPUT_ENV: &str = "VIDEO_DECODE_BENCH_INPUT";
const NUM_FRAMES_ENV: &str = "VIDEO_DECODE_BENCH_NUM_FRAMES";
const DEFAULT_NUM_FRAMES: u64 = 30;
const CONCURRENT_DECODE_LEVELS: [usize; 3] = [1, 8, 32];
const CI_VIDEO: &[u8] = include_bytes!("../tests/data/media/240p_100.mp4");

fn bench_video_decode(c: &mut Criterion) {
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    let num_frames = DEFAULT_NUM_FRAMES;
    let decoder_config = video_decoder(num_frames);
    let decoder = decoder_config.video.as_ref().unwrap();
    let mut group = c.benchmark_group("video_decode_h264_320x240_100_to_30");
    group.throughput(Throughput::Elements(num_frames));

    group.bench_function("video_rs", |b| {
        b.iter_batched(
            || EncodedMediaData::from_bytes(CI_VIDEO.to_vec()),
            |data| black_box(decoder.decode(data).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_video_concurrent_decode(c: &mut Criterion) {
    if std::env::var_os(RUN_CONCURRENT_SWEEP_ENV).is_none() {
        eprintln!(
            "skipping concurrent video decode sweep: set {RUN_CONCURRENT_SWEEP_ENV}=1 to run it"
        );
        return;
    }
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);

    let video = match std::env::var_os(BENCH_INPUT_ENV) {
        Some(path) => std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.to_string_lossy())),
        None => CI_VIDEO.to_vec(),
    };
    let num_frames = std::env::var(NUM_FRAMES_ENV)
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid {NUM_FRAMES_ENV}={value}"))
        })
        .unwrap_or(DEFAULT_NUM_FRAMES);
    let decoder_config = video_decoder(num_frames);
    let decoder = decoder_config.video.as_ref().unwrap();
    let mut group = c.benchmark_group("video_decode_concurrent");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for concurrency in CONCURRENT_DECODE_LEVELS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(concurrency)
            .build()
            .unwrap();
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("video_rs", format!("c{concurrency}")),
            &concurrency,
            |b, &concurrency| {
                b.iter_batched(
                    || {
                        (0..concurrency)
                            .map(|_| EncodedMediaData::from_bytes(video.clone()))
                            .collect::<Vec<_>>()
                    },
                    |inputs| {
                        pool.install(|| {
                            inputs.into_par_iter().for_each(|data| {
                                black_box(decoder.decode(data).unwrap());
                            });
                        });
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

fn video_decoder(num_frames: u64) -> MediaDecoder {
    serde_json::from_value(serde_json::json!({
        "video": {
            "num_frames": num_frames,
            "strict": true,
        },
    }))
    .unwrap()
}

criterion_group!(benches, bench_video_decode, bench_video_concurrent_decode);
criterion_main!(benches);
