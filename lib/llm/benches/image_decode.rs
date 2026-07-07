// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{hint::black_box, io::Cursor};

use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use dynamo_llm::preprocessor::media::{
    Decoder, EncodedMediaData, ImageDecoder, libjpeg_turbo_available,
};
use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use rayon::{ThreadPoolBuilder, prelude::*};

const RUN_CONCURRENT_SWEEP_ENV: &str = "RUN_IMAGE_DECODE_SWEEP";
const CONCURRENT_DECODE_WIDTH: u32 = 3840;
const CONCURRENT_DECODE_HEIGHT: u32 = 2160;
const CONCURRENT_DECODE_BATCH_SIZE: usize = 100;
const CONCURRENT_DECODE_LEVELS: [usize; 3] = [1, 8, 32];

fn bench_jpeg_decode(c: &mut Criterion) {
    if !libjpeg_turbo_available() {
        eprintln!("skipping image_decode benchmark: libturbojpeg is not available");
        return;
    }

    let jpeg = make_jpeg(2400, 1080);
    let decoders = image_decoders();

    let mut group = c.benchmark_group("image_decode_jpeg_2400x1080");
    group.throughput(Throughput::Bytes(jpeg.len() as u64));
    for (name, decoder) in decoders {
        group.bench_function(name, |b| {
            b.iter_batched(
                || EncodedMediaData::from_bytes(jpeg.clone()),
                |data| black_box(decoder.decode(data).unwrap()),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_jpeg_concurrent_decode(c: &mut Criterion) {
    if std::env::var_os(RUN_CONCURRENT_SWEEP_ENV).is_none() {
        eprintln!(
            "skipping concurrent image decode sweep: set {RUN_CONCURRENT_SWEEP_ENV}=1 to run it"
        );
        return;
    }
    if !libjpeg_turbo_available() {
        eprintln!("skipping concurrent image decode sweep: libturbojpeg is not available");
        return;
    }

    let jpeg = make_jpeg(CONCURRENT_DECODE_WIDTH, CONCURRENT_DECODE_HEIGHT);
    let decoders = image_decoders();
    let mut group = c.benchmark_group("image_decode_jpeg_3840x2160_batch_100");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(CONCURRENT_DECODE_BATCH_SIZE as u64));

    for (name, decoder) in decoders {
        for concurrency in CONCURRENT_DECODE_LEVELS {
            let pool = ThreadPoolBuilder::new()
                .num_threads(concurrency)
                .build()
                .unwrap();

            group.bench_with_input(
                BenchmarkId::new(name, format!("c{concurrency}")),
                &concurrency,
                |b, _| {
                    b.iter_batched(
                        || {
                            (0..CONCURRENT_DECODE_BATCH_SIZE)
                                .map(|_| EncodedMediaData::from_bytes(jpeg.clone()))
                                .collect::<Vec<_>>()
                        },
                        |inputs| {
                            pool.install(|| {
                                inputs.into_par_iter().for_each(|data| {
                                    let decoded = decoder.decode(data).unwrap();
                                    black_box(decoded);
                                });
                            });
                        },
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }
    group.finish();
}

fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        Rgb([
            ((x * 17 + y * 3) % 256) as u8,
            ((x * 5 + y * 29) % 256) as u8,
            ((x * 43 + y * 7) % 256) as u8,
        ])
    });

    let mut out = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut out, 87);
    encoder.encode_image(&img).unwrap();
    out.into_inner()
}

fn image_decoder(config: serde_json::Value) -> ImageDecoder {
    serde_json::from_value(config).unwrap()
}

fn image_decoders() -> [(&'static str, ImageDecoder); 2] {
    [
        ("image_reader", image_decoder(serde_json::json!({}))),
        (
            "libjpeg_turbo",
            image_decoder(serde_json::json!({"enable_libjpeg": true})),
        ),
    ]
}

criterion_group!(benches, bench_jpeg_decode, bench_jpeg_concurrent_decode);
criterion_main!(benches);
