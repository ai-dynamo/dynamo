// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unsafe_code)]

use std::io::Cursor;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_llm::preprocessor::media::{Decoder, EncodedMediaData, ImageDecoder};
use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use libloading::Library;

fn require_libturbojpeg() {
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.0.dylib",
        "libturbojpeg.dylib",
    ];

    if CANDIDATES
        .iter()
        .any(|name| unsafe { Library::new(name) }.is_ok())
    {
        return;
    }

    panic!("image_decode benchmark requires libturbojpeg; install libturbojpeg0-dev or equivalent");
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

fn bench_jpeg_decode(c: &mut Criterion) {
    require_libturbojpeg();

    let jpeg = make_jpeg(2400, 1080);
    let decoders = [
        ("image_reader", image_decoder(serde_json::json!({}))),
        (
            "libjpeg_turbo",
            image_decoder(serde_json::json!({"backend": "libjpeg_turbo"})),
        ),
    ];

    let mut group = c.benchmark_group("image_decode_jpeg_2400x1080");
    group.throughput(Throughput::Bytes(jpeg.len() as u64));
    for (name, decoder) in decoders {
        group.bench_function(name, |b| {
            b.iter_batched(
                || EncodedMediaData::from_bytes(jpeg.clone()),
                |data| criterion::black_box(decoder.decode(data).unwrap()),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_jpeg_decode);
criterion_main!(benches);
