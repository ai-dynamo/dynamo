<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored `llm-multimodal` 1.9.0 (AAC decoder removed)

This directory is a copy of the published
[`llm-multimodal` 1.9.0](https://crates.io/crates/llm-multimodal/1.9.0) crate
(Apache-2.0, upstream repository <https://github.com/lightseekorg/smg>).

Dynamo uses it through the `mm-routing` feature of `dynamo-llm`. Cargo
features are additive, so Dynamo cannot switch off a Symphonia codec that a
dependency enables. Upstream `llm-multimodal` depends on `symphonia` with
`features = ["all"]`, which includes the AAC decoder (`symphonia-codec-aac`).
Dynamo patches this crate in through `[patch.crates-io]` so that
`mm-routing` stays enabled while the Symphonia AAC decoder is not built.

## Differences from the published crate

- `Cargo.toml`, `symphonia` dependency: `features = ["all"]` is replaced by the
  same codec, format and metadata features minus `aac`.
- `Cargo.toml`: the `[[test]]` and `[[bench]]` targets are removed because
  `tests/`, `benches/` and `scripts/` are not vendored.
- `Cargo.toml`: a `[package.metadata.cargo-machete]` section ignores
  `serde_bytes`, which upstream declares but does not use from `src/`. Without
  it, the repository's `cargo machete` CI check fails on the vendored path crate.
- Not vendored: `tests/`, `benches/`, `scripts/`, `Cargo.lock`,
  `Cargo.toml.orig`, `.cargo_vcs_info.json`.
- `build.rs` and `src/` are unchanged.

## Effect

- Symphonia cannot decode AAC audio in-process. An AAC clip returns an audio
  decode error. WAV, FLAC, Ogg/Vorbis, MP3, ALAC, ADPCM and PCM are unchanged.
- Image preprocessing and MM-aware KV routing are unchanged.

## Updating

Keep this copy in sync with the version pinned by `lib/llm/Cargo.toml`
(`llm-multimodal = "=1.9.0"`). When that pin changes, re-vendor the matching
published crate and reapply the `symphonia` feature change above.
