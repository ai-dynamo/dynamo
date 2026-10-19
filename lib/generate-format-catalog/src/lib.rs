// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The empty catalog used when a custom image does not replace Dynamo's
//! `/generate` format catalog slot.
//!
//! To add a client format, replace this crate's dependency path in
//! `lib/bindings/python/Cargo.toml` and register an `Arc<dyn GenerateFormat>`
//! below. Implement `ClientCodec` with `SglangFormat::with_codec` to reuse
//! SGLang's joining and migration, or implement `GenerateFormat` and
//! `GenerateExchange` for another engine contract.
//!
//! Enable the `/generate` route with `DYN_SGLANG_ENABLE_GENERATE=1` (or
//! `--enable-engine-apis`) and select the registered name with
//! `DYN_GENERATE_FORMAT`. The default is `sglang`. The vLLM
//! `/inference/v1/generate` route always uses the built-in `vllm` format.
//! Unknown names and duplicate registrations fail at initialization.

use dynamo_llm::generate_format::Registry;

/// Register the formats linked into this image.
///
/// Custom catalogs replace this crate under the same dependency name in the
/// frontend extension. Worker parsing stays in the backend sidecars. Dynamo's
/// own formats are registered first, so a catalog cannot shadow one.
pub fn register(_registry: &mut Registry) -> anyhow::Result<()> {
    Ok(())
}
