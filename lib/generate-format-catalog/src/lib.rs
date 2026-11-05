// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default catalog of client-facing `/generate` formats.
//!
//! Backend maintainers own each implementation in `lib/generate-formats/`.
//! A custom frontend replaces this dependency in `lib/bindings/python/Cargo.toml`.
//! Its catalog can depend on this crate under an alias, call [`register`], then
//! add formats under distinct names. The frontend installs the resulting registry.
//!
//! `DYN_GENERATE_FORMAT` selects `/generate`'s format (default `sglang`).
//! `/inference/v1/generate` selects `vllm`. Enable engine APIs to expose the routes.
//! Unknown names and duplicate registrations fail at initialization.

use dynamo_llm::generate_format::Registry;
use std::sync::Arc;

/// Register the built-in formats. Custom catalogs may extend this registry.
pub fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(Arc::new(dynamo_sglang_generate::SglangFormat::native()))?;
    registry.register(Arc::new(dynamo_vllm_generate::VllmFormat))?;
    Ok(())
}
