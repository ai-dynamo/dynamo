// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the vLLM Rust backend.
//!
//! The engine implementation depends on vLLM's engine-core crates, which are
//! only available as git dependencies and are gated behind the `vllm_rs`
//! feature. Without the feature the crate builds to a stub so the default
//! workspace build does not require the git sources.

#[cfg(feature = "vllm_rs")]
mod backend;
#[cfg(feature = "vllm_rs")]
mod control;
#[cfg(feature = "vllm_rs")]
mod convert;
#[cfg(feature = "vllm_rs")]
mod error;

#[cfg(feature = "vllm_rs")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;

    let (engine, config) = backend::VllmBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}

#[cfg(not(feature = "vllm_rs"))]
fn main() {
    eprintln!(
        "dynamo-vllm-rs-backend was built without the `vllm_rs` feature. \
         Rebuild with `--features vllm_rs` to enable the vLLM engine-core backend."
    );
    std::process::exit(1);
}
