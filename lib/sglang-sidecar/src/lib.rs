// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo SGLang sidecar backend.
//!
//! A [`SglangSidecarEngine`] implements [`dynamo_backend_common::LLMEngine`] by
//! proxying inference to an out-of-process SGLang engine over the OpenEngine v1
//! gRPC contract. The sidecar is **endpoint-only**: it is configured with just
//! `--openengine-endpoint` and discovers model identity, disaggregation role,
//! parallelism, KV block sizing, and context length from the engine itself.
//!
//! The crate never depends on `sglang` or any engine crate — only
//! `dynamo-backend-common`, `tonic`/`prost`, `clap`, and tokio.

pub mod args;
pub mod client;
pub mod engine;
pub mod proto;

pub use engine::SglangSidecarEngine;

#[cfg(test)]
mod tests;
