// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo vLLM sidecar backend.
//!
//! A [`VllmSidecarEngine`] implements [`dynamo_backend_common::LLMEngine`] by
//! proxying inference to an out-of-process vLLM engine over the engine RPC v1
//! gRPC contract. The sidecar is **endpoint-only**: it is configured with just
//! `--engine-rpc-endpoint` and discovers model identity, disaggregation role,
//! parallelism, KV block sizing, and context length from the engine itself.
//!
//! The crate never depends on `vllm` or any engine crate — only
//! `dynamo-backend-common`, `tonic`/`prost`, `clap`, and tokio.

pub mod args;
pub mod client;
mod discovery;
pub mod engine;
pub mod proto;
mod request;
mod wire;

pub use engine::VllmSidecarEngine;

#[cfg(test)]
mod tests;
