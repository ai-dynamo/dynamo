// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for the LPX engines' OpenEngine (`openengine.v1`) gRPC
//! servers.
//!
//! A hybrid GPU+LPU engine serves two roles behind this sidecar: aggregated,
//! and the decode half of prefill/decode disaggregation, where a vLLM worker
//! runs prefill and the engine pulls its KV cache over NIXL. An LPU-only
//! engine serves aggregated requests. When one engine spans several Pods,
//! only the sidecar on rank 0 serves; the others stand by.

mod args;
mod client;
mod convert;
mod engine;
mod handoff;
mod model;
mod standby;

/// The generated `openengine.v1` types, reused from the TensorRT-LLM sidecar.
///
/// Both sidecars speak the same vendored contract, and one set of generated
/// types lets the TensorRT-LLM Mocker server stand in for the LPX engines in
/// tests without conversions. Move to a standalone `openengine` protocol crate
/// once one exists.
pub(crate) use dynamo_trtllm_sidecar::proto as pb;

pub use engine::{Launch, LpxSidecarEngine};
pub use standby::run_standby;

#[cfg(test)]
mod tests;
