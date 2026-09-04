// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC API.

mod args;
mod client;
mod convert;
/// Disaggregation wire contract, shared with the Mocker server so the two do
/// not carry separate copies of the same keys.
#[doc(hidden)]
pub mod disagg;
mod engine;
mod model;

/// Generated OpenEngine gRPC types, exposed for the Mocker server until
/// `openengine.v1` is published as a standalone protocol crate.
#[doc(hidden)]
pub mod proto;

pub use engine::TrtllmSidecarEngine;

#[cfg(test)]
mod tests;
