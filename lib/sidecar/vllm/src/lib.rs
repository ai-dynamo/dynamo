// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

mod args;
mod client;
mod convert;
mod engine;
mod json;
mod model;

/// Generated vLLM gRPC types, temporarily exposed for the Mocker server until
/// vLLM publishes its upstream protocol package.
#[doc(hidden)]
pub mod proto;

pub use engine::VllmSidecarEngine;

#[cfg(test)]
mod tests;
