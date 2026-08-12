// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's native `TrtllmService` gRPC API.

mod args;
mod client;
mod convert;
mod engine;
mod model;
mod proto;

pub use engine::TrtllmSidecarEngine;

#[cfg(test)]
mod tests;
