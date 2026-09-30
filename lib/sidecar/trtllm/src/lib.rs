// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC API.

mod args;
mod client;
mod convert;
mod disagg;
mod engine;
mod model;
mod proto;

pub use engine::TrtllmSidecarEngine;

#[cfg(test)]
mod tests;
