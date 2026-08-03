// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's native `TrtllmService` gRPC API.

mod args;
mod client;
mod convert;
mod direct;
mod direct_backend;
mod engine;
mod model;
mod proto;

pub use direct::{GrpcDispatch, TrtllmDirectDispatchProvider};
pub use direct_backend::{Launch, launch_from_env};
pub use engine::TrtllmSidecarEngine;

#[cfg(test)]
mod tests;
