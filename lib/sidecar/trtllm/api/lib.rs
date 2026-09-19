// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for TensorRT-LLM's native `TrtllmService` gRPC API.

mod args;
#[path = "../src/client.rs"]
mod client;
#[path = "../src/convert.rs"]
mod convert;
#[path = "../src/engine.rs"]
mod engine;
#[path = "../src/model.rs"]
mod model;
mod proto;
mod startup;

pub use engine::TrtllmSidecarEngine;

#[cfg(test)]
#[path = "../src/tests.rs"]
mod tests;
