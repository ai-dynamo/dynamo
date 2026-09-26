// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

mod args;
#[path = "../src/client.rs"]
mod client;
#[path = "../src/convert.rs"]
mod convert;
#[path = "../src/engine.rs"]
mod engine;
#[path = "../src/json.rs"]
mod json;
#[path = "../src/lora.rs"]
mod lora;
#[path = "../src/model.rs"]
mod model;
mod startup;

#[doc(hidden)]
pub use vllm_proto as proto;

pub use engine::VllmSidecarEngine;

#[cfg(test)]
#[path = "../src/tests.rs"]
mod tests;
