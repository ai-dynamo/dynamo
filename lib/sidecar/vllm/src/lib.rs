// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

mod args;
mod client;
mod convert;
mod engine;
mod json;
mod lora;
mod model;
mod run;

#[doc(hidden)]
pub use vllm_proto as proto;

pub use engine::VllmSidecarEngine;
pub use run::{RunError, run};

#[cfg(test)]
mod tests;
