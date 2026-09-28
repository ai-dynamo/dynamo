// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

#[cfg(test)]
#[macro_use]
#[path = "../../testkit/src/lanes.rs"]
mod test_support;

mod args;
mod client;
mod convert;
mod engine;
mod json;
mod lora;
mod model;

#[doc(hidden)]
pub use vllm_proto as proto;

pub use engine::VllmSidecarEngine;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_fixtures;
