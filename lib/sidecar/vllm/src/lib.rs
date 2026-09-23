// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for vLLM's released native gRPC API.

#[cfg(test)]
#[macro_use]
#[path = "../../testkit/tests/unit/support/mod.rs"]
mod test_support;

#[cfg(test)]
#[macro_use]
#[path = "../../testkit/tests/unit/shared.rs"]
mod test_shared;

#[cfg(test)]
#[macro_use]
#[path = "../../testkit/tests/unit/support/vllm.rs"]
mod test_vllm_support;

#[cfg(test)]
#[macro_use]
#[path = "../../testkit/tests/unit/vllm.rs"]
mod test_vllm;

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
use dynamo_sidecar_testkit::fixtures as unit_fixtures;

#[cfg(test)]
#[path = "../../testkit/tests/support/fixtures/vllm.rs"]
mod unit_vllm_fixtures;
