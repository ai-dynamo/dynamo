// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for upstream SGLang's SMG scheduler service.

pub mod args;
pub mod client;
pub mod engine;
pub mod proto;

pub use engine::SglangSmgSidecarEngine;

#[cfg(test)]
mod tests;
