// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for vLLM's released native gRPC API.

mod args;
mod client;
mod convert;
mod engine;
mod json;
mod model;
mod proto;

pub use engine::VllmRemoteEngine;

#[cfg(test)]
mod tests;
