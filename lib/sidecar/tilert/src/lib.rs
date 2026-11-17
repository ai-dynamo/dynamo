// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Dynamo sidecar for TileRT: one vLLM prefill, reached over
//! vLLM's native gRPC API, and one TileRT decode server, reached over HTTP,
//! registered with Dynamo as a single worker.

mod args;
mod decode;
mod engine;
mod error;
mod prefill;

pub use engine::TilertSidecarEngine;

#[cfg(test)]
mod tests;
