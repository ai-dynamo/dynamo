// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo vLLM SMG sidecar backend.
//!
//! This crate implements [`dynamo_backend_common::LLMEngine`] by proxying
//! inference to an out-of-process upstream vLLM `--grpc` engine over SMG's
//! `vllm.grpc.engine.VllmEngine` contract. It is intentionally separate from
//! the OpenEngine sidecar so users can run against upstream vLLM without any
//! vLLM fork changes.

pub mod args;
pub mod client;
pub mod engine;
pub mod proto;

pub use engine::VllmSmgSidecarEngine;

#[cfg(test)]
mod tests;
