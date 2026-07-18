// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust types for vLLM's released native gRPC API.
//!
//! The protocol is vendored unchanged so Dynamo's vLLM sidecar and its
//! Mocker-backed test server compile against exactly the same wire contract.

#![allow(clippy::all)]
#![allow(missing_docs)]

tonic::include_proto!("vllm");
