// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo bridge to SGLang's native gRPC server (`sglang.runtime.v1`).
//! Runs as the `dynamo-sglang-bridge` sidecar binary, or in-process under
//! the `dynamo.sglang_grpc` Python supervisor via PyO3.

pub mod engine;
pub mod proto;

pub use engine::{Args, SglangBridge};
