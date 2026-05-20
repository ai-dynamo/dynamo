// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar bridge to SGLang's native gRPC server
//! (`sglang.runtime.v1`, enabled by `sglang.launch_server --enable-grpc`).
//!
//! Registers as a normal Dynamo worker; the frontend's KvRouter and
//! PrefillRouter treat it as any other network worker.

pub mod engine;
pub mod proto;

pub use engine::{Args, SglangBridge};
