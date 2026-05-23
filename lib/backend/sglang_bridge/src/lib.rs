// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo bridge to SGLang's native gRPC server (`sglang.runtime.v1`).

mod args;
mod engine;
mod proto;
mod sampling;
mod server_info;

pub use engine::SglangBridge;
