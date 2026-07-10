// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf / gRPC types for the `openengine.v1` package.
//!
//! Codegen is driven by `build.rs` from the vendored `proto/openengine.proto`.

#![allow(clippy::all)]
#![allow(missing_docs)]

tonic::include_proto!("openengine.v1");

pub mod prime_rl {
    tonic::include_proto!("prime_rl.engine.v1");
}
