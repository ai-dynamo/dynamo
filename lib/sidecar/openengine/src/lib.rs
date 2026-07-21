// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral Dynamo sidecar for the OpenEngine gRPC contract.

mod args;
mod client;
mod convert;
mod engine;
mod kv;

pub use engine::OpenEngineSidecar;

/// Immutable OpenEngine source commit checked by build.rs for the local
/// development dependency.
pub const OPENENGINE_PROTO_COMMIT: &str = env!("OPENENGINE_PROTO_COMMIT");

pub mod proto {
    pub use openengine_proto::openengine::v1::*;
}

#[cfg(test)]
mod tests;
