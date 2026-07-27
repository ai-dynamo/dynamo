// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral Dynamo sidecar for the OpenEngine gRPC contract.

mod args;
mod client;
mod convert;
mod engine;
mod kv;

pub use engine::OpenEngineSidecar;

/// Immutable OpenEngine source or BSR commit used to generate these bindings.
pub const OPENENGINE_SCHEMA_RELEASE: &str = env!("OPENENGINE_SCHEMA_RELEASE");

pub mod proto {
    tonic::include_proto!("openengine.v1");
}

#[cfg(test)]
mod tests;
