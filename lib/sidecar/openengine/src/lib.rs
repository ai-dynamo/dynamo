// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for the OpenEngine gRPC contract.

mod args;
mod client;
mod convert;
mod engine;
mod proto;

pub use engine::OpenEngineSidecarEngine;

/// Immutable OpenEngine schema used to generate this client.
pub const OPENENGINE_SCHEMA_RELEASE: &str = "768a93c7b44e40f28c692ad0b471a8f2";
