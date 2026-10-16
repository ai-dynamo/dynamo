// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WAN gRPC transport for Relay pool publications.

mod config;
#[cfg(test)]
mod conformance;
mod identity;
mod load;
pub mod protocol;
mod server;
mod service;
mod source;

pub use config::KvDcRelayGrpcConfig;
pub(crate) use server::GrpcTransport;
