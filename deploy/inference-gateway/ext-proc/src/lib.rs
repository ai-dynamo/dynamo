// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc gRPC server for Dynamo inference routing.
//!
//! This crate implements the Envoy `ExternalProcessor.Process` bidirectional
//! streaming RPC, using Dynamo's native Rust KV-aware router directly
//! (no CGO/FFI boundary). It replaces the Go EPP ext_proc server path
//! while keeping the same Envoy wire protocol.

pub mod envoy_helpers;
pub mod proto;
pub mod router;
pub mod server;

pub use server::ExtProcServer;
