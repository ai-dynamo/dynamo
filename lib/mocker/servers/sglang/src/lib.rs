// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only SGLang gRPC test server backed by the Dynamo Mocker scheduler.

mod server;
mod unimplemented_rpcs;

pub use server::{MockerServerConfig, ServerMode, SglangMockerService};
#[doc(hidden)]
pub use unimplemented_rpcs::__deps;
