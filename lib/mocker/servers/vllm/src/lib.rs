// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only vLLM gRPC test server backed by the Dynamo Mocker scheduler.

mod server;

pub use server::{MockerServerConfig, ServerMode, VllmMockerService};
