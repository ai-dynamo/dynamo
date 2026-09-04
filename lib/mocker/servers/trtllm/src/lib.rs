// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker-backed TensorRT-LLM OpenEngine gRPC service.

mod server;

pub use server::{MockerServerConfig, ServerMode, TrtllmMockerService};
