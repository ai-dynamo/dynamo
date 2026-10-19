// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo sidecar for NVIDIA Triton Inference Server's KServe v2 gRPC and HTTP APIs.

mod args;
mod client;
mod convert;
mod engine;
mod grpc;
mod http;
mod run;

pub use run::from_cli;
