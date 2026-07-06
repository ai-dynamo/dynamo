// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo SGLang backend for SMG's scheduler gRPC service.

mod args;
mod engine;
mod proto;
mod sampling;
mod server_info;

pub use engine::SglangBackend;
