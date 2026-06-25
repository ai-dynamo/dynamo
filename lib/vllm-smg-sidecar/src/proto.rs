// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf / gRPC types for SMG's vLLM engine service.
//!
//! Codegen is driven by `build.rs` from the vendored SMG proto sources.

#![allow(clippy::all)]
#![allow(missing_docs)]

pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}

pub mod vllm {
    pub mod grpc {
        pub mod engine {
            tonic::include_proto!("vllm.grpc.engine");
        }
    }
}

pub use smg::grpc::common;
pub use vllm::grpc::engine;
