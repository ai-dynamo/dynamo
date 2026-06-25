// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf / gRPC types for SGLang's SMG scheduler service.

#![allow(clippy::all)]
#![allow(missing_docs)]

pub mod sglang {
    pub mod grpc {
        pub mod scheduler {
            tonic::include_proto!("sglang.grpc.scheduler");
        }
    }
}

pub use sglang::grpc::scheduler;
