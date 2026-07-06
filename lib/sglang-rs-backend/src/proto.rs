// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tonic-generated stubs for Shepherd Model Gateway's SGLang scheduler proto.

pub use sglang::grpc::scheduler;

pub mod smg {
    pub mod grpc {
        pub mod common {
            #![allow(dead_code)]
            tonic::include_proto!("smg.grpc.common");
        }
    }
}

pub mod sglang {
    pub mod grpc {
        pub mod scheduler {
            #![allow(dead_code)]
            tonic::include_proto!("sglang.grpc.scheduler");
        }
    }
}
