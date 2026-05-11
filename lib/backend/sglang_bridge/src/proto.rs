// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tonic-generated stubs for SGLang's native gRPC schema
//! (`sglang.runtime.v1`, defined in upstream `proto/sglang/runtime/v1/sglang.proto`).
//!
//! Single proto package; everything lives under `sglang::runtime::v1`.

/// Convenience re-export so callers can do `use crate::proto::v1::*`.
pub use sglang::runtime::v1;

pub mod sglang {
    pub mod runtime {
        pub mod v1 {
            tonic::include_proto!("sglang.runtime.v1");
        }
    }
}
