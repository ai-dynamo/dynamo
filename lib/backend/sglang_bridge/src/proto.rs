// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tonic-generated stubs for SGLang's gRPC schema, vendored from
//! `smg_grpc_proto-0.4.5` (the proto sgl-router uses today against upstream
//! SGLang's `--grpc-mode` server). Two proto packages:
//!
//! - `sglang.grpc.scheduler` — `SglangScheduler` service + request/response types
//! - `smg.grpc.common` — shared types (`GetTokenizer*`, `SubscribeKvEvents*`,
//!   `KvEventBatch`, `KvCacheEvent`, `KvBlocks*`)
//!
//! Module hierarchy mirrors the proto package paths so cross-package
//! references in tonic-generated code (e.g. `sglang.grpc.scheduler` referring
//! to `smg.grpc.common.GetTokenizerChunk`) resolve via `super::super::super::smg::*`.

/// Convenience re-export so callers can do `use crate::proto::scheduler::*`.
pub use sglang::grpc::scheduler;

pub mod sglang {
    pub mod grpc {
        pub mod scheduler {
            tonic::include_proto!("sglang.grpc.scheduler");
        }
    }
}

pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}
