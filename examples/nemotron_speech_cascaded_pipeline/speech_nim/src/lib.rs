// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example-local speech adapters. Backend RPCs and Dynamo serving contracts
//! are separate modules so the crate can later move to `lib/sidecar`.

pub mod asr;
pub mod client;
pub mod config;
mod events;
pub mod tts;
pub mod worker;

/// Bindings generated from the pinned upstream Riva protocol.
pub mod proto {
    tonic::include_proto!("nvidia.riva");
    pub mod asr {
        tonic::include_proto!("nvidia.riva.asr");
    }
    pub mod tts {
        tonic::include_proto!("nvidia.riva.tts");
    }
}
