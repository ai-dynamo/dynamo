// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Speech NIM adapters for realtime transcription and streaming synthesis.
//! ASR and TTS share gRPC transport and lifecycle configuration.

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
