// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration module.
//!
//! Provides trait-based abstractions for vLLM configuration and integration.

pub mod config;
pub mod layout;

pub mod connector;
// pub mod scheduler;

pub use config::{KvbmVllmConfig, VllmAttentionConfig, VllmParallelConfig};
