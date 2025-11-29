// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration modules for external frameworks.
//!
//! This module provides trait-based abstractions for integrating with
//! external serving frameworks like vLLM, allowing pure Rust code to
//! remain independent of framework-specific types.

pub mod config;
pub mod connector;
pub mod vllm;

// Re-export key types for convenience
pub use config::{AttentionConfig, IntegrationsConfig, ParallelConfig};
