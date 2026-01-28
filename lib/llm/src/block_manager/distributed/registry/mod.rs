// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed registry for KV cache block deduplication.
//!
//! Provides a pluggable architecture for tracking KV blocks across workers.

pub mod config;
pub mod core;

pub use config::{RegistryClientConfig, RegistryHubConfig};
pub use core::*;
