// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre-configured discovery profiles for common deployment patterns.
//!
//! This module provides ready-to-use discovery configurations optimized
//! for different environments and use cases.

mod worker;

pub use worker::{DevProfile, DevProfileConfig, WorkerProfile};
