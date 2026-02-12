// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API protocol types and conversion logic.
//!
//! This module provides types for the Anthropic Messages API (`/v1/messages`)
//! and conversion logic to/from the internal chat completions representation.

pub mod stream_converter;
pub mod types;

pub use types::*;
