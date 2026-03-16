// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock LLM scheduler and KV manager for testing.
//!
//! This crate provides a mock implementation of an LLM scheduler that simulates
//! KV cache management, request scheduling, and token generation timing without
//! requiring actual GPU resources or a full distributed runtime.

pub mod cache;
pub mod common;
pub mod kv_manager;
pub mod scheduler;
