// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration Python bindings.
//!
//! Thin PyO3 shims only. The pure-Rust impls live in
//! `kvbm_connector::vllm`. See `/CLAUDE.md` for the separation rule.

pub mod cache_manager;
pub mod config;
pub mod handle;

pub use cache_manager::PyRustKvCacheManager;
pub use config::PyKvbmVllmConfig;
pub use handle::PyG1BlockManagerHandle;
