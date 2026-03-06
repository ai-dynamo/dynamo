// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

pub mod vllm;

// Backward compatibility: re-export Scheduler from vllm module
pub use vllm::Scheduler;
