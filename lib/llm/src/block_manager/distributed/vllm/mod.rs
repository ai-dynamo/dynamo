// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration for distributed block management.
//!
//! Provides:
//! - TP-aware registry operations (consensus lookup, multi-rank registration)
//! - Offload flow orchestration (D2H, H2O)
//! - Transfer pipeline construction

mod integration;

pub use integration::*;

