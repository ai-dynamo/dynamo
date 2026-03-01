// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backwards-compatible integration facade.
//!
//! vLLM distributed integration logic has been split into focused modules:
//! - registry ops
//! - offload planning
//! - G4 onboard shaping
//! - checksum helpers

pub use super::checksum::*;
pub use super::g4_onboard::*;
pub use super::offload_planner::*;
pub use super::registry_ops::*;
