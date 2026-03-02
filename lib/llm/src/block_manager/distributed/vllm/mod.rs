// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration for distributed block management.
//!
//! Provides:
//! - TP-aware registry operations (consensus lookup, multi-rank registration)
//! - Offload flow orchestration (D2H, H2O)
//! - Transfer pipeline construction

mod checksum;
mod g4_onboard;
mod integration;
mod leader_core;
mod leader_utils;
mod offload_planner;
mod registry_ops;
mod slot_api;
mod slot_config;
mod slot_ops;
mod slot_runtime;
mod slot_support;
mod transfer_engine;
mod transfer_types;

pub use integration::*;
pub use leader_core::*;
pub use leader_utils::*;
pub use slot_api::*;
pub use slot_config::*;
pub use slot_ops::*;
pub use slot_runtime::*;
pub use slot_support::*;
pub use transfer_engine::*;
pub use transfer_types::*;
