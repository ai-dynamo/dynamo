// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-plugin protocol slices for the leader control plane.
//!
//! Each submodule is the protocol+client half of one togglable control
//! module; the matching service-impl half lives in
//! `kvbm-engine::leader::control::modules`. The shared identity is
//! [`super::ModuleId`].

pub mod core;
pub mod dev;
pub mod metrics;
pub mod transfer;
