// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared KVBM wire protocols.
//!
//! This crate intentionally contains only serializable protocol data (and,
//! behind the `client` feature, the thin velo client that speaks it). It is
//! shared by the connector, engine, and hub without making any one of those
//! crates depend on another.
//!
//! - [`disagg`] — disaggregation session/request types.
//! - [`control`] — the public leader control plane: the `ControlReply`
//!   envelope, per-module request/response types, the `ModuleId` registry,
//!   and (with `--features client`) the `LeaderControlClient`.

pub mod control;
pub mod disagg;
