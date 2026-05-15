// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service-impl halves of the opt-in / data-path control modules.
//!
//! Each submodule pairs with a `kvbm_protocols::control::modules` protocol
//! slice. The always-on `core` and opt-in `dev` modules live one level up
//! in [`super::core`] / [`super::dev`].
//!
//! - `metrics` — on-demand runtime snapshot (opt-in via `control.metrics`).
//! - `test` — test-only G2 block helpers (opt-in via `control.test`).
//! - `transfer` — G2 search → disagg-session creation (always-on).

pub mod metrics;
pub mod test;
pub mod transfer;
