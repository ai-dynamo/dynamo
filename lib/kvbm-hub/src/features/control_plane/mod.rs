// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub control-plane manager.
//!
//! Forwards HTTP control requests for a registered leader to that leader's
//! velo handlers via the typed [`LeaderControlClient`]. Each route is a thin
//! axum→typed-client wrapper; the client's decode path applies
//! [`ControlError::http_status`] mapping uniformly.
//!
//! Operators hit a stable hub URL like `PUT /v1/instances/{id}/reset`; the
//! manager looks up `{id}` in the registry and dispatches the typed call.
//!
//! [`LeaderControlClient`]: kvbm_protocols::control::LeaderControlClient
//! [`ControlError::http_status`]: kvbm_protocols::control::ControlError::http_status

mod manager;

pub use manager::ControlPlaneManager;
