// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, target-agnostic session placement coordination.
//!
//! Callers derive trusted placement keys, select eligible targets, and commit a target only after
//! dispatch is accepted. Target discovery, health validation, and stateless capacity fallback stay
//! in the calling routing policy.

mod coordinator;

pub use coordinator::{
    PlacementAcquire, PlacementInitialization, PlacementLease, SessionPlacement,
    SessionPlacementConfig, SessionPlacementError,
};

#[cfg(test)]
mod tests;
