// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local coordination for session placement state.

mod coordinator;

pub(crate) use coordinator::{
    PlacementAcquire, PlacementInitialization, PlacementLease, SessionPlacement,
    SessionPlacementConfig, SessionPlacementError, TargetGeneration,
};

#[cfg(test)]
mod tests;
