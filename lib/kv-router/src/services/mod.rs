// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone services built from brokerless transport primitives.

#[cfg(any(
    feature = "standalone-indexer",
    feature = "standalone-slot-tracker",
    feature = "standalone-selection"
))]
pub(crate) mod zmq;

#[cfg(feature = "standalone-indexer")]
pub mod indexer;

#[cfg(feature = "standalone-indexer")]
pub mod shared_cache;

#[cfg(feature = "standalone-selection")]
pub mod selection;

#[cfg(feature = "standalone-slot-tracker")]
pub mod slot_tracker;
