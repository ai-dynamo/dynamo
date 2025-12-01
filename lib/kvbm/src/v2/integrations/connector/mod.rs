// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector module shared across framework integrations.
//!
//! The shared slot state machine and transfer planning logic live in `slot`.
//! Framework-specific leaders (vLLM, etc.) can build on top of these pieces
//! while supplying their own scheduling semantics.

// pub mod slot;
// pub use slot::*;

pub mod coordinator;
pub mod leader;
pub mod worker;

pub use coordinator::{MockCoordinator, TransferCoordinator};

pub mod metadata;
pub use metadata::ConnectorMetadataBuilder;

pub use crate::v2::{G1, G2, G3};
