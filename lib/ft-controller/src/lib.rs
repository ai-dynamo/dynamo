// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! World controller for vLLM fault tolerance: one decider per world, outside
//! the GPU pods. It watches every rank's front end and recovers the world
//! through vLLM's own fault-tolerance API (`/v1/fault_tolerance/*`), or
//! escalates to the world's supervisor when it cannot. See `README.md`.
//!
//! - [`decision`]: the host-neutral decision core (no I/O, no clock).
//! - [`engine`], [`round`], [`escalate`]: the I/O behind small traits.
//! - [`controller`]: the loop that ties them together.
//! - [`protocol`]: vLLM's wire types; [`cli`]: the binary's arguments.

pub mod cli;
pub mod controller;
pub mod decision;
pub mod engine;
pub mod escalate;
pub mod protocol;
pub mod round;
