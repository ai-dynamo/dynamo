// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal synchronization primitives for the event system.
//!
//! This module contains the racy slot machinery used by event entries.
//! These types are frozen — do not modify.

mod active;
mod completion;
pub(crate) mod entry;
mod waiter;

pub(crate) use completion::{CompletionKind, PoisonArc, WaitRegistration};
pub(crate) use entry::{EventEntry, EventKey};
pub use waiter::EventAwaiter;
