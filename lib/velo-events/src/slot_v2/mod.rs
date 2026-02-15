// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-lock synchronization primitives for the event system (v2).
//!
//! This module replaces the original `slot` module, eliminating the
//! stale-completion race (Race 1) by consolidating all per-entry state —
//! generation tracking, completion status, and waker registration — under
//! a single `parking_lot::Mutex`.
//!
//! See `docs/slot-state-machine.md` for the formal state machine specification.

mod completion;
pub(crate) mod entry;
mod waiter;

pub(crate) use completion::{CompletionKind, PoisonArc, WaitRegistration};
pub(crate) use entry::{EventEntry, EventKey};
pub use waiter::EventAwaiter;
