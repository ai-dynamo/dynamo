// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-lock synchronization primitives for the event system.
//!
//! All per-entry state — generation tracking, completion status, and waker
//! registration — is consolidated under a single `parking_lot::Mutex`,
//! eliminating stale-completion races by construction.
//!
//! See `docs/slot-state-machine.md` for the formal state machine specification.

mod completion;
pub(crate) mod entry;
mod waiter;

pub(crate) use completion::{CompletionKind, PoisonArc, WaitRegistration};
pub(crate) use entry::{EventEntry, EventKey};
pub use waiter::EventAwaiter;
