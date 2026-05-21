// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Async search machinery (parked)
//!
//! This module homes the salvaged G4/object-storage search machinery extracted
//! from the now-removed `leader::session` peer protocol. It is intentionally
//! **unwired** — nothing in the engine calls it yet. It exists to seed the
//! upcoming async remote-search refactor, which will own orchestration and
//! re-enable G4 lookups.
//!
//! The G4 search path never used the peer `MessageTransport`; it works purely
//! through `ParallelWorkers::{has_blocks, get_blocks}` and the local G2
//! `BlockManager`. The internal `OnboardMessage` channel variants were replaced
//! with a private [`g4::G4Message`] enum since the old message protocol is gone.
#![allow(dead_code)]

mod g4;

pub use g4::{AsyncSearch, G4SearchState};
