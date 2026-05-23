// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the P2P feature.
//!
//! P2P is the layout-compatibility gate for peer-to-peer block transfers
//! between leaders. The [`P2pManager`] owns the `layout_compat` baseline —
//! the first P2P registration establishes it (after [`validate_self`]); every
//! subsequent registration is checked against the baseline via
//! [`check_layout_compat`].
//!
//! CD ([`super::disagg`]) is a specialisation of P2P: any
//! `RegisterRequest` containing `Feature::ConditionalDisagg` MUST also
//! contain `Feature::P2P` in the same request. The server enforces this
//! pre-dispatch — see `crate::server::register_instance`. Do not duplicate
//! the check inside this manager.

/// `kvbmctl p2p` action CLI for this feature. Gated behind the `kvbmctl` feature.
#[cfg(feature = "kvbmctl")]
pub mod cli;
mod client;
mod manager;

pub use client::P2pClient;
pub use manager::P2pManager;
