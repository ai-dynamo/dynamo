// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One ordered KV cache event stream, kept as what a consumer that joins late needs: the most
//! recent batches, and a block ledger that a copy of the state is cut from.
//!
//! The host is whatever reads an engine's KV events: Dynamo's `LocalKvIndexer`, or an engine's
//! own Rust frontend. It owns the transport and the wire types. This crate owns the state
//! behind them:
//!
//! | The host serves | It calls |
//! |---|---|
//! | the live stream | [`Replay::apply`], then publishes the batch it passed in |
//! | replay from a sequence | [`Replay::since`] |
//! | a copy of the state, then the batches after it | [`Replay::cut`] and [`Replay::tail`] |
//!
//! [`Replay`] folds each batch into a state the host chooses. [`Ledger`] is that state for the
//! blocks held in one place: it is keyed by the engine's own block hash, so nothing is hashed
//! here.
//!
//! Everything is synchronous and single-owner. There are no threads, locks, channels or async,
//! and no Dynamo types: the crate depends on `rustc-hash` only, so an engine can build it with
//! its own toolchain.

mod history;
mod ledger;
mod replay;

#[cfg(test)]
mod tests;

pub use ledger::{Block, Chain, Ledger};
pub use replay::{Applied, Effect, Replay};
