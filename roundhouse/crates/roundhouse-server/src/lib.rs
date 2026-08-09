// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The turn engine: the composition root where state, context, routing, and
//! execution meet.
//!
//! One turn is: claim the session, replay it into a context assembler, admit
//! the turn, price every target, choose, record the choice, execute, and settle.
//! Each of those steps lives in a lower layer; this module only sequences them.
//!
//! The sequencing itself carries two guarantees worth stating. The routing
//! decision is written to the log *before* execution starts, so the audit trail
//! records what was chosen even when execution then fails. And a booked local
//! reservation is always settled, including on the error path, because a leaked
//! reservation silently inflates the router's view of a worker forever.

pub mod engine;

pub use engine::{
    EchoLocalExecutor, Engine, EngineConfig, EngineError, LocalExecution, LocalExecutor, TurnResult,
};
