// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU Memory Service — out-of-process CUDA VMM memory manager.
//!
//! Enables zero-copy GPU memory sharing across processes, crash survival,
//! and model weight sharing for inference engines.
//!
//! # Architecture
//!
//! - **Server**: Manages physical GPU memory allocations via CUDA VMM.
//!   Never maps memory to virtual addresses — survives GPU driver restarts.
//! - **Client**: Connects to the server, acquires locks (RW or RO),
//!   imports allocations via FD passing, and maps them locally.
//! - **State Machine**: Enforces RW/RO lock semantics. Single writer,
//!   multiple readers. Writers must commit before readers can connect.
//! - **Failover Lock**: Process-level mutual exclusion via `flock(2)`.
//!   Automatically released on process death.
//!
//! # Protocol
//!
//! Uses MessagePack-encoded messages over Unix domain sockets with
//! SCM_RIGHTS for file descriptor passing (CUDA VMM handles).

pub(crate) mod ancillary;
pub mod client;
pub mod error;
pub mod failover;
pub mod protocol;
pub mod server;
pub mod state;

pub use error::{GmsError, GmsResult};
pub use protocol::{GrantedLockType, RequestedLockType};
pub use state::{ServerState, StateEvent};

pub use client::GmsClient;
pub use failover::FlockFailoverLock;
pub use server::GmsServer;
