// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Nova
//!
//! The `dynamo-nova` crate is a Rust library that provides a set of traits and types for building
//! distributed applications. Goal will be to interoperate with Realm for events and tasking in the future.

pub mod am;
pub mod events;

pub use am::Nova;
pub use events::EventHandle;

pub use dynamo_nova_backend::{PeerInfo, WorkerAddress};
