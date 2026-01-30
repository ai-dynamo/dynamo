// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LORA (Low-Rank Adaptation) utilities

pub mod rendezvous_hash;

pub use rendezvous_hash::{compute_replica_factor, compute_replica_set, RendezvousHasher};
