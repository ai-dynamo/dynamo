// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake Store G4 backend for KVBM block object storage.
//!
//! See [`client`] for the MooncakeObjectBlockClient implementing ObjectBlockOps,
//! and [`lock`] for the optimistic lock manager.

pub mod client;
pub mod lock;
